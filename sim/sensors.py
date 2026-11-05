"""
sim/sensors.py — the robot's forward RGB camera and single-ray depth sensor.
"""

import io
import math

import numpy as np
import pybullet as p
from PIL import Image

from .scene import CAMERA_BACKGROUND_HORIZON, CAMERA_BACKGROUND_TOP, CAMERA_LIGHTING

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
ASPECT = IMAGE_WIDTH / IMAGE_HEIGHT
HFOV_DEG = 90.0
VFOV_DEG = math.degrees(2.0 * math.atan(math.tan(math.radians(HFOV_DEG / 2.0)) / ASPECT))
CAMERA_HEIGHT_M = 0.5
CAMERA_FORWARD_OFFSET_M = 0.12
CAMERA_TILT_DEG = 10.0
NEAR_PLANE = 0.05
FAR_PLANE = 12.0
DEPTH_MAX_M = 10.0
JPEG_QUALITY = 90


def heading_vector(heading_deg):
    h = math.radians(heading_deg)
    return math.cos(h), -math.sin(h)


def camera_pose(x, y, heading_deg, yaw_offset_deg=0.0):
    fx, fy = heading_vector(heading_deg)
    ex = x + CAMERA_FORWARD_OFFSET_M * fx
    ey = y + CAMERA_FORWARD_OFFSET_M * fy
    eye = (ex, ey, CAMERA_HEIGHT_M)
    dx, dy = heading_vector(heading_deg + yaw_offset_deg)
    t = math.radians(CAMERA_TILT_DEG)
    view_dir = (math.cos(t) * dx, math.cos(t) * dy, -math.sin(t))
    up = (math.sin(t) * dx, math.sin(t) * dy, math.cos(t))
    return eye, view_dir, up


class RobotCamera:
    def __init__(self, lighting=None):
        self.lighting = dict(CAMERA_LIGHTING if lighting is None else lighting)
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=VFOV_DEG, aspect=ASPECT, nearVal=NEAR_PLANE, farVal=FAR_PLANE
        )
        rows = np.linspace(0.0, 1.0, IMAGE_HEIGHT, dtype=np.float32)[:, None]
        top = np.array(CAMERA_BACKGROUND_TOP, dtype=np.float32)
        horizon = np.array(CAMERA_BACKGROUND_HORIZON, dtype=np.float32)
        grad = top + (horizon - top) * np.clip(rows / 0.6, 0.0, 1.0)
        self._background = np.repeat(grad[:, None, :], IMAGE_WIDTH, axis=1).astype(np.uint8)

    def view_matrix(self, x, y, heading_deg):
        eye, d, up = camera_pose(x, y, heading_deg)
        target = (eye[0] + d[0], eye[1] + d[1], eye[2] + d[2])
        return p.computeViewMatrix(cameraEyePosition=eye, cameraTargetPosition=target, cameraUpVector=up)

    def render_rgb(self, x, y, heading_deg):
        result = p.getCameraImage(
            IMAGE_WIDTH,
            IMAGE_HEIGHT,
            viewMatrix=self.view_matrix(x, y, heading_deg),
            projectionMatrix=self.projection_matrix,
            renderer=p.ER_TINY_RENDERER,
            flags=p.ER_NO_SEGMENTATION_MASK,
            **self.lighting
        )
        width, height, rgb, depth = result[0], result[1], result[2], result[3]
        frame = np.reshape(np.asarray(rgb, dtype=np.uint8), (height, width, 4))[:, :, :3].copy()
        depth = np.reshape(np.asarray(depth, dtype=np.float32), (height, width))
        empty = (~np.isfinite(depth)) | (depth >= 0.99999) | (depth < -1.0)
        if empty.any() and frame.shape == self._background.shape:
            frame[empty] = self._background[empty]
        return frame

    def get_camera_image(self, x, y, heading_deg):
        img = Image.fromarray(self.render_rgb(x, y, heading_deg), "RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY)
        return buf.getvalue()


class DepthSensor:
    def __init__(self, ignored_bodies_fn):
        self._ignored = ignored_bodies_fn

    def get_depth_ray(self, x, y, heading_deg, yaw_offset_deg=0.0, max_dist=DEPTH_MAX_M):
        eye, d, _ = camera_pose(x, y, heading_deg, yaw_offset_deg)
        ignored = set(self._ignored())
        start = eye
        for _ in range(8):
            end = (start[0] + d[0] * max_dist, start[1] + d[1] * max_dist, start[2] + d[2] * max_dist)
            hit = p.rayTest(start, end)[0]
            body, fraction, pos = hit[0], hit[2], hit[3]
            if body < 0 or fraction >= 1.0:
                return None
            if body in ignored:
                start = (pos[0] + d[0] * 0.02, pos[1] + d[1] * 0.02, pos[2] + d[2] * 0.02)
                continue
            dist = math.sqrt(sum((pos[i] - eye[i]) ** 2 for i in range(3)))
            return round(float(dist), 2) if dist <= max_dist else None
        return None

    def get_depth_fan(self, x, y, heading_deg, angles=(-30, -15, 0, 15, 30), max_dist=DEPTH_MAX_M):
        return [
            {"angle_offset": a, "distance": self.get_depth_ray(x, y, heading_deg, a, max_dist)}
            for a in angles
        ]


# Compatibility wrapper for preview.py
class SensorSuite:
    def __init__(self, engine_or_lighting=None, lighting=None):
        if isinstance(engine_or_lighting, (dict, list, tuple)):
            lighting = engine_or_lighting
            engine_or_lighting = None
        self.camera = RobotCamera(lighting)
        self.engine = engine_or_lighting

    def get_camera_image(self, x, y, heading_deg):
        return self.camera.get_camera_image(x, y, heading_deg)

    def get_depth_ray(self, x, y, heading_deg, max_dist=10.0):
        # Fallback helper if needed directly
        return None