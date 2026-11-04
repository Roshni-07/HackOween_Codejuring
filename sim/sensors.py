# sim/sensors.py

import io

import math

import numpy as np

import pybullet as p

from PIL import Image


class SensorSuite:

    def __init__(self, engine):

        self.engine = engine

        self.width = 640

        self.height = 480

        self.aspect = self.width / self.height

       

        # 73.74 deg vertical FOV yields exactly 90.0 deg horizontal FOV at 4:3 aspect ratio

        self.fov_vert = 73.74

        self.proj_matrix = p.computeProjectionMatrixFOV(

            fov=self.fov_vert, aspect=self.aspect, nearVal=0.1, farVal=10.0

        )



    def get_camera_image(self) -> bytes:

        """Renders 640x480 JPEG frame."""

        rad = math.radians(self.engine.robot_heading)

        cam_height = 0.5

        tilt_rad = math.radians(10)



        dx = math.cos(rad)

        dy = math.sin(rad)

        dz = -math.tan(tilt_rad)



        cam_pos = [self.engine.robot_pos[0], self.engine.robot_pos[1], cam_height]

        cam_target = [cam_pos[0] + dx, cam_pos[1] + dy, cam_pos[2] + dz]

        cam_up = [0, 0, 1]



        view_matrix = p.computeViewMatrix(cam_pos, cam_target, cam_up)



        _, _, rgb_img, _, _ = p.getCameraImage(

            self.width,

            self.height,

            viewMatrix=view_matrix,

            projectionMatrix=self.proj_matrix,

            renderer=p.ER_TINY_RENDERER,

            flags=p.ER_NO_SEGMENTATION_MASK

        )



        frame = np.reshape(rgb_img, (self.height, self.width, 4))[:, :, :3]

        img = Image.fromarray(frame.astype("uint8"), "RGB")

        buf = io.BytesIO()

        img.save(buf, format="JPEG", quality=90)

        return buf.getvalue()



    def get_depth_ray(self, pitch_deg=10.0, max_dist=10.0):

        """Single depth ray straight ahead. Filters out ground hits (z <= 0.01m)."""

        rad = math.radians(self.engine.robot_heading)

        pitch_rad = math.radians(pitch_deg)



        # Start 0.26m forward to clear capsule radius (0.25m)

        start = [

            self.engine.robot_pos[0] + 0.26 * math.cos(rad),

            self.engine.robot_pos[1] + 0.26 * math.sin(rad),

            0.5

        ]

        end = [

            start[0] + max_dist * math.cos(rad) * math.cos(pitch_rad),

            start[1] + max_dist * math.sin(rad) * math.cos(pitch_rad),

            start[2] - max_dist * math.sin(pitch_rad)

        ]



        res = p.rayTest(start, end)[0]

        hit_frac, hit_pos = res[2], res[3]



        if hit_frac < 1.0:

            if hit_pos[2] <= 0.01:

                return None

            return round(float(hit_frac * max_dist), 2)

        return None



    def get_low_bump_ray(self, max_dist=2.0):

        """Low horizontal range ray (z=0.15m) for close target detection."""

        rad = math.radians(self.engine.robot_heading)

        start = [

            self.engine.robot_pos[0] + 0.26 * math.cos(rad),

            self.engine.robot_pos[1] + 0.26 * math.sin(rad),

            0.15

        ]

        end = [

            start[0] + max_dist * math.cos(rad),

            start[1] + max_dist * math.sin(rad),

            0.15

        ]

        res = p.rayTest(start, end)[0]

        return round(float(res[2] * max_dist), 2) if res[2] < 1.0 else None



    def get_depth_fan(self, angles=[-30, -15, 0, 15, 30], max_dist=10.0):

        """Multi-ray depth fan across camera FOV."""

        results = []

        for angle in angles:

            rad = math.radians(self.engine.robot_heading + angle)

            pitch_rad = math.radians(10)

            start = [

                self.engine.robot_pos[0] + 0.26 * math.cos(rad),

                self.engine.robot_pos[1] + 0.26 * math.sin(rad),

                0.5

            ]

            end = [

                start[0] + max_dist * math.cos(rad) * math.cos(pitch_rad),

                start[1] + max_dist * math.sin(rad) * math.cos(pitch_rad),

                start[2] - max_dist * math.sin(pitch_rad)

            ]

            res = p.rayTest(start, end)[0]

            hit_frac, hit_pos = res[2], res[3]

            dist = round(float(hit_frac * max_dist), 2) if (hit_frac < 1.0 and hit_pos[2] > 0.01) else None

            results.append({"angle_offset": angle, "distance": dist})

        return results 