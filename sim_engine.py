import math
import time
import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image


class SimulationEngine:

    def __init__(self, gui=True):
        mode = p.GUI if gui else p.DIRECT
        self.physics_client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        self.room_size = 5.0
        self.robot_pos = [0.0, 0.0, 0.25]
        self.robot_heading = 0.0  # Degrees (0 = facing +X)
        self.state_name = "Idle"
        self.held_object_id = None
        self.collided = False
        self.retries = 0

        self._build_room()
        self._spawn_shapes_and_dropzone()
        self._create_robot()

    def _build_room(self):
        """Build floor plane and 4 surrounding arena walls[cite: 2]."""
        p.loadURDF("plane.urdf")
        wall_height, wall_thick = 1.0, 0.1
        half_r = self.room_size / 2.0

        walls = [
            [0, half_r, half_r, wall_thick],
            [0, -half_r, half_r, wall_thick],
            [half_r, 0, wall_thick, half_r],
            [-half_r, 0, wall_thick, half_r],
        ]
        for wx, wy, hx, hy in walls:
            shape = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[hx, hy, wall_height / 2.0]
            )
            vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[hx, hy, wall_height / 2.0],
                rgbaColor=[0.7, 0.7, 0.7, 1],
            )
            p.createMultiBody(
                0, shape, vis, basePosition=[wx, wy, wall_height / 2.0]
            )

    def _spawn_shapes_and_dropzone(self):
        """Spawn green drop zone platform and primitive objects[cite: 2]."""
        self.drop_zone_pos = [1.8, 1.8, 0.005]
        drop_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.4, 0.4, 0.005],
            rgbaColor=[0.0, 0.8, 0.2, 0.6],
        )
        self.drop_zone_id = p.createMultiBody(
            0, -1, drop_vis, basePosition=self.drop_zone_pos
        )

        objects_config = [
            {
                "type": p.GEOM_BOX,
                "size": [0.15, 0.15, 0.15],
                "color": [1, 0, 0, 1],
                "pos": [1.5, 0.5, 0.15],
            },  # Red Box
            {
                "type": p.GEOM_BOX,
                "size": [0.15, 0.15, 0.15],
                "color": [0, 0, 1, 1],
                "pos": [1.0, -1.0, 0.15],
            },  # Blue Box
            {
                "type": p.GEOM_CYLINDER,
                "radius": 0.12,
                "length": 0.3,
                "color": [1, 0, 0, 1],
                "pos": [2.0, -0.4, 0.15],
            },  # Red Cylinder
            {
                "type": p.GEOM_SPHERE,
                "radius": 0.15,
                "color": [1, 0.8, 0, 1],
                "pos": [0.5, 1.5, 0.15],
            },  # Yellow Sphere
            {
                "type": p.GEOM_SPHERE,
                "radius": 0.15,
                "color": [0.5, 0, 0.5, 1],
                "pos": [-1.2, 1.0, 0.15],
            },  # Purple Sphere
        ]

        self.object_ids = []
        for cfg in objects_config:
            if cfg["type"] == p.GEOM_BOX:
                col = p.createCollisionShape(
                    p.GEOM_BOX, halfExtents=cfg["size"]
                )
                vis = p.createVisualShape(
                    p.GEOM_BOX, halfExtents=cfg["size"], rgbaColor=cfg["color"]
                )
            elif cfg["type"] == p.GEOM_SPHERE:
                col = p.createCollisionShape(
                    p.GEOM_SPHERE, radius=cfg["radius"]
                )
                vis = p.createVisualShape(
                    p.GEOM_SPHERE,
                    radius=cfg["radius"],
                    rgbaColor=cfg["color"],
                )
            elif cfg["type"] == p.GEOM_CYLINDER:
                col = p.createCollisionShape(
                    p.GEOM_CYLINDER, radius=cfg["radius"], height=cfg["length"]
                )
                vis = p.createVisualShape(
                    p.GEOM_CYLINDER,
                    radius=cfg["radius"],
                    length=cfg["length"],
                    rgbaColor=cfg["color"],
                )

            obj_id = p.createMultiBody(1.0, col, vis, basePosition=cfg["pos"])
            self.object_ids.append(obj_id)

    def _create_robot(self):
        """Capsule robot base[cite: 2]."""
        col = p.createCollisionShape(p.GEOM_CAPSULE, radius=0.25, height=0.2)
        vis = p.createVisualShape(
            p.GEOM_CAPSULE, radius=0.25, length=0.2, rgbaColor=[0.2, 0.2, 0.2, 1]
        )
        self.robot_id = p.createMultiBody(
            10.0, col, vis, basePosition=self.robot_pos
        )

    def get_camera_image(self):
        """Render 640x480 JPEG (90 FOV, 0.5m height, 10 deg downward tilt)[cite: 3, 4]."""
        rad = math.radians(self.robot_heading)
        cam_height = 0.5
        tilt_rad = math.radians(10)

        dx = math.cos(rad)
        dy = math.sin(rad)
        dz = -math.tan(tilt_rad)

        cam_pos = [self.robot_pos[0], self.robot_pos[1], cam_height]
        cam_target = [cam_pos[0] + dx, cam_pos[1] + dy, cam_pos[2] + dz]
        cam_up = [0, 0, 1]

        view_matrix = p.computeViewMatrix(cam_pos, cam_target, cam_up)
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=90, aspect=640 / 480, nearVal=0.1, farVal=10.0
        )

        _, _, rgb_img, _, _ = p.getCameraImage(
            640,
            480,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_TINY_RENDERER,
        )

        frame = np.reshape(rgb_img, (480, 640, 4))[:, :, :3]
        return Image.fromarray(frame.astype("uint8"), "RGB")

    def move_robot(self, distance):
        """Move robot with boundary collision detection and 0.1m recoil[cite: 3]."""
        rad = math.radians(self.robot_heading)
        target_x = self.robot_pos[0] + distance * math.cos(rad)
        target_y = self.robot_pos[1] + distance * math.sin(rad)

        if abs(target_x) > 2.25 or abs(target_y) > 2.25:
            self.collided = True
            self.state_name = "Blocked"
            self.robot_pos[0] -= 0.1 * math.cos(rad)
            self.robot_pos[1] -= 0.1 * math.sin(rad)
            p.resetBasePositionAndOrientation(
                self.robot_id, self.robot_pos, [0, 0, 0, 1]
            )
            return {"moved": round(distance * 0.5, 2), "collided": True}

        actual_moved = math.hypot(
            target_x - self.robot_pos[0], target_y - self.robot_pos[1]
        )
        self.robot_pos[0] = target_x
        self.robot_pos[1] = target_y
        p.resetBasePositionAndOrientation(
            self.robot_id, self.robot_pos, [0, 0, 0, 1]
        )
        self.collided = False

        if self.held_object_id:
            hold_x = self.robot_pos[0] + 0.35 * math.cos(rad)
            hold_y = self.robot_pos[1] + 0.35 * math.sin(rad)
            p.resetBasePositionAndOrientation(
                self.held_object_id, [hold_x, hold_y, 0.25], [0, 0, 0, 1]
            )

        return {"moved": round(actual_moved, 2), "collided": False}

    def turn_robot(self, degrees):
        """Turn robot in place (positive degrees = right / clockwise)[cite: 3, 4]."""
        self.robot_heading = (self.robot_heading + degrees) % 360
        return {"heading": round(self.robot_heading, 1)}

    def pickup_object(self):
        """Snap object to front if within 0.5m and +-30 deg heading[cite: 3]."""
        if self.held_object_id:
            return {"ok": False}

        rx, ry = self.robot_pos[0], self.robot_pos[1]
        for obj_id in self.object_ids:
            pos, _ = p.getBasePositionAndOrientation(obj_id)
            dist = math.hypot(pos[0] - rx, pos[1] - ry)

            angle_to_obj = math.degrees(math.atan2(pos[1] - ry, pos[0] - rx))
            heading_diff = abs((angle_to_obj - self.robot_heading + 180) % 360 - 180)

            if dist <= 0.5 and heading_diff <= 30:
                self.held_object_id = obj_id
                self.state_name = "Carry"
                return {"ok": True}

        return {"ok": False}

    def drop_object(self):
        """Release held object and check drop zone placement[cite: 3]."""
        if not self.held_object_id:
            return {"ok": False, "in_zone": False}

        pos, _ = p.getBasePositionAndOrientation(self.held_object_id)
        dx = abs(pos[0] - self.drop_zone_pos[0])
        dy = abs(pos[1] - self.drop_zone_pos[1])

        in_drop_zone = dx <= 0.4 and dy <= 0.4
        self.held_object_id = None
        self.state_name = "Done" if in_drop_zone else "Carry"
        return {"ok": True, "in_zone": in_drop_zone}


if __name__ == "__main__":
    engine = SimulationEngine(gui=True)
    img = engine.get_camera_image()
    img.save("camera_preview.jpg")
    print("Simulation engine running! Camera preview saved to camera_preview.jpg.")
    print("3D Window is active. Press Ctrl+C in Command Prompt to stop.")

    try:
        while p.isConnected(engine.physics_client):
            p.stepSimulation()
            time.sleep(1.0 / 60.0)
    except KeyboardInterrupt:
        print("Closing simulation.")
        p.disconnect()