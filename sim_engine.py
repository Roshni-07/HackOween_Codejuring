import io
import math
import random
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
        self.robot_heading = 0.0  # Degrees (0 = facing +X, positive = clockwise/right)
        self.state_name = "Idle"
        self.held_object_id = None
        self.pickup_constraint = None
        self.collided = False
        self.retries = 0
        self.debug_lines = []

        if gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=4.8,
                cameraYaw=45,
                cameraPitch=-45,
                cameraTargetPosition=[0, 0, 0],
            )
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(
                p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0
            )
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)

        self._build_room()
        self._spawn_shapes_and_dropzone()
        self._create_robot()
        self.update_debug_visuals()

    def _build_room(self):
        """Build floor plane and 4 surrounding arena walls."""
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
        """Spawn green drop zone platform and primitive objects."""
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
                "color": [0.9, 0.1, 0.1, 1],
                "pos": [1.5, 0.5, 0.15],
            },  # Red Box
            {
                "type": p.GEOM_BOX,
                "size": [0.15, 0.15, 0.15],
                "color": [0.1, 0.3, 0.9, 1],
                "pos": [1.0, -1.0, 0.15],
            },  # Blue Box
            {
                "type": p.GEOM_SPHERE,
                "radius": 0.15,
                "color": [0.9, 0.1, 0.1, 1],
                "pos": [1.8, -0.6, 0.15],
            },  # Red Sphere
            {
                "type": p.GEOM_CYLINDER,
                "radius": 0.12,
                "length": 0.3,
                "color": [0.9, 0.8, 0.0, 1],
                "pos": [0.5, 1.5, 0.15],
            },  # Yellow Cylinder
            {
                "type": p.GEOM_SPHERE,
                "radius": 0.15,
                "color": [0.6, 0.1, 0.8, 1],
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

        for obj_id in self.object_ids:
            p.changeDynamics(
                obj_id,
                -1,
                mass=1.0,
                lateralFriction=0.8,
                spinningFriction=0.1,
                rollingFriction=0.1,
                linearDamping=0.5,
                angularDamping=0.5,
            )

        p.changeDynamics(0, -1, lateralFriction=1.0)

    def _create_robot(self):
        """Capsule robot base."""
        col = p.createCollisionShape(p.GEOM_CAPSULE, radius=0.25, height=0.2)
        vis = p.createVisualShape(
            p.GEOM_CAPSULE, radius=0.25, length=0.2, rgbaColor=[0.2, 0.2, 0.2, 1]
        )
        self.robot_id = p.createMultiBody(
            10.0, col, vis, basePosition=self.robot_pos
        )

    def get_camera_image(self):
        """Render 640x480 JPEG (90 FOV, 0.5m height, 10 deg downward tilt)."""
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
        img = Image.fromarray(frame.astype("uint8"), "RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()

    def get_single_depth(self, offset_r=0.26, max_dist=10.0):
        """Returns raycast distance straight ahead without intersecting robot collision capsule (r=0.25m)."""
        rad = math.radians(self.robot_heading)
        tilt_rad = math.radians(10)

        start_x = self.robot_pos[0] + offset_r * math.cos(rad)
        start_y = self.robot_pos[1] + offset_r * math.sin(rad)
        start_z = 0.5

        end_x = start_x + max_dist * math.cos(rad)
        end_y = start_y + max_dist * math.sin(rad)
        end_z = start_z - max_dist * math.tan(tilt_rad)

        ray_res = p.rayTest([start_x, start_y, start_z], [end_x, end_y, end_z])
        hit_fraction = ray_res[0][2]

        if hit_fraction < 1.0:
            return round(float(hit_fraction * max_dist), 2)
        return None

    def get_depth_fan(self, angles=[-30, -15, 0, 15, 30], offset_r=0.26, max_dist=10.0):
        """Returns array of depth ray distances across a fan angle."""
        results = []
        for angle in angles:
            rad = math.radians(self.robot_heading + angle)
            tilt_rad = math.radians(10)

            start_x = self.robot_pos[0] + offset_r * math.cos(rad)
            start_y = self.robot_pos[1] + offset_r * math.sin(rad)
            start_z = 0.5

            end_x = start_x + max_dist * math.cos(rad)
            end_y = start_y + max_dist * math.sin(rad)
            end_z = start_z - max_dist * math.tan(tilt_rad)

            ray_res = p.rayTest([start_x, start_y, start_z], [end_x, end_y, end_z])
            hit_frac = ray_res[0][2]
            dist = round(float(hit_frac * max_dist), 2) if hit_frac < 1.0 else None
            results.append({"angle_offset": angle, "distance": dist})
        return results

    def update_debug_visuals(self):
        """Draw 3D FOV frustum lines and pickup radius in PyBullet GUI."""
        for line_id in self.debug_lines:
            p.removeUserDebugItem(line_id)
        self.debug_lines.clear()

        rx, ry = self.robot_pos[0], self.robot_pos[1]

        # 1. Draw 0.5m pickup sector lines (+-30 deg) in RED
        for angle_offset in [-30, 30]:
            cone_rad = math.radians(self.robot_heading + angle_offset)
            ex = rx + 0.5 * math.cos(cone_rad)
            ey = ry + 0.5 * math.sin(cone_rad)
            line_id = p.addUserDebugLine(
                [rx, ry, 0.25], [ex, ey, 0.25], [1, 0, 0], lineWidth=2
            )
            self.debug_lines.append(line_id)

        # 2. Draw 90 deg Camera FOV cone in CYAN
        for angle_offset in [-45, 45]:
            fov_rad = math.radians(self.robot_heading + angle_offset)
            ex = rx + 2.0 * math.cos(fov_rad)
            ey = ry + 2.0 * math.sin(fov_rad)
            line_id = p.addUserDebugLine(
                [rx, ry, 0.5], [ex, ey, 0.1], [0, 1, 1], lineWidth=1
            )
            self.debug_lines.append(line_id)

    def check_physical_collisions(self):
        """Detect real physical mesh contacts against walls, obstacles, or floor items."""
        contacts = p.getContactPoints(bodyA=self.robot_id)
        for c in contacts:
            touching_body = c[2]
            if touching_body != 0 and touching_body != self.held_object_id:
                return True
        return False

    def move_robot(self, distance):
        """Move robot using interpolated sub-stepping, collision detection, 0.1m recoil, and 15 deg rotational nudge."""
        rad = math.radians(self.robot_heading)
        num_steps = 15
        step_dist = distance / num_steps

        for _ in range(num_steps):
            target_x = self.robot_pos[0] + step_dist * math.cos(rad)
            target_y = self.robot_pos[1] + step_dist * math.sin(rad)

            if abs(target_x) > 2.25 or abs(target_y) > 2.25 or self.check_physical_collisions():
                self.collided = True
                self.state_name = "Blocked"
                self.retries += 1
                
                # Apply 0.1m recoil away from impact
                self.robot_pos[0] -= 0.1 * math.cos(rad)
                self.robot_pos[1] -= 0.1 * math.sin(rad)
                p.resetBasePositionAndOrientation(
                    self.robot_id, self.robot_pos, [0, 0, 0, 1]
                )
                
                # Rotational nudge (+15 deg) to prevent corner trapping loops
                self.turn_robot(15.0)
                self.update_debug_visuals()
                return {"moved": round(distance * 0.5, 2), "collided": True}

            self.robot_pos[0] = target_x
            self.robot_pos[1] = target_y
            p.resetBasePositionAndOrientation(
                self.robot_id, self.robot_pos, [0, 0, 0, 1]
            )
            p.stepSimulation()

        self.collided = False
        self.update_debug_visuals()
        return {"moved": round(distance, 2), "collided": False}

    def turn_robot(self, degrees):
        """Turn robot in place (positive degrees = right / clockwise)."""
        self.robot_heading = (self.robot_heading + degrees) % 360
        rad = math.radians(self.robot_heading)
        orn = p.getQuaternionFromEuler([0, 0, -rad])
        p.resetBasePositionAndOrientation(self.robot_id, self.robot_pos, orn)
        self.update_debug_visuals()
        return {"heading": round(self.robot_heading, 1)}

    def pickup_object(self):
        """Attach nearest object within 0.5m and +-30 deg using a rigid PyBullet constraint."""
        if self.held_object_id:
            return {"ok": False}

        rx, ry = self.robot_pos[0], self.robot_pos[1]
        for obj_id in self.object_ids:
            pos, _ = p.getBasePositionAndOrientation(obj_id)
            dist = math.hypot(pos[0] - rx, pos[1] - ry)

            angle_to_obj = math.degrees(math.atan2(pos[1] - ry, pos[0] - rx))
            heading_diff = abs(((angle_to_obj - self.robot_heading + 180) % 360) - 180)

            if dist <= 0.5 and heading_diff <= 30:
                self.held_object_id = obj_id
                self.state_name = "Carry"

                self.pickup_constraint = p.createConstraint(
                    parentBodyUniqueId=self.robot_id,
                    parentLinkIndex=-1,
                    childBodyUniqueId=obj_id,
                    childLinkIndex=-1,
                    jointType=p.JOINT_FIXED,
                    jointAxis=[0, 0, 0],
                    parentFramePosition=[0.35, 0, 0.1],
                    childFramePosition=[0, 0, 0],
                )
                return {"ok": True}

        return {"ok": False}

    def drop_object(self):
        """Release fixed physics constraint and check drop zone placement."""
        if not self.held_object_id:
            return {"ok": False, "in_zone": False}

        if hasattr(self, "pickup_constraint") and self.pickup_constraint is not None:
            p.removeConstraint(self.pickup_constraint)
            self.pickup_constraint = None

        pos, _ = p.getBasePositionAndOrientation(self.held_object_id)
        dx = abs(pos[0] - self.drop_zone_pos[0])
        dy = abs(pos[1] - self.drop_zone_pos[1])

        in_drop_zone = dx <= 0.4 and dy <= 0.4
        self.held_object_id = None
        self.state_name = "Done" if in_drop_zone else "Carry"
        return {"ok": True, "in_zone": in_drop_zone}

    def reset_scene(self, seed=None):
        """Resets robot state and scrambles object locations with 1.0m origin clearance."""
        if seed is not None:
            random.seed(seed)

        if hasattr(self, "pickup_constraint") and self.pickup_constraint is not None:
            p.removeConstraint(self.pickup_constraint)
            self.pickup_constraint = None

        placed_positions = []

        for obj_id in self.object_ids:
            valid_pos = False
            while not valid_pos:
                rx = round(random.uniform(-1.8, 1.8), 2)
                ry = round(random.uniform(-1.8, 1.8), 2)

                dist_to_origin = math.hypot(rx, ry)
                dist_to_drop = math.hypot(
                    rx - self.drop_zone_pos[0], ry - self.drop_zone_pos[1]
                )
                dist_to_others = [
                    math.hypot(rx - px, ry - py) for px, py in placed_positions
                ]

                if (
                    dist_to_origin >= 1.0
                    and dist_to_drop > 0.8
                    and all(d >= 0.6 for d in dist_to_others)
                ):
                    valid_pos = True
                    placed_positions.append([rx, ry])
                    p.resetBasePositionAndOrientation(
                        obj_id, [rx, ry, 0.15], [0, 0, 0, 1]
                    )

        self.robot_pos = [0.0, 0.0, 0.25]
        self.robot_heading = 0.0
        self.held_object_id = None
        self.state_name = "Idle"
        self.collided = False
        self.retries = 0
        p.resetBasePositionAndOrientation(
            self.robot_id, self.robot_pos, [0, 0, 0, 1]
        )
        self.update_debug_visuals()
        return {"status": "reset_complete"}

    def enable_obstacle_mode(self):
        """Spawns internal wall blocking direct path from origin to drop zone for stretch goal testing."""
        wall_col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.1, 0.8, 0.5]
        )
        wall_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.1, 0.8, 0.5],
            rgbaColor=[0.3, 0.3, 0.3, 1],
        )
        self.obstacle_id = p.createMultiBody(
            0, wall_col, wall_vis, basePosition=[0.9, 0.9, 0.5]
        )
        return {"obstacle_enabled": True}


if __name__ == "__main__":
    engine = SimulationEngine(gui=True)
    img_bytes = engine.get_camera_image()
    with open("camera_preview.jpg", "wb") as f:
        f.write(img_bytes)
    print("Simulation engine running! Camera preview saved to camera_preview.jpg.")

    try:
        while p.isConnected(engine.physics_client):
            p.stepSimulation()
            time.sleep(1.0 / 60.0)
    except KeyboardInterrupt:
        print("Closing simulation.")
        p.disconnect()