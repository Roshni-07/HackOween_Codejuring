# sim/scene.py
import random
import pybullet as p
import pybullet_data


class SceneManager:
    def __init__(self, engine):
        self.engine = engine
        self.drop_zone_pos = [1.8, 1.8, 0.005]
        self.object_ids = []
        self.obstacle_id = None
        self.drop_zone_id = None

    def build_room(self):
        """Loads room geometry with default PyBullet checkerboard floor grid."""
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")

        wall_height, wall_thick = 1.0, 0.1
        half_r = self.engine.room_size / 2.0
        walls = [
            [0, half_r, half_r, wall_thick],
            [0, -half_r, half_r, wall_thick],
            [half_r, 0, wall_thick, half_r],
            [-half_r, 0, wall_thick, half_r],
        ]
        for wx, wy, hx, hy in walls:
            shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, wall_height / 2.0])
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, wall_height / 2.0], rgbaColor=[0.7, 0.7, 0.7, 1])
            p.createMultiBody(0, shape, vis, basePosition=[wx, wy, wall_height / 2.0])

    def spawn_robot(self):
        col = p.createCollisionShape(p.GEOM_CAPSULE, radius=0.25, height=0.2)
        vis = p.createVisualShape(p.GEOM_CAPSULE, radius=0.25, length=0.2, rgbaColor=[0.2, 0.2, 0.2, 1])
        self.engine.robot_id = p.createMultiBody(10.0, col, vis, basePosition=self.engine.robot_pos)

    def spawn_drop_zone(self, pos=None):
        if pos:
            self.drop_zone_pos = pos
        drop_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.005], rgbaColor=[0.0, 0.8, 0.2, 0.6])
        self.drop_zone_id = p.createMultiBody(0, -1, drop_vis, basePosition=self.drop_zone_pos)

    def spawn_objects(self, seed=None):
        if seed is not None:
            random.seed(seed)

        # Clear existing objects without relying on self.engine.client_id
        for obj_id in self.object_ids:
            try:
                p.removeBody(obj_id)
            except Exception:
                pass
        self.object_ids.clear()

        # 5 primitive objects with height variation
        configs = [
            {"type": p.GEOM_BOX, "size": [0.15, 0.15, 0.15], "color": [0.9, 0.1, 0.1, 1]},       # Red Box
            {"type": p.GEOM_BOX, "size": [0.15, 0.15, 0.15], "color": [0.1, 0.3, 0.9, 1]},       # Blue Box
            {"type": p.GEOM_SPHERE, "radius": 0.15, "color": [0.9, 0.1, 0.1, 1]},                # Red Sphere
            {"type": p.GEOM_CYLINDER, "radius": 0.12, "length": 0.45, "color": [0.9, 0.8, 0.0, 1]}, # Yellow Cylinder
            {"type": p.GEOM_CYLINDER, "radius": 0.15, "length": 0.20, "color": [0.6, 0.1, 0.8, 1]}  # Purple Cylinder
        ]

        spawn_positions = [
            [1.0, 0.5],
            [1.2, -0.4],
            [0.8, 0.8],
            [1.5, 0.0],
            [0.5, -0.8],
        ]

        for i, cfg in enumerate(configs):
            x, y = spawn_positions[i]

            if cfg["type"] == p.GEOM_BOX:
                z = cfg["size"][2]
                col = p.createCollisionShape(p.GEOM_BOX, halfExtents=cfg["size"])
                vis = p.createVisualShape(p.GEOM_BOX, halfExtents=cfg["size"], rgbaColor=cfg["color"])
            elif cfg["type"] == p.GEOM_SPHERE:
                z = cfg["radius"]
                col = p.createCollisionShape(p.GEOM_SPHERE, radius=cfg["radius"])
                vis = p.createVisualShape(p.GEOM_SPHERE, radius=cfg["radius"], rgbaColor=cfg["color"])
            else:
                z = cfg["length"] / 2.0
                col = p.createCollisionShape(p.GEOM_CYLINDER, radius=cfg["radius"], height=cfg["length"])
                vis = p.createVisualShape(p.GEOM_CYLINDER, radius=cfg["radius"], length=cfg["length"], rgbaColor=cfg["color"])

            obj_id = p.createMultiBody(1.0, col, vis, basePosition=[x, y, z])
            p.changeDynamics(obj_id, -1, mass=1.0, lateralFriction=0.8, linearDamping=0.5)
            self.object_ids.append(obj_id)

    def spawn_obstacle(self):
        wall_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.8, 0.5])
        wall_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.8, 0.5], rgbaColor=[0.3, 0.3, 0.3, 1])
        self.obstacle_id = p.createMultiBody(0, wall_col, wall_vis, basePosition=[0.9, 0.9, 0.5])