# sim/core.py
import math
import pybullet as p
import pybullet_data
from sim.sensors import SensorSuite
from sim.kinematics import KinematicDriver
from sim.referee import Referee


class CoreEngine:
    def __init__(self, gui: bool = False):
        mode = p.GUI if gui else p.DIRECT
        self.physics_client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        self.room_size = 5.0
        self.robot_pos = [0.0, 0.0, 0.25]
        self.robot_heading = 0.0  # Degrees (0 = facing +X)
        self.held_object_id = None
        self.pickup_constraint = None
        self.collided = False
        self.retries = 0

        if gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=4.8, cameraYaw=45, cameraPitch=-45, cameraTargetPosition=[0, 0, 0]
            )
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)

        # Core subsystems
        self.sensors = SensorSuite(self)
        self.driver = KinematicDriver(self)
        self.referee = Referee(self)
        self.recorder = None

    def sync_robot_transform(self):
        """Syncs physical body orientation to robot_heading."""
        rad = math.radians(self.robot_heading)
        orn = p.getQuaternionFromEuler([0, 0, -rad])
        p.resetBasePositionAndOrientation(self.robot_id, self.robot_pos, orn)

    def check_physical_collisions(self) -> bool:
        """Checks for geometric collisions against walls, floor items, or obstacles."""
        contacts = p.getContactPoints(bodyA=self.robot_id)
        for c in contacts:
            touching_body = c[2]
            if touching_body != 0 and touching_body != self.held_object_id:
                return True
        return False