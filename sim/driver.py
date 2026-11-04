# sim/driver.py
import math
import time
import pybullet as p


class RobotDriver:
    """Handles smooth sub-step movement and rotation for the robot."""

    def __init__(self, engine):
        self.engine = engine

    def turn(self, degrees: float) -> dict:
        """Rotates the robot smoothly over sub-steps."""
        target_heading = (self.engine.robot_heading + degrees) % 360.0
        start_heading = self.engine.robot_heading

        steps = 20
        diff = degrees

        for i in range(1, steps + 1):
            curr_heading = (start_heading + (diff * i / steps)) % 360.0
            rad = math.radians(curr_heading)
            orientation = p.getQuaternionFromEuler([0, 0, rad])

            p.resetBasePositionAndOrientation(
                self.engine.robot_id,
                self.engine.robot_pos,
                orientation,
                physicsClientId=self.engine.client_id,
            )
            p.stepSimulation(physicsClientId=self.engine.client_id)
            time.sleep(0.01)

        self.engine.robot_heading = target_heading

        return {
            "status": "success",
            "heading": round(self.engine.robot_heading, 1),
        }

    def move(self, distance: float) -> dict:
        """Drives the robot along its current heading vector with smooth sub-stepping."""
        rad = math.radians(self.engine.robot_heading)
        dx = distance * math.cos(rad)
        dy = distance * math.sin(rad)

        start_pos = list(self.engine.robot_pos)
        target_x = start_pos[0] + dx
        target_y = start_pos[1] + dy
        orientation = p.getQuaternionFromEuler([0, 0, rad])

        steps = 20
        for i in range(1, steps + 1):
            curr_x = start_pos[0] + (dx * i / steps)
            curr_y = start_pos[1] + (dy * i / steps)
            curr_pos = [curr_x, curr_y, start_pos[2]]

            p.resetBasePositionAndOrientation(
                self.engine.robot_id,
                curr_pos,
                orientation,
                physicsClientId=self.engine.client_id,
            )
            p.stepSimulation(physicsClientId=self.engine.client_id)
            time.sleep(0.01)

        self.engine.robot_pos = [target_x, target_y, start_pos[2]]

        return {
            "status": "success",
            "moved": distance,
            "x": round(self.engine.robot_pos[0], 2),
            "y": round(self.engine.robot_pos[1], 2),
            "collided": False
        }