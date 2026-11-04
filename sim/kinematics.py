# sim/kinematics.py
import math
import pybullet as p


class KinematicDriver:
    def __init__(self, engine, fps=30, pybullet_substeps=8):
        self.engine = engine
        self.fps = fps
        self.dt = 1.0 / fps  # 0.0333s per frame tick
        self.substeps = pybullet_substeps  # 240Hz physics update rate

    def move(self, distance: float) -> dict:
        """Drives robot forward/backward at 1.0 m/s using fixed dt frames."""
        speed = 1.0 if distance >= 0 else -1.0
        total_time = abs(distance) / 1.0
        num_frames = max(1, int(total_time / self.dt))
        step_dist = distance / num_frames

        for frame in range(num_frames):
            rad = math.radians(self.engine.robot_heading)
            dx = step_dist * math.cos(rad)
            dy = step_dist * math.sin(rad)

            target_x = self.engine.robot_pos[0] + dx
            target_y = self.engine.robot_pos[1] + dy

            # Check arena boundaries & physical mesh contacts
            if abs(target_x) > 2.25 or abs(target_y) > 2.25 or self.engine.check_physical_collisions():
                # Apply exact 0.1m recoil along heading vector
                self.engine.robot_pos[0] -= 0.1 * math.cos(rad)
                self.engine.robot_pos[1] -= 0.1 * math.sin(rad)
                self.engine.sync_robot_transform()

                self.engine.collided = True
                self.engine.referee.state_name = "Blocked"
                self.engine.retries += 1

                if self.engine.recorder:
                    self.engine.recorder.capture_frame()

                return {"moved": round(frame * step_dist, 2), "collided": True}

            self.engine.robot_pos[0] = target_x
            self.engine.robot_pos[1] = target_y
            self.engine.sync_robot_transform()

            # Advance PyBullet physics substeps
            for _ in range(self.substeps):
                p.stepSimulation()

            if self.engine.recorder:
                self.engine.recorder.capture_frame()

        self.engine.collided = False
        return {"moved": round(distance, 2), "collided": False}

    def turn(self, degrees: float) -> dict:
        """Turns robot in place at 90 deg/s using fixed dt frames."""
        total_time = abs(degrees) / 90.0
        num_frames = max(1, int(total_time / self.dt))
        step_deg = degrees / num_frames

        for _ in range(num_frames):
            self.engine.robot_heading = (self.engine.robot_heading + step_deg) % 360.0
            self.engine.sync_robot_transform()

            for _ in range(self.substeps):
                p.stepSimulation()

            if self.engine.recorder:
                self.engine.recorder.capture_frame()

        return {"heading": round(self.engine.robot_heading, 1)}