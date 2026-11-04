# sim/referee.py
import math
import pybullet as p


class Referee:
    def __init__(self, engine):
        self.engine = engine
        self.state_name = "Idle"

    def pickup_object(self) -> dict:
        """Picks up the object directly in FRONT of the robot's heading field of view."""
        if getattr(self.engine, "held_object_id", None) is not None:
            return {"status": "error", "message": "Already holding an object"}

        rx, ry, _ = self.engine.robot_pos
        heading_rad = math.radians(self.engine.robot_heading)

        target_obj = None
        min_front_dist = 0.65  # Max reach forward

        for obj_id in self.engine.scene.object_ids:
            pos, _ = p.getBasePositionAndOrientation(obj_id)
            dx = pos[0] - rx
            dy = pos[1] - ry
            dist = math.hypot(dx, dy)

            # Check angle relative to robot heading
            obj_angle = math.degrees(math.atan2(dy, dx))
            angle_diff = (obj_angle - self.engine.robot_heading + 180) % 360 - 180

            # MUST be in front cone (within 35 degrees of front heading) and within reach
            if abs(angle_diff) <= 35.0 and dist < min_front_dist:
                min_front_dist = dist
                target_obj = obj_id

        if target_obj is None:
            return {"status": "failed", "message": "No object directly in front to pick up"}

        # Position object cleanly attached to front bumper
        front_x = rx + 0.35 * math.cos(heading_rad)
        front_y = ry + 0.35 * math.sin(heading_rad)

        p.resetBasePositionAndOrientation(
            target_obj,
            [front_x, front_y, 0.25],
            [0, 0, 0, 1],
        )

        # Attach object dynamically to robot using PyBullet joint constraint
        constraint_id = p.createConstraint(
            parentBodyUniqueId=self.engine.robot_id,
            parentLinkIndex=-1,
            childBodyUniqueId=target_obj,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0.35, 0, 0.1],
            childFramePosition=[0, 0, 0],
        )

        self.engine.held_object_id = target_obj
        self.engine.held_constraint_id = constraint_id
        self.state_name = "Carrying"

        return {"status": "success", "picked_up_id": target_obj}

    def drop_object(self) -> dict:
        """Releases the held object constraint so it drops cleanly onto the ground."""
        if getattr(self.engine, "held_object_id", None) is None:
            return {"status": "error", "message": "Not holding any object"}

        if (
            hasattr(self.engine, "held_constraint_id")
            and self.engine.held_constraint_id is not None
        ):
            p.removeConstraint(self.engine.held_constraint_id)
            self.engine.held_constraint_id = None

        self.engine.held_object_id = None
        self.state_name = "Idle"
        return {"status": "success", "message": "Object dropped"}


# Alias for backward compatibility
RefereeManager = Referee