"""
sim/driver.py — kinematic base motion + smooth joint articulation.

The humanoid is loaded with a fixed (kinematic) base. Every physics sub-step
the driver places the base, spins the wheels to match the distance travelled
and blends the neck / head / arm joints toward their pose targets using
S-curve easing. Nothing ever pops: each command is split into at least
MIN_FRAMES (20) eased frames, each frame into PHYSICS_HZ/FPS physics
sub-steps, and joints are written on every sub-step.
"""

import math
import time

import pybullet as p

from .scene import ROBOT_GEOMETRY

FPS = 30
PHYSICS_HZ = 240
MIN_FRAMES = 20

G = ROBOT_GEOMETRY
WHEEL_RADIUS = G["wheel_radius"]
HALF_TRACK = G["half_track"]

ARM_JOINTS = ("left_shoulder_joint", "right_shoulder_joint", "left_elbow_joint", "right_elbow_joint")
HEAD_JOINTS = ("neck_joint", "head_joint")
WHEEL_JOINTS = ("left_wheel_joint", "right_wheel_joint")
ALL_JOINTS = WHEEL_JOINTS + HEAD_JOINTS + ARM_JOINTS
EYE_LINKS = ("left_eye", "right_eye")


def smoothstep(s):
    s = min(1.0, max(0.0, s))
    return s * s * (3.0 - 2.0 * s)


def smootherstep(s):
    s = min(1.0, max(0.0, s))
    return s * s * s * (s * (6.0 * s - 15.0) + 10.0)


def lerp(a, b, t):
    return a + (b - a) * t


def lerp_pose(a, b, t):
    return {k: lerp(a.get(k, 0.0), b.get(k, 0.0), t) for k in set(a) | set(b)}


def arm_ik(x, z):
    l1, l2 = G["upper_arm"], G["forearm"]
    dx, dz = x - G["shoulder_x"], z - G["shoulder_z"]
    r = math.hypot(dx, dz)
    r_max, r_min = l1 + l2 - 1e-3, abs(l1 - l2) + 1e-3
    if r < 1e-9:
        dx, dz, r = 0.0, -r_min, r_min
    clamped = min(max(r, r_min), r_max)
    dx, dz, r = dx * clamped / r, dz * clamped / r, clamped
    cos_b = (r * r - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    beta = math.acos(min(1.0, max(-1.0, cos_b)))
    phi1 = math.atan2(dx, -dz) - math.atan2(l2 * math.sin(beta), l1 + l2 * math.cos(beta))
    return -phi1, -beta


def arm_fk(shoulder, elbow):
    l1, l2 = G["upper_arm"], G["forearm"]
    x = G["shoulder_x"] - l1 * math.sin(shoulder) - l2 * math.sin(shoulder + elbow)
    z = G["shoulder_z"] - l1 * math.cos(shoulder) - l2 * math.cos(shoulder + elbow)
    return x, z


def arms_pose(shoulder, elbow):
    return {"left_shoulder_joint": shoulder, "right_shoulder_joint": shoulder,
            "left_elbow_joint": elbow, "right_elbow_joint": elbow}


def arms_to(x, z):
    return arms_pose(*arm_ik(x, z))


def arm_side(side, shoulder, elbow):
    return {"%s_shoulder_joint" % side: shoulder, "%s_elbow_joint" % side: elbow}


def arm_side_to(side, x, z):
    return arm_side(side, *arm_ik(x, z))


REST_ARMS = arms_pose(-0.12, -0.55)
TUCKED_FREE_ARM = arm_side("left", -0.35, -1.10)
HEAD_NEUTRAL = {"neck_joint": 0.0, "head_joint": 0.0}
NEUTRAL_POSE = dict(REST_ARMS, **HEAD_NEUTRAL)


def yaw_quat(heading_deg):
    half = -math.radians(heading_deg) / 2.0
    return (0.0, 0.0, math.sin(half), math.cos(half))


class HumanoidDriver:
    def __init__(self, robot_id, realtime=False, on_frame=None):
        if hasattr(robot_id, "robot_id"):
            robot_id = robot_id.robot_id
        elif hasattr(robot_id, "engine") and hasattr(robot_id.engine, "robot_id"):
            robot_id = robot_id.engine.robot_id

        self.robot_id = robot_id
        self.realtime = realtime
        self.on_frame = on_frame
        self.joint_index, self.link_index = {}, {}
        for j in range(p.getNumJoints(robot_id)):
            info = p.getJointInfo(robot_id, j)
            self.joint_index[info[1].decode("utf-8")] = j
            self.link_index[info[12].decode("utf-8")] = j
        missing = [n for n in ALL_JOINTS if n not in self.joint_index]
        missing += [n for n in EYE_LINKS if n not in self.link_index]
        if missing:
            raise RuntimeError("HUMANOID_URDF is missing joints/links: %s" % missing)
        self.q = {name: 0.0 for name in ALL_JOINTS}
        self.apply(NEUTRAL_POSE)
        self.set_wheels(0.0, 0.0)

    def set_joint(self, name, value):
        idx = self.joint_index[name]
        p.resetJointState(self.robot_id, idx, value)
        p.setJointMotorControl2(self.robot_id, idx, p.POSITION_CONTROL, targetPosition=value, force=60)
        self.q[name] = value

    def apply(self, pose):
        for name, value in pose.items():
            self.set_joint(name, value)

    def set_wheels(self, left, right):
        self.set_joint("left_wheel_joint", left)
        self.set_joint("right_wheel_joint", right)

    def roll_wheels(self, left_dist, right_dist, start):
        self.set_wheels(start[0] + left_dist / WHEEL_RADIUS, start[1] + right_dist / WHEEL_RADIUS)

    def wheel_angles(self):
        return self.q["left_wheel_joint"], self.q["right_wheel_joint"]

    def current_pose(self, names=HEAD_JOINTS + ARM_JOINTS):
        return {n: self.q[n] for n in names}

    def set_base(self, x, y, heading_deg):
        p.resetBasePositionAndOrientation(self.robot_id, [x, y, 0.0], yaw_quat(heading_deg))

    def set_eyes(self, alert):
        from .scene import PALETTE
        colour = list(PALETTE["eye_alert" if alert else "eye_cyan"])
        for link in EYE_LINKS:
            p.changeVisualShape(self.robot_id, self.link_index[link], rgbaColor=colour)

    def run(self, duration, update, min_frames=MIN_FRAMES, step_physics=True):
        duration = max(0.0, float(duration))
        frames = max(min_frames, int(math.ceil(duration * FPS)))
        substeps = max(1, int(round(duration * PHYSICS_HZ / frames)))
        frame_dt = duration / frames if frames else 0.0
        last = 0.0
        for f in range(frames):
            t0 = time.perf_counter()
            for k in range(substeps):
                s = (f + (k + 1) / substeps) / frames
                if update(s) is False:
                    return last, True
                last = s
                if step_physics:
                    p.stepSimulation()
            if self.on_frame is not None:
                self.on_frame()
            if self.realtime:
                remaining = frame_dt - (time.perf_counter() - t0)
                if remaining > 0:
                    time.sleep(remaining)
        return 1.0, False

    def blend_to(self, target, duration, extra=None, step_physics=True, min_frames=MIN_FRAMES):
        start = self.current_pose(tuple(target))

        def update(s):
            e = smootherstep(s)
            self.apply(lerp_pose(start, target, e))
            if extra is not None:
                return extra(s, e)
            return None

        return self.run(duration, update, step_physics=step_physics, min_frames=min_frames)

    def scan(self, duration=1.6, yaw=0.55, pitch=0.12):
        start = self.current_pose(HEAD_JOINTS)

        def update(s):
            e = smootherstep(s)
            neck = yaw * math.sin(2.0 * math.pi * s) * math.sin(math.pi * s)
            head = pitch * math.sin(math.pi * s)
            self.set_joint("neck_joint", lerp(start["neck_joint"], 0.0, e) + neck)
            self.set_joint("head_joint", lerp(start["head_joint"], 0.0, e) + head)

        self.run(duration, update, step_physics=False)
        self.apply(HEAD_NEUTRAL)

    def recentre_head(self, duration=0.2):
        if abs(self.q["neck_joint"]) > 1e-4 or abs(self.q["head_joint"]) > 1e-4:
            self.blend_to(HEAD_NEUTRAL, duration, min_frames=6)

# Alias for preview.py compatibility
RobotDriver = HumanoidDriver