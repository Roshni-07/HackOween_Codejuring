"""
sim_engine.py — SimulationEngine: the facade server.py talks to.

Composes the sim/ package:
    sim.scene     arena, humanoid URDF, fetch objects, drop zone, obstacle
    sim.driver    kinematic base + eased joint animation
    sim.sensors   /look camera and /depth ray
    sim.referee   state machine and task rules

Conventions
    * heading in degrees, clockwise-positive: forward = (cos h, -sin h).
      turn_robot(+90) turns RIGHT.
    * robot_pos = [x, y, 0.0] (base origin on the floor between the wheels).
    * One re-entrant lock around every public call, because uvicorn runs sync
      endpoints in a thread pool and the remote heading sync runs in its own
      thread. PyBullet is not safe to drive from two threads at once.
"""

import functools
import math
import random
import threading

import pybullet as p

from sim import scene
from sim.driver import (
    HEAD_NEUTRAL,
    NEUTRAL_POSE,
    REST_ARMS,
    TUCKED_FREE_ARM,
    HumanoidDriver,
    arm_side_to,
    arms_to,
    lerp,
    smootherstep,
)
from sim.referee import Referee, in_drop_zone, within_pickup_reach
from sim.sensors import DepthSensor, RobotCamera, heading_vector

MOVE_SPEED_MPS = 1.0
TURN_SPEED_DPS = 90.0
BACKOFF_M = 0.05
COLLISION_TOL_M = 0.001          # penetration deeper than this counts as contact
CARRY_LOCAL_XY = (0.14, -0.40)   # right hip: outside the camera's 45° half-FOV
DROP_AHEAD_M = 0.42
SPAWN_WALL_MARGIN = 0.25
SPAWN_ROBOT_CLEAR = 0.35
SPAWN_GAP = 0.30


def _locked(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        with self.lock:
            return fn(self, *args, **kwargs)
    return wrapper


def _wrap_deg(a):
    return a % 360.0


class SimulationEngine:
    def __init__(self, gui=True, realtime=None, collision_nudge_deg=0.0):
        self.gui = bool(gui)
        self.lock = threading.RLock()
        self.collision_nudge_deg = float(collision_nudge_deg or 0.0)

        if self.gui:
            self.client_id = p.connect(p.GUI, options=scene.GUI_CONNECT_OPTIONS)
        else:
            self.client_id = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)
        if self.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            for flag in ("COV_ENABLE_GUI", "COV_ENABLE_SEGMENTATION_MARK_PREVIEW",
                         "COV_ENABLE_DEPTH_BUFFER_PREVIEW", "COV_ENABLE_RGB_BUFFER_PREVIEW"):
                if hasattr(p, flag):
                    p.configureDebugVisualizer(getattr(p, flag), 0)

        # --- world -------------------------------------------------------
        arena = scene.build_arena()
        self.floor_id = arena["floor"]
        self.wall_ids = list(arena["walls"])
        self.drop_zone_pos = list(scene.DROP_ZONE_POS)
        self.drop_zone_half = scene.DROP_ZONE_HALF
        self.drop_zone_id = scene.build_drop_zone(self.drop_zone_pos)
        self.obstacle_id = None

        created = scene.spawn_objects()
        self.object_ids = [body for body, _ in created]
        self._specs = {body: spec for body, spec in created}

        self.robot_id = scene.load_humanoid(0.0, 0.0, 0.0)
        self._robot_com = tuple(p.getDynamicsInfo(self.robot_id, -1)[3])

        # --- subsystems --------------------------------------------------
        live = self.gui if realtime is None else bool(realtime)
        self.driver = HumanoidDriver(self.robot_id, realtime=live)
        self.camera = RobotCamera()
        self.depth_sensor = DepthSensor(self._depth_ignored_bodies)
        self.referee = Referee()

        # --- robot state -------------------------------------------------
        self.robot_pos = [0.0, 0.0, 0.0]
        self.robot_heading = 0.0
        self.held_object_id = None
        self.pickup_constraint = None
        self._carry_local = None          # (lx, ly, z) while holding
        self._set_pose(0.0, 0.0, 0.0)

        if self.gui:
            scene.configure_gui_lighting()
            p.resetDebugVisualizerCamera(cameraDistance=4.6, cameraYaw=45, cameraPitch=-42,
                                         cameraTargetPosition=[0.3, 0.3, 0.0])
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    # ------------------------------------------------------------------
    # Read-only views used by server.py / tests
    # ------------------------------------------------------------------
    @property
    def state_name(self):
        return self.referee.state

    @property
    def retries(self):
        return self.referee.retries

    @property
    def collided(self):
        return self.referee.collided

    @property
    def holding(self):
        return self.held_object_id is not None

    @_locked
    def get_state(self):
        """Snapshot in the /state schema."""
        return {
            "name": self.state_name,
            "x": round(self.robot_pos[0], 3),
            "y": round(self.robot_pos[1], 3),
            "heading": round(self.robot_heading, 1),
            "holding": self.holding,
            "retries": self.retries,
        }

    # ------------------------------------------------------------------
    # Frames and pose helpers
    # ------------------------------------------------------------------
    def _forward(self, heading=None):
        return heading_vector(self.robot_heading if heading is None else heading)

    def _left(self, heading=None):
        h = math.radians(self.robot_heading if heading is None else heading)
        return math.sin(h), math.cos(h)

    def _to_local(self, xyz):
        """World point -> (forward, left, z) in the robot frame."""
        dx, dy = xyz[0] - self.robot_pos[0], xyz[1] - self.robot_pos[1]
        fx, fy = self._forward()
        lx, ly = self._left()
        return (dx * fx + dy * fy, dx * lx + dy * ly, xyz[2])

    def _from_local(self, fwd, left, z, x=None, y=None, heading=None):
        x = self.robot_pos[0] if x is None else x
        y = self.robot_pos[1] if y is None else y
        fx, fy = self._forward(heading)
        lx, ly = self._left(heading)
        return (x + fwd * fx + left * lx, y + fwd * fy + left * ly, z)

    def _object_quat(self, heading=None):
        h = self.robot_heading if heading is None else heading
        return p.getQuaternionFromEuler([0.0, 0.0, -math.radians(h)])

    def _put_object(self, body, xyz, quat=None):
        p.resetBasePositionAndOrientation(body, list(xyz), list(quat or (0.0, 0.0, 0.0, 1.0)))
        p.resetBaseVelocity(body, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    def _set_pose(self, x, y, heading):
        """Place the robot base (and anything it carries). No animation."""
        self.robot_pos = [float(x), float(y), 0.0]
        self.robot_heading = _wrap_deg(float(heading))
        self.driver.set_base(self.robot_pos[0], self.robot_pos[1], self.robot_heading)
        if self.held_object_id is not None and self._carry_local is not None:
            self._put_object(self.held_object_id, self._from_local(*self._carry_local), self._object_quat())

    @_locked
    def _place_robot(self, x, y, heading):
        """Test/setup helper: teleport without animation or referee changes."""
        self._set_pose(x, y, heading)

    # ------------------------------------------------------------------
    # Collision
    # ------------------------------------------------------------------
    def _collidable_bodies(self):
        bodies = list(self.wall_ids) + [b for b in self.object_ids if b != self.held_object_id]
        if self.obstacle_id is not None:
            bodies.append(self.obstacle_id)
        return bodies

    def _penetration(self):
        """Deepest overlap (metres, >= 0) between the torso (or held object) and the world."""
        deepest = 0.0
        sources = [self.robot_id]
        if self.held_object_id is not None:
            sources.append(self.held_object_id)
        for src in sources:
            for body in self._collidable_bodies():
                for pt in p.getClosestPoints(bodyA=src, bodyB=body, distance=0.0, linkIndexA=-1):
                    deepest = max(deepest, -pt[8])
        return deepest

    @_locked
    def check_physical_collisions(self):
        return self._penetration() > COLLISION_TOL_M

    def _depth_ignored_bodies(self):
        ignored = [self.robot_id]
        if self.held_object_id is not None:
            ignored.append(self.held_object_id)
        return ignored

    # ------------------------------------------------------------------
    # Sensors
    # ------------------------------------------------------------------
    @_locked
    def get_camera_image(self):
        self.referee.on_look(self.holding)
        return self.camera.get_camera_image(self.robot_pos[0], self.robot_pos[1], self.robot_heading)

    @_locked
    def get_single_depth(self):
        return self.depth_sensor.get_depth_ray(self.robot_pos[0], self.robot_pos[1], self.robot_heading)

    @_locked
    def get_depth_fan(self):
        return self.depth_sensor.get_depth_fan(self.robot_pos[0], self.robot_pos[1], self.robot_heading)

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------
    def _animate_translate(self, distance, swing_arms):
        """Eased straight-line move. Returns (net_distance, hit)."""
        sx, sy, h = self.robot_pos[0], self.robot_pos[1], self.robot_heading
        fx, fy = self._forward(h)
        w0 = self.driver.wheel_angles()
        rest = self.driver.current_pose(tuple(REST_ARMS))
        start_pen = self._penetration()
        cycles = max(1, int(round(abs(distance) / 0.35)))
        state = {"safe": 0.0, "hit": False}

        def update(s):
            d = distance * smootherstep(s)
            self._set_pose(sx + fx * d, sy + fy * d, h)
            self.driver.roll_wheels(d, d, w0)
            if swing_arms:
                a = 0.35 * math.sin(2.0 * math.pi * cycles * s) * math.sin(math.pi * s)
                self.driver.set_joint("left_shoulder_joint", rest["left_shoulder_joint"] + a)
                self.driver.set_joint("right_shoulder_joint", rest["right_shoulder_joint"] - a)
            pen = self._penetration()
            # Contact counts only if it is new or getting deeper, so a robot
            # that starts touching something can still drive away from it.
            if pen > COLLISION_TOL_M and pen > start_pen + 1e-4:
                state["hit"] = True
                return False
            state["safe"] = d
            return None

        self.driver.run(abs(distance) / MOVE_SPEED_MPS, update)
        if state["hit"]:
            d = state["safe"]
            self._set_pose(sx + fx * d, sy + fy * d, h)
            self.driver.roll_wheels(d, d, w0)
        if swing_arms:
            self.driver.apply(rest)
        return state["safe"], state["hit"]

    def _animate_turn(self, degrees):
        h0 = self.robot_heading
        delta = float(degrees)
        w0 = self.driver.wheel_angles()
        arc = math.radians(delta) * scene.ROBOT_GEOMETRY["half_track"]
        sign = 1.0 if delta >= 0 else -1.0
        lead = min(0.45, abs(math.radians(delta)) * 0.4)
        head0 = self.driver.current_pose(("neck_joint", "head_joint"))

        def update(s):
            e = smootherstep(s)
            self._set_pose(self.robot_pos[0], self.robot_pos[1], h0 + delta * e)
            # Right (clockwise) turn: left wheel forward, right wheel back.
            self.driver.roll_wheels(arc * e, -arc * e, w0)
            bump = math.sin(math.pi * s)
            self.driver.set_joint("neck_joint", lerp(head0["neck_joint"], 0.0, e) - sign * lead * bump)
            self.driver.set_joint("head_joint", lerp(head0["head_joint"], 0.0, e) + 0.06 * bump)

        self.driver.run(abs(delta) / TURN_SPEED_DPS, update)
        self.driver.apply(HEAD_NEUTRAL)
        self._set_pose(self.robot_pos[0], self.robot_pos[1], h0 + delta)

    def _blocked_reaction(self, travel_sign):
        """0.1 m back-off opposite the travel direction, head recoil, eyes flash."""
        self.driver.set_eyes(True)
        sx, sy, h = self.robot_pos[0], self.robot_pos[1], self.robot_heading
        fx, fy = self._forward(h)
        back = -travel_sign * BACKOFF_M
        w0 = self.driver.wheel_angles()
        state = {"safe": 0.0}
        start_pen = self._penetration()

        def update(s):
            d = back * smootherstep(s)
            self._set_pose(sx + fx * d, sy + fy * d, h)
            self.driver.roll_wheels(d, d, w0)
            self.driver.set_joint("head_joint", -0.25 * math.sin(math.pi * s))
            pen = self._penetration()
            if pen > COLLISION_TOL_M and pen > start_pen + 1e-4:
                return False
            state["safe"] = d
            return None

        self.driver.run(0.35, update, min_frames=10)
        d = state["safe"]
        self._set_pose(sx + fx * d, sy + fy * d, h)
        self.driver.apply(HEAD_NEUTRAL)
        self.driver.set_eyes(False)
        return d

    @_locked
    def move_robot(self, distance):
        distance = float(distance)
        self.referee.begin_move(self.holding)
        self.driver.recentre_head()
        moved, hit = self._animate_translate(distance, swing_arms=not self.holding)
        if not hit:
            return {"moved": round(moved, 3), "collided": False}
        self.referee.on_collision()
        sign = 1.0 if distance >= 0 else -1.0
        moved += self._blocked_reaction(sign)
        if self.collision_nudge_deg:
            self._animate_turn(self.collision_nudge_deg)
        return {"moved": round(moved, 3), "collided": True}

    @_locked
    def turn_robot(self, degrees):
        self.referee.begin_turn(self.holding)
        self._animate_turn(degrees)
        return {"heading": round(self.robot_heading, 1)}

    @_locked
    def turn_to_heading(self, target_deg):
        """Animated turn the short way round to an absolute heading."""
        delta = ((float(target_deg) - self.robot_heading + 180.0) % 360.0) - 180.0
        if abs(delta) < 1e-6:
            return {"heading": round(self.robot_heading, 1)}
        return self.turn_robot(delta)

    # ------------------------------------------------------------------
    # Manipulation
    # ------------------------------------------------------------------
    def _carry_height(self, spec):
        return max(0.24, scene.object_half_height(spec) + 0.10)

    def _set_held_collisions(self, body, enabled):
        if enabled:
            p.setCollisionFilterGroupMask(body, -1, 1, -1)
        else:
            p.setCollisionFilterGroupMask(body, -1, 0, 0)

    def _attach(self, body, local):
        self.held_object_id = body
        self._carry_local = local
        self._set_held_collisions(body, False)
        cx, cy, cz = self._robot_com
        self.pickup_constraint = p.createConstraint(
            parentBodyUniqueId=self.robot_id, parentLinkIndex=-1,
            childBodyUniqueId=body, childLinkIndex=-1,
            jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
            parentFramePosition=[local[0] - cx, local[1] - cy, local[2] - cz],
            childFramePosition=[0, 0, 0],
        )
        p.changeConstraint(self.pickup_constraint, maxForce=500)

    def _release(self):
        body = self.held_object_id
        if self.pickup_constraint is not None:
            p.removeConstraint(self.pickup_constraint)
        self.pickup_constraint = None
        self.held_object_id = None
        self._carry_local = None
        if body is not None:
            self._set_held_collisions(body, True)
        return body

    @_locked
    def pickup_object(self):
        if self.held_object_id is not None:
            return {"ok": False}
        self.referee.begin_pickup()
        self.driver.recentre_head()

        rxy = (self.robot_pos[0], self.robot_pos[1])
        best = None
        for body in self.object_ids:
            pos, _ = p.getBasePositionAndOrientation(body)
            ok, dist = within_pickup_reach(rxy, self.robot_heading, (pos[0], pos[1]))
            if ok and (best is None or dist < best[1]):
                best = (body, dist, pos)

        if best is None:
            # Reach out, find nothing, come back.
            self.driver.blend_to(dict(arms_to(0.32, 0.18), head_joint=0.30, neck_joint=0.0), 0.45)
            self.driver.blend_to(NEUTRAL_POSE, 0.40)
            self.referee.end_pickup(False)
            return {"ok": False}

        body, _, pos = best
        spec = self._specs[body]
        f0, l0, z0 = self._to_local(pos)
        reach_z = max(0.10, pos[2])

        # 1. reach: both arms toward the object, visor tilts down
        self.driver.blend_to(dict(arms_to(min(f0, 0.33), reach_z), head_joint=0.30, neck_joint=0.0), 0.5)

        # 2. lift and tuck under the right arm
        target = (CARRY_LOCAL_XY[0], CARRY_LOCAL_XY[1], self._carry_height(spec))
        quat = self._object_quat()
        tuck = dict(arm_side_to("right", 0.16, 0.24), **TUCKED_FREE_ARM)
        tuck.update(HEAD_NEUTRAL)

        def carry_obj(s, e):
            lift = smootherstep(min(1.0, s * 2.0))     # rise first, then slide to the hip
            slide = smootherstep(max(0.0, s * 2.0 - 0.6) / 1.4)
            z = lerp(z0, target[2], lift)
            self._put_object(body, self._from_local(lerp(f0, target[0], slide), lerp(l0, target[1], slide), z), quat)

        self.driver.blend_to(tuck, 0.7, extra=carry_obj, step_physics=False)
        self._attach(body, target)
        self._set_pose(self.robot_pos[0], self.robot_pos[1], self.robot_heading)
        self.referee.end_pickup(True)
        return {"ok": True}

    @_locked
    def drop_object(self):
        if self.held_object_id is None:
            return {"ok": False, "in_zone": False}
        self.referee.begin_drop()
        body = self.held_object_id
        spec = self._specs[body]
        hz = scene.object_half_height(spec)
        sx, sy, sz = self._carry_local
        quat = self._object_quat()

        # 1. swing from the hip to straight ahead
        def swing(s, e):
            self._carry_local = (lerp(sx, DROP_AHEAD_M, e), lerp(sy, 0.0, e), sz)
            self._put_object(body, self._from_local(*self._carry_local), quat)

        self.driver.blend_to(arms_to(0.33, sz), 0.5, extra=swing, step_physics=False)

        # 2. lower it to the floor
        floor_z = hz + 0.001

        def lower(s, e):
            self._carry_local = (DROP_AHEAD_M, 0.0, lerp(sz, floor_z, e))
            self._put_object(body, self._from_local(*self._carry_local), quat)

        self.driver.blend_to(dict(arms_to(0.33, max(0.10, floor_z)), head_joint=0.25, neck_joint=0.0),
                             0.5, extra=lower, step_physics=False)

        # 3. release and retract
        self._release()
        self._put_object(body, self._from_local(DROP_AHEAD_M, 0.0, floor_z), quat)
        self.driver.blend_to(NEUTRAL_POSE, 0.4)

        pos, _ = p.getBasePositionAndOrientation(body)
        landed = in_drop_zone((pos[0], pos[1]), self.drop_zone_pos, self.drop_zone_half)
        self.referee.end_drop(landed)
        return {"ok": True, "in_zone": bool(landed)}

    # ------------------------------------------------------------------
    # Scene control
    # ------------------------------------------------------------------
    def _spawn_ok(self, x, y, r, placed):
        robot_r = scene.ROBOT_GEOMETRY["collision_radius"]
        if math.hypot(x, y) < robot_r + r + SPAWN_ROBOT_CLEAR:
            return False
        # Keep the lane straight ahead of the start pose (heading 0 = +x) clear
        # for the first 1 m, so the opening move can never hit an object.
        if -0.3 < x < 1.0 + robot_r + r and abs(y) < robot_r + r + 0.1:
            return False
        zx, zy = self.drop_zone_pos[0], self.drop_zone_pos[1]
        if max(abs(x - zx), abs(y - zy)) < self.drop_zone_half + r + 0.2:
            return False
        if self.obstacle_id is not None:
            ox, oy, hx, hy = scene.OBSTACLE_FOOTPRINT
            dx = max(0.0, abs(x - ox) - hx)
            dy = max(0.0, abs(y - oy) - hy)
            if math.hypot(dx, dy) < r + 0.15:
                return False
        return all(math.hypot(x - px, y - py) >= r + pr + SPAWN_GAP for px, py, pr in placed)

    def _scatter_objects(self, rng):
        placed = []
        for body in self.object_ids:
            spec = self._specs[body]
            r = scene.object_footprint_radius(spec)
            lim = scene.WALL_INNER - r - SPAWN_WALL_MARGIN
            for _ in range(5000):
                x, y = round(rng.uniform(-lim, lim), 2), round(rng.uniform(-lim, lim), 2)
                if self._spawn_ok(x, y, r, placed):
                    break
            placed.append((x, y, r))
            self._put_object(body, (x, y, scene.object_half_height(spec) + 0.001))

    @_locked
    def reset_scene(self, seed=None):
        if self.held_object_id is not None:
            self._release()
        self._scatter_objects(random.Random(seed))
        self.driver.set_eyes(False)
        self.driver.apply(NEUTRAL_POSE)
        self._set_pose(0.0, 0.0, 0.0)
        self.referee.reset()
        self.driver.scan()
        return {"status": "reset_complete"}

    @_locked
    def enable_obstacle_mode(self):
        if self.obstacle_id is None:
            self.obstacle_id = scene.build_obstacle()
            ox, oy, hx, hy = scene.OBSTACLE_FOOTPRINT
            for body in self.object_ids:
                pos, _ = p.getBasePositionAndOrientation(body)
                r = scene.object_footprint_radius(self._specs[body])
                if abs(pos[0] - ox) < hx + r + 0.15 and abs(pos[1] - oy) < hy + r + 0.15:
                    self._scatter_objects(random.Random())
                    break
        return {"obstacle_enabled": True}

    @_locked
    def give_up(self):
        self.referee.give_up()
        return self.referee.summary()

    def close(self):
        with self.lock:
            if p.isConnected():
                p.disconnect()