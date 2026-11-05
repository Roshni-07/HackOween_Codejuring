"""
sim/referee.py — the run state machine and the task rules.

States (brief, section 01):
    Idle -> Locate -> GoToObject -> PickUp -> Carry -> Drop -> Done
    Blocked: entered on any collision; the next command leaves it.
             After MAX_RETRIES collisions the run fails (Done / fail_collision).

The engine cannot know *which* object Claude is aiming for, so the
navigation states are inferred from what the robot is doing:
    /look or /turn while empty-handed  -> Locate
    /move while empty-handed           -> GoToObject
    any motion while holding           -> Carry
Animations call begin_*/end_* so /state reports PickUp / Drop *while* the
arm animation is playing, then the settled state when it finishes.

Done is terminal until /reset: later commands still move the robot (so an
agent never crashes) but the recorded outcome cannot be overwritten.

Outcome codes: pass | fail_wrong_object | fail_not_reached | fail_collision.
fail_wrong_object needs the command's intended target, which the engine never
sees — it is judged from the logs / clip, so the referee never sets it.
"""

import math
import time


class State:
    IDLE = "Idle"
    LOCATE = "Locate"
    GO_TO_OBJECT = "GoToObject"
    PICK_UP = "PickUp"
    CARRY = "Carry"
    DROP = "Drop"
    DONE = "Done"
    BLOCKED = "Blocked"


class Outcome:
    PASS = "pass"
    FAIL_WRONG_OBJECT = "fail_wrong_object"
    FAIL_NOT_REACHED = "fail_not_reached"
    FAIL_COLLISION = "fail_collision"


PICKUP_REACH_M = 0.5
PICKUP_CONE_DEG = 30.0
MAX_RETRIES = 3


def angle_diff_deg(a, b):
    """Smallest absolute difference between two headings, in degrees."""
    return abs(((a - b + 180.0) % 360.0) - 180.0)


def bearing_cw_deg(from_xy, to_xy):
    """Bearing from one point to another in the clockwise-positive convention."""
    return (-math.degrees(math.atan2(to_xy[1] - from_xy[1], to_xy[0] - from_xy[0]))) % 360.0


def within_pickup_reach(robot_xy, heading_deg, obj_xy, reach=PICKUP_REACH_M, cone=PICKUP_CONE_DEG):
    """Returns (ok, distance). Brief: reach 0.5 m and within +-30° of heading."""
    dist = math.hypot(obj_xy[0] - robot_xy[0], obj_xy[1] - robot_xy[1])
    if dist > reach:
        return False, dist
    return angle_diff_deg(bearing_cw_deg(robot_xy, obj_xy), heading_deg) <= cone, dist


def in_drop_zone(obj_xy, zone_xy, half):
    return abs(obj_xy[0] - zone_xy[0]) <= half and abs(obj_xy[1] - zone_xy[1]) <= half


class Referee:
    def __init__(self, max_retries=MAX_RETRIES, history=500):
        self.max_retries = max_retries
        self._history = history
        self.reset()

    # -- lifecycle -------------------------------------------------------
    def reset(self):
        self.state = State.IDLE
        self.retries = 0
        self.collided = False
        self.outcome = None
        self.events = []
        self._log("reset", State.IDLE)

    @property
    def terminal(self):
        return self.state == State.DONE

    def _log(self, reason, new_state):
        self.events.append({"t": round(time.time(), 3), "from": getattr(self, "state", None),
                            "to": new_state, "reason": reason})
        if len(self.events) > self._history:
            del self.events[: len(self.events) - self._history]

    def _go(self, new_state, reason):
        if self.terminal:
            return
        if new_state != self.state:
            self._log(reason, new_state)
        self.state = new_state

    # -- command hooks ---------------------------------------------------
    def on_look(self, holding):
        if self.state == State.IDLE:
            self._go(State.CARRY if holding else State.LOCATE, "look")

    def begin_move(self, holding):
        self.collided = False
        self._go(State.CARRY if holding else State.GO_TO_OBJECT, "move")

    def begin_turn(self, holding):
        self.collided = False
        self._go(State.CARRY if holding else State.LOCATE, "turn")

    def on_collision(self):
        self.collided = True
        if self.terminal:
            return
        self.retries += 1
        if self.retries > self.max_retries:
            self.outcome = Outcome.FAIL_COLLISION
            self._go(State.DONE, "collision retries exhausted")
        else:
            self._go(State.BLOCKED, "collision")

    def begin_pickup(self):
        self._go(State.PICK_UP, "pickup")

    def end_pickup(self, ok):
        self._go(State.CARRY if ok else State.LOCATE, "pickup ok" if ok else "pickup missed")

    def begin_drop(self):
        self._go(State.DROP, "drop")

    def end_drop(self, landed_in_zone):
        if landed_in_zone:
            if not self.terminal:
                self.outcome = Outcome.PASS
            self._go(State.DONE, "delivered to zone")
        else:
            self._go(State.LOCATE, "dropped outside zone")

    def give_up(self):
        """For an agent / harness that stops before delivering."""
        if not self.terminal:
            self.outcome = Outcome.FAIL_NOT_REACHED
            self._go(State.DONE, "gave up")

    def summary(self):
        return {"state": self.state, "retries": self.retries, "outcome": self.outcome,
                "collided": self.collided}
