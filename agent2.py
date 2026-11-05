#!/usr/bin/env python3
"""
claude_agent.py — Claude agent for the Membrane Robot Hackathon "Fetch" task.

Claude sees the world only through the robot's camera (/look), the centre
depth ray (/depth) and the collision flag. No object data is ever sent to it:
the system prompt and tools describe the robot only.

    pip install anthropic requests pillow
    export ANTHROPIC_API_KEY=sk-ant-...

    python claude_agent.py "Bring the red cube to the orange square."
    python claude_agent.py --model claude-haiku-5-5 "..."   # cheap plumbing test
    python claude_agent.py --dummy "..."                     # no API calls at all
    ENGINE_URL=http://10.121.224.93:8080 python claude_agent.py "..."   # via the bridge

Per step (problem statement "Claude budget"):
    * one Claude call in, exactly one tool call out (tool_choice any, no parallel calls)
    * the latest /look frame goes back as an image block INSIDE the tool result;
      every earlier step is one line of text
    * /look happens automatically after every /move and /turn, never after
      /pickup or /drop, and never twice without motion in between
    * prompt caching: tools + system prompt are cached, and so is the
      append-only history, so each step only pays full price for the new frame

Memory (for the judges): when Claude reports a sighting of the drop zone or a candidate
target (bearing and floor distance read off the camera frame), the agent stores it as a
world point using the robot's own pose from /state (x, y, heading = wheel odometry), and
returns it relative to the robot in later results ("turn 120 deg, 2.1 m"). The engine is
never asked where anything is: every remembered location comes from Claude's own pixels.

Logs: logs/run_<time>/run_log.jsonl (one line per step: step, timestamp, tool,
args, result, frame_path, plus Claude's reason, token usage and cost) and the
frames Claude saw as PNG next to it. The last line holds the Done result:
pass | fail_wrong_object | fail_not_reached | fail_collision.
"""

import argparse
import base64
import io
import json
import math
import os
import sys
import time
from datetime import datetime

import requests
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageStat

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ENGINE_URL = os.environ.get("ENGINE_URL", "http://10.121.224.248:8080").rstrip("/")
DEFAULT_MODEL = os.environ.get("ROBOT_MODEL", "claude-sonnet-5-5")
MAX_STEPS = 60                  # hard stop: report fail at step 60
SOFT_STEPS = 25                 # target for a clean run
MAX_RETRIES = 3                 # 4th collision ends the run
MAX_OUTPUT_TOKENS = 2000        # cap on thinking + reply together; a step normally uses far less
EFFORT = os.environ.get("ROBOT_EFFORT", "low")   # low = little thinking: fast, cheap steps
HTTP_TIMEOUT_S = 30             # moves/turns animate in real time on the engine
STALE_FRAME_WAIT_S = 5.0        # after motion, wait up to this long for a changed frame

# Camera (data contract; change here if your engine differs)
IMG_W, IMG_H = 640, 480
HFOV_DEG = 90.0
VFOV_DEG = 73.74
CAM_HEIGHT_M = 0.5
CAM_TILT_DEG = 10.0
CAM_FORWARD_M = 0.12            # camera sits this far ahead of the robot centre

BEARING_TICKS = (-40, -30, -20, -10, 0, 10, 20, 30, 40)
FLOOR_TICKS_M = (0.6, 0.7, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0)

# $ per million tokens: input, output, 5-min cache write, cache read
PRICES = {
    "claude-sonnet-5-5": (2.00, 10.00, 2.50, 0.10),
    "claude-haiku-5-5": (0.10, 0.50, 0.125, 0.01)
}


SYSTEM_PROMPT = """You drive a wheeled robot using only its forward camera. The robot never tells you about objects: no names, colours, shapes, positions, counts or distances to objects. Everything you know about the world comes from the camera frames you are shown, one centre depth ray, and a collision flag.

THE ROBOT
- Wheeled base that turns in place. Its body is a capsule 0.44 m wide (0.22 m radius).
- Camera: front of the robot, 0.5 m above the floor, tilted 10 degrees down, 640x480 pixels, 90 degree horizontal field of view.
- Every frame carries a HUD drawn by the robot (it is not part of the world):
  * Top edge: bearing ticks in degrees from -40 to +40. Something under the +20 tick is 20 degrees to the right: turn(+20) puts it dead ahead. Negative is left.
  * Left and right edges: floor-distance ticks in metres, from the robot's centre to the point on the FLOOR at that image row. Read them where a thing touches the floor (its base), never at its top. The bottom edge of the frame is about 0.6 m away; anything closer is below the frame.
- Depth: one ray from the camera centre along the 10 degree tilt, in metres (max 10, null if nothing). On open floor it reads about 2.8 m because it hits the floor. Things lower than the camera can pass under the ray when closer than about 1.3 m, so judge distance to small things with the floor ticks, and use depth to spot anything tall straight ahead.
- pickup: snaps the nearest object whose closest surface is within 0.5 m of the robot centre and within 30 degrees of straight ahead. Size does not matter: big things can be picked up too. It only answers ok true or false.
- While carrying, the object rides at the robot's side or behind it, outside the camera view.
- drop: sets the carried object down straight ahead, with its centre about 0.42 m from the robot centre (further for big things, so it clears the body), just below the bottom edge of the frame. It answers ok and in_zone.
- Collisions: if the body touches anything, the move stops, the robot backs off 0.1 m and the result says collided true. After 3 retries, the next collision fails the run.

HOW TO WORK
1. Read the command. Decide which visible properties pick out the target object and the drop zone. For commands that need reasoning ("the round one", "the tallest object", "something to drink") compare every candidate you have seen before committing, and name your choice in reason.
2. Locate: look once. If a frame shows nothing useful, turn 80 degrees: the view is 90 degrees wide, so five turns cover a full circle. If something promising sits near an edge, turn toward it by its bearing instead. If nothing matches after a full circle, drive up to 2 m toward open floor (check that depth is long enough first) and sweep again.
3. Centre: read the target's bearing on the top scale and turn by exactly that amount (for example turn -17). One centring turn is enough: once the target is within 10 degrees of the centre, approach it. Never make two centring turns in a row without moving; small errors are corrected on the way.
4. Approach: read the floor distance D at the near edge of the target's base and move D - 0.35 m. That leaves its near side 0.35 m from the robot centre: in reach, not touching, whatever its size. If D is more than 1.8 m, move 1.5 m at most, then re-centre and measure again.
5. Something between you and where you want to go, or blocking the view: do not drive into it, and do not sweep again from the same spot. Go around it: turn about 60 degrees away, move 0.6 to 1.0 m, then turn back toward your goal (use memory turn_deg) and check the new frame.
6. Pick up when the target is centred and its near edge is at or below the bottom edge of the frame (a big target may still fill much of the view). If pickup answers ok false: if the target is still centred and near, move 0.1 m once and try again; otherwise re-centre first. Never creep forward more than once in a row: that ends in a collision.
7. Carry: if memory has the drop_zone, turn by its turn_deg straight away (no sweep), then approach and confirm it in the frame. Otherwise sweep for it. Centre on the middle of the zone and approach. Drop only when the zone runs off the bottom edge of the frame, its far edge sits between the 0.7 m and 1.0 m ticks, and it is centred left to right. If its far edge is further away, move (far edge distance - 0.85) m, re-centre, and check again.
8. in_zone true ends the run. If in_zone is false, the object now lies about 0.42 m ahead and is still within reach: call pickup straight away, then deliver it again.
9. After a collision, never repeat the same move. Study the new frame, turn 45 to 60 degrees away from what you hit, move a short way, then re-acquire your target.
10. Commands that only ask you to pick something up (no place to bring it) are complete once pickup answers ok true: then call finish.
11. Call give_up only after two full sweeps from different places show nothing that fits the command.

MEMORY
- Whenever the frame you were given shows the drop zone, or something that could be your target, add it to sightings: what, its bearing from the top scale, and the floor distance at the middle of its base from the side scale. Report the drop zone even while you are still looking for the target.
- The robot combines your sightings with its own wheel odometry and returns memory in every result: turn_deg and distance_m from where you are now to each thing you reported. It is approximate (about 10 degrees, 0.3 m); confirm with the camera before pickup or drop.

RULES
- Exactly one tool call per reply. Put what you see and why you act in reason, at most 25 words.
- Every turn and move returns a fresh frame. Do not call look again until you have turned or moved.
- Be direct: aim for under 25 steps. The run fails at step 60."""

_REASON = {"type": "string", "description": "What you see in the latest frame and why this action. Max 25 words."}
_SIGHTINGS = {
    "type": "array",
    "description": "Optional. Things in the frame you were just given that are worth remembering.",
    "items": {
        "type": "object",
        "properties": {
            "what": {"type": "string", "enum": ["drop_zone", "target"]},
            "bearing_deg": {"type": "number", "description": "From the top scale. Positive = right."},
            "distance_m": {"type": "number", "description": "Floor distance at the middle of its base (side scale)."},
        },
        "required": ["what", "bearing_deg", "distance_m"],
        "additionalProperties": False,
    },
}

TOOLS = [
    {
        "name": "look",
        "description": "Capture one frame from the robot camera (GET /look) plus the centre depth reading. "
                       "Use it once at the start. Turns and moves already return a fresh frame.",
        "input_schema": {"type": "object", "properties": {"reason": _REASON, "sightings": _SIGHTINGS},
                         "required": ["reason"], "additionalProperties": False},
    },
    {
        "name": "turn",
        "description": "Turn in place (POST /turn) at 90 deg/s. Positive degrees = right (clockwise), negative = left. "
                       "Returns the new heading and a fresh camera frame.",
        "input_schema": {
            "type": "object",
            "properties": {"reason": _REASON,
                           "degrees": {"type": "number", "description": "-180 to 180"},
                           "sightings": _SIGHTINGS},
            "required": ["reason", "degrees"],
            "additionalProperties": False,
        },
    },
    {
        "name": "move",
        "description": "Drive straight (POST /move) at 1 m/s. Positive = forward, negative = reverse, metres. "
                       "Stops early on contact (collided true, robot backs off 0.1 m). Returns a fresh camera frame.",
        "input_schema": {
            "type": "object",
            "properties": {"reason": _REASON,
                           "distance": {"type": "number", "description": "-1.0 to 2.0"},
                           "sightings": _SIGHTINGS},
            "required": ["reason", "distance"],
            "additionalProperties": False,
        },
    },
    {
        "name": "pickup",
        "description": "Snap the nearest object within 0.5 m and 30 degrees of straight ahead (POST /pickup). "
                       "Returns ok true/false only. No new frame.",
        "input_schema": {"type": "object", "properties": {"reason": _REASON}, "required": ["reason"],
                         "additionalProperties": False},
    },
    {
        "name": "drop",
        "description": "Release the carried object about 0.42 m ahead (POST /drop). Returns ok and in_zone. "
                       "No new frame.",
        "input_schema": {"type": "object", "properties": {"reason": _REASON}, "required": ["reason"],
                         "additionalProperties": False},
    },
    {
        "name": "finish",
        "description": "End the run as complete when the command needs nothing more: e.g. it only asked "
                       "to pick something up and pickup answered ok true. Delivery runs end by themselves "
                       "when drop answers in_zone true.",
        "input_schema": {"type": "object", "properties": {"reason": _REASON}, "required": ["reason"],
                         "additionalProperties": False},
    },
    {
        "name": "give_up",
        "description": "End the run as failed because the target or the drop zone cannot be found.",
        "input_schema": {"type": "object", "properties": {"reason": _REASON}, "required": ["reason"],
                         "additionalProperties": False},
    },
]

for _tool in TOOLS:
    _tool["strict"] = True          # inputs always match the schema

SYSTEM_BLOCKS = [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]
# Sonnet/Opus 5.5 reject forced tool use ("any"/"tool"), so: auto + at most one call.
# The system prompt asks for exactly one tool call; a text-only reply is re-asked once.
TOOL_CHOICE = {"type": "auto", "disable_parallel_tool_use": True}   # constant -> cache stays valid
NUDGE = "Reply with exactly one tool call now."


# ---------------------------------------------------------------------------
# Camera geometry + HUD
# ---------------------------------------------------------------------------
_FX = (IMG_W / 2) / math.tan(math.radians(HFOV_DEG / 2))
_TAN_HALF_V = math.tan(math.radians(VFOV_DEG / 2))


def bearing_to_x(deg):
    return IMG_W / 2 + _FX * math.tan(math.radians(deg))


def floor_distance_to_row(dist_m):
    """Image row of a floor point dist_m from the robot centre (None if above the frame)."""
    d_cam = dist_m - CAM_FORWARD_M
    if d_cam <= 0:
        return None
    below = math.degrees(math.atan2(CAM_HEIGHT_M, d_cam)) - CAM_TILT_DEG
    row = IMG_H / 2 + (IMG_H / 2) * math.tan(math.radians(below)) / _TAN_HALF_V
    return row if 0 <= row <= IMG_H else None


def _font():
    try:
        return ImageFont.load_default(size=13)
    except TypeError:               # Pillow < 10.1
        return ImageFont.load_default()


_FONT = None


def draw_hud(img):
    """Bearing scale on top, floor-distance scale on both sides. Camera intrinsics only."""
    global _FONT
    _FONT = _FONT or _font()
    img = img.convert("RGB")
    d = ImageDraw.Draw(img)
    ink, edge = (255, 235, 59), (0, 0, 0)

    def text(xy, s, anchor):
        d.text(xy, s, fill=ink, font=_FONT, anchor=anchor, stroke_width=2, stroke_fill=edge)

    for deg in BEARING_TICKS:
        x = bearing_to_x(deg)
        length = 16 if deg == 0 else 9
        d.line([(x, 0), (x, length)], fill=edge, width=4)
        d.line([(x, 0), (x, length)], fill=ink, width=2)
        text((x, length + 2), ("%+d" % deg) if deg else "0", "mt")
    d.line([(IMG_W / 2, IMG_H - 10), (IMG_W / 2, IMG_H)], fill=ink, width=2)    # centre mark at the bottom

    for dist in FLOOR_TICKS_M:
        row = floor_distance_to_row(dist)
        if row is None:
            continue
        label = "%gm" % dist
        for x0, x1, tx, anchor in ((0, 12, 15, "lm"), (IMG_W - 12, IMG_W, IMG_W - 15, "rm")):
            d.line([(x0, row), (x1, row)], fill=edge, width=4)
            d.line([(x0, row), (x1, row)], fill=ink, width=2)
            text((tx, row), label, anchor)
    return img


# ---------------------------------------------------------------------------
# Engine HTTP client (the only thing that talks to the robot)
# ---------------------------------------------------------------------------
class Engine:
    def __init__(self, base_url=ENGINE_URL):
        self.base = base_url
        self.http = requests.Session()
        self.depth_available = True

    def _get(self, path):
        r = self.http.get(self.base + path, timeout=HTTP_TIMEOUT_S)
        r.raise_for_status()
        return r

    def _post(self, path, body=None):
        r = self.http.post(self.base + path, json=body if body is not None else {}, timeout=HTTP_TIMEOUT_S)
        r.raise_for_status()
        return r.json()

    def state(self):
        try:
            return self._get("/state").json()
        except Exception:
            return {}

    def look_jpeg(self):
        return self._get("/look").content

    def depth(self):
        if not self.depth_available:
            return None
        try:
            return self._get("/depth").json().get("depth")
        except requests.HTTPError:
            self.depth_available = False          # optional sensor not offered
        except Exception:
            pass
        return None

    def move(self, distance):
        return self._post("/move", {"distance": distance})

    def turn(self, degrees):
        return self._post("/turn", {"degrees": degrees})

    def pickup(self):
        return self._post("/pickup")

    def drop(self):
        return self._post("/drop")


# ---------------------------------------------------------------------------
# Claude client (real, or a dummy for plumbing tests)
# ---------------------------------------------------------------------------
API_KEY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_key.txt")


def _clean_key(raw):
    """Undo common paste damage: spaces, line breaks, quotes, a leading 'ANTHROPIC_API_KEY='."""
    key = "".join(raw.split())
    if key.upper().startswith("ANTHROPIC_API_KEY="):
        key = key.split("=", 1)[1]
    return key.strip("\"'")


def load_api_key():
    """api_key.txt next to this script first (so a stale or half-pasted environment
    variable can't override it), else the ANTHROPIC_API_KEY environment variable.
    Returns (key, where_it_came_from). Keep api_key.txt out of git."""
    if os.path.exists(API_KEY_FILE):
        with open(API_KEY_FILE, encoding="utf-8-sig") as fh:
            key = _clean_key(fh.read())
        if key:
            return key, API_KEY_FILE
    key = _clean_key(os.environ.get("ANTHROPIC_API_KEY", ""))
    if key:
        return key, "ANTHROPIC_API_KEY environment variable"
    sys.exit("No API key found. Paste the key into %s (or set ANTHROPIC_API_KEY)." % API_KEY_FILE)


def describe_key(key, source):
    masked = "%s...%s" % (key[:12], key[-4:]) if len(key) > 20 else "(too short)"
    print("API key from %s: %s  (%d characters)" % (source, masked, len(key)))
    if not key.startswith("sk-ant-"):
        print("  ! Anthropic API keys start with 'sk-ant-'. This one does not: check what you pasted.")
    if len(key) < 90:
        print("  ! Anthropic API keys are usually about 108 characters. This one looks cut off.")
    if os.environ.get("ANTHROPIC_BASE_URL"):
        print("  Using ANTHROPIC_BASE_URL = %s" % os.environ["ANTHROPIC_BASE_URL"])


def check_key():
    """Free check: list models with the key (no tokens used)."""
    import anthropic
    key, source = load_api_key()
    describe_key(key, source)
    try:
        models = anthropic.Anthropic(api_key=key).models.list(limit=20)
        print("OK: the API accepted the key. Models available: %s"
              % ", ".join(m.id for m in models.data))
        return 0
    except anthropic.AuthenticationError as exc:
        print("REJECTED (401 authentication error): %s" % exc)
        print("  The key itself is wrong, revoked, or not a direct Anthropic API key.\n"
              "  Copy it again (or create a new one) in the Claude Console, paste it alone into\n"
              "  %s, and remove any ANTHROPIC_API_KEY from your PyCharm Run Configuration.\n"
              "  If the hackathon gave you a gateway URL with the key, set ANTHROPIC_BASE_URL to it."
              % API_KEY_FILE)
    except anthropic.APIError as exc:
        print("The key was not rejected, but the call failed: %s" % exc)
    return 1


def selftest(engine):
    """No Claude, no key: does this server really move the robot and update the camera?"""
    def frame():
        return FetchAgent._signature(Image.open(io.BytesIO(engine.look_jpeg())))

    def diff(a, b):
        return ImageStat.Stat(ImageChops.difference(a, b)).mean[0]

    def pose():
        st = engine.state()
        return st, (st.get("x"), st.get("y"), st.get("heading"))

    print("Self-test against %s\n" % engine.base)
    try:
        st, _ = pose()
    except Exception as exc:
        print("FAIL  cannot reach %s/state: %s" % (engine.base, exc))
        return 1
    print("GET /state  -> %s" % json.dumps(st))
    want = {"name", "x", "y", "heading", "holding", "retries"}
    print(("OK    " if want <= set(st) else "FAIL  ") + "state fields %s" % sorted(st))
    verdict = {}

    for label, call, undo, key in (("move 0.3 m", lambda: engine.move(0.3), lambda: engine.move(-0.3), "moved"),
                                   ("turn 30 deg", lambda: engine.turn(30), lambda: engine.turn(-30), "heading")):
        f0, (_, p0) = frame(), pose()
        try:
            reply = call()
        except Exception as exc:
            print("FAIL  %s: request failed: %s" % (label, exc))
            verdict[label] = False
            continue
        print("\nPOST %-11s -> %s" % (label, json.dumps(reply)))
        if key not in reply:
            print("FAIL  reply has no '%s' field: this server is not the robot engine (a bridge?)" % key)
        changed, waited = 0.0, 0.0
        while waited < 5.0:
            changed = diff(f0, frame())
            if changed >= 0.5:
                break
            time.sleep(0.5); waited += 0.5
        _, p1 = pose()
        print(("OK    " if changed >= 0.5 else "FAIL  ") + "camera view %s (difference %.1f, waited %.1fs)"
              % ("changed" if changed >= 0.5 else "did NOT change", changed, waited))
        print(("OK    " if p1 != p0 else "FAIL  ") + "/state pose %s: %s -> %s"
              % ("changed" if p1 != p0 else "did NOT change", p0, p1))
        verdict[label] = changed >= 0.5
        try:
            undo()
        except Exception:
            pass

    print("\nVERDICT")
    if all(verdict.values()):
        print("  The robot turns AND moves through this server. If runs still stall, send the run log.")
    elif verdict.get("turn 30 deg") and not verdict.get("move 0.3 m"):
        print("  The robot TURNS but does NOT MOVE through %s.\n"
              "  This server only relays heading. Point ENGINE_URL at the PC running server.py." % engine.base)
    else:
        print("  The robot does not respond properly through %s. Is server.py running there?" % engine.base)
    return 0 if all(verdict.values()) else 1


class ClaudeBrain:
    def __init__(self, model):
        import anthropic                          # imported here so --dummy needs no SDK
        self.anthropic = anthropic
        key, source = load_api_key()
        describe_key(key, source)
        self.client = anthropic.Anthropic(api_key=key, max_retries=4)
        self.model = model

        self.effort = EFFORT or None

    def _create(self, messages):
        extra = {"output_config": {"effort": self.effort}} if self.effort else None
        return self.client.messages.create(
            model=self.model,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=SYSTEM_BLOCKS,
            tools=TOOLS,
            tool_choice=TOOL_CHOICE,
            messages=messages,
            extra_body=extra,
        )

    def decide(self, messages):
        """One step. Returns (tool_use dict, usage, assistant blocks to replay next step)."""
        usage_total = {}
        ask = messages
        for attempt in range(4):
            try:
                msg = self._create(ask)
            except self.anthropic.BadRequestError as exc:
                text = str(exc)
                if self.effort and ("effort" in text or "output_config" in text):
                    print("[claude] effort setting not accepted here; continuing without it.")
                    self.effort = None
                    continue
                raise SystemExit("[claude] request rejected (400): %s" % text)
            except (self.anthropic.AuthenticationError, self.anthropic.PermissionDeniedError) as exc:
                raise SystemExit("[claude] %s\nRun with --check-key to test the key." % exc)
            except self.anthropic.APIError as exc:          # overload, rate limit, network
                print("[claude] API error (%s), retrying..." % exc)
                time.sleep(3 * (attempt + 1))
                continue

            for k, v in _usage_dict(msg.usage).items():
                usage_total[k] = usage_total.get(k, 0) + v
            replay, call = [], None
            for block in msg.content:
                # Thinking blocks are NOT replayed: they are bound to the exact conversation
                # prefix, and our prefix grows a history line every step. The API accepts
                # assistant turns without thinking.
                if block.type == "text" and block.text.strip():
                    replay.append({"type": "text", "text": block.text})
                elif block.type == "tool_use" and call is None:
                    call = {"type": "tool_use", "id": block.id, "name": block.name, "input": dict(block.input)}
                    replay.append(call)
            if call is not None:
                return call, usage_total, replay

            said = " ".join(b.text for b in msg.content if b.type == "text").strip()
            print("[claude] no tool call (stop_reason=%s): %s" % (msg.stop_reason, said[:120]))
            ask = _with_nudge(messages)
        raise RuntimeError("Claude did not return a tool call")


class DummyBrain:
    """Fixed moves, no API calls: checks the engine plumbing for free."""
    script = [("look", {}), ("turn", {"degrees": 45}), ("move", {"distance": 0.3}),
              ("turn", {"degrees": -45}), ("move", {"distance": -0.3}), ("pickup", {}), ("give_up", {})]

    def __init__(self):
        self.model = "dummy"
        self.i = 0

    def decide(self, messages):
        name, args = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        call = {"type": "tool_use", "id": "toolu_dummy_%03d" % self.i, "name": name,
                "input": dict(args, reason="dummy plumbing test")}
        return call, {}, [call]


def _as_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _with_nudge(messages):
    """Same request plus one line at the end of the last user turn (cached prefix unchanged)."""
    out = [dict(m) for m in messages]
    out[-1] = dict(out[-1], content=list(out[-1]["content"]) + [{"type": "text", "text": NUDGE}])
    return out


def _usage_dict(u):
    get = lambda k: int(getattr(u, k, 0) or 0)     # noqa: E731
    return {"input": get("input_tokens"), "output": get("output_tokens"),
            "cache_write": get("cache_creation_input_tokens"), "cache_read": get("cache_read_input_tokens")}


def step_cost(model, usage):
    p_in, p_out, p_write, p_read = PRICES.get(model, PRICES[DEFAULT_MODEL])
    return (usage.get("input", 0) * p_in + usage.get("output", 0) * p_out
            + usage.get("cache_write", 0) * p_write + usage.get("cache_read", 0) * p_read) / 1e6


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------
class FetchAgent:
    def __init__(self, command, brain, engine, hud=True, log_root="logs"):
        self.command = command
        self.brain = brain
        self.engine = engine
        self.hud = hud
        self.run_dir = os.path.join(log_root, "run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.run_dir, exist_ok=True)
        self.log_path = os.path.join(self.run_dir, "run_log.jsonl")

        self.history = []            # one text line per finished step (append-only -> cacheable)
        self.last_call = None        # assistant tool_use block of the latest step
        self.last_turn = None        # that step's full assistant content (thinking + tool_use)
        self.last_result = None      # its tool_result content blocks
        self.frame_is_current = False
        self.collisions = 0
        self.holding = False
        self.total_cost = 0.0
        self.warned_static = False
        self.warned_contract = False
        self.warned_stale = False
        self.frame_unchanged = False
        self._last_sig = None
        self.settle_s = 0.0          # extra wait before /look after motion (only for slow bridges)
        self._last_error = None
        self._summary_line = ""
        self.memory = {}             # what -> world (x, y, step): built from Claude's own sightings

    # -- request building ---------------------------------------------------
    def build_messages(self):
        head = [{"type": "text", "text": "Command: %s" % self.command}]
        head += [{"type": "text", "text": line} for line in self.history]
        head[-1] = dict(head[-1], cache_control={"type": "ephemeral"})
        messages = [{"role": "user", "content": head}]
        if self.last_call is None:
            return messages
        messages.append({"role": "assistant", "content": self.last_turn or [self.last_call]})
        messages.append({"role": "user", "content": [{
            "type": "tool_result", "tool_use_id": self.last_call["id"], "content": self.last_result,
            "is_error": bool(self._last_error)}]})
        return messages

    # -- sensing --------------------------------------------------------------
    @staticmethod
    def _signature(img):
        return img.convert("L").resize((64, 48))

    def _fetch_frame(self, expect_change):
        """GET /look. After a move/turn, an identical frame means the camera has not caught
        up yet (e.g. a bridge that relays the motion late): wait and fetch again.
        Returns (PIL image, changed?)."""
        deadline = time.time() + (STALE_FRAME_WAIT_S if expect_change else 0)
        while True:
            img = Image.open(io.BytesIO(self.engine.look_jpeg()))
            img.load()
            sig = self._signature(img)
            same = (self._last_sig is not None
                    and ImageStat.Stat(ImageChops.difference(sig, self._last_sig)).mean[0] < 0.5)
            if not (expect_change and same) or time.time() >= deadline:
                self._last_sig = sig
                return img, not same
            time.sleep(0.3)

    def _capture(self, step, expect_change=False):
        """One /look (+ depth). Returns (image block, frame_path, depth)."""
        img, changed = self._fetch_frame(expect_change)
        if expect_change and not changed:
            self.frame_unchanged = True
            if not self.warned_stale:
                self.warned_stale = True
                print("\n[WARNING] The camera frame did not change after a move/turn (waited %.0fs). "
                      "The robot is probably not executing commands from this agent.\n" % STALE_FRAME_WAIT_S)
        if img.size != (IMG_W, IMG_H):
            img = img.resize((IMG_W, IMG_H))
        if self.hud:
            img = draw_hud(img)
        frame_path = os.path.join(self.run_dir, "step_%03d.png" % step)
        img.save(frame_path)                       # exactly the image Claude sees
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=85)
        block = {"type": "image",
                 "source": {"type": "base64", "media_type": "image/jpeg",
                            "data": base64.b64encode(buf.getvalue()).decode("ascii")}}
        self.frame_is_current = True
        return block, frame_path, self.engine.depth()

    # -- one tool call ----------------------------------------------------------
    def execute(self, step, name, args):
        """Runs the tool. Returns (result dict, tool_result content, frame_path, done_result)."""
        result, frame_path, done, image = {}, None, None, None
        self._last_error = None
        pose_before = None
        if args.get("sightings"):
            self._remember(step - 1, args["sightings"], self._pose(self.engine.state()))

        try:
            if name == "look":
                if self.frame_is_current:
                    self._last_error = True
                    result = {"error": "You already have the current frame. Turn or move before looking again."}
                else:
                    image, frame_path, depth = self._capture(step)
                    result = {"depth_m": depth}

            elif name == "turn":
                deg = max(-180.0, min(180.0, float(args.get("degrees", 0))))
                result = dict(self.engine.turn(deg))
                self.frame_is_current = False
                self._check_engine_reply("turn", result, "heading")
                time.sleep(self.settle_s)
                image, frame_path, result["depth_m"] = self._capture(step, expect_change=abs(deg) >= 2)

            elif name == "move":
                dist = max(-1.0, min(2.0, float(args.get("distance", 0))))
                pose_before = self.engine.state()
                result = dict(self.engine.move(dist))
                self.frame_is_current = False
                self._check_engine_reply("move", result, "moved")
                time.sleep(self.settle_s)
                if result.get("collided"):
                    self.collisions += 1
                    result["note"] = ("COLLIDED: hit something and backed off 0.1 m. Retry %d of %d."
                                      % (self.collisions, MAX_RETRIES))
                self._check_robot_moved(pose_before, result)
                image, frame_path, result["depth_m"] = self._capture(
                    step, expect_change=abs(_as_float(result.get("moved"), dist)) >= 0.08)

            elif name == "pickup":
                result = dict(self.engine.pickup())
                if result.get("ok"):
                    self.holding = True
                    self.memory.pop("target", None)

            elif name == "drop":
                result = dict(self.engine.drop())
                if result.get("ok"):
                    self.holding = False
                if result.get("in_zone"):
                    done = "pass"
                elif result.get("ok"):
                    result["note"] = "Released OUTSIDE the drop zone. It lies about 0.42 m ahead, in reach."

            elif name == "finish":
                if self.holding:
                    result = {"status": "run complete"}
                    done = "pass"
                else:
                    self._last_error = True
                    result = {"error": "Not holding anything, so the command is not complete."}

            elif name == "give_up":
                result = {"status": "run ended by agent"}
                done = "fail_not_reached"

            else:
                self._last_error = True
                result = {"error": "unknown tool %r" % name}

        except requests.RequestException as exc:
            self._last_error = True
            result = {"error": "engine request failed: %s" % exc}

        if self.frame_unchanged:
            self.frame_unchanged = False
            result["note"] = (result.get("note", "") + " The camera view did NOT change after this "
                              "command: the robot may not have moved. Do not repeat it blindly.").strip()
        state = self.engine.state()
        if isinstance(state.get("holding"), bool):
            self.holding = state["holding"]
        retries = max(self.collisions, int(state.get("retries") or 0))
        if done is None and (retries > MAX_RETRIES or (result.get("collided") and state.get("name") == "Done")):
            done = "fail_collision"

        status = {"state": state.get("name"), "heading": state.get("heading"),
                  "holding": self.holding, "retries": retries,
                  "steps_used": "%d/%d" % (step, MAX_STEPS)}
        if step >= SOFT_STEPS:
            status["warning"] = "%d steps left before the run fails" % (MAX_STEPS - step)
        payload = dict(result, **{k: v for k, v in status.items() if v is not None})
        memory = self._memory_view(self._pose(state))
        if memory:
            payload["memory"] = memory

        content = [{"type": "text", "text": json.dumps(payload, separators=(",", ":"))}]
        if image is not None:
            content.append(image)
        return result, payload, content, frame_path, done

    # -- memory: Claude's own sightings + robot odometry (no engine object data) --------
    @staticmethod
    def _pose(state):
        try:
            return float(state["x"]), float(state["y"]), float(state["heading"])
        except (KeyError, TypeError, ValueError):
            return None

    def _remember(self, step, sightings, pose):
        """Sightings refer to the frame Claude was given, taken at the current pose."""
        if not sightings or pose is None:
            return
        x, y, h = pose
        for s in sightings:
            try:
                what, b, d = s["what"], float(s["bearing_deg"]), float(s["distance_m"])
            except (KeyError, TypeError, ValueError):
                continue
            if what not in ("drop_zone", "target") or not (0.2 <= d <= 8.0) or abs(b) > 60:
                continue
            w = math.radians(h + b)                 # heading is clockwise-positive
            self.memory[what] = (x + d * math.cos(w), y - d * math.sin(w), step)

    def _memory_view(self, pose):
        """Remembered things relative to the robot now: turn needed and distance."""
        if pose is None or not self.memory:
            return None
        x, y, h = pose
        view = {}
        for what, (wx, wy, seen) in self.memory.items():
            if what == "target" and self.holding:
                continue
            dist = math.hypot(wx - x, wy - y)
            bearing = (-math.degrees(math.atan2(wy - y, wx - x))) % 360.0
            view[what] = {"turn_deg": int(round(((bearing - h + 180.0) % 360.0) - 180.0)),
                          "distance_m": round(dist, 1), "seen_at_step": seen}
        return view or None

    def _check_engine_reply(self, tool, result, key):
        """The real engine answers /turn with heading and /move with moved. Anything else
        is a different server (e.g. a bridge) that may not drive the robot at all."""
        if key not in result and not self.warned_contract:
            self.warned_contract = True
            print("\n[WARNING] %s/%s answered %s: no '%s' field, so this is not the robot engine.\n"
                  "          Moves may never reach the robot and frames may lag behind turns.\n"
                  "          Point --engine / ENGINE_URL at the PC running server.py "
                  "(http://localhost:8080 if it is this PC).\n"
                  % (self.engine.base, tool, json.dumps(result), key))

    def _check_robot_moved(self, before, result):
        """Warn loudly if the engine reports motion but the pose never changes."""
        after = self.engine.state()
        try:
            asked = abs(float(result.get("moved", 0)))
            shift = math.hypot(after["x"] - before["x"], after["y"] - before["y"])
        except (KeyError, TypeError, ValueError):
            return
        if asked > 0.05 and shift < 0.3 * asked and not self.warned_static:
            self.warned_static = True
            print("\n[WARNING] /move answered moved=%.2f but the robot pose moved %.2f m. "
                  "The engine at %s is not executing moves (a bridge that only relays heading?). "
                  "Point ENGINE_URL at the engine itself.\n" % (asked, shift, self.engine.base))

    # -- main loop ----------------------------------------------------------------
    def run(self):
        print("Command: %s\nEngine: %s | model: %s | logs: %s"
              % (self.command, self.engine.base, self.brain.model, self.run_dir))
        self._last_error = None
        result_label = "fail_not_reached"
        for step in range(1, MAX_STEPS + 1):
            t0 = time.time()
            call, usage, turn = self.brain.decide(self.build_messages())
            think_s = time.time() - t0
            name, args = call["name"], dict(call["input"])
            reason = str(args.pop("reason", ""))[:300]

            result, payload, content, frame_path, done = self.execute(step, name, args)
            cost = step_cost(self.brain.model, usage) if usage else 0.0
            self.total_cost += cost

            self._log({"step": step, "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                       "tool": name, "args": args, "result": result, "frame_path": frame_path,
                       "reason": reason, "claude_s": round(think_s, 2), "usage": usage, "cost_usd": round(cost, 5)})
            print("%2d %-7s %-22s %-58s | %s" % (step, name, json.dumps(args)[:22],
                                                 json.dumps(result)[:58], reason[:70]))

            # roll the previous step into history; keep this one as the live tool_use/tool_result
            if self.last_call is not None:
                self.history.append(self._summary_line)
            self._summary_line = "Step %d: %s(%s) -> %s | %s" % (
                step, name, ", ".join("%s=%s" % kv for kv in args.items()),
                json.dumps({k: v for k, v in payload.items() if k != "memory"}, separators=(",", ":")), reason)
            self.last_call, self.last_turn, self.last_result = call, turn, content

            if done:
                result_label = done
                break
            if self.warned_contract and self.warned_stale:
                print("[STOPPED] %s answers commands but the robot is not moving. Stopping to save "
                      "API credit. Run --selftest, then point ENGINE_URL at the PC running server.py."
                      % self.engine.base)
                break
        else:
            print("[TERMINATED] Step %d reached." % MAX_STEPS)

        self._log({"step": step, "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                   "done": True, "result": result_label, "steps": step,
                   "collisions": self.collisions, "cost_usd": round(self.total_cost, 4)})
        print("\nDone: %s in %d steps, %d collisions, cost $%.4f. Log: %s"
              % (result_label, step, self.collisions, self.total_cost, self.log_path))
        return result_label

    def _log(self, entry):
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Claude agent for the Fetch task")
    ap.add_argument("command", nargs="*", help='e.g. "Bring the red cube to the green square."')
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--engine", default=ENGINE_URL)
    ap.add_argument("--dummy", action="store_true", help="fixed moves, no Claude calls")
    ap.add_argument("--no-hud", action="store_true", help="send raw frames without the bearing/distance scales")
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--check-key", action="store_true", help="test the API key (free) and exit")
    ap.add_argument("--selftest", action="store_true", help="check the engine really moves the robot (free) and exit")
    ap.add_argument("--settle", type=float, default=0.0,
                    help="seconds to wait before /look after a move or turn (only needed behind a bridge)")
    a = ap.parse_args(argv)
    if a.check_key:
        return check_key()
    if a.selftest:
        return selftest(Engine(a.engine.rstrip("/")))
    command = " ".join(a.command).strip() or input("Command: ").strip()
    brain = DummyBrain() if a.dummy else ClaudeBrain(a.model)
    agent = FetchAgent(command, brain, Engine(a.engine.rstrip("/")), hud=not a.no_hud, log_root=a.logs)
    agent.settle_s = max(0.0, a.settle)
    return 0 if agent.run() == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
