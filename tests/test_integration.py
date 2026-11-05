"""
End-to-end verification for the HackOween simulation (Stages 1-4).

Run from the project root (headless, no GUI window):

    python tests/test_integration.py            # everything
    python tests/test_integration.py --no-http  # skip the HTTP server check

Saves review frames to tests/output/ (robot POV + third-person overviews).
Plain-Python runner: no pytest needed. Exit code 0 = all checks passed.
"""

import io
import json
import math
import os
import socket
import subprocess
import sys
import time
import traceback
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "tests", "output")
os.makedirs(OUT, exist_ok=True)

import numpy as np  # noqa: E402
import pybullet as p  # noqa: E402
from PIL import Image  # noqa: E402

import sim.driver as driver_mod  # noqa: E402
import sim.sensors as sensors_mod  # noqa: E402
from sim import scene  # noqa: E402
from sim.driver import arm_fk, arm_ik  # noqa: E402
from sim.referee import Outcome, State  # noqa: E402
from sim_engine import SimulationEngine  # noqa: E402

FAKE = getattr(p, "IS_FAKE", False)
ENGINE = None
RESULTS = []


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def place(engine, x, y, heading):
    """Test-only setup helper (bypasses the API to stage a scenario)."""
    engine._place_robot(x, y, heading)


def find_clear(engine, heading, run=0.6):
    """A staging spot with no contact and a clear straight run along heading."""
    fx, fy = sensors_mod.heading_vector(heading)
    for gx in np.arange(-1.8, 1.81, 0.3):
        for gy in np.arange(-1.8, 1.81, 0.3):
            ex, ey = gx + fx * run, gy + fy * run
            if max(abs(ex), abs(ey)) > 1.9:
                continue
            if _clear_path(engine, (gx, gy), (ex, ey), None, 0.5):
                return float(gx), float(gy)
    raise AssertionError("no clear staging spot")


def obj_by_key(engine, key):
    for body, spec in engine._specs.items():
        if spec["key"] == key:
            return body
    raise KeyError(key)


def obj_xy(body):
    pos, _ = p.getBasePositionAndOrientation(body)
    return pos[0], pos[1], pos[2]


def face(engine, tx, ty):
    """Turn (through the API) to face a world point."""
    x, y = engine.robot_pos[0], engine.robot_pos[1]
    bearing = (-math.degrees(math.atan2(ty - y, tx - x))) % 360.0
    delta = ((bearing - engine.robot_heading + 180.0) % 360.0) - 180.0
    return engine.turn_robot(delta)


def purple_fraction(rgb):
    arr = rgb.astype(np.float32) / 255.0
    mx, mn = arr.max(axis=2), arr.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    d = np.maximum(mx - mn, 1e-6)
    hue = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) / 6.0
    mask = (sat > 0.25) & (mx > 0.20) & (hue >= 0.69) & (hue <= 0.93)
    return float(mask.mean())


def overview(engine, name, eye, target):
    """Third-person render (tests/review only — never served over HTTP)."""
    view = p.computeViewMatrix(cameraEyePosition=eye, cameraTargetPosition=target, cameraUpVector=[0, 0, 1])
    proj = p.computeProjectionMatrixFOV(fov=60, aspect=4 / 3, nearVal=0.05, farVal=20)
    w, h, rgb, depth, _ = p.getCameraImage(960, 720, view, proj, renderer=p.ER_TINY_RENDERER,
                                           flags=p.ER_NO_SEGMENTATION_MASK, **scene.CAMERA_LIGHTING)
    img = np.reshape(np.asarray(rgb, dtype=np.uint8), (h, w, 4))[:, :, :3]
    Image.fromarray(img).save(os.path.join(OUT, name))
    return img


# ---------------------------------------------------------------------------
# Stage 0: model + scene
# ---------------------------------------------------------------------------
def test_urdf_is_16_links_loaded_natively():
    e = ENGINE
    check(len(e.driver.link_index) + 1 == 16, "expected 16 links, got %d" % (len(e.driver.link_index) + 1))
    for j in driver_mod.ALL_JOINTS:
        check(j in e.driver.joint_index, "missing joint " + j)
    check("<robot" in scene.HUMANOID_URDF and "capsule" in scene.HUMANOID_URDF, "URDF string malformed")


def test_zero_purple_everywhere():
    scene.assert_palette_has_no_purple()
    bad = []
    for i in range(p.getNumBodies()):
        body = p.getBodyUniqueId(i)
        for vs in p.getVisualShapeData(body):
            if scene.is_purple(vs[7]):
                bad.append((body, vs[7]))
    check(not bad, "purple visual shapes: %s" % bad)


def test_object_set_meets_brief():
    specs = scene.active_object_specs()
    check(4 <= len(specs) <= 6, "brief wants 4-6 objects")
    colours = [tuple(s["color"]) for s in specs]
    shapes = [s["shape"] for s in specs]
    check(len(set(colours)) < len(colours), "at least two objects must share a colour")
    check(len(set(shapes)) < len(shapes), "at least two objects must share a shape")


# ---------------------------------------------------------------------------
# Stage 1: camera + depth
# ---------------------------------------------------------------------------
def test_camera_contract():
    e = ENGINE
    e.reset_scene(seed=3)
    captured = {}
    real_get = sensors_mod.p.getCameraImage

    def spy(*a, **kw):
        captured.update(kw)
        return real_get(*a, **kw)

    sensors_mod.p.getCameraImage = spy
    try:
        t0 = time.perf_counter()
        jpeg = e.get_camera_image()
        dt = time.perf_counter() - t0
    finally:
        sensors_mod.p.getCameraImage = real_get
    check(captured.get("flags") == p.ER_NO_SEGMENTATION_MASK, "segmentation mask must be disabled")
    img = Image.open(io.BytesIO(jpeg))
    check(img.format == "JPEG" and img.size == (640, 480), "expected 640x480 JPEG, got %s %s" % (img.format, img.size))
    check(abs(sensors_mod.VFOV_DEG - 73.74) < 0.01, "vertical fov should be 73.74")
    eye, d, _ = sensors_mod.camera_pose(0, 0, 0)
    check(abs(eye[2] - 0.5) < 1e-9, "camera must be 0.5 m high")
    check(abs(math.degrees(math.asin(-d[2])) - 10.0) < 1e-6, "camera must tilt 10 deg down")
    _, d90, _ = sensors_mod.camera_pose(0, 0, 90)
    check(d90[1] < -0.9, "heading 90 (right/clockwise) must look toward -y")
    if not FAKE:
        visor = p.getLinkState(e.robot_id, e.driver.link_index["visor_link"])[0]
        check(abs(visor[2] - 0.5) < 0.02, "visor centre %.3f should sit at ~0.5 m" % visor[2])
        check(visor[0] < eye[0], "camera must be in front of the visor")
        img.save(os.path.join(OUT, "pov_reset_seed3.jpg"))
        print("      /look render time %.0f ms" % (dt * 1000))


def test_depth_ray_from_visor_height():
    e = ENGINE
    e.reset_scene(seed=3)
    starts = []
    real_ray = sensors_mod.p.rayTest

    def spy(a, b, *rest, **kw):
        starts.append(a)
        return real_ray(a, b, *rest, **kw)

    sensors_mod.p.rayTest = spy
    try:
        place(e, 0.0, 0.0, 180.0)           # facing the bare west wall
        dist = e.get_single_depth()
    finally:
        sensors_mod.p.rayTest = real_ray
    check(abs(starts[0][2] - 0.5) < 1e-9, "depth ray must start at 0.5 m")
    expected = (2.4 - 0.12) / math.cos(math.radians(10))   # inner wall face 2.4 m
    check(dist is not None and abs(dist - expected) < 0.05, "depth %.3f vs expected %.3f" % (dist or -1, expected))
    fan = e.get_depth_fan()
    check(len(fan) == 5 and set(fan[0]) == {"angle_offset", "distance"}, "depth fan schema changed")


def test_pov_frames_and_no_purple_pixels():
    if FAKE:
        return
    e = ENGINE
    e.reset_scene(seed=5)
    worst = 0.0
    for k, hd in enumerate((0, 45, 90, 135, 180, 225, 270, 315)):
        place(e, 0.0, 0.0, hd)
        rgb = e.camera.render_rgb(0.0, 0.0, hd)
        worst = max(worst, purple_fraction(rgb))
        Image.fromarray(rgb).save(os.path.join(OUT, "pov_heading_%03d.jpg" % hd))
    check(worst < 0.0005, "purple pixels in POV frames: %.4f%%" % (worst * 100))
    place(e, 0.0, 0.0, 0.0)
    overview(e, "overview_arena.jpg", [-2.2, -2.2, 2.6], [0.4, 0.4, 0.2])
    overview(e, "overview_robot_closeup.jpg", [1.1, -0.7, 0.75], [0.0, 0.0, 0.35])
    overview(e, "overview_drop_zone.jpg", [0.9, 0.9, 1.4], [1.8, 1.8, 0.0])


# ---------------------------------------------------------------------------
# Stage 2: articulation
# ---------------------------------------------------------------------------
def test_arm_ik_round_trip():
    for x, z in ((0.28, 0.20), (0.25, 0.22), (0.20, 0.10), (0.31, 0.21), (0.16, 0.08)):
        s, el = arm_ik(x, z)
        fx, fz = arm_fk(s, el)
        check(abs(fx - x) < 1e-6 and abs(fz - z) < 1e-6, "IK round trip failed for %s" % ((x, z),))
    s, el = arm_ik(2.0, 0.0)   # unreachable -> clamped, still finite
    check(math.isfinite(s) and math.isfinite(el), "IK clamp produced NaN")


def test_wheels_spin_and_head_leads_turn():
    e = ENGINE
    e.reset_scene(seed=3)
    place(e, *find_clear(e, 0.0), 0.0)
    left0, right0 = e.driver.wheel_angles()
    e.move_robot(0.5)
    left1, right1 = e.driver.wheel_angles()
    check(abs((left1 - left0) - 0.5 / 0.08) < 1e-3 and abs((right1 - right0) - 0.5 / 0.08) < 1e-3, "wheel roll mismatch")

    neck_seen = []
    real_set = e.driver.set_joint

    def spy(name, value):
        if name == "neck_joint":
            neck_seen.append(value)
        real_set(name, value)

    e.driver.set_joint = spy
    try:
        l0, r0 = e.driver.wheel_angles()
        e.turn_robot(90)
    finally:
        e.driver.set_joint = real_set
    l1, r1 = e.driver.wheel_angles()
    check(l1 > l0 and r1 < r0, "right turn must drive left wheel forward, right wheel back")
    check(min(neck_seen) < -0.2, "neck should lead into a right turn")
    check(abs(e.driver.q["neck_joint"]) < 1e-6, "neck must re-centre so the camera matches the visor")


def test_heading_convention_and_motion():
    e = ENGINE
    e.reset_scene(seed=3)
    sx, sy = find_clear(e, 90.0)
    place(e, sx, sy, 0.0)
    e.turn_robot(90)                    # right = clockwise -> facing -y
    res = e.move_robot(0.5)
    check(res == {"moved": 0.5, "collided": False}, "unexpected move result %s" % res)
    check(abs(e.robot_pos[0] - sx) < 1e-6 and abs(e.robot_pos[1] - (sy - 0.5)) < 1e-6,
          "after +90 turn, forward should be -y (got %s)" % e.robot_pos)
    check(e.robot_heading == 90.0 and e.state_name == State.GO_TO_OBJECT, "state %s" % e.state_name)


def test_wall_collision_blocks_and_backs_off():
    e = ENGINE
    e.reset_scene(seed=3)
    place(e, -1.5, 0.0, 180.0)          # facing the west wall, 0.9 m of room minus robot radius
    res = e.move_robot(3.0)
    check(res["collided"] is True and e.state_name == State.BLOCKED and e.retries == 1, "no collision: %s" % res)
    expected = (2.4 - 0.22) - 1.5 - 0.1
    check(abs(res["moved"] - expected) < 0.03, "moved %.3f, expected ~%.3f" % (res["moved"], expected))
    check(e.check_physical_collisions() is False, "robot should be clear of the wall after backing off")
    res2 = e.move_robot(-0.3)           # reversing away is allowed
    check(res2["collided"] is False, "reversing away from a wall should not collide")


def test_retries_exhausted_fails_run():
    e = ENGINE
    e.reset_scene(seed=3)
    for _ in range(4):
        place(e, -2.0, 0.0, 180.0)
        e.move_robot(1.0)
    check(e.state_name == State.DONE and e.referee.outcome == Outcome.FAIL_COLLISION,
          "4th collision should fail the run (state %s, outcome %s)" % (e.state_name, e.referee.outcome))


def _clear_path(e, start, goal, ignore, margin=0.5):
    sx, sy = start
    gx, gy = goal
    for body in e.object_ids:
        if body == ignore:
            continue
        ox, oy, _ = obj_xy(body)
        vx, vy = gx - sx, gy - sy
        t = max(0.0, min(1.0, ((ox - sx) * vx + (oy - sy) * vy) / (vx * vx + vy * vy)))
        if math.hypot(sx + t * vx - ox, sy + t * vy - oy) < margin:
            return False
    return True


def test_pick_and_place_end_to_end():
    e = ENGINE
    key = "teal_cylinder"
    for seed in range(200):
        e.reset_scene(seed=seed)
        body = obj_by_key(e, key)
        ox, oy, _ = obj_xy(body)
        zx, zy = e.drop_zone_pos[0], e.drop_zone_pos[1]
        d = math.hypot(zx - ox, zy - oy)
        ux, uy = (zx - ox) / d, (zy - oy) / d
        sx, sy = ox - ux * 0.43, oy - uy * 0.43          # approach from the far side
        if max(abs(sx), abs(sy)) < 2.0 and _clear_path(e, (sx, sy), (zx, zy), body) \
                and _clear_path(e, (sx - ux * 0.3, sy - uy * 0.3), (sx, sy), body, 0.45):
            break
    else:
        raise AssertionError("no usable seed found")

    place(e, sx - ux * 0.3, sy - uy * 0.3, 0.0)
    face(e, ox, oy)
    res = e.move_robot(0.3)
    check(not res["collided"], "approach collided: %s" % res)

    states, shoulder = [], []
    real_frame = e.driver.on_frame
    real_set = e.driver.set_joint

    def frame_spy():
        states.append(e.state_name)
        if real_frame:
            real_frame()

    def set_spy(name, value):
        if name == "right_shoulder_joint":
            shoulder.append(value)
        real_set(name, value)

    e.driver.on_frame, e.driver.set_joint = frame_spy, set_spy
    try:
        res = e.pickup_object()
    finally:
        e.driver.on_frame, e.driver.set_joint = real_frame, real_set
    check(res == {"ok": True}, "pickup failed: %s" % res)
    check(State.PICK_UP in states and e.state_name == State.CARRY, "state did not sync with pickup animation")
    steps = np.abs(np.diff(shoulder))
    check(len(shoulder) >= 40 and steps.max() < 0.08, "arm motion not smooth (n=%d, max step %.3f)" % (len(shoulder), steps.max()))
    check(e.driver.q["right_elbow_joint"] < -0.8 and e.driver.q["left_elbow_joint"] < -0.8,
          "arms should be bent into the tucked carry posture")
    off = e._to_local(obj_xy(body))
    check(off[1] < -0.3 and 0.1 < off[0] < 0.4, "object should be tucked at the right hip, got %s" % (off,))
    if not FAKE:
        Image.open(io.BytesIO(e.get_camera_image())).save(os.path.join(OUT, "pov_carrying.jpg"))

    before = obj_xy(body)
    face(e, zx, zy)
    dist = math.hypot(zx - e.robot_pos[0], zy - e.robot_pos[1]) - 0.42
    res = e.move_robot(dist)
    after = obj_xy(body)
    check(not res["collided"], "carry move collided: %s" % res)
    check(math.hypot(after[0] - before[0], after[1] - before[1]) > dist * 0.9, "held object did not follow the robot")
    check(after[2] > before[2] - 1e-6 and after[2] > 0.15, "held object should stay lifted")
    if not FAKE:
        overview(e, "overview_carry_at_zone.jpg", [0.6, 0.9, 1.3], [e.robot_pos[0], e.robot_pos[1], 0.25])

    res = e.drop_object()
    check(res == {"ok": True, "in_zone": True}, "drop result %s" % res)
    check(e.state_name == State.DONE and e.referee.outcome == Outcome.PASS, "run should pass")
    check(e.held_object_id is None and e.pickup_constraint is None, "constraint not released")


def test_pickup_out_of_reach_fails_cleanly():
    e = ENGINE
    e.reset_scene(seed=3)
    place(e, -1.8, 1.9, 0.0)
    nearest = min(math.hypot(obj_xy(b)[0] + 1.8, obj_xy(b)[1] - 1.9) for b in e.object_ids)
    res = e.pickup_object()
    if nearest > 0.5:
        check(res == {"ok": False} and e.state_name == State.LOCATE, "out-of-reach pickup %s %s" % (res, e.state_name))
    check(e.drop_object() == {"ok": False, "in_zone": False}, "drop with empty hands")


def test_stress_spawn_and_motion():
    e = ENGINE
    for seed in range(30):
        e.reset_scene(seed=seed)
        check(not e.check_physical_collisions(), "spawn clipping on seed %d" % seed)
        res = e.move_robot(0.5)
        e.turn_robot(45)
        if res["collided"]:
            check(e.state_name == State.BLOCKED, "collision without Blocked on seed %d" % seed)


def test_obstacle_wall_collision():
    """Runs last among engine tests: the obstacle stays in this world."""
    e = ENGINE
    check(e.enable_obstacle_mode() == {"obstacle_enabled": True}, "/obstacle result")
    e.enable_obstacle_mode()                         # idempotent
    for seed in range(11, 300):
        e.reset_scene(seed=seed)
        check(not e.check_physical_collisions(), "spawn overlaps obstacle")
        if _clear_path(e, (0.2, 0.9), (0.8, 0.9), None, 0.45):
            break
    place(e, 0.2, 0.9, 0.0)                          # facing the partition at x=0.8
    res = e.move_robot(1.0)
    expected = (0.8 - 0.22) - 0.2 - 0.1
    check(res["collided"] and abs(res["moved"] - expected) < 0.03,
          "obstacle not detected (static-vs-kinematic contact?): %s" % res)


# ---------------------------------------------------------------------------
# Stage 4: HTTP endpoints (separate process: one PyBullet world per process)
# ---------------------------------------------------------------------------
def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _call(base, method, path, body=None):
    data = json.dumps(body).encode() if body is not None else (b"" if method == "POST" else None)
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data is not None else {})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status, r.headers.get("content-type"), r.read()


def test_http_endpoints():
    port = _free_port()
    env = dict(os.environ, SIM_GUI="0")
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "server:app", "--port", str(port), "--log-level", "warning"],
                            cwd=ROOT, env=env)
    base = "http://127.0.0.1:%d" % port
    try:
        for _ in range(120):
            try:
                _call(base, "GET", "/state")
                break
            except Exception:
                time.sleep(0.25)
        js = lambda m, path, b=None: json.loads(_call(base, m, path, b)[2])  # noqa: E731
        check(set(js("GET", "/state")) == {"name", "x", "y", "heading", "holding", "retries"}, "/state schema")
        status, ctype, img = _call(base, "GET", "/look")
        check(status == 200 and ctype == "image/jpeg" and Image.open(io.BytesIO(img)).size == (640, 480), "/look")
        check(set(js("GET", "/depth")) == {"depth"}, "/depth schema")
        check(set(js("GET", "/depth_fan")) == {"fan"}, "/depth_fan schema")
        check(set(js("POST", "/turn", {"degrees": 45})) == {"heading"}, "/turn schema")
        check(set(js("POST", "/move", {"distance": 0.3})) == {"moved", "collided"}, "/move schema")
        check(set(js("POST", "/pickup")) == {"ok"}, "/pickup schema")
        check(set(js("POST", "/drop")) == {"ok", "in_zone"}, "/drop schema")
        check(js("POST", "/reset", {"seed": 7}) == {"status": "reset_complete"}, "/reset")
        check(js("POST", "/reset") == {"status": "reset_complete"}, "/reset without body")
        st = js("GET", "/state")
        check(st["name"] == "Idle" and st["retries"] == 0 and st["holding"] is False, "state after reset %s" % st)
    finally:
        proc.terminate()
        proc.wait(timeout=10)


# ---------------------------------------------------------------------------
def main():
    global ENGINE
    skip_http = "--no-http" in sys.argv
    t0 = time.time()
    ENGINE = SimulationEngine(gui=False)
    print("Engine ready in %.1fs (%s pybullet)\n" % (time.time() - t0, "FAKE" if FAKE else "real"))
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        if fn is test_http_endpoints and skip_http:
            continue
        try:
            t = time.time()
            fn()
            print("PASS  %-45s %.1fs" % (fn.__name__, time.time() - t))
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print("FAIL  %-45s %s" % (fn.__name__, exc))
            traceback.print_exc()
    print("\n%d/%d passed. Review frames in %s" % (len(tests) - failed - (1 if skip_http else 0),
                                                   len(tests) - (1 if skip_http else 0), OUT))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
