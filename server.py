"""
server.py — HTTP API for the HackOween simulation (localhost:8080).

    python server.py                 # GUI window + API
    SIM_GUI=0 python server.py       # headless

Remote heading sync (Sharvesh's server)
---------------------------------------
The AI sends POST /turn to Sharvesh's server, not to this one, so the
heading change has to be pulled from there. A background thread polls
    GET {SHARVESH_URL}/state
and whenever the "heading" it reports changes, turns the PyBullet robot
(animated, the short way round) to that heading, then pushes a fresh frame
back so the AI sees the new view.

    SHARVESH_URL=http://127.0.0.1:8000   remote server to poll / push to
    REMOTE_SYNC=0                            turn the poller off
    REMOTE_POLL_S=0.2                        poll interval in seconds
    REMOTE_HEADING_CCW=1                     remote heading is counter-clockwise-
                                             positive (this engine is clockwise)
    PUSH_FRAMES=0                            stop pushing /look frames outbound

GET /sync_status shows what the poller is seeing.
"""

import base64
import math
import os
import queue
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional

import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from sim_engine import SimulationEngine


def _env_flag(name, default):
    return os.environ.get(name, default).strip().lower() not in ("0", "false", "no", "off", "")


GUI = _env_flag("SIM_GUI", "1")
SHARVESH_URL = os.environ.get("SHARVESH_URL", "http://127.0.0.1:8000").rstrip("/")
REMOTE_SYNC = _env_flag("REMOTE_SYNC", "1")
REMOTE_POLL_S = float(os.environ.get("REMOTE_POLL_S", "0.2"))
REMOTE_HEADING_CCW = _env_flag("REMOTE_HEADING_CCW", "0")
PUSH_FRAMES = _env_flag("PUSH_FRAMES", "1")
HEADING_EPS_DEG = 0.5
FORWARD_HEADER = "X-Hackoween-Engine"   # marks our own outbound posts (loop guard)

engine = SimulationEngine(gui=GUI)


def _angle_diff(a, b):
    return abs(((a - b + 180.0) % 360.0) - 180.0)


# ---------------------------------------------------------------------------
# Outbound frame push (never blocks a request)
# ---------------------------------------------------------------------------
class FramePusher(threading.Thread):
    """Keeps only the newest frame and posts it to Sharvesh's /update_frame."""

    def __init__(self, base_url):
        super().__init__(daemon=True, name="frame-pusher")
        self.base_url = base_url
        self._slot = queue.Queue(maxsize=1)
        self.sent = 0
        self.last_error = None

    def submit(self, jpeg_bytes, render_complete=False):
        item = (jpeg_bytes, render_complete)
        try:
            self._slot.put_nowait(item)
        except queue.Full:
            try:
                self._slot.get_nowait()
            except queue.Empty:
                pass
            self._slot.put_nowait(item)

    def run(self):
        while True:
            jpeg, render_complete = self._slot.get()
            try:
                headers = {FORWARD_HEADER: "1"}
                requests.post(f"{self.base_url}/update_frame", headers=headers,
                              json={"image_base64": base64.b64encode(jpeg).decode("ascii")}, timeout=1.0)
                if render_complete:
                    requests.post(f"{self.base_url}/engine/render_complete", headers=headers, timeout=1.0)
                self.sent += 1
                self.last_error = None
            except Exception as exc:  # network hiccups must never hurt the sim
                self.last_error = str(exc)


# ---------------------------------------------------------------------------
# Remote heading sync: poll Sharvesh's GET /state, apply heading to the robot
# ---------------------------------------------------------------------------
def extract_heading(data):
    """Pulls a heading (degrees) out of a /state-style payload, or None."""
    if not isinstance(data, dict):
        return None
    for key in ("heading", "robot_heading", "yaw", "heading_deg"):
        if key in data:
            try:
                value = float(data[key])
            except (TypeError, ValueError):
                return None
            return value if math.isfinite(value) else None
    for nested in ("state", "robot", "data"):
        if isinstance(data.get(nested), dict):
            return extract_heading(data[nested])
    return None


class RemoteHeadingSync(threading.Thread):
    def __init__(self, eng, base_url, interval=0.2, ccw=False, pusher=None):
        super().__init__(daemon=True, name="remote-heading-sync")
        self.engine = eng
        self.url = f"{base_url}/state"
        self.interval = max(0.05, float(interval))
        self.ccw = ccw
        self.pusher = pusher
        self._stop = threading.Event()
        self.last_remote_heading = None
        self.last_raw = None
        self.applied_turns = 0
        self.polls = 0
        self.last_error = None
        self.last_ok_time = None

    def stop(self):
        self._stop.set()

    def status(self):
        return {
            "enabled": True,
            "url": self.url,
            "polls": self.polls,
            "applied_turns": self.applied_turns,
            "last_remote_heading": self.last_remote_heading,
            "last_remote_payload": self.last_raw,
            "local_heading": round(self.engine.robot_heading, 1),
            "seconds_since_ok": None if self.last_ok_time is None else round(time.time() - self.last_ok_time, 1),
            "last_error": self.last_error,
            "heading_convention": "ccw->cw converted" if self.ccw else "clockwise (same as engine)",
        }

    def poll_once(self, session=requests):
        """One poll. Returns True if the robot was turned."""
        self.polls += 1
        data = session.get(self.url, timeout=1.0).json()
        self.last_raw = data
        self.last_ok_time = time.time()
        self.last_error = None
        remote = extract_heading(data)
        if remote is None:
            self.last_error = "no 'heading' field in remote /state"
            return False
        target = (-remote if self.ccw else remote) % 360.0

        # Act only when the remote heading changes (and once at start-up), so
        # local commands like dashboard A/D are not undone every 200 ms.
        changed = self.last_remote_heading is None or _angle_diff(target, self.last_remote_heading) > HEADING_EPS_DEG
        self.last_remote_heading = round(target, 2)
        if not changed or _angle_diff(target, self.engine.robot_heading) <= HEADING_EPS_DEG:
            return False

        self.engine.turn_to_heading(target)          # animated, takes the engine lock
        self.applied_turns += 1
        print(f"[sync] remote heading {remote:.1f} -> robot turned to {self.engine.robot_heading:.1f}")
        if self.pusher is not None:
            self.pusher.submit(self.engine.get_camera_image(), render_complete=True)
        return True

    def run(self):
        session = requests.Session()
        backoff = self.interval
        while not self._stop.is_set():
            try:
                self.poll_once(session)
                backoff = self.interval
            except Exception as exc:
                if self.last_error != str(exc):
                    print(f"[sync] cannot read {self.url}: {exc}")
                self.last_error = str(exc)
                backoff = min(2.0, backoff * 2)
            self._stop.wait(backoff)


pusher = FramePusher(SHARVESH_URL) if PUSH_FRAMES else None
heading_sync = RemoteHeadingSync(engine, SHARVESH_URL, REMOTE_POLL_S, REMOTE_HEADING_CCW, pusher) if REMOTE_SYNC else None


@asynccontextmanager
async def lifespan(_app):
    if pusher is not None and not pusher.is_alive():
        pusher.start()
    if heading_sync is not None and not heading_sync.is_alive():
        heading_sync.start()
        print(f"[sync] polling {heading_sync.url} every {heading_sync.interval}s")
    yield
    if heading_sync is not None:
        heading_sync.stop()


app = FastAPI(title="Humanoid Robot Control API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request models ---------------------------------------------------------
class MoveRequest(BaseModel):
    distance: float = 1.0


class TurnRequest(BaseModel):
    degrees: float = 45.0
    turn_speed: Optional[float] = None


class ResetRequest(BaseModel):
    seed: Optional[int] = None


class FrameRequest(BaseModel):
    image_base64: str = ""


class VelocityRequest(BaseModel):
    velocity: float = 1.0   # positive forward, negative reverse


class TurnVelRequest(BaseModel):
    turn_speed: float = 1.0  # positive right, negative left


# --- Contract endpoints (schemas as in README) ---------------------------------
@app.get("/state")
def state():
    return engine.get_state()


@app.get("/look")
def look():
    img = engine.get_camera_image()
    if pusher is not None:
        pusher.submit(img)
    return Response(content=img, media_type="image/jpeg")


@app.get("/depth")
def depth():
    return {"depth": engine.get_single_depth()}


@app.get("/depth_fan")
def depth_fan():
    return {"fan": engine.get_depth_fan()}


@app.post("/move")
def move(req: MoveRequest):
    return engine.move_robot(req.distance)


@app.post("/turn")
def turn(req: TurnRequest):
    degrees = req.degrees
    if req.turn_speed is not None and "degrees" not in req.model_fields_set:
        degrees = req.turn_speed * 45.0
    return engine.turn_robot(degrees)


@app.post("/pickup")
def pickup():
    return engine.pickup_object()


@app.post("/drop")
def drop():
    return engine.drop_object()


@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    return engine.reset_scene(seed=req.seed if req is not None else None)


@app.post("/obstacle")
def obstacle():
    return engine.enable_obstacle_mode()


# --- Team integration extras -------------------------------------------------
@app.get("/sync_status")
def sync_status():
    if heading_sync is None:
        return {"enabled": False, "url": f"{SHARVESH_URL}/state"}
    out = heading_sync.status()
    out["frames_pushed"] = pusher.sent if pusher is not None else 0
    out["push_error"] = pusher.last_error if pusher is not None else None
    return out


@app.post("/update_frame")
def update_frame(req: FrameRequest):
    if not req.image_base64 or req.image_base64 == "string":
        return {"status": "ignored", "message": "Dummy frame received and safely ignored."}
    return {"status": "processed"}


@app.post("/engine/render_complete")
def render_complete(request: Request):
    if pusher is not None and FORWARD_HEADER.lower() not in request.headers:
        pusher.submit(engine.get_camera_image(), render_complete=True)
    return {"status": "ok"}


@app.post("/move_start")
def move_start(req: VelocityRequest):
    return engine.move_robot(0.5 * req.velocity)


@app.post("/turn_start")
def turn_start(req: TurnVelRequest):
    return engine.turn_robot(25.0 * req.turn_speed)


@app.post("/stop")
def stop():
    # Every motion command finishes before it returns, so there is nothing to cancel.
    return {"status": "stopped"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
