import threading
import time
import pybullet as p
from fastapi import FastAPI, Response
from pydantic import BaseModel
from sim.core import CoreEngine
from sim.scene import SceneManager
from sim.bridge import push_frame_and_unlock

app = FastAPI(title="Membrane Robot Agent API")

engine = CoreEngine(gui=True)
scene = SceneManager(engine)
engine.scene = scene

scene.build_room()
scene.spawn_robot()
scene.spawn_drop_zone()
scene.spawn_objects(seed=42)

# --- ROBUST BACKGROUND STREAMING WORKER ---
def continuous_stream_worker():
    """Continuously captures frames and pushes them to Sharvesh's backend safely."""
    time.sleep(2.0)  # Give the server a moment to settle
    print("Background continuous bridge worker started!")
    while True:
        try:
            # Check if PyBullet is still connected to avoid crash errors
            if p.isConnected(engine.client_id) if hasattr(engine, 'client_id') else True:
                jpeg_bytes = engine.sensors.get_camera_image()
                push_frame_and_unlock(jpeg_bytes)
        except Exception as e:
            # Suppress noisy disconnect errors during reloads/resets
            pass
        time.sleep(0.1)  # ~10 FPS stream frequency

@app.on_event("startup")
def startup_event():
    thread = threading.Thread(target=continuous_stream_worker, daemon=True)
    thread.start()
# ------------------------------------------

class MoveRequest(BaseModel):
    distance: float


class TurnRequest(BaseModel):
    degrees: float


@app.get("/state")
def get_state():
    return {
        "name": engine.referee.state_name,
        "x": round(engine.robot_pos[0], 2),
        "y": round(engine.robot_pos[1], 2),
        "heading": round(engine.robot_heading, 1),
        "holding": engine.held_object_id is not None,
        "retries": engine.retries,
    }


@app.get("/look")
def get_look():
    jpeg_bytes = engine.sensors.get_camera_image()
    return Response(content=jpeg_bytes, media_type="image/jpeg")


@app.post("/move")
def post_move(req: MoveRequest):
    return engine.driver.move(req.distance)


@app.post("/turn")
def post_turn(req: TurnRequest):
    return engine.driver.turn(req.degrees)


@app.post("/pickup")
def post_pickup():
    return engine.referee.pickup_object()


@app.post("/drop")
def post_drop():
    return engine.referee.drop_object()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)