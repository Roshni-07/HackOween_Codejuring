import argparse
import io
import math
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image
from pydantic import BaseModel

from sim_engine import SimulationEngine


# ---------------------------------------------------------
# EXTENDED ENGINE CLASS WITH PROPER RAYCAST OFFSET
# ---------------------------------------------------------
class ServerSimulationEngine(SimulationEngine):

    def get_camera_bytes(self) -> bytes:
        """Render 640x480 JPEG frame for HTTP response[cite: 3, 5]."""
        rad = math.radians(self.robot_heading)
        cam_height = 0.5
        tilt_rad = math.radians(10)

        dx = math.cos(rad)
        dy = math.sin(rad)
        dz = -math.tan(tilt_rad)

        cam_pos = [self.robot_pos[0], self.robot_pos[1], cam_height]
        cam_target = [cam_pos[0] + dx, cam_pos[1] + dy, cam_pos[2] + dz]
        cam_up = [0, 0, 1]

        view_matrix = p.computeViewMatrix(cam_pos, cam_target, cam_up)
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=90, aspect=640 / 480, nearVal=0.1, farVal=10.0
        )

        _, _, rgb_img, _, _ = p.getCameraImage(
            640,
            480,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_TINY_RENDERER,
        )

        frame = np.reshape(rgb_img, (480, 640, 4))[:, :, :3]
        img = Image.fromarray(frame.astype("uint8"), "RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=80)
        return buffer.getvalue()

    def get_depth_reading(self):
        """Single ray distance straight ahead up to 10m with 0.26m origin offset[cite: 3]."""
        rad = math.radians(self.robot_heading)
        start_x = self.robot_pos[0] + 0.26 * math.cos(rad)
        start_y = self.robot_pos[1] + 0.26 * math.sin(rad)
        start_p = [start_x, start_y, 0.5]

        end_p = [
            start_p[0] + 10.0 * math.cos(rad),
            start_p[1] + 10.0 * math.sin(rad),
            0.5 - 10.0 * math.tan(math.radians(10)),
        ]
        ray_res = p.rayTest(start_p, end_p)[0]
        hit_obj_id = ray_res[0]
        hit_fraction = ray_res[2]

        if hit_obj_id != -1 and hit_obj_id != self.robot_id:
            return round(hit_fraction * 10.0, 2)
        return None


# ---------------------------------------------------------
# CLI ARGUMENT PARSER FOR HEADLESS TESTING
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description="Simulation REST API Engine Server")
parser.add_argument("--headless", action="store_true", help="Run without PyBullet GUI")
cli_args, _ = parser.parse_known_args()

# ---------------------------------------------------------
# FASTAPI SERVER INITIALIZATION (localhost:8080)
# ---------------------------------------------------------
app = FastAPI(title="Membrane Robot Hackathon Simulation Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = ServerSimulationEngine(gui=not cli_args.headless)


class MoveRequest(BaseModel):
    distance: float


class TurnRequest(BaseModel):
    degrees: float


@app.get("/state")
def get_state():
    """Returns robot state without leaking object information[cite: 3, 4]."""
    return {
        "name": engine.state_name,
        "x": round(engine.robot_pos[0], 2),
        "y": round(engine.robot_pos[1], 2),
        "heading": round(engine.robot_heading, 1),
        "holding": engine.held_object_id is not None,
        "retries": engine.retries,
    }


@app.get("/look")
def get_look():
    """Returns 640x480 JPEG frame[cite: 3, 5]."""
    frame_bytes = engine.get_camera_bytes()
    return Response(content=frame_bytes, media_type="image/jpeg")


@app.get("/depth")
def get_depth():
    """Returns single-ray straight-ahead float distance in meters[cite: 3, 6]."""
    return {"depth": engine.get_depth_reading()}


@app.get("/depth_fan")
def get_depth_fan():
    """Optional 3-ray depth scan (Left -15 deg, Center 0 deg, Right +15 deg)[cite: 3]."""
    angles = [-15, 0, 15]
    readings = {}
    for a in angles:
        rad = math.radians((engine.robot_heading + a) % 360)
        start_p = [
            engine.robot_pos[0] + 0.26 * math.cos(rad),
            engine.robot_pos[1] + 0.26 * math.sin(rad),
            0.5,
        ]
        end_p = [
            start_p[0] + 10.0 * math.cos(rad),
            start_p[1] + 10.0 * math.sin(rad),
            0.5 - 10.0 * math.tan(math.radians(10)),
        ]
        res = p.rayTest(start_p, end_p)[0]
        readings[f"ray_{a}"] = (
            round(res[2] * 10.0, 2) if res[0] != -1 and res[0] != engine.robot_id else None
        )
    return readings


@app.post("/move")
def move(req: MoveRequest):
    return engine.move_robot(req.distance)


@app.post("/turn")
def turn(req: TurnRequest):
    return engine.turn_robot(req.degrees)


@app.post("/pickup")
def pickup():
    return engine.pickup_object()


@app.post("/drop")
def drop():
    return engine.drop_object()


if __name__ == "__main__":
    import uvicorn

    print("Starting Simulation HTTP Server on http://localhost:8080...")
    uvicorn.run(app, host="0.0.0.0", port=8080)