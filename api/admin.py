# api/admin.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from api.server import engine, scene

admin_app = FastAPI(title="Harness Admin API")


class ResetRequest(BaseModel):
    seed: Optional[int] = None


class CommandRequest(BaseModel):
    command: str
    target_object_index: Optional[int] = None


@admin_app.post("/reset")
def post_reset(req: ResetRequest = ResetRequest()):
    scene.spawn_objects(seed=req.seed)
    engine.robot_pos = [0.0, 0.0, 0.25]
    engine.robot_heading = 0.0
    engine.held_object_id = None
    engine.referee.state_name = "Idle"
    engine.referee.outcome = None
    engine.retries = 0
    engine.sync_robot_transform()
    return {"status": "reset_complete", "seed": req.seed}


@admin_app.post("/command")
def post_command(req: CommandRequest):
    engine.referee.state_name = "Locate"
    if req.target_object_index is not None and req.target_object_index < len(scene.object_ids):
        engine.referee.target_object_id = scene.object_ids[req.target_object_index]
    if engine.recorder:
        engine.recorder.set_command(req.command)
    return {"status": "command_registered"}


@admin_app.post("/obstacle")
def post_obstacle():
    scene.spawn_obstacle()
    return {"status": "obstacle_enabled"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(admin_app, host="127.0.0.1", port=8081)