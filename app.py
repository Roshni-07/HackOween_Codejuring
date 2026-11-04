from fastapi import FastAPI, Response
from pydantic import BaseModel
import os
import json
from datetime import datetime
import math
from pythonagent.logger import StepLogger

app = FastAPI(title="Hackathon Engine API")
logger = StepLogger("hackathon_run")

# State Machine tracking
robot_state = {
    "x": 0.0, "y": 0.0, "heading": 0.0, 
    "holding": False, "retry_count": 0, "state": "Idle",
    "is_rendering": False  # Clock-sync flag added
}

class MoveRequest(BaseModel):
    distance: float
    simulate_collision: bool = False

class TurnRequest(BaseModel):
    degrees: float

# --- JSONL LOGGER ---
# --- JSONL LOGGER ---
def log_step(action, args, result, frame_path=None):
    return logger.log_step(
        tool=action,
        args=args,
        result=result,
        frame_path=frame_path
    )

# --- GET ENDPOINTS ---
@app.get("/state")
def get_state():
    return robot_state

@app.get("/look")
def get_look():
    if not os.path.exists("dummy.jpg"):
        return Response(
            content=b"fake_image_bytes",
            media_type="image/jpeg"
        )

    with open("dummy.jpg", "rb") as f:
        image_bytes = f.read()

    try:
        next_step = logger.step_count + 1

        frame_path = logger.save_frame(
            next_step,
            image_bytes
        )

        log_step(
            action="look",
            args={},
            result={"success": True},
            frame_path=frame_path
        )

    except Exception as e:
        log_step(
            action="look",
            args={},
            result={"success": False, "error": str(e)}
        )

    return Response(content=image_bytes, media_type="image/jpeg")

@app.get("/depth")
def get_depth():
    return {"distance": 5.0}

# --- ENGINE CLOCK-SYNC ---
@app.post("/engine/render_complete")
def render_complete():
    """Teammate 2 calls this when the 3D engine finishes rendering."""
    global robot_state
    robot_state["is_rendering"] = False
    return {"success": True, "message": "Engine ready for next command"}

# --- ACTION ENDPOINTS ---
@app.post("/move")
def move(req: MoveRequest):
    global robot_state
    
    # 1. Clock-sync check
    if robot_state.get("is_rendering"):
        result = {"success": False, "message": "Engine is rendering. Wait."}
        log_step("move", {"distance": req.distance}, result)
        return result
        
    # 2. Blocked state check
    if robot_state["state"] == "Blocked":
        result = {"success": False, "message": "Robot is blocked. Must turn to reset."}
        log_step("move", {"distance": req.distance}, result)
        return result

    # 3. Collision simulation
    if req.simulate_collision:
        robot_state["retry_count"] += 1
        robot_state["x"] -= 0.1
        
        if robot_state["retry_count"] >= 3:
            robot_state["state"] = "Blocked"
            
        result = {"success": False, "collided": True, "distance_moved": -0.1, "retries": robot_state["retry_count"]}
        log_step("move", {"distance": req.distance, "simulate_collision": True}, result)
        return result
    
    # 4. Successful move
    robot_state["retry_count"] = 0
    robot_state["state"] = "Go to object"
    
    rad = math.radians(robot_state["heading"])
    robot_state["x"] += req.distance * math.sin(rad)
    robot_state["y"] += req.distance * math.cos(rad)
    
    robot_state["is_rendering"] = True  # Lock until 3D engine renders
    
    result = {"success": True, "collided": False, "distance_moved": req.distance}
    log_step("move", {"distance": req.distance}, result)
    return result

@app.post("/turn")
def turn(req: TurnRequest):
    global robot_state
    
    if robot_state.get("is_rendering"):
        result = {"success": False, "message": "Engine is rendering. Wait."}
        log_step("turn", {"degrees": req.degrees}, result)
        return result
        
    robot_state["heading"] += req.degrees
    
    if robot_state["state"] == "Blocked":
        robot_state["state"] = "Idle"
        robot_state["retry_count"] = 0
        
    robot_state["is_rendering"] = True
    
    result = {"success": True, "degrees_turned": req.degrees}
    log_step("turn", {"degrees": req.degrees}, result)
    return result

@app.post("/pickup")
def pickup():
    global robot_state
    
    if robot_state.get("is_rendering"):
        result = {"success": False, "message": "Engine is rendering. Wait."}
        log_step("pickup", {}, result)
        return result
        
    robot_state["holding"] = True
    robot_state["state"] = "Carry"
    robot_state["is_rendering"] = True
    
    result = {"success": True}
    log_step("pickup", {}, result)
    return result

@app.post("/drop")
def drop():
    global robot_state
    
    if robot_state.get("is_rendering"):
        result = {"success": False, "message": "Engine is rendering. Wait."}
        log_step("drop", {}, result)
        return result
        
    robot_state["holding"] = False
    robot_state["state"] = "Done"
    robot_state["is_rendering"] = True
    
    result = {"success": True, "in_drop_zone": True}
    log_step("drop", {}, result)
    return result