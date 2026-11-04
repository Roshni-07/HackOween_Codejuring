from fastapi import FastAPI, Response
from pydantic import BaseModel
import os
import json
from datetime import datetime
import math
import base64

app = FastAPI(title="Hackathon Engine API")

# State Machine tracking
robot_state = {
    "x": 0.0, "y": 0.0, "heading": 0.0, 
    "holding": False, "retry_count": 0, "state": "Idle",
    "is_rendering": False  # Clock-sync flag added
}

# Live frame stored in RAM to prevent file-locking crashes
current_frame_bytes = None

class MoveRequest(BaseModel):
    distance: float
    simulate_collision: bool = False

class TurnRequest(BaseModel):
    degrees: float

class FrameUpdate(BaseModel):
    image_base64: str

# --- JSONL LOGGER ---
def log_step(action, args, result):
    with open("run_log.jsonl", "a") as f:
        log_entry = {
            "timestamp": datetime.now().isoformat(), 
            "action": action, 
            "args": args, 
            "result": result
        }
        f.write(json.dumps(log_entry) + "\n")

# --- GET ENDPOINTS ---
@app.get("/state")
def get_state():
    return robot_state

@app.get("/look")
def get_look():
    global current_frame_bytes
    
    # 1. Serve the live RAM frame from Teammate 2 if available
    if current_frame_bytes:
        return Response(content=current_frame_bytes, media_type="image/jpeg")
        
    # 2. Fallback to the static file if the engine hasn't sent a frame yet
    if os.path.exists("dummy.jpg"):
        with open("dummy.jpg", "rb") as f:
            return Response(content=f.read(), media_type="image/jpeg")
            
    return Response(content=b"fake_image_bytes", media_type="image/jpeg")

@app.get("/depth")
def get_depth():
    return {"distance": 5.0}

# --- ENGINE CLOCK-SYNC & FRAME PIPELINE ---
@app.post("/engine/render_complete")
def render_complete():
    """Teammate 2 calls this when the 3D engine finishes rendering."""
    global robot_state
    robot_state["is_rendering"] = False
    return {"success": True, "message": "Engine ready for next command"}

@app.post("/update_frame")
def update_frame(req: FrameUpdate):
    """Teammate 2 sends live base64 camera frames here."""
    global current_frame_bytes
    current_frame_bytes = base64.b64decode(req.image_base64)
    return {"success": True, "message": "Frame updated in memory"}

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