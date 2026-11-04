from fastapi import FastAPI, Response
from pydantic import BaseModel
import os

app = FastAPI(title="Hackathon Engine API")

# Your State Machine tracking
robot_state = {
    "x": 0.0, "y": 0.0, "heading": 0.0, 
    "holding": False, "retry_count": 0, "state": "Idle"
}

class MoveRequest(BaseModel):
    distance: float
    simulate_collision: bool = False

class TurnRequest(BaseModel):
    degrees: float

@app.get("/state")
def get_state():
    return robot_state

@app.get("/look")
def get_look():
    # Returns a dummy image for now. Prevents crashing if the file is missing.
    if not os.path.exists("dummy.jpg"):
        return Response(content=b"fake_image_bytes", media_type="image/jpeg")
    with open("dummy.jpg", "rb") as f:
        return Response(content=f.read(), media_type="image/jpeg")

@app.get("/depth")
def get_depth():
    return {"distance": 5.0}

@app.post("/move")
@app.post("/move")
def move(req: MoveRequest):
    global robot_state
    
    if robot_state["state"] == "Blocked":
        return {"success": False, "message": "Robot is blocked. Must turn to reset."}

    if req.simulate_collision:
        robot_state["retry_count"] += 1
        robot_state["x"] -= 0.1
        
        if robot_state["retry_count"] >= 3:
            robot_state["state"] = "Blocked"
            
        return {"success": False, "collided": True, "distance_moved": -0.1, "retries": robot_state["retry_count"]}
    
    robot_state["retry_count"] = 0
    robot_state["state"] = "Go to object"
    
    import math
    rad = math.radians(robot_state["heading"])
    robot_state["x"] += req.distance * math.sin(rad)
    robot_state["y"] += req.distance * math.cos(rad)
    
    return {"success": True, "collided": False, "distance_moved": req.distance}
@app.post("/turn")
def turn(req: TurnRequest):
    global robot_state
    robot_state["heading"] += req.degrees
    
    if robot_state["state"] == "Blocked":
        robot_state["state"] = "Idle"
        robot_state["retry_count"] = 0
        
    return {"success": True, "degrees_turned": req.degrees}

@app.post("/pickup")
def pickup():
    robot_state["holding"] = True
    robot_state["state"] = "Carry"
    return {"success": True}

@app.post("/drop")
def drop():
    robot_state["holding"] = False
    robot_state["state"] = "Done"
    return {"success": True, "in_drop_zone": True}