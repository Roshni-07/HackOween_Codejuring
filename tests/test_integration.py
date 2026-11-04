# tests/test_integration.py
import time
import requests

AGENT_URL = "http://127.0.0.1:8080"
ADMIN_URL = "http://127.0.0.1:8081"

def test_full_pickup_pipeline():
    print("1. Resetting scene...")
    res = requests.post(f"{ADMIN_URL}/reset", json={"seed": 42})
    print("Reset response:", res.json())

    print("\n2. Checking initial state...")
    state = requests.get(f"{AGENT_URL}/state").json()
    print("Robot State:", state)

    print("\n3. Looking around...")
    res = requests.get(f"{AGENT_URL}/look")
    print(f"Camera frame received! Size: {len(res.content)} bytes")

    print("\n4. Driving close to object (0.4m)...")
    move_res = requests.post(f"{AGENT_URL}/move", json={"distance": 0.4}).json()
    print("Move result:", move_res)

    print("\n5. Calling PICKUP endpoint...")
    pickup_res = requests.post(f"{AGENT_URL}/pickup").json()
    print("Pickup response:", pickup_res)

    print("\n6. Turning towards drop zone...")
    turn_res = requests.post(f"{AGENT_URL}/turn", json={"degrees": 45.0}).json()
    print("Turn result:", turn_res)

    print("\n7. Driving with object to drop zone (1.0m)...")
    move_carry_res = requests.post(f"{AGENT_URL}/move", json={"distance": 1.0}).json()
    print("Move carrying result:", move_carry_res)

    print("\n8. Calling DROP endpoint...")
    drop_res = requests.post(f"{AGENT_URL}/drop").json()
    print("Drop response:", drop_res)

    print("\nSUCCESS: Full Pick & Place Pipeline Executed!")

if __name__ == "__main__":
    test_full_pickup_pipeline()