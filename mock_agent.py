import requests

BASE_URL = "http://127.0.0.1:8080"

print("1. Crashing 1:", requests.post(f"{BASE_URL}/move", json={"distance": 1.0, "simulate_collision": True}).json())
print("2. Crashing 2:", requests.post(f"{BASE_URL}/move", json={"distance": 1.0, "simulate_collision": True}).json())
print("3. Crashing 3:", requests.post(f"{BASE_URL}/move", json={"distance": 1.0, "simulate_collision": True}).json())
print("4. Moving while Blocked:", requests.post(f"{BASE_URL}/move", json={"distance": 1.0}).json())
print("5. Current State:", requests.get(f"{BASE_URL}/state").json())
print("6. Turning to unblock:", requests.post(f"{BASE_URL}/turn", json={"degrees": 45.0}).json())
print("7. Final State:", requests.get(f"{BASE_URL}/state").json())