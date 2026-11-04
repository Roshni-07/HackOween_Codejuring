import requests

BASE_URL = "http://localhost:8080"


def run_api_test():
    print("--- 1. Testing GET /state ---")
    res = requests.get(f"{BASE_URL}/state")
    print("State Response:", res.json())

    print("\n--- 2. Testing GET /depth ---")
    res = requests.get(f"{BASE_URL}/depth")
    print("Depth Reading:", res.json())

    print("\n--- 3. Testing GET /depth_fan ---")
    res = requests.get(f"{BASE_URL}/depth_fan")
    print("Depth Fan Reading:", res.json())

    print("\n--- 4. Testing GET /look ---")
    res = requests.get(f"{BASE_URL}/look")
    print(
        "Look Image Status:",
        res.status_code,
        "| Content-Type:",
        res.headers.get("content-type"),
    )
    with open("api_look_test.jpg", "wb") as f:
        f.write(res.content)
    print("Saved look frame to api_look_test.jpg")

    print("\n--- 5. Testing POST /turn (Turn 45 deg) ---")
    res = requests.post(f"{BASE_URL}/turn", json={"degrees": 45.0})
    print("Turn Response:", res.json())

    print("\n--- 6. Testing POST /move (Move 0.5m) ---")
    res = requests.post(f"{BASE_URL}/move", json={"distance": 0.5})
    print("Move Response:", res.json())

    print("\n--- 7. Testing POST /pickup ---")
    res = requests.post(f"{BASE_URL}/pickup")
    print("Pickup Response:", res.json())

    print("\n--- 8. Testing POST /drop ---")
    res = requests.post(f"{BASE_URL}/drop")
    print("Drop Response:", res.json())


if __name__ == "__main__":
    try:
        run_api_test()
        print("\nAll endpoints tested successfully!")
    except Exception as e:
        print("\nError connecting to server:", e)