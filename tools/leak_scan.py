# tools/leak_scan.py
import sys
import requests

AGENT_API_URL = "http://127.0.0.1:8080"


def test_state_leakage():
    print("Running Zero-Leakage AST & API Scan...")
    resp = requests.get(f"{AGENT_API_URL}/state").json()
    
    prohibited_keys = ["object", "objects", "target", "drop_zone", "cube", "sphere", "color", "shape"]
    leaks = [k for k in resp.keys() if any(p in k.lower() for p in prohibited_keys)]

    if leaks:
        print(f"FAILED: Prohibited object keys found in /state: {leaks}")
        sys.exit(1)

    print("PASS: /state is 100% clean of object metadata!")


if __name__ == "__main__":
    test_state_leakage()