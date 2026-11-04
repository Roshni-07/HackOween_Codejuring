# sim/bridge.py
import base64
import requests

SHARVESH_BACKEND_URL = "http://10.121.224.93:8080"


def push_frame_and_unlock(jpeg_bytes: bytes):
    """Sends base64 camera frame to Sharvesh's server and unlocks the robot."""
    try:
        # 1. Base64 encode the JPEG frame
        encoded_frame = base64.b64encode(jpeg_bytes).decode("utf-8")

        # 2. Push frame to POST /update_frame with key "image_base64"
        requests.post(
            f"{SHARVESH_BACKEND_URL}/update_frame",
            json={"image_base64": encoded_frame},  # <--- Exact match for your backend key!
            timeout=3.0,
        )

        # 3. Notify render complete to unlock robot
        requests.post(
            f"{SHARVESH_BACKEND_URL}/engine/render_complete",
            timeout=3.0,
        )
    except requests.exceptions.RequestException as e:
        print(f"Warning: Sharvesh hotspot sync failed: {e}")