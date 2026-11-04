
import json
from pathlib import Path
from io import BytesIO
from PIL import Image

from logger import StepLogger


# Create a fresh test run
logger = StepLogger("logger_upgrade_test")


# Test 1: Successful robot action
logger.log_step(
    tool="move",
    args={"distance": 1.0},
    result={"success": True}
)


# Test 2: Failed robot action
logger.log_step(
    tool="turn",
    args={"degrees": 90},
    result={"success": False, "error": "Test failure"}
)


# Test 3: Create a sample JPEG camera image
sample_image = Image.new("RGB", (640, 480), color="blue")

image_buffer = BytesIO()
sample_image.save(image_buffer, format="JPEG")

jpeg_bytes = image_buffer.getvalue()


# Save JPEG image as PNG
frame_path = logger.save_frame(2, jpeg_bytes)


# Test 4: Record the camera frame path
logger.log_step(
    tool="look",
    args={},
    result={"success": True},
    frame_path=frame_path
)


# Verify the log file
log_file = Path("Logs/logger_upgrade_test/steps.jsonl")

with open(log_file, "r", encoding="utf-8") as file:
    entries = [json.loads(line) for line in file]


print("Total logged steps:", len(entries))
print("Step numbers:", [entry["step"] for entry in entries])
print("Failed action recorded:", entries[1]["result"]["success"] is False)

# Verify actual PNG format
saved_image = Image.open(frame_path)

print("Saved image format:", saved_image.format)
print("Saved image dimensions:", saved_image.size)

print("\nALL TESTS COMPLETED!")

