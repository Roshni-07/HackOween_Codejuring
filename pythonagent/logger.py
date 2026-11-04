
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from io import BytesIO
from PIL import Image


class StepLogger:

    def __init__(self, run_name):

        self.run_dir = Path("Logs") / run_name
        self.frames_dir = self.run_dir / "frames"

        self.frames_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.run_dir / "steps.jsonl"

        self.step_count = 0
        self.lock = threading.Lock()

    def log_step(self, tool, args, result, step=None, frame_path=None):

        with self.lock:

            if step is None:
                self.step_count += 1
                step = self.step_count
            else:
                self.step_count = max(self.step_count, step)

            entry = {
                "step": step,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tool": tool,
                "args": args,
                "result": result,
                "frame_path": frame_path
            }

            with open(self.log_file, "a", encoding="utf-8") as file:
                file.write(json.dumps(entry) + "\n")

        return entry

    def save_frame(self, step, image_bytes):

        image = Image.open(BytesIO(image_bytes))
        image.load()

        image = image.convert("RGB")

        frame_file = self.frames_dir / f"frame_{step:04d}.png"

        image.save(frame_file, format="PNG")

        return str(frame_file)
