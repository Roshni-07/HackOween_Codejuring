import json
from datetime import datetime, timezone
from pathlib import Path


class StepLogger:

    def __init__(self, run_name):

        self.run_dir = Path("Logs") / run_name
        self.frames_dir = self.run_dir / "frames"

        self.frames_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.run_dir / "steps.jsonl"

    def log_step(self, step, tool, args, result, frame_path=None):

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