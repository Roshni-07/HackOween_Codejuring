# sim/recorder.py
import cv2
import numpy as np
from sim.bridge import push_frame_and_unlock


class VideoRecorder:
    def __init__(self, engine, output_path="robot_pov.mp4", fps=30):
        self.engine = engine
        self.output_path = output_path
        self.fps = fps
        self.frames = []
        self.command_text = ""
        self.overlay_frames = fps * 2  # Show command text for first 2 seconds

    def set_command(self, command: str):
        self.command_text = command

    def capture_frame(self):
        # 1. Render clean frame from sensors
        jpeg_bytes = self.engine.sensors.get_camera_image()
        
        # 2. Automatically sync with Sharvesh's hotspot backend
        push_frame_and_unlock(jpeg_bytes)

        # 3. Process locally for MP4 video recording
        nparr = np.frombuffer(jpeg_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if len(self.frames) < self.overlay_frames and self.command_text:
            cv2.putText(
                frame,
                f"Cmd: {self.command_text}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        self.frames.append(frame)

    def save_video(self):
        if not self.frames:
            return
        height, width, _ = self.frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (width, height))
        for frame in self.frames:
            out.write(frame)
        out.release()