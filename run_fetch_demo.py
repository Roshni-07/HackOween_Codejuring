# run_fetch_demo.py
import time
from sim.core import CoreEngine
from sim.scene import SceneManager
from sim.recorder import VideoRecorder

def run_demo():
    print("Initializing Core Engine with GUI & Video Recorder...")
    engine = CoreEngine(gui=True)
    scene = SceneManager(engine)
    engine.scene = scene

    # Setup recorder
    recorder = VideoRecorder(engine, output_path="robot_pov.mp4", fps=30)
    recorder.set_command("Bring the red box to the green square")
    engine.recorder = recorder

    # Build scene & spawn robot
    scene.build_room()
    scene.spawn_robot()
    scene.spawn_drop_zone()
    scene.spawn_objects(seed=42)

    print("Executing robot work routine...")
    time.sleep(1.0)

    # 1. Turn to locate
    engine.driver.turn(45.0)
    time.sleep(0.5)

    # 2. Drive towards objects
    engine.driver.move(1.5)
    time.sleep(0.5)

    # 3. Attempt pickup
    res = engine.referee.pickup_object()
    print("Pickup result:", res)
    time.sleep(1.0)

    # Save the POV MP4 clip
    recorder.save_video()
    print("Demo complete! POV video saved to robot_pov.mp4")

if __name__ == "__main__":
    run_demo()