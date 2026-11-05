# preview.py — open the PyBullet GUI with the full scene and keep it running.
#
#     python preview.py            # seed 42
#     python preview.py 7          # another layout
import sys
import time

import pybullet as p

from sim_engine import SimulationEngine


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    print("Launching PyBullet humanoid preview (seed %d)..." % seed)
    engine = SimulationEngine(gui=True)
    engine.reset_scene(seed=seed)
    print("Window open. Ctrl+C in this terminal to stop.")
    try:
        while p.isConnected():
            with engine.lock:
                p.stepSimulation()
            time.sleep(1.0 / 60.0)
    except KeyboardInterrupt:
        pass
    finally:
        engine.close()
        print("Preview closed.")


if __name__ == "__main__":
    main()
