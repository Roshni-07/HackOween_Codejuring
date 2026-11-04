# tests/test_camera_fov.py
from sim.sensors import SensorSuite

class DummyEngine:
    robot_heading = 0.0

def test_fov_aspect_ratio():
    sensors = SensorSuite(DummyEngine())
    assert sensors.fov_vert == 73.74
    assert round(sensors.aspect, 2) == 1.33