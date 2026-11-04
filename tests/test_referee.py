# tests/test_referee.py
from sim.referee import Referee

class DummyEngine:
    held_object_id = None
    pickup_constraint = None

def test_initial_state():
    ref = Referee(DummyEngine())
    assert ref.state_name == "Idle"
    assert ref.outcome is None