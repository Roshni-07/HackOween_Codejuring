from sim_engine import SimulationEngine

engine = SimulationEngine(gui=False)
print("Running 100-seed simulation stress test...")

for seed in range(100):
    engine.reset_scene(seed=seed)
    # Simulate random motion
    engine.move_robot(0.5)
    engine.turn_robot(45)
    assert not engine.collided, f"Unexpected spawn collision on seed {seed}"

print("Physics engine passed 100/100 stress runs!")