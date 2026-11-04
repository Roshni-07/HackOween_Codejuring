from sim_engine import SimulationEngine

engine = SimulationEngine(gui=False)
print("Running 100-seed simulation stress test...")

spawn_clipping_errors = 0
movement_collisions = 0

for seed in range(100):
    engine.reset_scene(seed=seed)

    # 1. Assert no instant spawn overlap or clipping at t=0
    assert (
        not engine.check_physical_collisions()
    ), f"Spawn clipping detected on seed {seed}"

    # 2. Test movement physics and recoil handling
    res = engine.move_robot(0.5)
    engine.turn_robot(45)

    if res["collided"]:
        movement_collisions += 1
        assert (
            engine.state_name == "Blocked"
        ), f"Failed to enter Blocked state on seed {seed}"

print("PASS: 100/100 seeds executed cleanly!")
print(f"  - Spawn clipping errors: {spawn_clipping_errors}")
print(
    f"  - Obstacle contacts safely detected & recoiled: {movement_collisions}"
)