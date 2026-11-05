"""
HackOween simulation package.

    scene.py    arena, Halloween set dressing, HUMANOID_URDF, palette/lighting
    sensors.py  forward RGB camera (/look) and depth ray (/depth)
    driver.py   kinematic base + S-curve joint articulation
    referee.py  state machine, pickup reach / drop-zone rules, outcomes
    meshes.py   procedural OBJ builder for visual-only decorations

sim_engine.SimulationEngine composes these behind the API used by server.py.
"""
