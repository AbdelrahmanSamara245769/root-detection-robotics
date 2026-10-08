# OT-2 simulation (not included)

The robotics scripts run on a PyBullet digital twin of the Opentrons OT-2. The twin was provided by the course, so it isn't part of this repository. To run the scripts, put these files in this folder:

```
simulation/
├── sim_class.py              # Simulation class (PyBullet)
├── custom.urdf               # specimen plate (150 mm x 150 mm)
├── ot_2_simulation_v6.urdf   # robot model
├── meshes/                   # robot meshes
└── textures/
    ├── 01.png, 02.png, ...   # plate textures shown in the simulation
    └── _plates/              # full-size plate image matching each texture
```

`sim_class.py` loads these files with paths relative to the working directory. `robotics/project_paths.py` switches to this folder before the simulation starts, so you can run the scripts from the repository root.

## Notes

- **Integration scripts:** `integration_pid.py`, `integration_rl.py` and `telemetry.py` identify the loaded plate by its file name in `textures/_plates/`. They expect `test_image_10.png`, `test_image_14.png` and `test_image_16.png`, each with a matching texture. They also need `data/root_tip_coordinates.json`, which `notebooks/03_root_tip_pipeline.ipynb` writes.
- **RL training speed:** comment out the `time.sleep(1./240.)` call in `Simulation.run`. This raised throughput from about 24 to about 800 steps per second.
