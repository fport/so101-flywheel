# SO-101 model

`so101_new_calib.xml` and `assets/*.stl` were copied unmodified from the `Simulation/SO101/` folder of the
[TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) repo (commit `63eede5a636e548eb8f2854e558bd343c21db9f7`,
Apache-2.0, see `LICENSE`).

The files are not edited by hand. Scene-specific changes (joint names, jaw collision boxes, wrist camera,
`pinch` site) are applied at build time in `flywheel/sim/scene.py`.
