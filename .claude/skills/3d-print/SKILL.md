---
name: 3d-print
description: Use in this project when designing or changing a 3D-printed part, or before sending a print file (hardware/print/*.3mf) to the printer. Bambu X2D, PLA, Gridfinity feet, fits, tolerances, Bambu Studio steps and the pre-send checklist.
---

1. Read `hardware/3DPRINT.md` and apply its design rules. The rules come from mistakes that actually happened in this project.
2. Change the geometry under `python/flywheel/hardware/`; take dimensions from `hardware/params.yaml`.
3. Before sending, run the following in `python/` and report the results to the user with numbers:
   - `uv run flywheel-cad build`
   - `uv run pytest -q`
   - `uv run flywheel-cad slicecheck ../hardware/print/<job>.3mf`
4. For new or changed jobs, if you can use agents, have an independent agent measure the geometry. Give the agent the files and the rules,
   not your own conclusion.
5. Don't add parts the user didn't ask for; ask before adding. For Bambu Studio, write the steps using the names of the buttons
   to click. When interpreting a screenshot, check the layer slider's height; if unsure, run
   `slicecheck`, don't guess.
