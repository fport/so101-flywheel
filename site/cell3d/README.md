# 3D cell viewer

The interactive three.js view of the cell at [flywheel.getporti.com](https://flywheel.getporti.com): orbit the scene,
play a simulated episode, browse the parts set. It is a static site; the step-by-step guide lives at
[zero2hero.getporti.com](https://zero2hero.getporti.com) (`site/build.py`).

    cd site/cell3d && npx wrangler deploy

`public/data.js` and `public/scene/` are a snapshot exported from the simulator on 2026-11-21. The exporter was
removed with the old recorder; the "Recording" section still shows the earlier session-based plan.
