# Bambu Lab print plan

> Print-ready files and slicer notes: [hardware/README.md](../hardware/README.md)

Everything to print on the X2D for the flywheel project: workspace fixtures, bins, camera mounts, the QC part set and a feeder for autonomous batches. The goal is for every eval to be repeatable under the same physical conditions.

## Measurements first

Everything is derived from a single `params` file. The values you measure on your own arm go here, and all parts are generated parametrically from them. If the setup changes, only this file is updated and the STLs are re-exported.

Sources give the SO-101 roughly 500 mm reach and ~500 g payload ([roboticscenter.ai](https://roboticscenter.ai/hardware/so-101/specs)). But these are the limits with the arm fully extended in an ideal pose. For repeatable pick-and-place you work well inside these limits, so the values below are measured on your own arm.

| Parameter | How to measure | What it is used for |
| --- | --- | --- |
| `gripper_open_max` | Distance between the jaws with the gripper fully open, with calipers | Part size (widest edge ≤ ~60% of this) |
| `jaw_depth` | Depth from the jaw tip to the palm | Part height, tray pocket depth |
| `reach_comfort` | Extend the arm straight forward via teleop; the farthest point reached comfortably with the wrist straight, minus 20% | The arc where the tray and bins are placed |
| `reach_min` | The closest point the arm reaches without hitting its own body | Near limit |
| `base_footprint` | Footprint of the base plate and clamp on the table | Arm seat in the baseplate |
| `wrist_cam_mount` | Compare with the wrist camera part in the SO-ARM100 repo | Don't redraw if the repo part works |
| `cam_top_height` | Height needed for the top camera to see the tray + both bins in one frame | Top camera mast |

Practical tip: in teleop mode, attach a pen to the end of the arm and draw the reach arc on a sheet of paper spread on the table. `reach_comfort` and `reach_min` can be read straight off the paper.

The CAD tool is up to you. For parametric code, build123d / CadQuery; for a GUI, the user parameter table in Fusion 360 or Onshape's variable studio. All STLs and source files are versioned in the repo under `hardware/`.

## Table layout

The tray and the two bins sit in the arm's comfortable working band, i.e. between `reach_min` and `reach_comfort`, at roughly equal distance from the base. That way the motion to either bin is of similar difficulty and bin choice does not introduce a bias for the policy.

![Table layout: sim view from the top camera](media/sim-scene.png)

*table layout · top camera (left) and wrist camera (right) in the MuJoCo scene; dimensions from `hardware/params.yaml`*

The inspection pose is in the air: the arm picks the part from the tray and shows it to the top camera. The inspection backdrop is placed below or behind this pose. Exact positions are computed from the measurements in the `params` file.

## Print list

The most critical part is the baseplate. The tray, bins and camera mast seat onto it, so if something gets knocked it returns to the same position and evals stay comparable. Gridfinity (42 mm) is recommended as the grid: it is a common standard with plenty of ready-made models, and since it is modular, parts can be printed tile by tile to fit the bed.

| # | Part | Material | Why / note | Phase |
| --- | --- | --- | --- | --- |
| 1 | Gridfinity baseplate tiles + arm seat | PLA | Repeatable position for all fixtures. Arm seat based on `base_footprint`; tiles lock together | 0 |
| 2 | Input tray (numbered pockets) | PLA | Each pocket = a fixed start position in the eval protocol. Pockets chamfered, numbers embossed | 0 |
| 3 | OK and reject bins | PETG (green / red) | Wide mouth, low and outward-flared wall; the top camera must be able to see inside the bin | 0 |
| 4 | Bin floor liner | TPU | Stops the part bouncing out (reduces the `out_of_bin` failure) | 0 |
| 5 | Top camera mast + head | PETG | Based on `cam_top_height`. Printed body, or a wood/aluminium rod + printed connectors; 1/4" screw seat | 0 |
| 6 | Wrist camera mount | PETG | Try the one in the SO-ARM100 repo first; if it doesn't fit, draw one for your camera's dimensions | 0 |
| 7 | AprilTag / ArUco plate holders | PLA | 4 at the table corners. For camera extrinsic calibration and detecting camera shift (drift) | 0 |
| 8 | Cable clips + USB strain relief | PETG | Keeps servo cables from snagging as the arm moves; USB disconnects stop the batch | 0 |
| 9 | E-stop box | PETG | A mushroom button cuts the servo power line. The button has to be bought separately | 0 |
| 10 | Gripper TPU finger pads | TPU | Increases grip friction (reduces `grasp_fail` and `drop`). Must fit the jaw geometry in the repo | 1 |
| 11 | Inspection backdrop | PLA (matte, single colour) | A fixed, plain background behind the part in the inspection pose; improves classifier accuracy | 1 |
| 12 | Gravity feeder (magazine) | PETG | Vertical tube, bottom exit; when the arm takes a part the next one drops down. Removes the need for a human to refill the tray in autonomous batches | 4 |
| 13 | Bin emptying ramp (optional) | PETG | Bins empty to a single collection point via a sloped floor; shortens reset time | 4 |

The e-stop button, TPU filament, camera mounting screws and heat-set inserts are non-printed materials. These can be sourced in Turkey.

## QC part set

4 types × 15 = 60 training parts and 16 held-out; the scenario is end-of-line inspection at an automotive supplier. Each type is printed as 4 sound + 2 missing hole + 2 broken in white and 3 + 2 + 2 in orange: defects exist in both colours so that colour alone does not tell the bin.

| Type | Sound version | Why this shape |
| --- | --- | --- |
| Wheel rim | Ø34 × 14 mm, five spokes, hub hole | Round, low; the spokes are clearly visible from above |
| Gear | Ø33 mm, 14 teeth, with hub and shaft hole | Gripped by the hub; round surface, risk of slipping |
| Brake pad | 34 × 24 mm backing plate + friction block, two ear holes | Low and rectangular; the block is narrower than the plate |
| Mounting bracket | L profile with two mounting holes | Asymmetric; orientation matters |

All dimensions are scaled to `gripper_open_max`. The part's widest gripping edge should not exceed ~60% of this value.

| Defect class | How it is made | Note |
| --- | --- | --- |
| Missing hole | Version with the hub, shaft or mounting hole not machined | Line equivalent: a skipped drilling operation |
| Broken | Snapped spoke on the rim, two missing teeth on the gear, corner break on the pad and bracket | The defect's location varies from part to part |
| Wrong variant | No separate model: the same parts are printed in white and orange | The work order says which one is wrong; see [experiment design](experiment-design.md) |

Things to watch:

- The defect's location and orientation must differ on every part. If the same corner is always broken, the classifier learns the location instead of the defect.
- Put a small embossed ID on the bottom of every part (e.g. `J-W-03`; the label is not part of the ID). Labels stay tied to the physical part and matching with episodes becomes easier.
- There should be slight variation among sound parts too (slightly different shade, layer height). Otherwise "anything that looks different is defective" gets memorized.
- Print a separate held-out set for eval (2-3 parts per type). These parts never enter the training data.

## Material and print settings

The rule is simple: fixtures that need dimensional accuracy are PLA, parts that take impact and load are PETG, parts that need grip and damping are TPU.

| Material | Parts | Settings note |
| --- | --- | --- |
| PLA | Baseplate, tray, tag holders, inspection backdrop, QC parts | 0.2 mm layers; QC parts 0.16 mm. Matte filament reduces reflections on camera |
| PETG | Bins, camera mast, mounts, feeder, cable clips | 4 walls, 30-40% infill; print mounts laid down according to the load direction |
| TPU | Finger pads, bin liner | Slow speed, low retraction. With dual nozzles, can be printed onto a PETG body in a single print |

Fits and tolerances:

- Leave +0.2 mm clearance on fitted parts. Before the first big print, print a small tolerance test piece and write the result to params if needed.
- Enable elephant foot compensation on the baseplate and tray; otherwise tiles and pockets will not seat properly.
- For screw joints, use M3 heat-set inserts instead of screwing directly into plastic.
- Avoid glossy filament for camera-facing surfaces and bins; glare misleads both the classifier and the verifier.

## Print order

First dimensions and tolerances, then the parts that fix positions, the dataset last. That way no part is printed twice.

**Day 1 — Measure and test**

- [ ] Measure all values in the `params` table and write them to the file
- [ ] Tolerance test piece (fit, hole, heat-set insert)
- [ ] A single baseplate tile + arm seat; check that the arm sits in it

**Phase 0 — Workspace**

- [ ] The remaining baseplate tiles
- [ ] Input tray
- [ ] OK and reject bins + TPU liners
- [ ] Top camera mast and wrist camera mount
- [ ] AprilTag holders (4)
- [ ] Cable clips, USB strain relief, e-stop box
- [ ] Assemble the setup and verify that the tray + both bins appear in the same top camera frame

**Phase 1 — Dataset and grasping**

- [ ] TPU finger pads; do 10 teleop attempts and check for slipping
- [ ] QC part set (60 parts) + held-out eval set
- [ ] Inspection backdrop

**Phase 4 — Autonomy**

- [ ] Gravity feeder; 10-part jam test
- [ ] Bin emptying ramp (optional)

To run the print queue overnight, put the long tile and QC set prints in the evening and the test pieces during the day.
