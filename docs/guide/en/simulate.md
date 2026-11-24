# 0 · Simulate first

**Goal:** see the whole task run before you print or wire anything.

The repo contains a MuJoCo copy of the cell, built from the same `hardware/params.yaml` as the printed parts. A
scripted expert solves the task in it. You need a laptop and nothing else.

There is also an interactive 3D version in the browser: [flywheel.getporti.com](https://flywheel.getporti.com). Orbit
the cell and play a recorded episode.

## Run it

```bash
git clone https://github.com/fport/so101-flywheel && cd so101-flywheel
uv sync
uv run flywheel-sim build                  # params.yaml → scene + 76-part catalogue
uv run flywheel-sim view                   # interactive viewer: drag to orbit
uv run flywheel-sim render --out cell.png  # one frame: top camera | wrist camera
uv run flywheel-sim episode --part J-T-06  # one episode: video + json + arrays
uv run flywheel-sim batch                  # the 120-episode plan, about 80 s
```

`batch` should end like this:

```
                       success  Wilson 95%
correct command         60/60   [0.94, 1.00]
opposite (followed)     60/60   [0.94, 1.00]
  rim                   30/30   [0.89, 1.00]
  gear                  30/30   [0.89, 1.00]
  pad                   30/30   [0.89, 1.00]
  bracket               30/30   [0.89, 1.00]
failure_type: none
```

Outputs land in `outputs/sim/<run>/`: for every episode a `.json` with the part and the outcome, and a `.npz` with
the joint states and actions (about 200 steps × 6 joints).

## What you are looking at

| In the scene | It is |
| --- | --- |
| The arm | The official SO-101 model: six position-controlled joints with STS3215 servo parameters |
| The black tray | Six numbered pockets, 2 × 3. Pocket 1 is near-left as the arm sees it |
| Two bins | Grey is OK, orange is REJECT |
| Four tags | AprilTags the top camera uses to locate the tray |
| Two cameras | `top` on the mast, `wrist` on the gripper. The same two the real cell records |
| The parts | Rims, gears, brake pads, brackets, in white and orange, some with a missing hole or a broken corner |

Part IDs read like `J-W-03`: a rim (`J`), white (`W`), number 3. Whether a part is OK or REJECT is not a property of
the part. It depends on the *work order*: if the order asks for white, a perfect orange part is a reject.

## The two rows of the table

- **Correct command.** The part goes to the bin the work order implies.
- **Opposite (followed).** The arm is told the *wrong* bin on purpose, and success means it obeyed.

The second row is why the dataset has 12 sentences and not 6. A policy that always sends good-looking parts to OK
has learned to judge parts. We want one that follows the sentence, and we leave the judging to the inspection.

## What the simulator already taught us

These came out of the sim before a single part was printed, and each one changed the real cell.

| Found in sim | Change in the real cell |
| --- | --- |
| On low parts the pocket wall stops the jaws reaching the floor | An 18 mm finger slot on both sides of every pocket |
| Jaws closing along the short axis hit the neighbouring part | Closing axis parallel to the tray's long edge; brackets sit rotated 90° |
| The arm sags 5 to 7 mm under gravity | Grasp points leave margin; nothing depends on millimetre height |
| Parts dropped at one point stack above the bin edge | The drop point is spread inside the bin |
| Bins level with the tray block the jaw at the edge pockets | Bins sit one tile behind the tray |
| The bin wall hides a tag corner from the top camera | The two near tags moved outward in their tiles |
| Holding a part up to the top camera barely works with this wrist | Inspection reads the whole tray once, from above, before any pick |

That is what a simulator is for at this stage: it is a cheap place to be wrong.

## What it cannot tell you

- **How the real servos behave.** The simulated arm reaches its target; the real one has backlash and lag.
- **How things look.** Flat colours and one clean light, against webcam noise and glare.
- **Whether a learned policy works.** The expert here reads the part's exact position from the simulator. It is a
  script, not a policy, and it proves only that the layout is solvable.

Those gaps are the subject of [sim to real](learn-sim2real.md), and making this simulator part of training is
[phase 8 of the route](roadmap.md#phase-8-the-simulator-as-a-twin).

## Done when

- [ ] `flywheel-sim batch` prints 60/60 on both rows.
- [ ] You have watched one episode in the viewer or on the 3D site, from both cameras.
- [ ] You can explain why the "opposite" row exists.

More detail on the scene, the ground-truth labels and the limits of the model: [`docs/sim.md`](../../sim.md).
