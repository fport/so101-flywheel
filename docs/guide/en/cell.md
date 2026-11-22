# 1 · Build the cell

**Goal:** a scene where nothing moves between recording and running the policy. The policy learns pixels; if the
tray, a bin or a camera shifts by a centimetre, the pixels it learned no longer mean the same thing.

## What is on the desk

| Piece | Why |
| --- | --- |
| Baseplate (Gridfinity tiles) | Locks arm, tray, bins and mast to one grid |
| Tray, 2 × 3 pockets, matte black | Light parts on black: easy for a camera and for you |
| OK bin (grey, arm's right) and REJECT bin (orange, arm's left) | Different colours so the two targets never look alike |
| Mast with the top camera, 66 cm up | Sees the tray, both bins and the arm in one frame |
| Wrist camera on the gripper | Sees the part between the jaws: this is what makes grasps precise |
| Four AprilTags on the baseplate | Let the inspection find the pockets in the image, in millimetres |
| 60 parts: rim, gear, brake pad, bracket × white/orange × sound / missing hole / broken | Variety to learn from, defects to inspect |

Everything is generated from one file, [`hardware/params.yaml`](../../../hardware/params.yaml): the print files and
the simulation use the same numbers. Print and assembly: [`hardware/README.md`](../../../hardware/README.md).

## No printer?

Use any dark tray with six marked spots and two boxes of different colours, and tape or screw everything down. You
lose `flywheel-inspect` (it needs the tags and the pocket positions), not the rest of the path.

## Check

- Push each piece with a finger: nothing slides.
- Cables cannot fall into the camera view or snag the arm.
- Lighting comes from above or from the front, not from a window behind the cell; no hard shadow of the arm on the tray.
