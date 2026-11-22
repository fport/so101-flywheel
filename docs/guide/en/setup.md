# 2 · Set up the arm

**Goal:** the follower copies the leader smoothly and both cameras show the right view.

## Install

```bash
git clone https://github.com/fport/so101-flywheel && cd so101-flywheel/python
uv sync --extra robot          # LeRobot 0.6.1 with the Feetech servo driver
hf auth login                  # token with write access
```

## Tell the scripts about your rig

```bash
uv run lerobot-find-port               # unplug one arm when asked; note both ports
uv run lerobot-find-cameras opencv     # saves one frame per camera, note the two indices
```

Put the four values and your Hugging Face user name in `scripts/env.local.sh` (your own file, not committed; the
template is at the top of [`scripts/env.sh`](../../../scripts/env.sh)). Every script reads them from there and prints the
full `lerobot-*` command before running it, so you always see what runs.

## Calibrate (once)

```bash
scripts/calibrate.sh follower
scripts/calibrate.sh leader
```

Move the arm to the middle of every joint, press Enter, then sweep each joint end to end. Calibration maps raw servo
ticks to the same angle on both arms; it is saved in `hardware/calibration/`. Recalibrating after you have recorded
changes what the numbers in your dataset mean: do it before, not in between.

## Teleoperate

```bash
scripts/teleop.sh           # leader drives follower
scripts/cams.sh             # in a second terminal: both cameras at http://localhost:8642
```

Hold the leader near the follower's pose before you start: the follower jumps to the leader on connect.

## Check

- Top camera: tray, both bins and the whole reach of the arm are in frame, nothing else moves in it.
- Wrist camera: both jaw tips and the space between them are visible.
- Do the task once while looking only at the two camera windows, not at the desk. If you cannot do it from the
  images, the policy cannot either. Move the cameras until you can.
- Practise until one pick-and-drop takes about 10 seconds without fumbling.

**Why not Rerun?** `scripts/teleop.sh --cams` opens LeRobot's Rerun viewer, but on a Mac it has stalled the control loop
("Sender has been blocked… waiting for space in channel"). The browser preview is one small process and cannot block
the arm. Stop it before recording: the recording needs the cameras for itself.
