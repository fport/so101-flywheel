# Sim to real: Cosmos 3 Sim-Transfer

Teleoperating 120 episodes takes an afternoon. A simulator can produce 12 000 overnight, with perfect labels, and
nobody has to reset the tray. The catch has a name, the *reality gap*, and this page is about one specific way to
narrow it: repainting simulated video so it looks like your desk.

Nothing here has been run yet. The model facts are from the
[Cosmos3-Nano-Sim-Transfer card](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Transfer); the pipeline is our design.

## The gap has three parts

| Gap | In our cell | Who can close it |
| --- | --- | --- |
| **Appearance** | MuJoCo's flat colours and hard shadows against a webcam's noise, glare and blur | Randomisation, or a generative model: this page |
| **Dynamics** | A simulated servo follows its target exactly; an STS3215 has backlash, sags under load and lags | System identification, or real data |
| **Sensing and timing** | Simulated frames are instant and synchronised; USB cameras are neither | Real data |

A repainted video fixes the first row only. That decides the whole design: generated episodes are an *addition* to
real ones, never a replacement.

## What we already have

`flywheel-sim` builds a MuJoCo copy of the cell from the same `hardware/params.yaml` as the printed parts: tray,
bins, mast, both cameras at their real positions. A scripted expert solves it.

```bash
uv run flywheel-sim batch      # every part to both bins, prints a success table
```

So the simulator already gives us what a human cannot: unlimited episodes with exact actions, and every part in
every pocket.

## The classic fix: randomise

Domain randomisation changes colours, lights, textures and camera pose in every episode, so that the real world
looks like just another variation. It is cheap and it works for simple scenes, but the images stay unmistakably
synthetic, and a pretrained VLM has seen millions of real photos and very few flat-shaded renders.

## The generative fix: keep the geometry, repaint the pixels

Cosmos3-Nano-Sim-Transfer takes a *control video* and a text prompt, and generates a photoreal video that follows
the control.

| | |
| --- | --- |
| Control (one of) | Depth maps, segmentation masks, edge maps, or blurred images |
| Prompt | A text description of how the scene should look |
| Output | 480p, 30 fps, up to 600 frames (20 s) |
| Model | 16 B parameters, a few-step distilled version of Cosmos 3 Nano |
| Speed | 14.6 frames/s on an H100, 24.5 on a GB200 |
| Runs as | A Docker server (vLLM-Omni or SGLang) you send requests to. Linux |
| Licence | OpenMDW 1.1, commercial use allowed |

The division of labour is the point: *the simulator stays responsible for physics, state and labels; the model only
paints.* The action at frame 137 is still the action the simulator executed at frame 137.

## How well is "follows the control"?

NVIDIA reports how closely the output matches the control it was given (PAI-Bench, Sim-Transfer against the plain
Cosmos 3 Nano):

| Check | Sim-Transfer | Base model | Reads as |
| --- | --- | --- | --- |
| Segmentation mIoU ↑ | 0.74 | 0.64 | Objects stay roughly where the mask says |
| Depth error (SI-RMSE) ↓ | 0.62 | 0.84 | Depth is followed better than by the base model |
| Edge F1 ↑ | 0.34 | 0.21 | Better than the base model, but fine edges are followed only loosely |

That last row matters for a quality-control cell. A missing 7 mm hole or a chipped corner is an edge-level detail,
and it may not survive the repaint. For the *policy* this is fine: the sentence already says which pocket and which
bin, so the arm has to see where the part is, not what is wrong with it. For the *inspection* it is not fine: do not
train or test `flywheel-inspect` on generated frames.

## The pipeline for this cell

```
flywheel-sim (scripted expert)
   ├─ actions, joint states ───────────────────────────────┐
   └─ depth or segmentation video, top and wrist           │
            │                                              │
            ▼  Sim-Transfer + prompt                       │
      photoreal top and wrist video                        │
            │                                              ▼
            └──────────────► LeRobot dataset (same keys, same sentences as fport/so101-qc)
                                              │
                    real episodes ───────────►│ train on both
                                              ▼
                                   the same real 24-trial test
```

Four things have to be built, in this order:

1. **Depth and segmentation from the sim.** MuJoCo's renderer produces both (`enable_depth_rendering`,
   `enable_segmentation_rendering`); `flywheel-sim` saves only RGB today.
2. **A sim episode writer** in the LeRobot format, with the real dataset's feature names, fps and task sentences.
3. **The repaint job.** For 120 episodes of 10 s on two cameras that is 72 000 frames, about 80 minutes on one H100
   at the published speed.
4. **A prompt** that describes *our* desk: matte black tray, grey and orange bins, a white SO-101, the light. A few
   real frames are the reference for writing it.

## What is likely to go wrong

- **The two cameras disagree.** The card says nothing about multi-view consistency. Top and wrist are repainted
  separately, so the same part may get a different texture in each view. One thing to try: tile both views into one
  control video, the way Cosmos 3 policies tile their input.
- **The arm does not look like our arm.** A depth map does not say "white PLA with black servos". The prompt has to,
  and it may still drift between episodes. Segmentation control with a consistent colour per body is the better bet.
- **The dynamics gap is untouched.** If the simulated gripper closes in 0.2 s and the real one in 0.5 s, the policy
  learns the wrong timing from perfectly realistic pictures.
- **It was built for something else.** NVIDIA describes closed-loop RL and policy evaluation in Isaac Lab, with the
  model painting frames live. Offline repainting of an imitation dataset is our use, not a documented one.
- **Hardware.** Tested on H100-class GPUs and the RTX PRO 6000 only; memory is not stated. With 16 B parameters, plan
  for a 48 GB or 80 GB GPU.

## The experiment that settles it

Three policies, same architecture, same real test:

| Train on | Tells you |
| --- | --- |
| 120 real | The baseline |
| 60 real | How much the second 60 real episodes were worth |
| 60 real + 600 generated | Whether generated data can stand in for the missing 60 |

If the third beats the second, sim data helps. If it also matches the first, half of the teleoperation was
replaceable. If it is *worse* than the second, the generated data is pulling the policy away from reality, and the
cheaper fix is randomised sim or simply more real episodes.

## What to take to the desk

- Sim-Transfer changes how frames look. It cannot fix how the simulated robot moves.
- Always evaluate on the real arm. Success in a repainted simulator is not a result.
- Keep generated episodes in a separate dataset. You will want to change the mixing ratio.

**Sources:** [nvidia/Cosmos3-Nano-Sim-Transfer](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Transfer) model card ·
[cosmos-framework](https://github.com/NVIDIA/cosmos-framework) ·
[Domain Randomization](https://arxiv.org/abs/1703.06907) · [MuJoCo rendering](https://mujoco.readthedocs.io/en/stable/python.html)
