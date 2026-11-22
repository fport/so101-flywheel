# What to try next

Each of these is one experiment with a clear question. Do them after the SmolVLA baseline has a score: without a
baseline you cannot tell whether anything helped.

## A bigger policy: Cosmos 3 Edge

[nvidia/Cosmos3-Edge](https://huggingface.co/nvidia/Cosmos3-Edge) is a 4 B-parameter *world model*: it generates
video, text and actions with one network, so it is not a drop-in VLA like SmolVLA. LeRobot (0.7 and later) can
fine-tune it as a policy:

```bash
lerobot-train --policy.type=cosmos3_edge --policy.pretrained_path=lerobot/Cosmos3-Edge \
  --policy.load_action_modules=false --dataset.repo_id=fport/so101-qc ...
```

- **Question:** does a 10× larger model follow the 12 sentences better on the same 120 episodes?
- **Cost:** LoRA fine-tuning needs about 21 GB of GPU memory (an A100, not an A10G); full fine-tuning 47 GB. Linux only.
- **Caveat:** the model card does not list the SO-101; NVIDIA's blog does. Read the
  [LeRobot Cosmos 3 page](https://huggingface.co/docs/lerobot/cosmos3) for the camera description it expects.

## Sim to real: Cosmos 3 Nano Sim-Transfer

[nvidia/Cosmos3-Nano-Sim-Transfer](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Transfer) turns simulator output
(depth, segmentation or edges) plus a text prompt into photoreal video. This repo has a MuJoCo copy of the cell and a
scripted expert that solves it (`uv run flywheel-sim batch`).

- **Question:** can sim episodes, repainted to look real, replace some of the teleoperated ones?
- **Cost:** 15–16 B parameters, tested by NVIDIA on H100-class GPUs only. Expect more than 24 GB.
- **Path:** render depth + segmentation from the sim → Sim-Transfer → a LeRobot dataset with the sim's actions →
  train on real + generated → same 24-trial test.

## Grounding: LocateAnything {#grounding-locateanything}

[nvidia/LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B) returns boxes or points (0–1000
coordinates) for a text prompt such as "gear with a missing tooth".
[Demo](https://huggingface.co/spaces/nvidia/LocateAnything).

- **Question:** can a prompt replace the hand-written measurements in `flywheel-inspect`, and the AprilTags?
- **Test:** the frames in `outputs/inspect/*/frame.png` with known part IDs are a ready-made benchmark.
- **Caveats:** non-commercial licence; the public demo may store what you upload.

## Reinforcement learning

Imitation stops at the quality of your demonstrations. RL lets the arm improve past them from its own attempts and a
success signal. LeRobot ships HIL-SERL (human-in-the-loop RL on the real arm). The inspection camera could give the
reward: "is the part in the commanded bin?". Start this only when imitation plateaus.

## A newer LeRobot

This repo pins LeRobot 0.6.1 (the arm is calibrated and tested on it). 0.7 adds Cosmos 3 policies and remote
inference. The dataset format (v3.0) is the same, so you can record on 0.6.1 and train on 0.7.
