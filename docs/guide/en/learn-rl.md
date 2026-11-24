# Reinforcement learning on a real arm

Imitation learning copies you. It cannot become better than you, and it never finds out whether the part reached
the bin. Reinforcement learning (RL) learns from the *outcome*. This page explains how that works on a physical arm
without thousands of hours, and what it would take in this cell.

Nothing here has been run on our arm. The workflow is from the LeRobot
[HIL-SERL](https://huggingface.co/docs/lerobot/hilserl) and reward-model documentation.

## What changes compared with imitation

| | Imitation | Reinforcement learning |
| --- | --- | --- |
| Learns from | What the human did | What happened afterwards |
| Signal | An action for every frame | A reward, often only 1 or 0 at the end |
| Data comes from | You, teleoperating | The policy's own attempts |
| Ceiling | Your demonstrations | The task itself |
| Main risk | Drifting off the demonstrated path | Exploring: the arm tries things, including bad ones |

The last row is why RL on real hardware was rare for so long. A policy that learns by trial needs many trials, and
early trials look like flailing.

## The three ideas that make it practical

**1. Off-policy learning.** The algorithm (Soft Actor-Critic, SAC) can learn from any stored transition, no matter
which policy produced it. Every attempt goes into a *replay buffer* and is reused many times. So can your
demonstrations: they are just more transitions.

**2. A human in the loop.** When the arm is about to do something useless or dangerous, you take over, fix it, and
give control back. Your correction is stored as data. Exploration stops being random and becomes "try, get rescued,
learn the rescue".

**3. A learned reward.** Someone has to say whether an attempt succeeded. Pressing a key works for the first runs;
after that a small image classifier, trained on success and failure frames, does it automatically.

Together these are **HIL-SERL** (Luo et al., 2024), which LeRobot implements.

## How it is wired

```
           ┌──────────── Learner (GPU) ────────────┐
           │  SAC updates the policy               │
           │  from the replay buffers              │
           └───────▲───────────────────┬──────────┘
        transitions│                   │ new policy weights, every few seconds
           ┌───────┴───────────────────▼──────────┐
           │  Actor (next to the arm)             │
           │  runs the policy, sends what happened │◄── you: take over, hand back
           └───────────────────────────────────────┘
                reward classifier: success? → reward 1, episode ends
```

Two processes that talk over gRPC. The actor never stops moving the arm while the learner trains; fresh weights
arrive every few seconds (4 s by default; the docs suggest 1 to 2).

## The workflow

Five steps, each with its own command.

| | Step | Command |
| --- | --- | --- |
| 1 | Find the workspace limits by moving the arm through the task space | `lerobot-find-joint-limits` |
| 2 | Record about 15 demonstrations with a success key | `python -m lerobot.rl.gym_manipulator --config_path rl/env_config.json` |
| 3 | Crop each camera to the region that matters | `python -m lerobot.rl.crop_dataset_roi` |
| 4 | Train the success classifier | `lerobot-train --config_path reward_classifier_train_config.json` |
| 5 | Train: start the learner, then the actor | `python -m lerobot.rl.learner`, `python -m lerobot.rl.actor` |

## Why it looks nothing like our recording setup

| | Our imitation setup | HIL-SERL as documented |
| --- | --- | --- |
| Action space | 6 joint angles | End-effector position; the docs say some tasks are "nearly impossible to learn in joint space" |
| Needs | Nothing extra | A URDF and inverse kinematics; a bounding box the gripper may not leave |
| Rate | 30 fps | 10 fps in every example config |
| Images | 640 × 480 into a VLM | Cropped, then 128 × 128 or 64 × 64 into a small ResNet |
| Language | 12 sentences | None: one policy, one task |
| Task length | Up to 30 s | "Tasks that can be completed in 5-10 seconds" |
| Compute | Rented, afterwards | An NVIDIA GPU running during the session |

So RL does not replace the VLA. It fits *one short, hard piece* of the task, the one imitation keeps getting wrong.

## What it would be in this cell

**The skill.** "Grasp the part in pocket 3 and lift it 5 cm." Short, visually simple, and the step where a
millimetre matters.

**The reward.** The top camera already computes where every pocket and bin is. "Pocket 3 is empty and nothing lies
on the tray" is a success detector that needs no training. That is an advantage this cell has over a bare desk.

**The interventions.** The SO-101 leader works for this, and the docs prefer it to the SO-100's: its geared joints
let it follow the robot during exploration, so taking over is smooth. Keys: space to take over and hand back, `s`
for success, `esc` for failure.

**The honest costs.** A CUDA GPU on the same network as the arm; an afternoon to set bounds, crops and the
classifier before any learning; several restarts, because the docs warn that learning "is very sensitive to the
intervention strategy".

How to intervene, from the docs: let the policy explore for a few episodes first; do not hold control for long;
later, limit yourself to quick fixes such as a grasp. The number to watch is the **intervention rate**. It should
fall steadily. If it does not, the policy is not learning from you.

## Reward models: a judge you do not have to train

A success classifier knows one task. Current LeRobot ships three general *reward models* that score a video against an
instruction.

| Model | How it works | Output | Trained on your data? |
| --- | --- | --- | --- |
| TOPReward | Asks a frozen 8 B VLM whether the instruction is complete and reads the probability of "True" | One score per clip | No, zero-shot |
| ROBOMETER | A 4 B VLM fine-tuned to rate trajectories | Progress from 0 to 1 and a success probability, per frame | No, zero-shot in LeRobot |
| SARM | A small model on CLIP features that learns the stages of *your* task | Stage and progress from 0 to 1 | Yes |

Three uses, cheapest first:

1. **Grade trials.** Run the model on the video of each rollout and compare with your own verdict. If they agree,
   you have an automatic judge for every later experiment.
2. **Grade your demonstrations.** A progress curve that stalls or runs backwards marks a hesitant episode. That is
   the "watch every episode" step of [review](review.md), done by a model.
3. **Weight the training data.** *Reward-aligned behaviour cloning* (RA-BC) trains the same policy on the same data
   but gives more weight to the frames where progress was being made:

```bash
lerobot-train --reward_model.type=sarm --reward_model.annotation_mode=single_stage ...   # 1. fit SARM
python -m lerobot.rewards.sarm.compute_rabc_weights --dataset-repo-id ... --reward-model-path ...   # 2. score every frame
lerobot-train --policy.path=... --sample_weighting.type=rabc --sample_weighting.kappa=0.01 ...   # 3. train
```

RA-BC is supported for SmolVLA, π0, π0.5 and FastWAM. The docs add a caveat that applies to us: if the dataset is
already clean, it will not help much. It pays off after DAgger rounds, when the data is a mix of good episodes,
recoveries and hesitations.

## Where RL sits among the other ways to improve

| Method | Human effort | Gets you |
| --- | --- | --- |
| More demonstrations | High, every time | More coverage, same ceiling |
| DAgger corrections | Medium, targeted | The states the policy actually reaches |
| RA-BC | None after setup | More from the data you have |
| HIL-SERL | High at first, falling | Past your own skill, on one short task |

Do them in that order. Each one is cheaper than the next and tells you whether the next is needed.

## What to take to the desk

- RL needs a reward. In this cell the top camera can provide one; build that first
  ([phase 6](roadmap.md#phase-6-better-eyes)), because every experiment benefits from an automatic judge.
- Keep the RL task under ten seconds.
- A zero-shot reward model costs an hour to try on recorded videos. Do that long before HIL-SERL.

**Sources:** [HIL-SERL](https://arxiv.org/abs/2410.21845) · LeRobot docs:
[Train a robot with RL](https://huggingface.co/docs/lerobot/hilserl), [SARM](https://huggingface.co/docs/lerobot/sarm),
[ROBOMETER](https://huggingface.co/docs/lerobot/robometer), [TOPReward](https://huggingface.co/docs/lerobot/topreward) ·
[SARM paper](https://arxiv.org/abs/2509.25358) · [TOPReward paper](https://arxiv.org/abs/2602.19313)
