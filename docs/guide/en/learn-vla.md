# Inside SmolVLA

SmolVLA is 450 M parameters: a small VLM that reads the scene, and a 100 M *action expert* that writes motion.
This page follows one training step and one control step through it, with the real numbers from this project.

```
top 640×480 ─┐                                   ┌──────────────────────────┐
wrist 640×480├─► SigLIP + pixel shuffle ─► 128 ─►│ SmolVLM2, first 16 layers │──► keys/values (cached)
"put the part from pocket 3…" ─► tokens ─► 48 ──►│        (frozen)           │          │
6 joint angles ─► pad to 32 ─► linear ─► 1 ─────►└──────────────────────────┘          ▼
                                                 ┌──────────────────────────┐   cross-attention
noise (50 × 32) ─► + time embedding ─► 50 ──────►│  action expert, 16 layers │◄─────────┘
                                                 │        (trained)          │──► 50 actions × 6 joints
                                                 └──────────────────────────┘
```

## What goes in

| Input | Processing | Tokens |
| --- | --- | --- |
| Each camera frame | Resized to fit 512 × 512 (ours become 512 × 384, padded), scaled to [−1, 1], SigLIP, pixel shuffle | 64 |
| Sentence | A newline is appended, tokenised, padded to 48 | 48 |
| Joint angles | Normalised with the dataset mean and std, zero-padded from 6 to 32, one linear layer | 1 |

So the VLM reads **177 tokens** per decision. The pretrained model calls its cameras `camera1..3`; `--rename_map`
renames `top` and `wrist` onto the first two, and the third is simply left out.

The padding to 32 is why one checkpoint works on arms with different joint counts.

## Two transformers, side by side

- **The VLM** is SmolVLM2-500M with its upper half removed: only the first 16 of 32 language layers are used.
  Middle-layer features turn out to be enough for control, and half the layers is half the time.
- **The action expert** has 16 layers too, at 0.75 × the width. Its layers alternate: even layers attend over
  everything jointly, odd layers *cross-attend*, meaning the action tokens look up the VLM's cached keys and values.

During one decision the VLM runs **once**. Its output is cached and the expert reuses it for all ten denoising steps.
That cache is what makes a VLA fast enough.

## Flow matching: how noise becomes a motion

This section is the short version. [Flow matching from scratch](learn-flow.md) derives it and runs it on a toy problem.

The expert does not output actions directly. It learns a *velocity field*: a rule that says, for any half-noisy
action chunk, which direction leads to a clean one.

**Training**, for one sample (a random frame, plus the 50 actions that followed it):

```
A  = the 50 real actions                    (50 × 6, normalised; zero-padded to 50 × 32)
ε  = random noise of the same shape
t  = a random number in (0, 1], biased toward 1 (the noisy end)

x_t = t·ε + (1 − t)·A          # mix: t = 1 is pure noise, t = 0 is the real chunk
u   = ε − A                    # the direction from data to noise
loss = mean( (expert(x_t, t, scene) − u)² )
```

The network sees a blurred version of what you did and must say which way the blur points. Steps past the end of an
episode are masked out of the loss.

**Inference** runs the rule backwards from pure noise, in ten Euler steps:

```
x = ε                           # start: noise
for t in 1.0, 0.9, … 0.1:
    x = x − 0.1 · expert(x, t, scene)
return x                        # 50 actions
```

Why not just regress the actions? Because of the "several right answers" problem in
[how imitation learning works](learn-imitation.md): regression averages two valid grasps into an invalid one. Starting
from a random ε, flow matching ends at *one* of the demonstrated behaviours, a different one for a different ε.

## From 50 actions to a moving arm

| | |
| --- | --- |
| Chunk | 50 actions = 1.67 s at 30 fps |
| Default (`--inference.type=sync`) | Play all 50, then stop and think. The arm pauses at every chunk boundary when the model is slow |
| Real-time chunking (`rtc`, used by `scripts/rollout.sh`) | A background thread starts the next chunk when 30 actions are left |

RTC has one subtlety. The new chunk is computed from an observation that is already old by the time it arrives.
So while the new chunk is being denoised, its first actions are steered toward what is left of the old chunk: fully
for the steps that will already have run by the time it arrives, fading to nothing by step 10. Those stale steps are
then dropped. The result is continuous motion and a new plan every 20 actions (0.67 s), or less often if the model
is slower than that.

## What fine-tuning changes

`lerobot-train --policy.path=lerobot/smolvla_base` keeps the vision encoder and the VLM frozen and trains about
100 M parameters: the action expert, the projections in and out of it, and the one layer that embeds the joint state.

| Setting | Value | Meaning |
| --- | --- | --- |
| Optimiser | AdamW, lr 1e-4 | Warm-up, then cosine decay to 2.5e-6 |
| Batch × steps | 64 × 20 000 | 1.28 M samples: each of our ~36 000 frames is seen about 35 times |
| Normalisation | mean/std from `meta/stats.json` | Your dataset's statistics replace the pretrained ones |
| Augmentation | off | Lighting robustness has to come from your data |

The model was pretrained on 481 community datasets (22.9 K episodes, mostly SO-100 arms), with cameras mapped to
top / wrist / side. That is why your top and wrist views are a good match, and why 10 episodes per sentence can be
enough.

## How it compares

| Policy | Size | Reads language | Action head | Needs |
| --- | --- | --- | --- | --- |
| ACT | ~52 M in LeRobot (80 M in the paper) | no (in LeRobot) | CVAE transformer, chunks of 100, L1 loss | Any GPU, even a laptop |
| **SmolVLA** | 450 M | yes | Flow matching, ~100 M expert | One consumer GPU |
| π0 / π0.5 | 3–4 B | yes | Flow matching, ~300 M expert | 24–40 GB |
| GR00T N1.7 | 3 B | yes | Flow-matching diffusion transformer under a slower VLM | Large GPU, Jetson to run |
| [Cosmos 3 Edge](learn-world-models.md) | 4.6 B | yes | Actions denoised together with the future video | 21 GB with LoRA, LeRobot 0.7 |

On real SO-100 tasks with 50 demonstrations each, the SmolVLA paper reports 78 % average success, against 62 % for π0
and 48 % for ACT, and 52 % for SmolVLA without its pretraining. Our task has 12 sentences that differ in one
word, which is harder than three separate tasks: expect the pocket number to be the first thing that fails.

**Sources:** LeRobot 0.6.1 source (`policies/smolvla/modeling_smolvla.py`, `smolvlm_with_expert.py`,
`policies/rtc/`) · [SmolVLA paper](https://arxiv.org/abs/2506.01844) ·
[Flow matching](https://arxiv.org/abs/2210.02747) · [π0](https://arxiv.org/abs/2410.24164) ·
[Real-time chunking](https://arxiv.org/abs/2506.07339) · *Vision Language Models*, ch. 11, p. 504–514
