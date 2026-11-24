# World models as policies: Cosmos 3 Edge

SmolVLA looks at the scene and writes actions. A *world action model* looks at the scene and writes two things at
once: the actions, and the video of what the cameras will see while those actions run. This page explains why that
is worth 10× the parameters, what is inside [nvidia/Cosmos3-Edge](https://huggingface.co/nvidia/Cosmos3-Edge), and
exactly how our dataset would go through it.

Nothing on this page has been run on our arm yet. Numbers are from the model cards and the
[LeRobot Cosmos 3 docs](https://huggingface.co/docs/lerobot/cosmos3); where a choice is ours, it says so.

## Why predict video at all

A demonstration frame gives a policy six numbers to learn from: the joint targets. The same frame, as video, is
thousands of numbers, and they carry the physics: the part does not move until the jaws close, the arm occludes the
tray, a dropped part falls. A model that must predict the next frames is forced to represent all of that, and it can
learn it from ordinary video with no robot in it.

So the bet is: pretrain on video at scale, then attach actions as one more thing to predict. Cosmos 3 was trained on
about 1.3 billion samples, of which 348 M are video and only 7 M contain actions.

## What is inside

```
"put the part from pocket 3 in the REJECT bin"            top | wrist  (one canvas, current frame)
        │                                                        │
        ▼                                                        ▼ Wan2.2 VAE
┌─────────────────────┐      shared        ┌───────────────────────────────────────┐
│ understanding path  │◄──── attention ───►│ generation path (trained, flow matching)│
│ text tokens, causal │                    │ video latents + action tokens, noisy   │
└─────────────────────┘                    └───────────────────────────────────────┘
                                                 │                    │
                                                 ▼                    ▼
                                        future frames (latents)   next N actions
```

- **A Mixture-of-Transformers.** Every layer has two sets of weights, one per path, that meet in one shared
  attention. Edge has 28 layers at width 2048: 4.6 B parameters in total, 1.4 B trained by default.
- **One canvas.** Camera views are tiled into a single video: two views side by side, three as one on top and two
  below. The model was never told "this is camera 2"; a sentence in the prompt describes the layout.
- **Joint denoising.** The current frame is kept clean. The future video latents and the action chunk start as noise
  and are denoised together with [flow matching](learn-flow.md), using the UniPC solver.
- **The prompt is longer than the task.** It carries the instruction, the frame rate, the canvas size, the layout
  sentence and the number of idle frames in the chunk.

## SmolVLA and Cosmos 3 Edge, side by side

| | SmolVLA | Cosmos 3 Edge |
| --- | --- | --- |
| Parameters (trained) | 450 M (~100 M) | 4.6 B (1.4 B; less with LoRA) |
| Predicts | 50 actions | Actions and the future video |
| Pretraining | 481 community robot datasets, mostly SO-100 | Web-scale video and images; robot data from Franka, UR, WidowX, UMI, AgiBot and others |
| Has seen an SO-100/101 | Yes | Not listed on the model card |
| Fine-tuning memory | A consumer GPU | 21 GB with LoRA, 47 GB in full (both with gradient checkpointing) |
| One decision | Fast enough to run on the Mac next to the arm | 1.25 s on an H100 in NVIDIA's DROID benchmark |
| Runs on | macOS, Linux | Linux; bfloat16 only |
| LeRobot | 0.6.1 | 0.7, extra `cosmos3` |
| Licence | Apache 2.0 | OpenMDW 1.1 (commercial use allowed) |

The row that matters most for us is the fourth. SmolVLA starts with a prior for this exact arm. Cosmos 3 starts with
a prior for *how the world moves* and action modules that are untrained for our robot: the docs tell you to start
them from scratch with `--policy.load_action_modules=false`.

## Our dataset through it

Our recording already fits the requirements: LeRobot format v3.0, two cameras, 6 joint targets, one sentence per frame.

| Requirement | Ours | Setting |
| --- | --- | --- |
| 1 to 3 cameras, in layout order | `top`, `wrist` | Side-by-side canvas |
| A sentence describing the canvas | Must be written by hand; a wrong one is not detected | `--policy.view_description` |
| Real control rate | 30 fps | Read from the dataset |
| Action width ≤ 64 | 6 | Read from the dataset |
| `q01` / `q99` statistics | Written by `lerobot-record` | Default `QUANTILES` normalisation |
| Embodiment row | Any; all rows start untrained | `--policy.domain_id=0` |

A first run, LoRA on one GPU (flags from the LeRobot docs; chunk size and the layout sentence are our choices):

```bash
pip install 'lerobot[cosmos3,training,peft]'      # Linux; install LeRobot from source if your release has no cosmos3 extra

lerobot-train \
  --policy.type=cosmos3_edge \
  --policy.pretrained_path=lerobot/Cosmos3-Edge \
  --policy.load_action_modules=false \
  --policy.gradient_checkpointing=true \
  --policy.domain_id=0 \
  --policy.chunk_size=48 --policy.n_action_steps=48 \
  --policy.view_description='The left half shows a fixed camera looking down at the tray and the two bins; the right half shows the wrist-mounted camera.' \
  --peft.method_type=lora --peft.r=16 --peft.lora_alpha=32 \
  --policy.optimizer_lr=5e-4 --policy.optimizer_weight_decay=0 \
  --accelerator.mixed_precision=bf16 --tf32=false \
  --ema.type=power --ema.sigma_rel=0.1 \
  --batch_size=8 --steps=16000 \
  --dataset.repo_id=fport/so101-qc \
  --policy.repo_id=fport/so101-qc-cosmos3-edge
```

Evaluate the EMA weights (`checkpoints/last/pretrained_model_ema`) with `--policy.dtype=bfloat16`.

## The three things most likely to go wrong

**1. Time.** NVIDIA measures 1.25 s per decision on an H100 for the DROID policy (32 actions at 15 fps, 4 solver
steps). At our 30 fps a 32-action chunk is 1.07 s of motion: the arm would finish the chunk before the next one
exists and stop at every boundary. Two levers: a longer chunk (48 actions = 1.6 s, the choice above) and few solver
steps (the LIBERO default is 30, the DROID recipe uses 4). Expect to tune both.

**2. Where it runs.** 15 GB of GPU memory in bfloat16 and Linux only, so not on the Mac next to the arm. The policy
has to run on a GPU machine and send chunks back. Current LeRobot ships this split as *remote inference*:
`lerobot-policy-server` on the GPU machine, `lerobot-rollout --inference.type=remote` next to the arm. The docs say
only ACT, SmolVLA, X-VLA, EVO1 and LaWAM have run on real robots through it; other policies are supported by the
protocol but not validated on hardware. Network delay then adds to point 1.

**3. Data scale.** The public recipes fine-tune on 379 to 1 693 episodes with an effective batch of 2 048. We have
about 120 episodes and would use a batch of 8. Nobody has published what happens at that scale. The LIBERO result
itself (Edge 97.95 %, Nano 97.90 % average) says the small model loses nothing there, which is why Edge and not Nano
is the one to try.

## The predicted video is a debugging tool

`--policy.save_predicted_video=true` decodes what the model expects to see for every chunk. No other policy in this
guide can show you its plan. Use it:

- Predicted video shows the arm going to the wrong pocket: the model misread the sentence. More language variety or
  more data per sentence.
- Predicted video is right and the arm does something else: the action modules are the weak part. Train longer, or
  check normalisation.
- Predicted video is blurred or the part vanishes: the scene is outside what the model can imagine. Check the canvas
  size and the layout sentence first.

## What to take to the desk

- Run this only after SmolVLA has a score on the fixed test. Otherwise there is nothing to compare with.
- The honest question is narrow: *with the same 120 episodes, does a world model follow the pocket number better?*
- Budget: one 48 GB or 80 GB GPU for training, and a GPU machine reachable from the arm for rollout.

**Sources:** [LeRobot Cosmos 3 docs](https://huggingface.co/docs/lerobot/cosmos3) ·
[nvidia/Cosmos3-Edge](https://huggingface.co/nvidia/Cosmos3-Edge) and
[lerobot/Cosmos3-Edge](https://huggingface.co/lerobot/Cosmos3-Edge) model cards ·
[Cosmos 3 technical report](https://arxiv.org/abs/2606.02800)
