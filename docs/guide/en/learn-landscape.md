# The field in 2026

When this project started, "a policy for a small arm" meant ACT or SmolVLA. LeRobot's documentation in November
2026 (release 0.7.0 and the main branch after it) covers more than twenty. This page sorts them into families, says what actually differs, and picks the ones worth running
on an SO-101.

Everything here is taken from the [LeRobot documentation](https://huggingface.co/docs/lerobot) of each policy.
Where a page gives no number, this page gives none either.

## Five families

Every policy answers the same question, *given what I see and what I was told, what do I do for the next second?*
They differ in what they predict on the way.

```
                     ┌─ 1. Small behaviour cloning ── actions only, no language model
                     ├─ 2. VLA, flow head ────────── a VLM reads; a head denoises an action chunk
observation ─► model ├─ 3. VLA, action tokens ────── the VLM writes actions as tokens, like text
+ instruction        ├─ 4. Reasoning VLA ─────────── writes a subtask (or boxes) first, then acts
                     └─ 5. World-action model ────── predicts the future video together with the actions
```

### 1. Small behaviour cloning

| Policy | Size | Note |
| --- | --- | --- |
| ACT | ~80 M per the docs | No language. Trained from scratch in a few hours on one GPU. Still the recommended first model |
| Multitask DiT | ~450 M | A diffusion transformer conditioned on CLIP. Wants batch 64+ and, per its docs, more than 300 examples |

Use these as the floor. If ACT on one sentence matches your VLA on that sentence, the VLA is not adding much yet.

### 2. VLA with a flow-matching head

The family this guide is built on: [inside SmolVLA](learn-vla.md), [flow matching](learn-flow.md).

| Policy | Size | What sets it apart |
| --- | --- | --- |
| SmolVLA | 450 M | Small enough for a consumer GPU; pretrained on community SO-100 data |
| π0 | 3 B | The design the others copy. 50-action chunks |
| π0.5 | not stated | Co-trained on mixed data for open-world generalisation; joint state written into the prompt as text |
| GR00T N1.7 | 3 B | NVIDIA. 16-action chunks, relative actions. The docs use the SO-101 as the "new embodiment" example |
| X-VLA | 0.9 B | A learned *soft prompt* per robot; adapting can mean tuning 1 % of the weights |
| EO-1 | 3 B | Pretrained on interleaved vision, text and action; can also answer questions |
| EVO1 | ~1 B VLM | Two training stages: the head first, then the VLM |
| WALL-OSS, Wall-OSS-0.5 | up to 4 B | Flow or token head on a Qwen VLM |
| LingBot-VLA 2.0 | 6 B per its repo name | A mixture-of-experts action head; depth and video distilled in during training |

### 3. VLA that writes actions as tokens

| Policy | Size | What sets it apart |
| --- | --- | --- |
| π0-FAST | PaliGemma, 2 B language model | Compresses a chunk with a DCT and byte-pair encoding, then predicts it like text. Trains up to 5× faster than π0 |
| NAT VLA | 450 M | Tokens ordered in time, so actions can be streamed while they are generated |

No denoising loop: one pass of the language model writes the chunk. The price is a tokenizer that must fit your
action statistics.

### 4. Reasoning VLAs

These write something human-readable before the action.

| Policy | It writes first | Needs |
| --- | --- | --- |
| G0.5 | The current subtask, optionally bounding boxes | Subtask (and box) annotations for the reasoning to train |
| FineART-VLA | Task → subtask → action, with optional memory | Subtask annotations; without them it behaves like π0.5 |
| MolmoAct2 | Depth reasoning in its "Think" variant | The Think variant is not in LeRobot yet |

Current LeRobot has the tooling they depend on: `lerobot-annotate` has a VLM watch your episodes and write subtasks
into new dataset columns, about 50 VLM calls per 30-second episode.

Why this matters for our cell: "put the part from pocket 3 in the REJECT bin" has natural subtasks (reach, grasp,
lift, move, release). A policy that says which one it is in is a policy you can debug.

### 5. World-action models

They predict future video and actions together: [world models as policies](learn-world-models.md).

| Policy | Size | What sets it apart |
| --- | --- | --- |
| Cosmos 3 Edge / Nano | 4.6 B / 16.5 B | NVIDIA. Video and actions denoised jointly; the predicted video can be saved |
| FLUX 3 Action | not stated | Black Forest Labs. Predicts video latents and actions, returns only actions. Ships an SO-101 checkpoint |
| FastWAM | 5 B backbone | Learns from video in training, predicts actions directly at inference |
| Long-WAM | not stated | NVIDIA. 2.4 s of history as context |
| LingBot-VA | ~5 B | Interleaves video latents and actions autoregressively |
| VLA-JEPA | 2 B VLM | Predicts the future in *latent* space, and only during training |
| LaWAM | not stated | Latent actions learned from video |
| Praxis-1 | 27 B | Runway. Needs an 80 GB GPU just to run |

## Four things that changed since SmolVLA

1. **The future as a training signal.** Half the new models predict video or its latents. Several (FastWAM,
   VLA-JEPA, FLUX 3 Action) throw that prediction away at inference: it was only there to teach the network how
   scenes move.
2. **Thinking out loud.** Subtasks and boxes as an intermediate output, with tooling to annotate your own data.
3. **LoRA as the default.** Models are too big to fine-tune in full on one GPU, so the docs lead with adapters.
4. **The policy moved off the robot.** LeRobot replaced the old async module with *remote inference*: a policy server on
   a GPU machine, the rollout on the computer next to the arm. Large models are unusable on a desk without it.

## The benchmark everyone quotes

| Policy | LIBERO average |
| --- | --- |
| Long-WAM | 99.0 % |
| LaWAM | 98.5 % |
| MolmoAct2 | 98.25 % |
| Cosmos 3 Edge | 97.95 % |
| π0.5 | 97.5 % |
| EVO1 | 96.65 % |
| GR00T N1.7 | 96.5 % |
| VLA-JEPA | 96.5 % |
| FastWAM | 96.4 % |
| Multitask DiT | 90.6 % |
| π0-FAST | 82.5 % |

Read this table for one thing only: the benchmark is saturated. Ten models sit within three points of each other,
on simulated tasks with hundreds of demonstrations each, evaluated by their own authors under slightly different
protocols. It cannot tell you which one will find pocket 3 on your desk. Your 24 trials can.

## What each one costs to train

From the LeRobot hardware guide (peak memory at batch 8):

| Group | Policies | GPU memory |
| --- | --- | --- |
| Light | ACT | 2 to 6 GB |
| Diffusion | Multitask DiT | 8 to 14 GB |
| Small VLA | SmolVLA | 10 to 16 GB |
| Large VLA | π0, π0-FAST, π0.5, X-VLA, WALL-OSS, GR00T, EO-1 | 24 to 40 GB; 24 GB is tight even at batch 1 |
| World-action | Cosmos 3 Edge | 21 GB with LoRA, 47 GB in full |
| | Praxis-1 | 62 GB with LoRA |

The guide's own rule of thumb: training converges in 5 to 10 passes over the data.

## Which have already seen this arm

A prior for your exact robot is worth more than parameters. These have one:

| Policy | Evidence in the docs | Fits our data as it is? |
| --- | --- | --- |
| SmolVLA | Pretrained on community SO-100 data | Yes: this guide |
| GR00T N1.7 | SO-101 is the documented `new_embodiment` example | Yes |
| G0.5 | Checkpoint `lerobot/g05_so101` | Yes for actions; the reasoning needs subtask annotations |
| MolmoAct2 | Checkpoint `lerobot/MolmoAct2-SO100_101-LeRobot` | Yes, with a camera rename |
| FLUX 3 Action | Checkpoint `black-forest-labs/flux-3-action-so101`, 30 Hz, six joints, scene + wrist | Yes, with a camera rename |
| NAT VLA | Rollout example on an SO-100; a 24 GB training recipe | Yes |

And two that do not fit without work: LingBot-VA and Praxis-1 expect end-effector poses, not joint angles.

## A shortlist for this project

| Order | Policy | The question it answers |
| --- | --- | --- |
| 1 | ACT, one sentence | Is the VLA doing anything a small model cannot? |
| 2 | GR00T N1.7 | Does a 3 B VLA with relative actions follow the pocket number better? |
| 3 | G0.5 with annotated subtasks | Does saying the subtask first help, and can I read where it fails? |
| 4 | FLUX 3 Action | Does video prediction help when the model already knows the arm? |
| 5 | Cosmos 3 Edge | The same, without the SO-101 prior but with a visible predicted video |

Run them through the same 24 paired trials ([the experiment plan](next.md)). One at a time.

## What to take to the desk

- A model family is a choice of *what else to predict*: nothing, text, or the future.
- Benchmarks rank models on someone else's task. Rank them yourself on yours.
- Start from a checkpoint that has seen an SO-100 or SO-101 whenever one exists.

**Sources:** LeRobot documentation pages for each policy, the
[compute hardware guide](https://huggingface.co/docs/lerobot/hardware_guide),
[remote inference](https://huggingface.co/docs/lerobot/remote_inference) and the
[annotation pipeline](https://huggingface.co/docs/lerobot/annotation_pipeline). Sizes and results are as stated
there in November 2026.
