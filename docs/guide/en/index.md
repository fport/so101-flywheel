# SO-101 zero to hero

From an empty desk to a robot arm that sorts parts on a spoken-style command, using a policy you trained on data you
recorded. Eight steps. Each one is a few plain `lerobot-*` commands; the page behind it explains what the command
does and what to watch for.

**The task.** A tray holds six 3D-printed parts. A camera above looks once and decides, per pocket, OK or REJECT.
The arm is then told, one part at a time: *"put the part from pocket 3 in the REJECT bin"*.

| | Step | You do | You learn |
| --- | --- | --- | --- |
| 1 | [Build the cell](cell.md) | Print, assemble, mount two cameras | Why the scene must not move |
| 2 | [Set up the arm](setup.md) | Install, calibrate, teleoperate | Leader/follower, joint space |
| 3 | [Inspect the tray](inspect.md) | `flywheel-inspect` | Classic vision vs a VLM, grounding |
| 4 | [Record](record.md) | `scripts/record.sh 1 ok` × 12 blocks | What a demonstration dataset is |
| 5 | [Data quality](data-quality.md) | Follow the checklist while recording | Why policies fail: it is the data |
| 6 | [Review and upload](review.md) | Watch, delete bad episodes, push | The LeRobot dataset format |
| 7 | [Train](train.md) | `scripts/train.sh` (SmolVLA, rented GPU) | VLA, action chunks, flow matching |
| 8 | [Run and improve](rollout.md) | `scripts/rollout.sh 3 ok`, score, correct | Evaluation, DAgger, the data flywheel |

## Understand what you are doing

Each step works without these. Read them to know *why* it works, and what to change when it does not.

| Read | Before step | It answers |
| --- | --- | --- |
| [How imitation learning works](learn-imitation.md) | 4 | Why small errors snowball, why demos must be consistent, what a chunk is |
| [Inside the dataset](learn-dataset.md) | 6 | What exactly is stored, how one training sample is built, why timing matters |
| [Inside a vision-language model](learn-vlm.md) | 7 | How an image becomes 64 tokens, what is frozen, how models say *where* |
| [Inside SmolVLA](learn-vla.md) | 7 | The two transformers, flow matching in six lines, real-time chunking |
| [Measuring a policy](learn-eval.md) | 8 | What 18 out of 24 really tells you, and what to log |

Then: [what to try next](next.md) (Cosmos 3, sim-to-real transfer, grounding models, RL) and
[the concepts in one page](concepts.md).

## What you need

- SO-101 leader + follower arm, two USB webcams (one above, one on the gripper)
- A 3D printer for the cell, or any tray and two boxes you fix to the desk
- A Mac or Linux laptop for recording; a rented GPU for training (about 4 h of an A100, roughly $10)
- A free [Hugging Face](https://huggingface.co) account: the dataset and the policy live there

## Today, in order

```bash
scripts/teleop.sh --cams      # 1. both cameras see what they should
uv run flywheel-inspect       # 2. the top camera reads the tray
scripts/record.sh 1 ok        # 3. first block of 10 episodes
scripts/status.sh             # 4. what is left; repeat until 120
scripts/train.sh              # 5. SmolVLA on a rented GPU
scripts/rollout.sh 1 ok       # 6. the policy moves the arm
```
