# Inside the dataset

The dataset is the program. This page opens the folder LeRobot writes and follows one training sample out of it.

## On disk (format v3.0)

```
outputs/datasets/so101-qc/
  meta/info.json                         fps, feature shapes, episode and frame counts
  meta/stats.json                        mean, std, min, max, quantiles per feature
  meta/tasks.parquet                     the 12 sentences, numbered
  meta/episodes/chunk-000/file-000.parquet   one row per episode: its frames, its sentence, where its video starts
  data/chunk-000/file-000.parquet        one row per frame: state, action, timestamp, indices
  videos/observation.images.top/chunk-000/file-000.mp4
  videos/observation.images.wrist/chunk-000/file-000.mp4
```

Many episodes share one video file. An episode row stores the second at which it starts in that file; a frame is
found at *start + its own timestamp*. Video is AV1: a 30 s episode from two cameras is a few megabytes, not hundreds.

## One row

| Column | Shape | What |
| --- | --- | --- |
| `observation.state` | 6 | Follower joint positions, in degrees (gripper in %) |
| `action` | 6 | Leader joint positions at the same instant: the command sent to the follower |
| `observation.images.top`, `.wrist` | 480 × 640 × 3 | Decoded from the mp4 on demand |
| `timestamp`, `frame_index`, `episode_index`, `task_index` | scalars | Bookkeeping |

`action` is not "what the follower did next". It is where you held the leader. The follower lags by a few frames,
and that lag is what the policy learns to command through.

## One training sample

```
pick a random frame i (any episode)
observation  = images and state at frame i, plus that episode's sentence
target       = actions at frames i, i+1, … i+49
```

Near the end of an episode there are fewer than 50 actions left. The last one is repeated and those steps are marked
`action_is_pad`, so they do not count in the loss. This is why **ending every episode at the start pose** matters: the
last real actions teach the policy how a task finishes.

## Normalisation: why `stats.json` matters

Joint 1 moves over 180°, the gripper over 100 %. Fed raw, the large numbers would dominate the loss. So state and
action are normalised per dimension:

```
x_normalised = (x − mean) / std
```

The policy predicts normalised actions and they are converted back before being sent to the motors. Two consequences:

- A joint that barely moved in your data has a tiny std, and small noise on it is amplified. Demonstrate with the
  whole arm.
- **Recalibrating after recording silently breaks the policy.** The same pose now reads as different numbers, so
  every normalised value is shifted.

## Timing is part of the data

A frame is only useful if the image and the joint reading belong to the same instant. The book's example: a frame
from t = 1.50 s paired with joints from t = 1.58 s teaches the wrong mapping, and the error repeats in every episode.

What can go wrong here, and how to see it:

| Problem | Cause | Check |
| --- | --- | --- |
| Dropped or late frames | CPU busy (a live viewer, a browser, video encoding) | Record with no viewer and nothing heavy running; replay an episode and look for stutter |
| One camera lags | USB hub bandwidth, auto-exposure in low light | In `scripts/review.sh N`, the gripper must close in both views in the same frame |
| A joint that never moves | A servo that was not read | The `observation.state` curve is flat |
| Follower does not follow | Low supply voltage, overload | `action` and `observation.state` curves drift apart |

## Tasks are just strings

`--dataset.single_task` is stored once in `tasks.parquet` and referenced by index from every frame. Recording another
block with a different sentence adds a row to that table. If you mistype a sentence, fix it afterwards:

```bash
uv run lerobot-edit-dataset --repo_id=you/so101-qc --root=outputs/datasets/so101-qc \
  --new_root=outputs/datasets/so101-qc --operation.type=modify_tasks \
  --operation.episode_tasks='{"12": "put the part from pocket 3 in the OK bin"}'
```

The model reads the sentence through a tokenizer, so `pocket 3` and `pocket three` are different inputs. Twelve
sentences that differ in exactly one or two tokens are a clean experiment: whatever the policy does differently, it did
because of those tokens.

**Sources:** LeRobot 0.6.1 source (`datasets/`, `processor/normalize_processor.py`) ·
[LeRobot dataset v3 docs](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) · *Vision Language Models*, ch. 4
p. 147–160 and ch. 11 p. 506–507
