# 8 · Run and improve

**Goal:** a number for how good the policy is, and a way to make it better with exactly the data it lacks.

## Run it

```bash
scripts/rollout.sh 3 ok       # the policy has 30 s to do: "put the part from pocket 3 in the OK bin"
```

It loads the policy named in `scripts/env.sh` (`POLICY`, the "Model repo" that training printed).

Keep one hand near the follower's power supply. The first run is often jerky or wrong; that is information.
On a Mac the model runs on the Apple GPU (`DEVICE=mps`) and is slow, so the script uses *real-time chunking* (`--inference.type=rtc`):
the arm keeps executing the last predicted chunk while the next one is computed.

## Score it

Do not judge by a few lucky runs. Fix the test before you look:

| | |
| --- | --- |
| Trials | 2 per sentence = 24, with parts you held back from recording |
| Scene | Tray full for the first, half empty for the second |
| Success | Part is in the commanded bin and the arm is back, within 30 s, without your help |
| Write down | pocket, bin, success, and the failure type: wrong pocket · missed grasp · dropped · wrong bin · stuck |

Report successes out of 24, and per pocket and per bin. 24 trials only separate "works" from "does not": 18/24 and
20/24 are the same result. The failure types are what you act on:

| Failure | Usually means |
| --- | --- |
| Goes to the wrong pocket | Sentence not used: check balance per sentence, record more for the confused pair |
| Right pocket, misses the grasp | Inconsistent grasps in the data, or the wrist camera cannot see the part |
| Freezes mid-way | A state it never saw: more variety in tray fill, or dead time in the demos |
| Works at noon, fails at night | Lighting: record a few blocks in the failing light |

## The whole loop

```bash
cd python && uv run flywheel-inspect     # prints one sentence per pocket
scripts/rollout.sh 5 ok                  # run them one by one
```

## Improve: the flywheel

Record more of *exactly what failed*, retrain, score again with the same test.

- **More demonstrations** for the failing sentences: `scripts/record.sh 4 reject 10`.
- **Corrections (DAgger):** let the policy run, take over with the leader the moment it goes wrong, and record your
  correction. That gives the policy the recovery data it cannot get from clean demos:
  `lerobot-rollout --strategy.type=dagger ...` ([LeRobot docs](https://huggingface.co/docs/lerobot/inference)).

Each turn of record → train → score → correct is one revolution. Keep a table of the score per revolution; that curve
is the project.
