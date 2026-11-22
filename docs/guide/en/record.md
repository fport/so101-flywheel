# 4 · Record

**Goal:** 120 demonstrations: you move the leader, the follower does the task, LeRobot stores both camera streams,
the joint angles and your commands 30 times a second, with the task sentence attached.

## The plan: 12 blocks of 10

One block = one sentence, e.g. *"put the part from pocket 3 in the OK bin"*. Six pockets × two bins = 12 sentences,
10 episodes each. About 8 minutes per block, under two hours in total.

```bash
scripts/record.sh 1 ok          # 10 episodes, pocket 1 -> OK bin
scripts/status.sh               # what is done, and the next command to run
scripts/record.sh 1 reject      # ...
```

Every block is appended to the same dataset in `outputs/datasets/so101-qc` and pushed to your Hugging Face account
(private) when the block ends, so a crash costs at most one block.

## One block, step by step

1. Put a part in the block's pocket. Arm at the start pose (gripper open, hovering between the base and the tray).
2. Run the command. A voice says **"Recording episode N"**: recording has started.
3. Pick the part, drop it in the block's bin, return to the start pose.
4. Press **→**. The episode is saved; the voice says **"Reset the environment"**.
5. Nothing is recorded now. Put the next part in the pocket, hand back on the leader, arm at the start pose.
6. Press **→** to start the next episode (or wait 15 s, it starts by itself).

| Key | Does |
| --- | --- |
| → | End this phase now (episode done / reset done) |
| ← | Throw this episode away and record it again |
| Esc | Stop after this episode, save and upload |

Fumbled the grasp, bumped the tray, forgot to return? Press **←**. Never keep an episode "because it mostly worked".

## Inside a block, change everything except the sentence

- **The part:** a different one every episode. Mix types, colours, sound and broken ones. A broken part goes to the OK
  bin when the block says OK: the arm obeys the sentence, the inspection decides what the sentence is.
- **Its angle:** turn it differently each time.
- **The other pockets:** episodes 1–4 full tray, 5–7 two or three neighbours, 8–10 the part alone. At run time the tray
  empties as the arm works, and the policy must have seen all of that.

Why this matters, and the rest of the rules: [data quality](data-quality.md).

## If something goes wrong

| Symptom | Fix |
| --- | --- |
| `TimeoutError` on a camera | Wrong index (an iPhone nearby shifts them): `uv run lerobot-find-cameras opencv`, fix `scripts/env.sh` |
| "no status packet" from a servo | Follower's 12 V supply, or a loose servo cable |
| Arrow keys do nothing | Give the terminal Accessibility permission in macOS settings, or keep the terminal focused |
| Dataset refuses to resume | The last run died mid-write. Re-download the last pushed copy: `hf download fport/so101-qc --repo-type dataset --local-dir outputs/datasets/so101-qc` |
