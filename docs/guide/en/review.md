# 6 · Review and upload

**Goal:** know what is in the dataset before you spend GPU hours on it.

```bash
scripts/status.sh              # episodes per sentence
scripts/review.sh 12           # replay episode 12: both cameras and all joints, in Rerun
scripts/review.sh delete 12 40 # remove bad episodes; the previous version is kept as so101-qc_old
scripts/review.sh push         # upload after deleting: training reads the Hub copy, not your disk
```

Online, without installing anything: open
`https://huggingface.co/spaces/lerobot/visualize_dataset` and enter `fport/so101-qc` (you must be logged in for a
private dataset).

## What to look for

Watch at least two episodes of every block, and every episode you were unsure about.

- Both videos play, are not black, and show the same moment (the gripper closes in both at once).
- The `action` curve (leader) and the `observation.state` curve (follower) overlap with a small lag. A flat line is a
  servo that was not read.
- The episode starts moving within about half a second and ends at the start pose.
- No hand, no leader arm in frame.

## What is in the folder

```
outputs/datasets/so101-qc/
  meta/info.json        fps, features (6 joints in, 6 out, two cameras), episode count
  meta/tasks.parquet    the 12 sentences
  meta/episodes/        which frames and which sentence belong to which episode
  meta/stats.json       mean/std per feature: used to normalise during training
  data/                 joint angles and actions, 30 rows per second (parquet)
  videos/               one mp4 stream per camera
```

One frame = `observation.images.top`, `observation.images.wrist`, `observation.state` (where the follower is),
`action` (where the leader told it to go), and the task sentence. Training is: given the first three and the sentence,
predict the next 50 actions.
