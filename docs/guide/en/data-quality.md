# 5 · Data quality

Imitation learning copies what is in the data: your hesitations, your shortcuts, and every accident of the scene.
A policy that fails is nearly always showing you something about the dataset. These rules are ordered by how often
breaking them ruins a model.

## 1. Be consistent in *how*, varied in *what*

| Keep the same | Vary on purpose |
| --- | --- |
| Start pose and end pose of the arm | Which part is in the pocket |
| How you grasp each part type (rim by the edge, pad across the short side…) | The part's rotation in the pocket |
| The path: up, across, down, release, back | Which other pockets are filled |
| Speed: smooth, about 10 s per episode | Light (record a few blocks at another time of day) |
| Camera positions, bin positions, tray position | How full the bins are |

One way of doing the task, shown many times, is learnable from 10 episodes. Two ways shown 5 times each is not.

## 2. The sentence must be the only clue to the target

The policy takes any shortcut the data allows. If white parts always went right, it learns "white → right" and ignores
the words. So in every block, put sound *and* defective parts, white *and* orange ones, into the commanded bin. The
only thing that predicts the pocket and the bin is then the sentence.

For the same reason, record every pocket with both bins, and the same number of episodes per sentence.
`scripts/status.sh` shows the balance.

## 3. No dead time, no hands

- Start moving right after recording starts; press → as soon as you are back at the start pose. Frames where nothing
  happens teach the policy to wait.
- Your hands, the leader arm and you must be out of both camera views while recording.
- Nothing else changes in view: no phone, no cup, no cable that is there in half the episodes.

## 4. Only keep clean successes (at first)

A clean episode: one grasp, no bump against the tray, part lands in the bin, arm returns. Anything else: **←** and
redo. Recovery behaviour (re-grasping after a slip) is valuable, but add it later, deliberately, from the failures the
trained policy actually makes ([step 8](rollout.md)).

## 5. You must be able to do it from the cameras

The policy sees two 640 × 480 images, nothing else. If the part is hidden behind the gripper in both views at the
moment of grasping, there is nothing to learn from. Check in [setup](setup.md), before the first episode.

## 6. Do not touch the scene between record and run

Same camera angle, same bin positions, same calibration files. If a camera was bumped, re-aim it using a frame from
the dataset as reference before recording more.

## How much

| Episodes per sentence | Expect |
| --- | --- |
| 5 | Moves toward the right area, often misses the grasp |
| 10 (this plan, 120 total) | A working baseline; the floor LeRobot recommends per location |
| 20–30 | Clearly more reliable; add where the evaluation shows failures, not everywhere |

More data only helps where the policy is weak. Train on 120, evaluate, then record more for the sentences that fail.

## Before each block

- [ ] Arm at the start pose, gripper open
- [ ] Hands and leader out of both camera views
- [ ] The right pocket has a part; you know which bin
- [ ] Bins not overflowing
- [ ] Nothing new on the desk
