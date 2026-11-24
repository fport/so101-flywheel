# The route: ten phases

The eight steps on the [front page](index.md) get one policy onto one arm. This page is the whole project: where
those steps sit, what comes after them, and which current work from the field you get to try in each phase.

Every phase has an **exit test**. Do not start the next phase until the current one passes: each one produces the
number or the tool the next one needs.

**Status, so nothing here is oversold.** Phases 0 to 3 are built and documented in this repo; the simulator, the
inspection and the scripts have been run. Phases 4 to 9 are a plan assembled from the current LeRobot documentation
and the model cards. None of those has been run on this arm yet.

## The map

| | Phase | You end up with | Time | Cost |
| --- | --- | --- | --- | --- |
| 0 | [Simulate](#phase-0-simulate) | The task solved in MuJoCo, before any hardware | An evening | Free |
| 1 | [Build and see](#phase-1-build-and-see) | A printed cell, a calibrated arm, a tray the top camera can read | A week of printing, a day of work | Filament |
| 2 | [Collect](#phase-2-collect) | 120 clean episodes on the Hub | An afternoon | Free |
| 3 | [Baseline](#phase-3-baseline) | A SmolVLA checkpoint and a score out of 24 | A day | ~$10 of GPU |
| 4 | [Squeeze the baseline](#phase-4-squeeze-the-baseline) | Knowing whether data or the model is the limit | Two days | ~$20 |
| 5 | [Other policies](#phase-5-other-policies) | The same test on two or three current models | A week | $20 to $100 |
| 6 | [Better eyes](#phase-6-better-eyes) | Grounding in the inspection, an automatic judge | Three days, no arm | A few GPU hours |
| 7 | [World models](#phase-7-world-models) | A policy that also predicts video | A week | A 48 GB+ GPU |
| 8 | [The simulator as a twin](#phase-8-the-simulator-as-a-twin) | Policies evaluated in sim, sim data in training | Two weeks | An H100-class GPU for repainting |
| 9 | [Learning from its own attempts](#phase-9-learning-from-its-own-attempts) | RL on one sub-skill, weighted imitation | Two weeks | A CUDA GPU next to the arm |

The closing phase is not a model: [the full loop](#the-full-loop), a tray sorted end to end with nobody typing sentences.

## Phase 0: simulate

**Goal.** Understand the task and the layout before printing anything.

- **Do.** [Simulate first](simulate.md): build the scene, watch the scripted expert, run the 120-episode plan.
- **Learn.** What an episode is, what "success" means for this task, why the tray and bins are where they are.
- **Exit test.** `flywheel-sim batch` prints 60/60 for both rows, and you can say out loud what the arm has to do for
  "put the part from pocket 3 in the REJECT bin".

## Phase 1: build and see

**Goal.** A cell that is identical every day, and a camera that reads the tray.

- **Do.** [The cell](cell.md), [arm and cameras](setup.md), [inspect the tray](inspect.md).
- **Learn.** Calibration, camera placement, how a homography turns pixels into millimetres.
- **Exit test.** Teleoperation follows the leader with no stutter, both cameras show what the guide says they should,
  and `flywheel-inspect` names the right bin for all six pockets of a tray you laid out yourself.

## Phase 2: collect

**Goal.** A dataset you would trust if someone else had recorded it.

- **Do.** [Record](record.md) the 12 blocks, follow [data quality](data-quality.md), [review and upload](review.md).
- **Learn.** [How imitation learning works](learn-imitation.md), [inside the dataset](learn-dataset.md).
- **Exit test.** `scripts/status.sh` shows 12 blocks of 10, you have watched at least two episodes of every block,
  and the dataset loads from the Hub on another machine.

## Phase 3: baseline

**Goal.** One number, measured properly, that everything later is compared with.

- **Do.** [Train](train.md) SmolVLA, [run and score](rollout.md) the fixed 24 trials.
- **Learn.** [Inside a VLM](learn-vlm.md), [inside SmolVLA](learn-vla.md),
  [flow matching from scratch](learn-flow.md), [measuring a policy](learn-eval.md).
- **Exit test.** A score with its interval, and a table of failure types. A low score is a valid exit: it is the
  baseline.

## Phase 4: squeeze the baseline

**Goal.** Find out what limits the score, at almost no cost.

- **Do.** Experiments 1 to 3 of [the experiment plan](next.md): inference settings, the learning curve, one DAgger round.
- **Learn.** How chunk execution, data amount and corrections each move the score on *your* arm.
- **Exit test.** You can finish this sentence with evidence: "the main limit is ___ (too little data / the wrong
  data / the model / the control loop)".

## Phase 5: other policies

**Goal.** See how much of the result is SmolVLA and how much is the data.

Current LeRobot documents more than twenty policies. The ones worth your time first are those that have already seen this
arm, because they start with a prior for it.

| Policy | Why this one | Starting checkpoint |
| --- | --- | --- |
| ACT | The floor: the smallest policy here, no language. Train it on a single sentence | From scratch |
| GR00T N1.7 | NVIDIA's 3 B VLA; the docs use the SO-101 as their example | `nvidia/GR00T-N1.7-3B` |
| G0.5 | Writes a subtask before it acts; has an SO-101 checkpoint | `lerobot/g05_so101` |
| MolmoAct2 | LoRA by default, advice for under 200 demonstrations | `lerobot/MolmoAct2-SO100_101-LeRobot` |
| NAT VLA | 450 M, autoregressive tokens, a documented 24 GB recipe | `ahad-j/nat-vla-450m` |

- **Do.** Set up a current LeRobot environment on the GPU machine (this repo stays on 0.6.1 for recording), learn *remote inference* (the model on a GPU, the
  arm on your desk), then train one or two of the table on the same dataset.
- **Learn.** [The field in 2026](learn-landscape.md): the families, and what differs between them.
- **Exit test.** A paired comparison on the same 24 trials, with the count of trials where exactly one policy succeeded.

## Phase 6: better eyes

**Goal.** Make perception less hand-written, and stop judging trials by eye.

- **Do.** The three grounding experiments on saved frames ([grounding](learn-grounding.md)). Then an automatic
  judge: the top camera already knows where the bins are, so "is the part in the commanded bin" is a measurement.
  Compare it with a zero-shot reward model (TOPReward, ROBOMETER) on recorded trial videos.
- **Learn.** Where a general model stops seeing your defects; how reward models score progress.
- **Exit test.** A table of agreement between your own verdicts and each automatic judge on at least 48 trials.

## Phase 7: world models

**Goal.** Run a policy that predicts the future as video, and look at what it imagines.

- **Do.** Start with FLUX 3 Action, which ships an SO-101 checkpoint and wants exactly our data shape (30 Hz, six
  commanded joints, a scene and a wrist camera). Then Cosmos 3 Edge with LoRA
  ([world models as policies](learn-world-models.md)).
- **Learn.** Joint video and action denoising, what the predicted video tells you about a failure.
- **Exit test.** The same 24 trials, plus three failures explained by looking at the predicted video.

## Phase 8: the simulator as a twin

**Goal.** Make the simulator part of the training and evaluation loop, not a demo.

- **Do.**
  1. Package the MuJoCo cell as an EnvHub environment: one `env.py` with a `make_env()` function. LeRobot can then
     evaluate any checkpoint in it (`lerobot-eval --env.type=envhub`) and drive it as a robot
     (`--robot.type=sim`), including with the real leader arm.
  2. Measure the gap: the same policy, scored in sim and on the real arm.
  3. Close the appearance gap with repainted episodes ([sim to real](learn-sim2real.md)).
- **Learn.** What transfers between sim and real, and what does not.
- **Exit test.** A policy trained on real + generated data, scored on the real 24 trials against the phase 3 baseline.

## Phase 9: learning from its own attempts

**Goal.** Go past the quality of your own demonstrations on one small skill.

- **Do.** HIL-SERL on a single short sub-skill, such as "grasp the part in pocket 3 and lift it", with the top camera
  as the success detector. Separately, score your dataset with a reward model and retrain with weighted imitation
  (RA-BC). Both are in [reinforcement learning on a real arm](learn-rl.md).
- **Learn.** Rewards, off-policy learning, why interventions matter, why RL wants a short task.
- **Exit test.** The intervention rate falls over a session, and the learned skill is at least as reliable as the
  imitation policy on that sub-skill.

## The full loop

Everything above is a part. The project is finished when the parts run in sequence with no human in between:

```
tray on the desk
   → top camera frame → inspection (measurement, grounding as second opinion)
   → decide(): OK or REJECT for each pocket
   → six sentences, one per pocket
   → policy executes each one
   → automatic judge: is the part in the right bin?
   → failures are saved → corrections → next training round
```

That last arrow is the flywheel the repo is named after: the cell produces its own next dataset.

## If you only have a month

Phases 0 to 4 are the course. They cover the full path from nothing to a measured, improved policy, and they cost
about thirty dollars. Then pick **one** of 5 to 9 by what you want to be good at:

| You care about | Go to |
| --- | --- |
| Models and training | 5, then 7 |
| Perception | 6 |
| Simulation | 8 |
| Control and RL | 9 |
