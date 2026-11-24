# After the baseline: the experiment plan

You have a dataset, a SmolVLA checkpoint and a score. This page is what to do next, in the order that teaches the
most for the least money. Each experiment changes **one thing** and is judged on the **same fixed test**
([measuring a policy](learn-eval.md)).

## The cell is four layers

Every model in this guide belongs to one of them. Knowing which keeps the experiments separate.

```
1. SEE        top camera frame ─► where is each part, what is wrong with it
                 today: AprilTags + measurement          try: LocateAnything
2. DECIDE     work order + what was seen ─► OK or REJECT per pocket
                 today: decide() in flywheel.qc          (rules; nothing to learn)
3. SAY        one sentence per pocket: "put the part from pocket 3 in the REJECT bin"
4. DO         sentence + two cameras + joints ─► 30 actions per second
                 today: SmolVLA                          try: Cosmos 3 Edge
   ─────────────────────────────────────────────────────────────────────────────
   DATA       what layer 4 learns from
                 today: teleoperated episodes            try: DAgger corrections, Sim-Transfer, RL
```

Why not one model for everything? Because of resolution. The policy sees each camera as 64 tokens; a 7 mm hole is
far below that. Inspection reads a straightened crop at 4 pixels per millimetre. Keeping "what is wrong" out of the
policy is what lets a small VLA do the job: it only has to find a pocket and a bin.

## The ladder

| | Experiment | Changes | Cost | Needs |
| --- | --- | --- | --- | --- |
| 0 | SmolVLA baseline | | ~4 h of an A100 | Done first |
| 1 | Inference settings | Chunk execution only | Free | The baseline checkpoint |
| 2 | Learning curve | Amount of data | 2 more trainings | The dataset |
| 3 | DAgger round | 20 to 30 correction episodes | One afternoon + 1 training | The arm |
| 4 | Grounding | Layer 1 | A GPU for an hour, no arm | Saved frames |
| 5 | Cosmos 3 Edge | Layer 4 | A 48 GB+ GPU, and a GPU machine for rollout | LeRobot 0.7 |
| 6 | Sim-Transfer data | The data | An H100-class GPU, a week of building | The simulator |
| 7 | Reinforcement learning | How layer 4 improves | Days on the arm | A reward signal |

Do them top to bottom. The first three cost almost nothing and tell you whether the expensive ones are worth it.

This ladder is phases 4 to 9 of [the route](roadmap.md). Between experiments 3 and 5 there is one more option: the
same test on another current policy that already knows the SO-101. Which ones, and why: [the field in 2026](learn-landscape.md).

### 1. Inference settings (free)

Same checkpoint, same test, three runs:

| Run | Setting | Looks for |
| --- | --- | --- |
| a | `--inference.type=sync` | Pauses at chunk boundaries |
| b | `--inference.type=rtc` (what `scripts/rollout.sh` uses) | Smooth motion, late reactions |
| c | `rtc`, fewer actions executed per chunk | Faster reaction to a slipped part, more compute |

This is the cheapest lesson in how a chunked policy behaves. Background: [inside SmolVLA](learn-vla.md).

### 2. Learning curve

Train the same model on half the data, then compare with the full one:

```bash
# the first 5 episodes of each 10-episode block: 5 per sentence instead of 10
HALF=$(python3 -c "print(str([b*10+i for b in range(12) for i in range(5)]).replace(' ',''))")
scripts/train.sh --dataset.episodes="$HALF" --job_name=so101-qc-smolvla-half
```

The list assumes 12 blocks of exactly 10 episodes in recording order. If you deleted episodes, build it from
`scripts/status.sh` instead.

| Result | Meaning | Next |
| --- | --- | --- |
| 60 episodes scores about the same as 120 | Data amount is not the limit | Fix what fails: DAgger, or a different model |
| 120 clearly better than 60 | Still on the rising part of the curve | Record more before anything else |

This one result decides whether experiments 3 and 6 (more data) or 5 (a bigger model) come first.

### 3. One DAgger round

Run the policy, take over when it fails, save only the corrections, retrain on old + new, retest. Twenty good
corrections aimed at the most common failure type usually move the score more than sixty new clean episodes. How:
[run and improve](rollout.md); why: [how imitation learning works](learn-imitation.md).

### 4. Grounding in the inspection

Replace or double-check the hand-written measurements with a grounding model, on saved frames, with no robot.
Start here if you care more about perception than about control. Full page: [grounding](learn-grounding.md).

### 5. A world model as the policy

Fine-tune Cosmos 3 Edge with LoRA on the same episodes and run the same test. The question is narrow: does a 10×
larger model that also predicts video follow the pocket number better? Full page:
[world models as policies](learn-world-models.md).

### 6. Simulated episodes, repainted

Generate episodes in the MuJoCo cell, repaint them with Cosmos 3 Sim-Transfer, mix with real data. Worth doing only
if experiment 2 said more data helps. Full page: [sim to real](learn-sim2real.md).

### 7. Reinforcement learning

Imitation stops at the quality of your demonstrations. RL lets the arm improve past them from its own attempts and a
success signal. LeRobot ships HIL-SERL (human-in-the-loop RL on the real arm). The top camera can give the reward:
"is the part in the commanded bin?". Start this only when imitation has plateaued and you can leave the arm running.

## Comparing two policies with 24 trials

Twenty-four trials can tell *broken* from *working*. They cannot tell 70 % from 80 %. So:

- **Pair the trials.** Both policies get the same 24 trials: same part, pocket, bin, tray fill. Count the trials
  where exactly one of them succeeded. If A wins 7 of those and B wins 1, that is a real difference. If it is 4
  against 3, it is not.
- **Compare failure types, not only totals.** "Wrong pocket dropped from 6 to 1" is a finding even when the total
  barely moves.
- **Change one thing.** A new model *and* new data *and* new settings is three experiments with one answer.

## What each one costs to get wrong

| | If it fails you lose | You still learn |
| --- | --- | --- |
| 1, 2, 3 | An afternoon | How chunking, data amount and corrections behave on your own arm |
| 4 | An hour of GPU | Where a general model stops seeing your defects |
| 5 | A day and tens of dollars | How a world model is fine-tuned and served |
| 6 | A week | A simulator that writes LeRobot datasets, which RL needs anyway |

## A newer LeRobot

This repo pins LeRobot 0.6.1: the arm is calibrated and tested on it. Cosmos 3 needs 0.7 or later. The dataset
format (v3.0) is the same, so keep recording on 0.6.1 and train the Cosmos experiment in a separate environment.
