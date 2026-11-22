# How imitation learning works

The path pages tell you what to do. This page is why. It explains the three ideas the whole project stands on:
behaviour cloning, compounding error, and action chunks.

## Behaviour cloning is supervised learning

Your dataset is a long table. Each row is one thirtieth of a second:

```
observation o_t                                   action a_t
  top image, wrist image, 6 joint angles, sentence  →  6 target joint angles (where the leader was)
```

A policy is a function `π(o) → a`. Training adjusts its weights so that `π(o_t)` is close to `a_t` on every row.
There is no reward and no notion of "the part reached the bin". The model is only ever asked: *in this situation,
what did the human do next?* That is why a low training loss does not mean a working robot, and why what you
demonstrate, including your hesitations, is exactly what you get.

## Why small mistakes snowball

In ordinary supervised learning a 2 % error stays 2 %. On a robot it does not, because the policy's own output
decides what it sees next:

```
demonstrations:   s0 → s1 → s2 → s3 → …            (states the human visited)
policy:           s0 → s1'→ s2''→ s3'''→ …         (each small error moves it further off the data)
```

Once the arm is 1 cm off the demonstrated path, it is in a state that is not in the dataset, where its prediction
is worse, which moves it further off. This is *covariate shift*. Ross and Bagnell (2010) showed the cost: with an
error rate ε per step over T steps, plain behaviour cloning can lose up to **T² · ε**, not T · ε.

Three things fight this, and all three are in this project:

| Remedy | How | Where |
| --- | --- | --- |
| Cover more states | Vary part, rotation and tray fill, so "slightly off" is still inside the data | [Data quality](data-quality.md) |
| Fewer decisions | Predict 50 actions at once: a 300-step episode becomes 6 to 15 decisions | Action chunks, below |
| Show the way back | Let the policy fail, take over, record the correction | DAgger, below |

## One situation, several right answers

You can grasp a gear from the left or from the right. Both are correct. A network trained with a plain squared-error
loss on both will predict the *average*: the middle of the gear, a grasp nobody demonstrated.

```
demo A: approach from the left   ←●
demo B: approach from the right     ●→
MSE average:                        ●      (closes on nothing)
```

Two defences:

1. **You** remove the ambiguity: one grasp per part type, one path, every time. This is rule 1 of data quality and it is
   the cheapest fix there is.
2. **The model** is built to cope with several answers. ACT trains with a variational autoencoder that absorbs the
   differences between demos, but at run time it outputs one deterministic chunk. SmolVLA and π0 use *flow matching*,
   which turns random noise into one coherent action sequence and really does land on one mode or the other, not
   between them. See [inside SmolVLA](learn-vla.md).

## Action chunks

Instead of one action per camera frame, modern policies predict a *chunk*: the next 50 actions in one go.

- **Fewer chances to drift.** The ACT paper measured it in simulation (two tasks, scripted and human data): 1 % success predicting one step at a time,
  44 % with chunks of 100.
- **Pauses stop being a trap.** If a demo pauses for half a second, a one-step policy sees identical observations
  with "stay still" as the label and can freeze forever. Inside a chunk the pause has a beginning and an end.
- **It is affordable.** A VLA needs about ten network passes to produce one chunk. Fifty actions per call means
  the heavy model runs under once a second while the arm still moves at 30 Hz.

The price: while a chunk plays, the arm is not looking. At 30 fps, SmolVLA's 50 actions are **1.67 s of open-loop
motion**. If the part slips in that time the arm finishes the old plan first. Real-time chunking (in
[inside SmolVLA](learn-vla.md)) is the fix.

## DAgger: learning from your own mistakes

Clean demonstrations only show the good path. DAgger (Ross, Gordon, Bagnell 2011) collects the states the *policy*
actually reaches:

1. Run the policy.
2. When it goes wrong, a human supplies the right action from that state.
3. Add those pairs to the dataset and retrain. Repeat.

This brings the error bound back from T² · ε to roughly T · ε. LeRobot implements a human-gated version:

```bash
lerobot-rollout --strategy.type=dagger --policy.path=... --teleop.type=so101_leader ...
```

| Key | Does |
| --- | --- |
| Space | Pause / resume the policy |
| Tab | Start / stop your correction with the leader arm |
| Enter | Upload |
| Esc | Stop |

By default only your correction windows are saved, each as an episode. That is the "correct" spoke of the flywheel
in [step 8](rollout.md).

## What to take to the desk

- The policy can only be as consistent as you were.
- Variety in the data is what lets it survive its own small errors.
- A failure on the robot is a state missing from the dataset. Go and record that state.

**Sources:** Ross & Bagnell, *Efficient Reductions for Imitation Learning* (2010) ·
[DAgger](https://arxiv.org/abs/1011.0686) · [ACT](https://arxiv.org/abs/2304.13705) ·
*Vision Language Models* (Noyan, Farré, Marafioti), ch. 11, p. 500–505 ·
[LeRobot human-in-the-loop docs](https://huggingface.co/docs/lerobot/hil_data_collection)
