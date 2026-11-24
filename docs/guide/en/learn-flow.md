# Flow matching from scratch

SmolVLA, π0, GR00T and Cosmos 3 all write actions the same way: they start from random noise and move it, step by
step, until it is a motion. This page builds that idea from nothing, runs it on a problem small enough to hold in
your head, and then maps every symbol onto the code you train in [step 7](train.md).

## The problem it solves

One situation, two correct actions: a gear you may grasp from the left (−1) or from the right (+1).

A regression head is trained to minimise the squared error to whatever the human did. The value that minimises a
squared error is the *mean*. The mean of −1 and +1 is 0: the middle of the gear, a grasp nobody showed.

What we want instead is a model that outputs −1 sometimes and +1 other times, and never the middle. In other words:
a model that *samples* from the distribution of demonstrated actions instead of summarising it with one number.

## The idea: learn a direction, not an answer

Take a real action `A` and a blob of random noise `ε`. Draw a straight line between them:

```
x_t = t·ε + (1 − t)·A          t = 0 is the action, t = 1 is pure noise
```

Along that line the direction of travel is constant: `u = ε − A`. Train a network `v(x_t, t)` to output that
direction for random triples `(A, ε, t)`:

```
loss = mean( (v(x_t, t) − (ε − A))² )
```

To generate, start at noise and walk the line backwards in small steps:

```
x = ε
for t in 1.0, 0.9, … 0.1:
    x = x − 0.1 · v(x, t)
```

That is the whole method. It is still a regression, so why does it not average the two grasps?

## Why the average disappears

The network cannot know which `A` a noisy point `x_t` came from, so it learns the *average direction over all the
actions that could have produced this point*. What "could have" means changes along the path:

| Where | What the network sees | What it outputs |
| --- | --- | --- |
| t ≈ 1 (pure noise) | Either grasp is equally likely | A direction toward the middle, but only a small step is taken |
| t ≈ 0.5 | A point left of centre almost certainly came from −1 | A direction toward −1 |
| t ≈ 0 | The point is next to one grasp | A fine correction onto it |

The average over directions is taken *separately at every point*, and points on the left are dominated by the left
grasp. A sample that starts slightly left of centre is pushed further left at each step and commits. The random
starting point `ε` decides which grasp you get; the network decides that it is a real one.

## Run it

`scripts/toy_flow.py` is this page in 50 lines: one situation, demonstrations at −1 and +1, a regression head and a
flow-matching head with the same tiny network.

```bash
uv run --extra robot python scripts/toy_flow.py
```

```
regression predicts      -0.01   (the average: closes on nothing)
flow matching, 1000 runs 51% near -1, 49% near +1
  mean |x| = 0.98, samples between -0.5 and 0.5: 2.1%
```

The 2 % that end up in the middle are the cost of ten coarse steps and a small network: they started almost exactly
on the fence. More steps or a better network shrinks that number; it does not reach zero. On a robot this is the
occasional hesitant grasp, and it is one more reason to demonstrate *one* way per part.

## From the toy to SmolVLA

Nothing changes except the sizes.

| Toy | SmolVLA (`policies/smolvla/modeling_smolvla.py`) |
| --- | --- |
| `A`: one number | 50 future actions × 6 joints, normalised, zero-padded to 50 × 32 |
| `v(x_t, t)` | The action expert. It also sees the scene: 177 cached tokens from the two images, the sentence and the joint state |
| `t` uniform in (0, 1) | `t` drawn from a Beta(1.5, 1) distribution, scaled into (0.001, 1]: more training at the noisy end |
| 10 Euler steps | 10 Euler steps (`num_steps`) |
| The situation never changes | The scene tokens are computed once per decision and reused in all ten steps |

Why train more at the noisy end? For images, the hard part is fine detail, which lives near t = 0. For actions it is
the other way round: once the chunk is mostly denoised, the scene pins down the rest. The decision that matters
(which grasp, which bin) is taken early, in the noise.

## How this differs from diffusion

Diffusion policies (and image generators before them) learn the same kind of denoiser, with two practical differences.

| | Diffusion (DDPM) | Flow matching |
| --- | --- | --- |
| Path from data to noise | A curved schedule of many small noising steps | A straight line |
| Network predicts | The noise that was added | The velocity along the line |
| Steps at inference | Typically 50 to 100 (10 or so with a DDIM sampler) | About 10 with plain Euler steps |

Straight paths are why ten steps are enough, and ten steps are why a VLA can produce a chunk in a fraction of a second.

## Three knobs you will meet

- **Steps.** Fewer steps is faster and coarser. SmolVLA uses 10. Cosmos 3 uses a better solver (UniPC) with 30 steps
  in its LIBERO recipe and 4 in its DROID recipe.
- **Guidance.** Run the network twice, with and without the sentence, and push the result away from the unconditioned
  one. It makes the model follow the instruction harder at twice the cost. SmolVLA does not use it; the Cosmos 3
  DROID recipe does (scale 3).
- **The starting noise.** Same observation, different `ε`, different chunk. That is why two rollouts from an
  identical start can differ, and why [real-time chunking](learn-vla.md) has to steer a new chunk toward the old one.

## What to take to the desk

- A low flow-matching loss means the *direction* is predicted well on your data. It still says nothing about
  success on the arm; see [measuring a policy](learn-eval.md).
- If two demonstrations disagree, the model will reproduce both, and occasionally something in between.
- Sampling is random. Judge a policy over many trials, never over one.

**Sources:** [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) ·
[Rectified Flow](https://arxiv.org/abs/2209.03003) · [π0](https://arxiv.org/abs/2410.24164) (the Beta time
sampling) · [Diffusion Policy](https://arxiv.org/abs/2303.04137) · LeRobot 0.6.1 source ·
[LeRobot Cosmos 3 docs](https://huggingface.co/docs/lerobot/cosmos3)
