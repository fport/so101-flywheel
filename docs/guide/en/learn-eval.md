# Measuring a policy

A robot demo that works once proves almost nothing. This page is the small amount of statistics that keeps you from
fooling yourself.

## Loss is not success

Training loss measures how well the model reproduces recorded motions on recorded states. Success depends on states
the policy creates itself ([compounding error](learn-imitation.md)). Two checkpoints with the same loss can score
30 % and 80 % on the arm. There is no shortcut: you count outcomes on the real robot.

## How sure is "18 out of 24"?

Every trial is a coin flip with an unknown bias. The Wilson interval gives the range of true success rates that fit
what you saw (95 %):

```
centre = (p + z²/2n) / (1 + z²/n)
half   = z · sqrt( p(1−p)/n + z²/4n² ) / (1 + z²/n)        with p = successes/n, z = 1.96
```

| Result | 95 % interval |
| --- | --- |
| 18 / 24 (75 %) | 55 % – 88 % |
| 20 / 24 (83 %) | 64 % – 93 % |
| 6 / 24 (25 %) | 12 % – 45 % |
| 45 / 60 (75 %) | 63 % – 84 % |

18 and 20 out of 24 overlap almost completely: that is not an improvement you can claim. 6 versus 18 is. To separate
75 % from 85 % you need a few hundred trials, so do not try: use 24 trials to tell *broken* from *working*, and use
failure types to decide what to do next.

## Fix the test before you run it

Decide all of this first and keep it identical between policy versions:

- which parts (hold some back from recording: a policy that only handles parts it has seen has memorised them),
- which pocket and bin for each trial (2 per sentence, written down),
- how full the tray is,
- the time limit (30 s) and what counts as success (part in the commanded bin, arm back, no help).

If you choose the trials after seeing the policy move, you will unconsciously pick the ones it can do.

## Two questions, two numbers

The task has two separable skills. Score them separately:

| Question | Number | A low value means |
| --- | --- | --- |
| Did it go to the commanded pocket and bin? | *followed the command* | The sentence is not being used: a data balance problem |
| Did it complete the pick and drop? | *success* | A manipulation problem: grasp consistency, camera view |

A policy that always empties pocket 1 into the OK bin perfectly has high manipulation skill and scores 1 in 12 on
command following. Knowing which of the two failed tells you which block to record again.

## Log failures by type

```
trial  pocket  bin     result   failure
1      3       ok      ✓
2      3       reject  ✗        wrong bin
3      5       ok      ✗        missed grasp
```

After 24 trials the failure column is your recording plan. This is the same idea the book describes for large
systems: judge the *end state* ("is the part in the bin?"), not each step, and feed the outcome back into training.
π*0.6 does it at scale by labelling every attempt as better or worse than expected and conditioning the policy on that
label.

## Automating the judge

The top camera already knows how to read the tray and could read the bins. A check like "after the episode, pocket 3
is empty and the REJECT bin has one more part" is a verifier you can run on every attempt, which is what turns
record → train → evaluate into a loop that runs without you watching. That is the flywheel this repo is named after,
and it is not built yet.

**Sources:** Wilson (1927) score interval · [SmolVLA paper](https://arxiv.org/abs/2506.01844) (evaluation protocol) ·
*Vision Language Models*, ch. 11 p. 474–475 and 509–512
