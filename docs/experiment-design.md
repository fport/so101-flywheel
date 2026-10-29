# Experiment design: breaking memorization

With 60 parts and a hundred-odd demos, the model can memorize a shortcut that happens to sit in the data instead of learning the defect.
This page describes which shortcuts are possible, what closes each one, and how we will measure that it is closed.

## Three shortcuts

| Shortcut | What the model memorizes | What closes it |
| --- | --- | --- |
| Colour | "Orange part goes to the reject bin" | Colour is a variant, not a defect; the work order says which one is correct. Defects exist in both colours |
| Appearance to bin | The policy ignores the command and picks the bin by looking at the part | Every part goes to both bins in the demos (reversed-command attempts) |
| Part identity | Recognizes that particular part's print marks, not the defect | Held-out parts, defect location different on every part, shuffled pocket and order |

## Colour is a variant, not a defect

White and orange finish are both valid products. Whether a part is OK or REJECT depends on the work order, not on the part:

| Part | Work order white | Work order orange |
| --- | --- | --- |
| White, sound | OK | REJECT (mixed-in part) |
| Orange, sound | REJECT (mixed-in part) | OK |
| Any colour, missing hole or broken | REJECT | REJECT |

The set is factorial: every type, in every colour, has sound, missing-hole and broken parts (per type white
4 + 2 + 2, orange 3 + 2 + 2). Colour alone does not tell the bin; a structural defect does not tell the colour either.

That is why the part ID carries no label: `J-W-03`, not `J-03-OK`.

## The decision is in a rule, not in the model

The inspection classifier does not predict OK/REJECT; it predicts four attributes: type, variant, whether a hole is missing, whether it is broken.
The decision comes from a one-line rule that compares these attributes with the work order (`flywheel.sim.parts.decide`).

This has two benefits. When the work order changes, the model is not retrained. And when a part goes into the wrong bin, you can see which
attribute was misread: variant, hole or break.

## Reversed-command attempts

The policy is only told "put the part in the OK bin" or "put the part in the REJECT bin". If defective parts always go to the
reject bin in the demos, the policy can skip the command and learn the bin from the image; then when inspection is wrong
the policy "corrects" it and the source of the error cannot be told apart.

So in the recording plan every part is picked twice: once into the correct bin according to the work order, once deliberately into the opposite bin.
Across the whole plan, both commands occur equally often for every type, every colour and every defect class; no link remains
between appearance and command.

```bash
cd python
uv run flywheel-sim plan --order W            # plans/record-W-seed0.md: 20 trays, 120 episodes
uv run flywheel-sim plan --order W --heldout  # plans/eval-heldout-W-seed0.md: 6 trays, 32 episodes
uv run flywheel-sim batch --order W           # runs the same plan in sim with the scripted policy
```

The plan gives the operator the tray layouts and pick order to follow: which part goes in which pocket, in what order,
with which command. A part's pocket differs between the two passes; the layout and order are shuffled by a seed.

## What is measured

A single success rate hides the shortcut. Four numbers are kept separately:

| Metric | What it shows | How |
| --- | --- | --- |
| Attribute accuracy | Does the classifier see the defect | On held-out parts, type, variant, hole, break separately |
| Command-following rate | Does the policy listen to the command or to the image | In reversed-command attempts, is the part in the commanded bin |
| Manipulation success | Grasping and transport | Correct-command attempts, per type, Wilson CI |
| End-to-end success | The system | Inspection decision + policy, on the held-out plan |

The held-out eval plan is 16 parts × 2 commands = 32 attempts; it meets the PRD's "at least 30 attempts per policy" condition.

If the command-following rate is clearly lower than correct-command success, the policy has memorized from the image. If attribute accuracy is
high on training parts and low on held-out, the classifier has memorized the parts.

## Next experiment

Because labels are exact in sim, we can measure the shortcut itself: train the same classifier once on data from the old design
(orange is always REJECT), once on data from this design, and test both on "orange and sound, work order orange"
parts. The first should collapse, the second should hold. Not done yet; the inspection crops
(`QCSortSim.inspection_crop`) are ready for it.

## What it does not close

- The bin positions are fixed: OK is always on the left. The policy learns "OK = left"; this is the intended behaviour, but what happens
  when the bins swap places is a separate generalization test.
- Backdrop and lighting are fixed. Robustness to environment changes is not the subject of this design.
- The scripted policy in sim follows the command by definition; the command-following rate only becomes meaningful with a trained policy.
