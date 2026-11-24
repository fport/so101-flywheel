# Grounding: asking a model where

A classifier answers *what*. A robot needs *where*: which pocket, which pixel, which part has the broken corner.
Turning a phrase into coordinates is called *grounding*. This page explains how
[nvidia/LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B) does it, and where it could replace
hand-written code in this cell.

Nothing here has been run on our frames yet. Model facts are from the model card.

## How inspection finds things today

`flywheel-inspect` uses no learning to locate anything:

1. Four AprilTags give the tray's position in the image (a homography).
2. Each of the six pockets is cut out and straightened at 4 pixels per millimetre.
3. Holes, spokes, teeth and corners are measured from the silhouette.

It is exact and fast, and it knows only what was programmed: four part types, two defects, tags that must stay
visible. Show it a fifth part and it has nothing to say.

## What a grounding model is

A detector such as YOLO has a fixed list of classes, decided at training time. A grounding model takes the class as
*text at run time*: "gear", "the orange part", "a part with a broken corner".

LocateAnything is a vision-language model built for this:

| Part | What |
| --- | --- |
| Vision encoder | MoonViT |
| Language model | Qwen2.5-3B-Instruct |
| Connector | An MLP projector, as in [inside a VLM](learn-vlm.md) |
| Size | 3 B parameters |
| Training data | 12 M images, about 140 M queries, 785 M boxes; natural scenes, robotics, driving, GUIs and documents |

Its one new idea is how it writes coordinates. An ordinary VLM spells a box out one token after another:
`x1`, then `y1`, then `x2`, then `y2`. LocateAnything predicts all four in one parallel step (*parallel box
decoding*), which the card credits with up to 2.5× the throughput of earlier approaches.

## Talking to it

The prompts are fixed sentences. Coordinates come back as integers from 0 to 1000, relative to the image.

| You want | Prompt |
| --- | --- |
| Every instance of some classes | `Locate all the instances that matches the following description: gear</c>rim.` |
| One thing | `Locate a single instance that matches the following description: the orange gear.` |
| Everything matching a phrase | `Locate all the instances that match the following description: a part with a broken corner.` |
| A point instead of a box | `Point to: the gear in the top-left pocket.` |

```python
worker = LocateAnythingWorker("nvidia/LocateAnything-3B")       # helper class from the model card
result = worker.detect(image, ["gear", "rim", "brake pad", "bracket"])
boxes = LocateAnythingWorker.parse_boxes(result["answer"], width, height)   # 0..1000 → pixels
```

Two details from the card that change results:

- The example code **samples** (temperature 0.7). The same image can give different boxes twice. For a measurement,
  turn sampling off.
- There are three modes, `fast`, `slow` and `hybrid` (the default). `fast` is described as good for simple scenes,
  which a tray on a black background is.

## From a box to a pocket

A box in pixels is not yet a decision. The tray homography we already compute turns any pixel into millimetres on
the tray, and from there into a pocket number:

```
"Point to: the gear with a missing tooth."  →  (412, 630) in 0..1000      (illustrative numbers)
        → pixel (264, 302) in the 640 × 480 frame
        → tray homography → (31 mm, −42 mm)
        → pocket 2
        → "put the part from pocket 2 in the REJECT bin"   → the policy
```

This is the pattern to remember: the grounding model never drives the arm. It produces a *pocket number*, and the
sentence carries that to the policy.

## Three experiments, cheapest first

All three run on saved frames, with no robot. `outputs/inspect/*/frame.png` plus the known part in each pocket is
the answer key.

| | Question | Score with |
| --- | --- | --- |
| **G1. Find the parts** | Does it return exactly one box per occupied pocket, with the right type? | Per-pocket accuracy over all saved frames |
| **G2. Find the defects** | Does "a part with a missing hole" select the right pockets and no others? | Precision and recall against the part IDs |
| **G3. Follow a description** | Does "the orange rim" land in the right pocket? | Accuracy over a written list of 30 phrases |

Compare each with `flywheel-inspect` on the same frames. The measured pipeline is the baseline to beat, and it is a
strong one: it reads the bin correctly for all six pockets on the reference tray.

## What to expect, honestly

- **G1 should work.** Parts on a matte black tray are an easy scene.
- **G2 is the hard one.** At the recording resolution the top camera sees about 1.3 pixels per millimetre, so a 7 mm
  hole is 9 pixels across. A model that resizes the frame may not see it. Feed it the straightened 4 px/mm pocket
  crops, or capture at 1280 × 960, before concluding it cannot do it.
- **"Missing" is a hard word.** Grounding models locate things that are present. A hole that is not there is an
  absence. Asking for "a rim with five holes" and treating the others as defective may work better.
- **It can answer when it should not.** The card does not say how the model behaves when nothing matches. Include
  frames with no defective part and count false boxes.

## Limits that are not about accuracy

| | |
| --- | --- |
| Licence | NVIDIA non-commercial: research and evaluation only. Fine for this guide, not for a product |
| Hardware | Linux and an NVIDIA GPU, bfloat16. The only published memory figure is 11.7 GB for a batch of four 4K images on an A100 |
| Public demo | [The Space](https://huggingface.co/spaces/nvidia/LocateAnything) is the fastest first look. Treat anything you upload as public |

## Where it would sit in the cell

```
top camera frame
   ├─ AprilTags → homography ──────────────┐
   ├─ measurement (today) ─► defect type ──┤
   └─ grounding (experiment) ─► box, point ┤
                                           ▼
                              decide(): OK or REJECT per pocket
                                           ▼
                              one sentence per pocket → policy
```

Grounding is a second opinion next to the measurement, not a replacement for it. Where the two disagree is the
interesting data: those are the frames to look at by hand.

## What to take to the desk

- Grounding turns language into coordinates. The homography turns coordinates into a pocket.
- Test on saved frames first. It costs nothing and needs no arm.
- Fix the phrases and the frames before the first run, exactly as with the [policy test](learn-eval.md).

**Sources:** [nvidia/LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B) model card ·
[Eagle](https://github.com/NVlabs/Eagle) · *Vision Language Models* (Noyan, Farré, Marafioti), localisation
p. 87–94
