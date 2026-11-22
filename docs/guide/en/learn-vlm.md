# Inside a vision-language model

A VLA is a VLM with a robot attached, so start here. A VLM has three parts and one trick.

```
image ──► vision encoder ──► projector ──► ┐
                                           ├──► language model ──► text
text  ──► tokenizer ─────► embeddings ───► ┘
```

## 1. The vision encoder turns pixels into tokens

A Vision Transformer cuts the image into square patches and turns each patch into one vector, a *visual token*.
The count is simple arithmetic:

| Image | Patch | Tokens |
| --- | --- | --- |
| 256 × 256 | 16 × 16 | 16 × 16 = 256 |
| 512 × 512 (SmolVLA's input) | 16 × 16 | 32 × 32 = 1 024 |

The encoder in the SmolVLM family is **SigLIP**, trained on image–caption pairs so that its vectors already mean
something to a language model.

## 2. The projector makes them look like words

A language model expects vectors of a fixed width (960 for SmolVLM2-500M). The projector, often a single linear
layer, reshapes each visual token to that width. After it, an image is just a run of "visual words" placed in the
same sequence as the text. This design is called *unified sequence* or early fusion: no special machinery, the
language model's ordinary self-attention mixes the two.

The older alternative (Flamingo) keeps the language model frozen and adds cross-attention layers that look at the
image. The book compares both and concludes that data and training settings matter more than which one you pick.

## 3. The language model does the reasoning

A normal transformer decoder. It reads the visual words and the text tokens and predicts the next token. Everything a
VLM "knows" about gears, bins and the word *left* lives here and in the encoder. A VLA inherits all of it for free.

## The trick: fewer tokens per image

Attention cost grows with the *square* of the sequence length. 1 024 tokens per camera, two cameras, every control
step, is too slow for a robot. **Pixel shuffle** folds neighbouring tokens into one wider token:

```
32 × 32 tokens × 1 width   →   16 × 16 × 4   →   8 × 8 × 16
       1 024                      256               64 tokens
```

Pixel shuffle itself throws nothing away, it rearranges: space becomes width. The projector then squeezes each wide
token down to the language model's 960 numbers, and that is where fine detail is lost. SmolVLA uses a factor of 4 per side, so each camera
frame costs **64 tokens**. Our two cameras are 128 tokens, less than a paragraph of text.

The same trade-off drives *tiling* in document models (cut a large image into 256-pixel tiles: 15 tiles are already
3 840 tokens). A robot does the opposite: low resolution, high frequency.

## What "frozen" and "fine-tuned" mean here

| Part | Parameters (SmolVLA) | In our fine-tune |
| --- | --- | --- |
| SigLIP vision encoder | 86 M | frozen |
| Language model (first 16 of 32 layers), embeddings, projector | ~260 M | frozen |
| Action expert + small projections | ~100 M | **trained** |

Frozen weights do not change, so the model cannot forget what a gear looks like while it learns your arm. The book's
line on the lighter alternative, LoRA (small trainable matrices added to frozen layers): it "learns less and forgets
less". You would use LoRA to adapt a VLM as a judge, or for the larger Cosmos 3 policy.

## Asking a VLM *where*, not *what*

The inspection step needs positions. Models answer that in different formats:

| Model family | Output |
| --- | --- |
| OWLv2, GroundingDINO | Boxes in pixels with a confidence score. The threshold is yours to tune: in the book's example 0.3 gives 3 boxes, 0.01 gives 78 |
| PaliGemma | Location tokens in the text, `<loc0324><loc0242>…`, coordinates normalised to 0–1023 |
| LocateAnything | Boxes or points on a 0–1000 scale |
| General VLMs | Coordinates written as text or JSON: validate them with a schema before use |

`flywheel-inspect` avoids the question with AprilTags: the pocket positions are known in millimetres, so the VLM only
has to say *what* is in a crop. Replacing the tags with a grounding model is experiment 3 in
[what to try next](next.md).

**Sources:** *Vision Language Models* (Noyan, Farré, Marafioti): tokens and pixel shuffle p. 131–134, architectures
p. 229–249, localisation p. 87–94, LoRA p. 191–199 · LeRobot 0.6.1 source, `policies/smolvla/smolvlm_with_expert.py`
