# 7 · Train

**Goal:** fine-tune [SmolVLA](https://huggingface.co/docs/lerobot/smolvla), a 450 M-parameter vision-language-action
model, on your 120 episodes. It already knows how to see and read; you teach it your arm, your cell and your 12
sentences.

```bash
scripts/train.sh
```

That runs, on a rented Hugging Face GPU (billed per minute to your account; see `hf jobs hardware` for prices):

```bash
lerobot-train --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=fport/so101-qc --policy.repo_id=fport/so101-qc-smolvla \
  --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
  --batch_size=64 --steps=20000 --policy.device=cuda --job.target=a100-large
```

| Flag | Why |
| --- | --- |
| `--policy.path=lerobot/smolvla_base` | Start from the pretrained model, do not train from zero |
| `--rename_map` | The pretrained model calls its cameras `camera1..3`; ours are `top` and `wrist`. Needed again at run time |
| `--steps=20000` | LeRobot's documented recipe: about 4 h on an A100 |
| `--batch_size=64` | Lower it if the GPU runs out of memory |
| `--job.target=a100-large` | Which rented GPU: $2.50/h, so about $10 for the run. Leave the flag out to train on your own machine |

**Colab instead:** open the official
[SmolVLA notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/lerobot/training-smolvla.ipynb),
choose an L4 or A100 runtime, and use the same flags without `--job.target`. The free T4 is too small for this batch
size.

## What happens inside

1. Each camera frame becomes about 64 tokens, the sentence becomes tokens, the joint angles become one more token.
2. The frozen vision-language model reads them once.
3. A small *action expert* (about 100 M parameters, the only part you train) turns noise into the next 50 actions in
   10 denoising steps. This is *flow matching*.
4. The loss is how far the predicted 50 actions are from the 50 you actually made. Nothing else: no reward, no notion
   of success.

So a falling loss means "imitates the recorded motions better", not "sorts parts". Only the real arm tells you that.

## Check

- The loss falls quickly in the first thousand steps, then slowly. If it is flat from the start, the dataset or the
  camera map is wrong.
- When training ends the policy is uploaded to `fport/so101-qc-smolvla` (private).

## Do a short dry run first

After the first two blocks, run `STEPS=200 scripts/train.sh`. It will not produce a useful policy; it proves that the
dataset loads, the camera names map and the upload works, before you have recorded two hours of data.
