#!/usr/bin/env bash
# Fine-tune SmolVLA on the recorded dataset. Runs where you call it; on a Mac that is too slow, so by default the job
# is sent to a Hugging Face GPU, billed per minute to your HF account (`hf jobs hardware` lists flavors and prices).
#   scripts/train.sh                 HF Jobs, a100-large: the recipe below takes about 4 h there
#   STEPS=200 scripts/train.sh       dry run: proves the dataset loads and the camera names map
#   TARGET=local scripts/train.sh    this machine (needs a CUDA GPU)
# A job uploads the policy to $HF_USER/$DATASET-smolvla_<time> and prints that name as "Model repo:". Put it in
# scripts/env.sh as POLICY. A local run uploads to $HF_USER/$DATASET-smolvla.
set -euo pipefail
source "$(dirname "$0")/env.sh"

cmd=(lerobot-train --policy.path=lerobot/smolvla_base \
  "--dataset.repo_id=$HF_USER/$DATASET" "--job_name=$DATASET-smolvla" \
  "--policy.repo_id=$HF_USER/$DATASET-smolvla" --policy.private=true \
  "--rename_map=$RENAME_MAP" \
  "--batch_size=${BATCH_SIZE:-64}" "--steps=${STEPS:-20000}" \
  --policy.device=cuda --wandb.enable=false)
[[ "${TARGET:-a100-large}" != "local" ]] && cmd+=("--job.target=${TARGET:-a100-large}")
run "${cmd[@]}"
