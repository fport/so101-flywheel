#!/usr/bin/env bash
# Look at recorded episodes, drop bad ones, upload.
#   scripts/review.sh 12              play episode 12 in Rerun (both cameras, joints)
#   scripts/review.sh delete 12 40    remove episodes 12 and 40 (the rest are renumbered)
#   scripts/review.sh push            upload the local dataset to the Hub (private)
set -euo pipefail
source "$(dirname "$0")/env.sh"

case "${1:-}" in
  delete)
    shift; ids="$(IFS=,; echo "$*")"
    # same root in and out = edit in place; lerobot keeps the previous version next to it as ${DATASET}_old
    run lerobot-edit-dataset "--repo_id=$HF_USER/$DATASET" "--root=$DATASET_ROOT" "--new_root=$DATASET_ROOT" \
      --operation.type=delete_episodes "--operation.episode_indices=[$ids]" ;;
  push)
    run hf upload "$HF_USER/$DATASET" "$DATASET_ROOT" . --repo-type dataset --private ;;
  ''|*[!0-9]*)
    echo "usage: scripts/review.sh <episode> | delete <episode>... | push" >&2; exit 2 ;;
  *)
    run lerobot-dataset-viz "--repo-id=$HF_USER/$DATASET" "--root=$DATASET_ROOT" "--episode-index=$1" --display-compressed-images ;;
esac
