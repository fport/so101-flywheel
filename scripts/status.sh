#!/usr/bin/env bash
# What is recorded and which block to record next.   scripts/status.sh [episodes per block = 10]
set -euo pipefail
source "$(dirname "$0")/env.sh"

if [[ ! -f "$DATASET_ROOT/meta/info.json" ]]; then
  echo "nothing recorded yet. Start with: scripts/record.sh 1 ok"; exit 0
fi
(cd "$REPO_ROOT/python" && uv run --extra robot python "$REPO_ROOT/scripts/status.py" "$HF_USER/$DATASET" "$DATASET_ROOT" "${1:-10}")
