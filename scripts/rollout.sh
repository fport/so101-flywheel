#!/usr/bin/env bash
# Let the trained policy move the arm for one pick. Keep a hand near the power switch.
#   scripts/rollout.sh 3 ok
#   POLICY=fport/other-policy scripts/rollout.sh 3 reject   (default: POLICY in scripts/env.local.sh)
#   DEVICE=cuda scripts/rollout.sh 3 ok        (default mps: Apple GPU)
set -euo pipefail
source "$(dirname "$0")/env.sh"

pocket="${1:-}"; bin="${2:-}"
if [[ ! "$pocket" =~ ^[1-6]$ || ! "$bin" =~ ^(ok|reject)$ ]]; then
  echo "usage: scripts/rollout.sh <pocket 1-6> <ok|reject>" >&2; exit 2
fi
if [[ -z "$POLICY" ]]; then
  echo "set POLICY in scripts/env.local.sh to the model repo that scripts/train.sh printed" >&2; exit 2
fi
task="put the part from pocket $pocket in the $(echo "$bin" | tr '[:lower:]' '[:upper:]') bin"

run lerobot-rollout --strategy.type=base "--policy.path=$POLICY" \
  "--policy.device=${DEVICE:-mps}" --inference.type=rtc "--rename_map=$RENAME_MAP" \
  "${ROBOT_ARGS[@]}" "--robot.cameras=$CAMERAS" \
  "--task=$task" "--duration=${DURATION:-30}" "--fps=$FPS"
