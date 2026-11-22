#!/usr/bin/env bash
# Recalibrate one arm. The result is written to hardware/calibration/<follower|leader>.json.
#   scripts/calibrate.sh follower
#   scripts/calibrate.sh leader
set -euo pipefail
source "$(dirname "$0")/env.sh"

case "${1:-}" in
  follower) run lerobot-calibrate "${ROBOT_ARGS[@]}" ;;
  leader)   run lerobot-calibrate "${TELEOP_ARGS[@]}" ;;
  *) echo "usage: scripts/calibrate.sh follower|leader" >&2; exit 2 ;;
esac
