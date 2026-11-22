#!/usr/bin/env bash
# Drive the follower with the leader. Ctrl+C to quit.
#   scripts/teleop.sh          arms only
#   scripts/teleop.sh --cams   also opens Rerun with both cameras (use it to aim the cameras, not while recording)
set -euo pipefail
source "$(dirname "$0")/env.sh"

extra=()
if [[ "${1:-}" == "--cams" ]]; then
  extra=("--robot.cameras=$CAMERAS" --display_data=true --display_compressed_images=true)
fi
run lerobot-teleoperate "${ROBOT_ARGS[@]}" "${TELEOP_ARGS[@]}" --fps=60 "${extra[@]}"
