#!/usr/bin/env bash
# Drive the follower with the leader. Ctrl+C to quit.
#   scripts/teleop.sh          arms only
#   scripts/teleop.sh --cams   also opens Rerun with both cameras. Rerun can stall the control loop on a Mac;
#                              scripts/cams.sh in a second terminal shows the cameras in the browser without it
set -euo pipefail
source "$(dirname "$0")/env.sh"

cmd=(lerobot-teleoperate "${ROBOT_ARGS[@]}" "${TELEOP_ARGS[@]}" --fps=60)
if [[ "${1:-}" == "--cams" ]]; then
  cmd+=("--robot.cameras=$CAMERAS" --display_data=true --display_compressed_images=true)
fi
run "${cmd[@]}"
