#!/usr/bin/env bash
# Both cameras side by side in the browser (http://localhost:8642), to aim them. No Rerun, no robot.
# Run it next to `scripts/teleop.sh`; stop it before recording.
set -euo pipefail
source "$(dirname "$0")/env.sh"
(cd "$REPO_ROOT" && uv run flywheel-cams "$TOP_CAM" "$WRIST_CAM")
