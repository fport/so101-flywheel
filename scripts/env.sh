# Rig settings shared by every script here; nothing else hides a port or a camera index.
# Put your own values in scripts/env.local.sh (not committed), e.g.
#   FOLLOWER_PORT=/dev/cu.usbmodem5AB0...   LEADER_PORT=...   TOP_CAM=1   WRIST_CAM=0   HF_USER=you
# Find them with:  uv run lerobot-find-port   and   uv run lerobot-find-cameras opencv

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$_here/env.local.sh" ]] && source "$_here/env.local.sh"

FOLLOWER_PORT="${FOLLOWER_PORT:-}"   # 12 V arm
LEADER_PORT="${LEADER_PORT:-}"       # 5 V arm
TOP_CAM="${TOP_CAM:-1}"       # OpenCV index of the camera on the mast
WRIST_CAM="${WRIST_CAM:-0}"   # OpenCV index of the camera on the gripper
FPS="${FPS:-30}"

HF_USER="${HF_USER:-}"
DATASET="${DATASET:-so101-qc}"          # -> huggingface.co/datasets/$HF_USER/$DATASET (private)

# The trained policy on the Hub. A Hugging Face job names it <job name>_<time> and prints it as "Model repo:" when it
# starts: put it in scripts/env.local.sh as POLICY=... (or pass POLICY=... to scripts/rollout.sh).
POLICY="${POLICY:-}"

REPO_ROOT="$(cd "$_here/.." && pwd)"
CALIBRATION_DIR="$REPO_ROOT/hardware/calibration"
DATASET_ROOT="$REPO_ROOT/outputs/datasets/$DATASET"

CAMERAS="{ top: {type: opencv, index_or_path: $TOP_CAM, width: 640, height: 480, fps: $FPS, warmup_s: 3}, wrist: {type: opencv, index_or_path: $WRIST_CAM, width: 640, height: 480, fps: $FPS, warmup_s: 3} }"

# SmolVLA's pretrained weights call their cameras camera1..3; ours are top and wrist. Same map for training and rollout.
RENAME_MAP='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'

ROBOT_ARGS=(--robot.type=so101_follower "--robot.port=$FOLLOWER_PORT" --robot.id=follower "--robot.calibration_dir=$CALIBRATION_DIR")
TELEOP_ARGS=(--teleop.type=so101_leader "--teleop.port=$LEADER_PORT" --teleop.id=leader "--teleop.calibration_dir=$CALIBRATION_DIR")

if [[ -z "$FOLLOWER_PORT" || -z "$LEADER_PORT" || -z "$HF_USER" ]]; then
  echo "set FOLLOWER_PORT, LEADER_PORT and HF_USER in scripts/env.local.sh (see the top of scripts/env.sh)" >&2
  exit 2
fi

# Prints the command, then runs it inside the project's virtualenv.
run() {
  printf '\n\033[2m$'; printf ' %q' "$@"; printf '\033[0m\n\n'
  (cd "$REPO_ROOT" && uv run --extra robot "$@")
}
