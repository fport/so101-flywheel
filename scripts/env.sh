# Rig settings shared by every script here. Edit once; nothing else hides a port or a camera index.
# Find yours with:  uv run lerobot-find-port   and   uv run lerobot-find-cameras opencv

FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/cu.usbmodem5AB01794851}"   # 12 V arm
LEADER_PORT="${LEADER_PORT:-/dev/cu.usbmodem5A7C1232561}"       # 5 V arm
TOP_CAM="${TOP_CAM:-1}"       # OpenCV index of the camera on the mast
WRIST_CAM="${WRIST_CAM:-0}"   # OpenCV index of the camera on the gripper
FPS="${FPS:-30}"

HF_USER="${HF_USER:-fport}"
DATASET="${DATASET:-so101-qc}"          # -> huggingface.co/datasets/$HF_USER/$DATASET (private)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CALIBRATION_DIR="$REPO_ROOT/hardware/calibration"
DATASET_ROOT="$REPO_ROOT/outputs/datasets/$DATASET"

CAMERAS="{ top: {type: opencv, index_or_path: $TOP_CAM, width: 640, height: 480, fps: $FPS, warmup_s: 3}, wrist: {type: opencv, index_or_path: $WRIST_CAM, width: 640, height: 480, fps: $FPS, warmup_s: 3} }"

# SmolVLA's pretrained weights call their cameras camera1..3; ours are top and wrist. Same map for training and rollout.
RENAME_MAP='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'

ROBOT_ARGS=(--robot.type=so101_follower "--robot.port=$FOLLOWER_PORT" --robot.id=follower "--robot.calibration_dir=$CALIBRATION_DIR")
TELEOP_ARGS=(--teleop.type=so101_leader "--teleop.port=$LEADER_PORT" --teleop.id=leader "--teleop.calibration_dir=$CALIBRATION_DIR")

# Prints the command, then runs it inside the project's virtualenv.
run() {
  printf '\n\033[2m$'; printf ' %q' "$@"; printf '\033[0m\n\n'
  (cd "$REPO_ROOT/python" && uv run --extra robot "$@")
}
