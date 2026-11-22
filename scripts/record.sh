#!/usr/bin/env bash
# Record one block of episodes with plain `lerobot-record`: one pocket, one bin, one task sentence.
#
#   scripts/record.sh 3 ok          10 episodes of "put the part from pocket 3 in the OK bin"
#   scripts/record.sh 3 reject 5    5 more episodes for pocket 3 -> REJECT
#
# Every block is appended to the same dataset (outputs/datasets/$DATASET) and pushed to the Hub, private, when the
# block ends. Keys:  → end this phase (episode or reset)   ← redo the episode   Esc stop now (keeps the current episode:
# press it during a reset, not mid-pick).
set -euo pipefail
source "$(dirname "$0")/env.sh"

pocket="${1:-}"; bin="${2:-}"; episodes="${3:-10}"
if [[ ! "$pocket" =~ ^[1-6]$ || ! "$bin" =~ ^(ok|reject)$ ]]; then
  echo "usage: scripts/record.sh <pocket 1-6> <ok|reject> [episodes=10]" >&2; exit 2
fi
task="put the part from pocket $pocket in the $(echo "$bin" | tr '[:lower:]' '[:upper:]') bin"

cmd=(lerobot-record "${ROBOT_ARGS[@]}" "--robot.cameras=$CAMERAS" "${TELEOP_ARGS[@]}" \
  "--dataset.repo_id=$HF_USER/$DATASET" "--dataset.root=$DATASET_ROOT" --dataset.no_stamp=true \
  "--dataset.single_task=$task" "--dataset.num_episodes=$episodes" "--dataset.fps=$FPS" \
  "--dataset.episode_time_s=${EPISODE_TIME_S:-30}" "--dataset.reset_time_s=${RESET_TIME_S:-15}" \
  --dataset.streaming_encoding=true --dataset.encoder_threads=2 \
  --dataset.push_to_hub=true --dataset.private=true \
  --display_data=false)
# the first block creates the dataset, every later one appends to it
[[ -f "$DATASET_ROOT/meta/info.json" ]] && cmd+=(--resume=true)
run "${cmd[@]}"
