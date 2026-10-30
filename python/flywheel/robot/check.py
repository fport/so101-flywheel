"""Checks a recorded dataset before it goes to the Hub: servos, command channel, plan log and videos.

    uv run flywheel-robot check                    # outputs/datasets/so101-qc
    uv run flywheel-robot check --repo-id fport/so101-qc-test

The recorder runs it before every push and does not push when it fails.
"""

import json
import sys
from pathlib import Path

import numpy as np

from .session import PHASE1_TASK, command_onehot, phase2_task

B, DIM, GREEN, ORANGE, RST = "\033[1m", "\033[2m", "\033[38;5;78m", "\033[38;5;208m", "\033[0m"
TRACK_LIMIT = 15.0  # mean |action - state| per joint (degrees / gripper %); above this the follower is not following
STILL_LIMIT = 1.0  # a joint whose range over the whole dataset is below this never moved: servo not read


def check(repo_id, root):
    """Prints a report and returns the list of problems (empty: safe to push)."""
    from lerobot.datasets import LeRobotDataset

    info = Path(root) / "meta" / "info.json"
    if not info.exists() or not json.loads(info.read_text())["total_episodes"]:
        # LeRobotDataset would go looking for it on the Hub
        print(f"{B}{repo_id}{RST} {DIM}{root}{RST}\n{ORANGE}   no saved episodes{RST}")
        return ["dataset has no episodes"]
    ds = LeRobotDataset(repo_id, root=root, video_backend="pyav")  # torchcodec does not load with Homebrew FFmpeg
    meta, h = ds.meta, ds.hf_dataset.with_format("numpy")
    problems = []
    names = [n.removesuffix(".pos") for n in meta.features["observation.state"]["names"]]
    cams = [k for k, v in meta.features.items() if v["dtype"] == "video"]
    print(f"{B}{repo_id}{RST} {DIM}{root}{RST}\n   {meta.total_episodes} episodes · {meta.total_frames} frames · "
          f"{meta.fps} fps · {meta.total_frames / meta.fps / 60:.1f} min · cameras {', '.join(c.split('.')[-1] for c in cams)}")

    state, action = np.asarray(h["observation.state"]), np.asarray(h["action"])
    env, ep, fi = np.asarray(h["observation.environment_state"]), np.asarray(h["episode_index"]), np.asarray(h["frame_index"])
    tasks = {int(i): t for t, i in meta.tasks["task_index"].items()}
    task = np.array([tasks[int(i)] for i in h["task_index"]])

    print(f"\n   {'joint':<14}{'min':>8}{'max':>8}{'|act-obs|':>11}")
    for j, name in enumerate(names):
        lo, hi = state[:, j].min(), state[:, j].max()
        err = np.abs(action[:, j] - state[:, j]).mean()
        flag = ""
        if hi - lo < STILL_LIMIT:
            flag = f"{ORANGE}never moved{RST}"
            problems.append(f"{name}: never moved over the whole dataset (servo not read?)")
        elif err > TRACK_LIMIT:
            flag = f"{ORANGE}follower not following{RST}"
            problems.append(f"{name}: mean |action - state| {err:.1f}, follower is not following the leader")
        print(f"   {name:<14}{lo:8.1f}{hi:8.1f}{err:11.1f}  {flag}")

    log = Path(root) / "flywheel" / "episodes.jsonl"
    rows = {r["episode_index"]: r for r in map(json.loads, log.read_text().splitlines())} if log.exists() else {}
    if set(rows) != set(range(meta.total_episodes)):
        problems.append(f"plan log has {len(rows)} rows, dataset has {meta.total_episodes} episodes")
    for e in range(meta.total_episodes):
        m = ep == e
        if not np.array_equal(fi[m], np.arange(m.sum())):
            problems.append(f"episode {e}: frame_index is not 0..{m.sum() - 1}")
        r = rows.get(e)
        if r is None:
            continue
        if r["frames"] != m.sum():
            problems.append(f"episode {e}: log says {r['frames']} frames, dataset has {m.sum()}")
        sw = r["phase_switch_frame"]
        want_env = np.array([[0.0, 0.0]] * sw + [command_onehot(r["command"])] * (m.sum() - sw))
        want_task = [PHASE1_TASK] * sw + [phase2_task(r["command"])] * (m.sum() - sw)
        if len(want_env) != m.sum() or not np.array_equal(env[m], want_env) or list(task[m]) != want_task:
            problems.append(f"episode {e} (plan #{r['index'] + 1}, {r['command']}): command channel or task text "
                            "does not match the plan")
    commands = [r["command"] for r in rows.values()]
    print(f"\n   command channel: {commands.count('ok')} OK · {commands.count('reject')} REJECT · "
          f"task texts {len(tasks)}")

    last = int(meta.episodes["dataset_to_index"][-1]) - 1
    for i in (0, last):
        item = ds[i]
        for c in cams:
            img = item[c].numpy()
            if img.mean() < 0.02:
                problems.append(f"{c.split('.')[-1]} camera frame {i} is black")
    print(f"   videos: first and last frame of {', '.join(c.split('.')[-1] for c in cams)} decoded")

    if problems:
        print(f"\n{ORANGE}{len(problems)} problems:{RST}")
        for p in problems:
            print(f"   - {p}")
    else:
        print(f"\n{GREEN}✓ dataset looks right{RST}")
    return problems


def main(args):
    sys.exit(1 if check(args.repo_id, args.root) else 0)
