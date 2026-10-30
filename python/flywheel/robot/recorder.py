"""Plan-driven teleop recording: two-phase task text, ACT command channel, private Hub push at the end of a session.

Three levels; numbers are positions in the plan and match the tracking site:
  episode (1–120)  one part: pick, show to the camera, drop into the bin
  tray (1–20)      6 parts laid in the tray; after 6 episodes the tray is empty, move on to the next tray
  session (1–4)    one `record --session N` run: 5 trays, 30 episodes, Hub push at the end

lerobot-record takes one task text per episode; here each frame's task is written by phase:
phase 1 "pick the part and show it to the camera" + environment_state [0, 0], phase 2 "put the part in the
<OK|REJECT> bin" + one-hot command. The operator advances with one key (ENTER or space): start → shown to camera →
dropped. Which plan row is which episode is kept in <root>/flywheel/episodes.jsonl; an interrupted session
resumes where it left off with the same command.

If the process dies before the LeRobot dataset is finalize()d, a half-written parquet makes all of it unreadable. So
at the end of each tray the dataset is closed and copied incrementally to <root>.snapshot; on startup a broken
dataset is restored from there and the log is trimmed to match. At worst only that tray is re-recorded.
"""

import json
import logging
import math
import os
import select
import shutil
import signal
import subprocess
import sys
import termios
import time
import tty
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

from ..sim.params import REPO_ROOT
from ..sim.parts import TYPE_NAMES, VARIANTS
from .session import PHASE1_TASK, command_onehot, phase2_task, picks, sessions, tray_layout

os.environ.setdefault("SVT_LOG", "1")  # keep the AV1 encoder from flooding the terminal every episode
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore", message="Cannot enable progress bars")
# torchcodec won't load on this Mac, LeRobot falls back to pyav for reading; recording is unaffected, the warning is noise
logging.getLogger("lerobot.utils.import_utils").setLevel(logging.ERROR)

DATASETS_DIR = REPO_ROOT / "outputs" / "datasets"
CATALOG = REPO_ROOT / "python/flywheel/sim/assets/generated/catalog.json"
ENV_STATE = "observation.environment_state"
DEFECT_NAMES = {"missing_hole": "missing hole", "broken_corner": "broken"}
HUB_TAGS = ["so101", "lerobot", "flywheel", "quality-control"]

B, DIM, RST = "\033[1m", "\033[2m", "\033[0m"
ORANGE, GREY, BLUE, GREEN, REJECT = "\033[38;5;208m", "\033[38;5;250m", "\033[38;5;75m", "\033[38;5;78m", "\033[38;5;203m"


def episode_log(root):
    return Path(root) / "flywheel" / "episodes.jsonl"


def snapshot_dir(root):
    return Path(str(root) + ".snapshot")


def writing_marker(root):
    """File that exists while recording; if it is there on startup, the previous run didn't shut down cleanly."""
    return Path(str(root) + ".writing")


def recorded(root):
    """Recorded episodes: [{plan, index, episode_index, ...}, ...]."""
    f = episode_log(root)
    return [json.loads(line) for line in f.read_text().splitlines() if line] if f.exists() else []


def describe(part_id, catalog):
    p = catalog.get(part_id)
    if not p:
        return part_id
    return f"{TYPE_NAMES[p['type']]} · {VARIANTS[p['variant']]} · {DEFECT_NAMES.get(p['defect'], 'sound')}"


def load_catalog():
    return {p["id"]: p for p in json.loads(CATALOG.read_text())} if CATALOG.exists() else {}


# --- keyboard ------------------------------------------------------------------------------------------------

class Keys:
    """Reads single keys while the terminal has focus (cbreak); no accessibility permission needed, unlike pynput."""

    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)

    def poll(self):
        if select.select([sys.stdin], [], [], 0)[0]:
            ch = os.read(self.fd, 1).decode(errors="ignore")
            return "go" if ch in ("\n", "\r", " ") else ch.lower()
        return None


class AutoKeys:
    """For --fake --auto: mimics the operator (tray ready, start, phase 2 after 1 s, drop after 1 s)."""

    def __init__(self, fps):
        self.fps, self.n = fps, 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def poll(self):
        self.n += 1
        return "go" if self.n % self.fps == 0 else None


# --- hardware -----------------------------------------------------------------------------------------------

def make_rig(args):
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.so_follower import SO101FollowerConfig
    from lerobot.teleoperators import make_teleoperator_from_config
    from lerobot.teleoperators.so_leader import SO101LeaderConfig

    from .cli import CALIBRATION_DIR

    cams = {name: OpenCVCameraConfig(index_or_path=idx, width=640, height=480, fps=args.fps)
            for name, idx in (("top", args.top), ("wrist", args.wrist)) if idx is not None}
    robot = make_robot_from_config(SO101FollowerConfig(
        port=args.follower_port, id="follower", calibration_dir=CALIBRATION_DIR, cameras=cams))
    teleop = make_teleoperator_from_config(SO101LeaderConfig(
        port=args.leader_port, id="leader", calibration_dir=CALIBRATION_DIR))
    return robot, teleop


class FakeRig:
    """End-to-end run without hardware: the real robot object's feature schema, synthetic joints and images."""

    def __init__(self, robot):
        self.robot, self.t = robot, 0
        self.name = robot.name
        self.is_connected = False

    @property
    def observation_features(self):
        return self.robot.observation_features

    @property
    def action_features(self):
        return self.robot.action_features

    def connect(self):
        self.is_connected = True

    def disconnect(self):
        self.is_connected = False

    def get_action(self):
        self.t += 1
        return {k: 30 * math.sin(self.t / 25 + i) for i, k in enumerate(self.robot.action_features)}

    def get_observation(self):
        obs = {k: 30 * math.sin(self.t / 25 + i) for i, k in enumerate(self.robot.action_features)}
        for k, shape in self.robot.observation_features.items():
            if isinstance(shape, tuple):
                img = np.zeros(shape, dtype=np.uint8)
                img[..., 0] = (self.t * 4) % 255
                img[: shape[0] // 2, :, 1] = 120
                obs[k] = img
        return obs

    def send_action(self, action):
        return action


# --- display -------------------------------------------------------------------------------------------------

def say(text, enabled):
    if enabled and shutil.which("say"):
        subprocess.Popen(["say", "-v", "Samantha", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def tray_box(plan, tray, done_pockets, catalog):
    ids = tray_layout(plan, tray)

    def cell(n):
        pid = ids[n - 1]
        if n in done_pockets:
            return f"{DIM}{n} {'—':<8}{RST}"
        color = ORANGE if catalog.get(pid, {}).get("variant") == "T" else GREY
        return f"{B}{n}{RST} {color}{pid:<8}{RST}"

    far = "   ".join(cell(n) for n in (4, 5, 6))
    near = "   ".join(cell(n) for n in (1, 2, 3))
    return (f"   {DIM}far row (toward the mast){RST}\n"
            f"   ┌────────────────────────────────────────┐\n"
            f"   │  {far}  │\n"
            f"   │  {near}  │\n"
            f"   └────────────────────────────────────────┘\n"
            f"   {DIM}row nearest the arm · view from behind the arm: left = REJECT bin, right = OK bin{RST}")


def pick_card(p, n_plan, n_trays, session, catalog):
    """Episode and tray numbers are positions in the plan (same as "Episode N" and "Tray N" on the site)."""
    bin_color = GREY if p.command == "ok" else ORANGE
    probe = f"   {BLUE}{B}DELIBERATELY OPPOSITE{RST}{BLUE} · correct is {'REJECT' if p.command == 'ok' else 'OK'}{RST}" if p.probe else ""
    return (f"\n{B}EPISODE {p.index + 1}{RST}{DIM}/{n_plan} · session {session} · tray {p.tray}/{n_trays} · "
            f"part {p.slot} from the tray{RST}\n"
            f"   pocket {B}{p.pocket}{RST}   {B}{p.part}{RST}  {DIM}{describe(p.part, catalog)}{RST}\n"
            f"   → {bin_color}{B}{p.command.upper()} bin{RST}{probe}\n")


def status(msg, end="\n"):
    sys.stdout.write("\r\033[2K" + msg + end)
    sys.stdout.flush()


# --- dataset -----------------------------------------------------------------------------------------------

def dataset_features(robot):
    from lerobot.datasets import aggregate_pipeline_dataset_features, create_initial_features
    from lerobot.processor import make_default_processors
    from lerobot.utils.feature_utils import combine_feature_dicts

    procs = make_default_processors()
    features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(pipeline=procs[0], use_videos=True,
                                            initial_features=create_initial_features(action=robot.action_features)),
        aggregate_pipeline_dataset_features(pipeline=procs[2], use_videos=True,
                                            initial_features=create_initial_features(observation=robot.observation_features)),
    )
    features[ENV_STATE] = {"dtype": "float32", "shape": (2,), "names": ["ok", "reject"]}
    return features, procs


def resume(repo_id, root):
    from lerobot.datasets import LeRobotDataset

    return LeRobotDataset.resume(repo_id, root=root, image_writer_threads=8)


def restore(root):
    """Moves aside a dataset that wasn't closed cleanly, restores the last checkpoint (dataset + log)."""
    snap = snapshot_dir(root)
    broken = root.with_name(f"{root.name}.broken-{datetime.now():%Y%m%d-%H%M%S}")
    if root.exists():
        root.rename(broken)
    if (snap / "meta" / "info.json").exists():
        shutil.copytree(snap, root)
        print(f"{ORANGE}Previous recording didn't close cleanly; restored the checkpoint from the end of the last tray.{RST}")
    else:
        print(f"{ORANGE}Previous recording didn't close cleanly and there is no checkpoint; starting the dataset from scratch.{RST}")
    print(f"{DIM}old copy: {broken}{RST}")
    writing_marker(root).unlink(missing_ok=True)


def open_dataset(repo_id, root, robot, fps):
    """Resumes an existing dataset or creates a new one; if it wasn't closed cleanly, rolls back to the checkpoint first.

    LeRobot buffers its metadata: if the process dies, info.json and the episode table can diverge and resume()
    won't notice. So we check the writing marker, not just "does it open".
    """
    from lerobot.datasets import LeRobotDataset

    root = Path(root)
    features, procs = dataset_features(robot)
    if writing_marker(root).exists():
        restore(root)
    if (root / "meta" / "info.json").exists():
        try:
            return resume(repo_id, root), procs
        except Exception:  # corruption without a marker: roll back to the checkpoint anyway
            restore(root)
            if (root / "meta" / "info.json").exists():
                return resume(repo_id, root), procs
    if root.exists():
        sys.exit(f"{root} exists but is not a LeRobot dataset; pass a different --root")
    n_cams = sum(isinstance(v, tuple) for v in robot.observation_features.values())
    ds = LeRobotDataset.create(repo_id, fps, root=root, robot_type=robot.name, features=features,
                               use_videos=True, image_writer_threads=4 * max(n_cams, 1))
    return ds, procs


def reconcile(root, n_episodes):
    """Removes episodes that are in the log but not in the dataset (lost in a crash) from the log; returns their count."""
    rows = recorded(root)
    keep = [r for r in rows if r["episode_index"] < n_episodes]
    if len(keep) != len(rows):
        episode_log(root).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep))
    return len(rows) - len(keep)


def snapshot(root):
    """Incrementally copies the closed (finalized) dataset to <root>.snapshot."""
    snap = snapshot_dir(root)
    if shutil.which("rsync"):
        subprocess.run(["rsync", "-a", "--delete", f"{root}/", f"{snap}/"], check=True)
    else:
        shutil.rmtree(snap, ignore_errors=True)
        shutil.copytree(root, snap)


def checkpoint(dataset, repo_id, root):
    """Closes the dataset, copies it and reopens it for writing."""
    dataset.finalize()
    snapshot(root)
    return resume(repo_id, root)


def log_episode(root, p, session, episode_index, frames, switch_frame):
    f = episode_log(root)
    f.parent.mkdir(parents=True, exist_ok=True)
    row = {"episode_index": episode_index, "plan": p.plan, "order": p.order, "index": p.index, "session": session,
           "tray": p.tray, "slot": p.slot, "pocket": p.pocket, "part": p.part, "command": p.command,
           "probe": p.probe, "frames": frames, "phase_switch_frame": switch_frame,
           "recorded_at": datetime.now().isoformat(timespec="seconds")}
    with f.open("a") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def push(repo_id, root):
    from lerobot.datasets import LeRobotDataset

    print(f"{DIM}Uploading to Hugging Face (private): {repo_id} …{RST}")
    LeRobotDataset(repo_id, root=root).push_to_hub(private=True, tags=HUB_TAGS)
    print(f"{GREEN}✓ https://huggingface.co/datasets/{repo_id}{RST}")


# --- recording loop -----------------------------------------------------------------------------------------

def _sigterm(*_):
    raise KeyboardInterrupt


def run(args):
    from lerobot.utils.constants import ACTION, OBS_STR
    from lerobot.utils.feature_utils import build_dataset_frame
    from lerobot.utils.robot_utils import precise_sleep
    from lerobot.utils.visualization_utils import init_visualization, log_visualization_data, shutdown_visualization

    all_sessions = sessions(args.plan)
    if not 1 <= args.session <= len(all_sessions):
        sys.exit(f"plan {args.plan} has {len(all_sessions)} sessions; --session 1..{len(all_sessions)}")
    root = Path(args.root)
    trays = set(all_sessions[args.session - 1])
    plan_picks = picks(args.plan)
    in_session = [p for p in plan_picks if p.tray in trays]
    n_trays = len({p.tray for p in plan_picks})
    catalog = load_catalog()
    print(f"\n{B}SO-101 recording · {args.plan} · session {args.session}/{len(all_sessions)}{RST}  "
          f"{DIM}{args.repo_id} → {root}{RST}")

    if args.display:  # same Rerun panel as lerobot-record --display_data: cameras and joint angles
        init_visualization("rerun", session_name="flywheel_record")
    robot, teleop = make_rig(args)
    if args.fake:
        robot = teleop = FakeRig(robot)
    dataset, (teleop_proc, robot_proc, obs_proc) = open_dataset(args.repo_id, root, robot, args.fps)
    if dataset.fps != args.fps:
        sys.exit(f"Dataset is {dataset.fps} fps, recording is {args.fps} fps: they must match.")
    lost = reconcile(root, dataset.meta.total_episodes)
    if lost:
        print(f"{ORANGE}{lost} episodes were recorded after the checkpoint; they will be re-recorded.{RST}")
    done = {r["index"] for r in recorded(root) if r["plan"] == args.plan}
    todo = [p for p in in_session if p.index not in done]
    if not todo:
        dataset.finalize()
        print(f"{GREEN}All {len(in_session)} episodes of this session are already recorded.{RST}")
        if not args.no_push:
            push(args.repo_id, root)
        return

    hw_features = {k: v for k, v in dataset.features.items() if k != ENV_STATE}  # we write the command channel ourselves
    try:
        robot.connect()  # cameras first: a wrong or sleeping index (iPhone Continuity camera) times out here
    except TimeoutError as e:
        dataset.finalize()
        if args.display:
            shutdown_visualization("rerun")
        sys.exit(f"{ORANGE}{e}{RST}\nCamera indices: --top {args.top} --wrist {args.wrist}. Check them with "
                 "`uv run lerobot-find-cameras opencv` (frames go to outputs/captured_images); "
                 "an iPhone nearby can shift the order. Nothing was recorded, run the command again.")
    teleop.connect()
    signal.signal(signal.SIGTERM, _sigterm)
    writing_marker(root).touch()
    saved, state = 0, "tray"
    try:
        with AutoKeys(args.fps) if args.auto else Keys() as keys:
            i, tray_shown, card_shown = 0, None, None
            frames = switch_frame = 0
            while i < len(todo):
                t0 = time.perf_counter()
                p = todo[i]
                obs = robot.get_observation()
                act = teleop.get_action()
                obs_p = obs_proc(obs)
                act_p = teleop_proc((act, obs))
                robot.send_action(robot_proc((act_p, obs)))
                if args.display:
                    log_visualization_data("rerun", observation=obs_p, action=act_p)

                if state == "tray" and tray_shown != p.tray:
                    done_pockets = {q.pocket for q in plan_picks if q.tray == p.tray and q.index in done}
                    print(f"\n{B}TRAY {p.tray}{RST}{DIM}/{n_trays} · lay these 6 parts in the tray, then ENTER{RST}\n"
                          + tray_box(args.plan, p.tray, done_pockets, catalog))
                    say(f"Tray {p.tray}. Lay out the tray.", args.voice)
                    tray_shown = p.tray

                if state in ("phase1", "phase2"):
                    phase2 = state == "phase2"
                    dataset.add_frame({
                        **build_dataset_frame(hw_features, obs_p, prefix=OBS_STR),
                        **build_dataset_frame(hw_features, act_p, prefix=ACTION),
                        ENV_STATE: np.array(command_onehot(p.command) if phase2 else [0.0, 0.0], dtype=np.float32),
                        "task": phase2_task(p.command) if phase2 else PHASE1_TASK,
                    })
                    frames += 1
                    if frames % max(args.fps // 3, 1) == 0:
                        label = f"{ORANGE}phase 2 · put in bin{RST}" if phase2 else f"{BLUE}phase 1 · pick and show{RST}"
                        status(f"   {REJECT}●{RST} rec {frames / args.fps:5.1f} s  {label}  "
                               f"{DIM}ENTER: {'dropped, save' if phase2 else 'shown to camera'} · r: redo · q: quit{RST}", end="")
                    if frames / args.fps >= args.episode_time:
                        status(f"   {REJECT}time is up ({args.episode_time:.0f} s){RST}: ENTER save · r redo")
                        state = "timeout"

                key = keys.poll()
                if key == "q":
                    break
                if key == "r" and state in ("phase1", "phase2", "timeout"):
                    dataset.clear_episode_buffer()
                    status(f"   {DIM}discarded; return the arm to start, ENTER to restart{RST}")
                    state = "ready"
                elif key == "go":
                    if state == "tray":
                        state = "ready"
                    elif state == "ready":
                        state, frames, switch_frame = "phase1", 0, 0
                        say("Start", args.voice)
                    elif state == "phase1":
                        state, switch_frame = "phase2", frames
                        say(f"{p.command.upper()} bin", args.voice)
                    elif state in ("phase2", "timeout"):
                        switch_frame = switch_frame or frames  # if phase 2 was never entered, there is no switch
                        dataset.save_episode()
                        log_episode(root, p, args.session, dataset.meta.total_episodes - 1, frames, switch_frame)
                        done.add(p.index)
                        saved += 1
                        status(f"   {GREEN}✓ episode {p.index + 1} saved{RST} {DIM}dataset #{dataset.meta.total_episodes - 1} · "
                               f"{frames} frames{RST}")
                        i += 1
                        state = "ready"
                        if i == len(todo) or todo[i].tray != p.tray:
                            state = "tray"
                            if i < len(todo):  # last tray is finalized on shutdown anyway
                                status(f"   {DIM}tray {p.tray} done · checkpoint…{RST}", end="")
                                dataset = checkpoint(dataset, args.repo_id, root)
                                status(f"   {GREEN}✓ tray {p.tray} is safe{RST}")

                if state == "ready" and card_shown != p.index:
                    print(pick_card(p, len(plan_picks), n_trays, args.session, catalog)
                          + f"   {DIM}return the arm to start · ENTER to start recording{RST}")
                    say(f"Pocket {p.pocket}. {p.command.upper()} bin.", args.voice)
                    card_shown = p.index

                precise_sleep(max(1 / args.fps - (time.perf_counter() - t0), 0.0))
    except KeyboardInterrupt:
        pass
    finally:
        if dataset.has_pending_frames():
            dataset.clear_episode_buffer()
            status(f"   {DIM}partial episode discarded{RST}")
        dataset.finalize()
        writing_marker(root).unlink(missing_ok=True)  # dataset is consistent after finalize
        try:
            snapshot(root)
        except Exception as e:  # backup failed; dataset is intact, retried at the end of the next tray
            print(f"{ORANGE}warning: could not copy checkpoint: {e}{RST}")
        for dev in {id(robot): robot, id(teleop): teleop}.values():
            if dev.is_connected:
                dev.disconnect()
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        if args.display:
            shutdown_visualization("rerun")

    left = len([p for p in in_session if p.index not in done])
    print(f"\n{B}Session summary{RST}: {saved} episodes this run · {left} left in session · "
          f"dataset total {dataset.meta.total_episodes}")
    if left:
        print(f"{DIM}run the same command again to continue{RST}")
    if saved and not args.no_push:
        push(args.repo_id, root)


def print_sessions(args):
    root = Path(args.root)
    done = {r["index"] for r in recorded(root) if r["plan"] == args.plan}
    plan_picks = picks(args.plan)
    catalog = load_catalog()
    all_sessions = sessions(args.plan)
    print(f"\n{B}{args.plan}{RST}  {DIM}{args.repo_id} · {root}{RST}")
    nxt = None
    for s, trays in enumerate(all_sessions, 1):
        ps = [p for p in plan_picks if p.tray in trays]
        n = sum(p.index in done for p in ps)
        mark = f"{GREEN}✓{RST}" if n == len(ps) else (f"{ORANGE}…{RST}" if n else " ")
        print(f" {mark} session {s}: episodes {ps[0].index + 1}–{ps[-1].index + 1} · trays {trays[0]}–{trays[-1]} · "
              f"{n}/{len(ps)} recorded")
        if nxt is None and n < len(ps):
            nxt = s
    if nxt:
        parts = []
        for p in plan_picks:
            if p.tray in all_sessions[nxt - 1] and p.part not in parts:
                parts.append(p.part)
        print(f"\nNext: session {nxt}. Set aside from the box ({len(parts)} parts):")
        for t in ("rim", "gear", "pad", "bracket"):
            ids = sorted(x for x in parts if catalog.get(x, {}).get("type") == t)
            if ids:
                print(f"   {TYPE_NAMES[t]:<7} {' '.join(ids)}")
        print(f"\n   uv run flywheel-robot record --session {nxt}")
