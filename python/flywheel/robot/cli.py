"""flywheel-robot: calibrate the real SO-101 arm and drive it with the leader (lerobot CLI wrapper)."""

import argparse
import filecmp
import glob
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from ..sim.params import REPO_ROOT

CALIBRATION_DIR = REPO_ROOT / "hardware" / "calibration"
# from the USB chip's serial number, usually unchanged when the cable is replugged
FOLLOWER_PORT = "/dev/cu.usbmodem5AB01794851"
LEADER_PORT = "/dev/cu.usbmodem5A7C1232561"


def _cameras(top, wrist):
    cams = [f"{name}: {{type: opencv, index_or_path: {idx}, width: 640, height: 480, fps: 30}}"
            for name, idx in (("top", top), ("wrist", wrist)) if idx is not None]
    return "{ " + ", ".join(cams) + " }"


def teleop_argv(args):
    argv = [
        "--robot.type=so101_follower", f"--robot.port={args.follower_port}",
        "--robot.id=follower", f"--robot.calibration_dir={CALIBRATION_DIR}",
        "--teleop.type=so101_leader", f"--teleop.port={args.leader_port}",
        "--teleop.id=leader", f"--teleop.calibration_dir={CALIBRATION_DIR}",
        f"--fps={args.fps}",
    ]
    if args.top is not None or args.wrist is not None:
        argv += [f"--robot.cameras={_cameras(args.top, args.wrist)}", "--display_data=true"]
    if args.seconds:
        argv.append(f"--teleop_time_s={args.seconds}")
    return argv + args.extra


def _check_ports(*ports):
    missing = [p for p in ports if not os.path.exists(p)]
    if missing:
        found = ", ".join(sorted(glob.glob("/dev/cu.usbmodem*"))) or "none"
        sys.exit(f"Port not found: {', '.join(missing)}\nConnected: {found}\n"
                 "Pass with --follower-port / --leader-port (follower ~12 V, leader ~5 V).")


def _lerobot(name):
    exe = Path(sys.executable).parent / name
    if not exe.exists():
        sys.exit("lerobot is not installed: `uv sync --extra robot`")
    return str(exe)


def calibrate_argv(device, port):
    kind = "robot" if device == "follower" else "teleop"
    return [f"--{kind}.type=so101_{device}", f"--{kind}.port={port}",
            f"--{kind}.id={device}", f"--{kind}.calibration_dir={CALIBRATION_DIR}"]


def cmd_calibrate(args):
    devices = ["follower", "leader"] if args.device == "both" else [args.device]
    ports = {"follower": args.follower_port, "leader": args.leader_port}
    _check_ports(*(ports[d] for d in devices))
    exe = _lerobot("lerobot-calibrate")
    for device in devices:
        current = CALIBRATION_DIR / f"{device}.json"
        backup = None
        if current.exists():
            backup = CALIBRATION_DIR / "old" / f"{device}-{time.strftime('%Y%m%d-%H%M%S')}.json"
            backup.parent.mkdir(exist_ok=True)
            shutil.copy2(current, backup)
        print(f"\n== {device} ({ports[device]}) ==\n"
              "At the existing-file prompt, type `c` and ENTER to recalibrate; ENTER alone writes the file to the motors.\n"
              "1) Move the arm to the middle of each joint's range, ENTER.\n"
              "2) Move each joint except wrist_roll from one end to the other in turn, ENTER.", flush=True)
        code = subprocess.run([exe, *calibrate_argv(device, ports[device])]).returncode
        if backup and current.exists() and filecmp.cmp(backup, current, shallow=False):
            backup.unlink()
        elif backup:
            print(f"Old calibration backed up: {backup}")
        if code:
            sys.exit(code)


def _dataset_args(args):
    from .recorder import DATASETS_DIR

    name = args.repo_id.split("/")[-1]
    args.root = args.root or str(DATASETS_DIR / name)
    return args


def cmd_record(args):
    from .recorder import run

    if args.fake:
        args.repo_id = args.repo_id + "-fake" if not args.repo_id.endswith("-fake") else args.repo_id
        args.no_push = args.no_push or not args.push_fake
    else:
        _check_ports(args.follower_port, args.leader_port)
    if args.auto and not args.fake:
        sys.exit("--auto only with --fake")
    run(_dataset_args(args))


def cmd_sessions(args):
    from .recorder import print_sessions

    print_sessions(_dataset_args(args))


def cmd_check(args):
    from .check import main as check_main

    check_main(_dataset_args(args))


def cmd_inspect(args):
    from .inspection import run as inspect_run

    if not (args.sim or args.camera is not None or args.image):
        sys.exit("give --sim N, --camera INDEX or --image FILE ...")
    inspect_run(args)


def cmd_teleop(args):
    _check_ports(args.follower_port, args.leader_port)
    argv = [_lerobot("lerobot-teleoperate"), *teleop_argv(args)]
    print(" ".join(argv), flush=True)
    os.execv(argv[0], argv)


def main():
    parser = argparse.ArgumentParser(prog="flywheel-robot")
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("teleop", help="drive the follower with the leader; Ctrl+C to quit")
    t.add_argument("--follower-port", default=FOLLOWER_PORT)
    t.add_argument("--leader-port", default=LEADER_PORT)
    t.add_argument("--fps", type=int, default=60)
    t.add_argument("--top", type=int, help="top camera OpenCV index (opens Rerun if given)")
    t.add_argument("--wrist", type=int, help="wrist camera OpenCV index")
    t.add_argument("--seconds", type=float, help="stop automatically after this many seconds")
    t.add_argument("extra", nargs=argparse.REMAINDER, help="everything after `--` is passed to lerobot-teleoperate as is")
    t.set_defaults(func=cmd_teleop)
    c = sub.add_parser("calibrate", help="recalibrate the follower, the leader or both")
    c.add_argument("device", choices=["follower", "leader", "both"])
    c.add_argument("--follower-port", default=FOLLOWER_PORT)
    c.add_argument("--leader-port", default=LEADER_PORT)
    c.set_defaults(func=cmd_calibrate)
    for name, fn, help_ in (("record", cmd_record, "plan-driven recording: record one session (5 trays, 30 episodes), private Hub push at the end"),
                            ("sessions", cmd_sessions, "session status and the parts list for the next session"),
                            ("check", cmd_check, "check a recorded dataset (servos, command channel, videos); runs before every push")):
        r = sub.add_parser(name, help=help_)
        r.add_argument("--plan", default="record-W-seed0", help="plans/<name>.json")
        r.add_argument("--repo-id", default="fport/so101-qc")
        r.add_argument("--root", help="local dataset folder (default outputs/datasets/<repo name>)")
        r.set_defaults(func=fn)
        if name in ("sessions", "check"):
            continue
        r.add_argument("--session", type=int, required=True, help="starts at 1; `flywheel-robot sessions` lists them")
        r.add_argument("--top", type=int, default=1, help="top camera OpenCV index")
        r.add_argument("--wrist", type=int, default=0, help="wrist camera OpenCV index")
        r.add_argument("--fps", type=int, default=30)
        r.add_argument("--episode-time", type=float, default=60, help="max seconds per episode")
        r.add_argument("--follower-port", default=FOLLOWER_PORT)
        r.add_argument("--leader-port", default=LEADER_PORT)
        r.add_argument("--no-push", action="store_true", help="don't upload to the Hub at the end of the session")
        r.add_argument("--no-voice", dest="voice", action="store_false", help="turn off spoken prompts")
        r.add_argument("--no-display", dest="display", action="store_false",
                       help="do not open the Rerun panel (cameras and joint angles; on by default)")
        r.add_argument("--fake", action="store_true", help="no hardware: synthetic joints and images")
        r.add_argument("--auto", action="store_true", help=argparse.SUPPRESS)
        r.add_argument("--push-fake", action="store_true", help=argparse.SUPPRESS)
    i = sub.add_parser("inspect", help="ask Claude which bin a part goes to, from a wrist-camera frame; prints the time")
    i.add_argument("--sim", type=int, metavar="N", help="N sim parts in the look pose, scored against ground truth")
    i.add_argument("--camera", type=int, metavar="INDEX", help="one frame from the real wrist camera (0)")
    i.add_argument("--image", nargs="+", metavar="FILE", help="image files")
    i.add_argument("--model", action="append", help="repeat to compare (default claude-sonnet-5-5)")
    i.add_argument("--order", choices=["W", "T"], default="W", help="variant in the work order")
    i.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"], help="lower is faster")
    i.add_argument("--seed", type=int, default=0)
    i.set_defaults(func=cmd_inspect)
    args = parser.parse_args()
    if args.cmd == "teleop" and args.extra[:1] == ["--"]:
        args.extra = args.extra[1:]
    args.func(args)


if __name__ == "__main__":
    main()
