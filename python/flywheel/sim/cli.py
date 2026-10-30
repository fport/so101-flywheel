"""flywheel-sim: build and view the scene, run scripted episodes and batch trials."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .env import FPS, QCSortSim
from .params import CATALOG_JSON, REPO_ROOT, SCENE_XML
from .parts import TYPE_NAMES, VARIANTS, load_catalog
from .plan import TASK, build_plan, to_markdown
from .scene import DEFAULT_PARTS, JOINTS, build_scene
from .scripted import ScriptedSorter

POLICY_VERSION = "scripted-v0"


def _run_id():
    return time.strftime("sim-%Y%m%d-%H%M%S")


def wilson(successes, n, z=1.96):
    """Wilson score interval (low, high)."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


class Recorder:
    """Collects state/action over an episode and, optionally, frames from both cameras."""

    def __init__(self, video=False):
        self.video = video
        self.states, self.actions, self.phases, self.frames = [], [], [], []

    def __call__(self, env, phase, action):
        self.states.append(env.data.qpos[:6].copy())
        self.actions.append(action)
        self.phases.append(phase)
        if self.video:
            self.frames.append(np.hstack([env.render("top"), env.render("wrist")]))

    def save(self, out, name, meta):
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / f"{name}.npz", state=np.array(self.states), action=np.array(self.actions),
                            phase=np.array(self.phases), joints=np.array(JOINTS), fps=FPS)
        (out / f"{name}.json").write_text(json.dumps(meta, indent=1))
        if self.video and self.frames:
            import imageio.v2 as imageio

            imageio.mimsave(out / f"{name}.mp4", self.frames, fps=FPS, macro_block_size=1)


def run_episode(env, pid, commanded_bin, out, run_id, video, probe=False, name=None):
    rec = Recorder(video=video)
    env.frame = 0
    outcome = ScriptedSorter(env, on_frame=rec).run(pid, commanded_bin, probe=probe)
    meta = {
        "run_id": run_id,
        "policy_version": POLICY_VERSION,
        "task": TASK[outcome["commanded_bin"]],
        "frames": len(rec.states),
        "fps": FPS,
        **outcome,
    }
    rec.save(out, name or pid, meta)
    return meta


def cmd_build(args):
    model, catalog = build_scene(part_ids=args.parts or DEFAULT_PARTS)
    print(f"scene: {SCENE_XML}")
    print(f"catalog: {CATALOG_JSON} ({len(catalog)} parts)")
    print(f"  {'type':8s} {'color':7s}  sound missing hole broken  held-out")
    for ptype in dict.fromkeys(c.type for c in catalog):
        for variant in VARIANTS:
            rows = [c for c in catalog if c.type == ptype and c.variant == variant]
            n = Counter(c.defect for c in rows if not c.heldout)
            print(f"  {TYPE_NAMES[ptype]:8s} {VARIANTS[variant]:7s} {n[None]:6d} {n['missing_hole']:12d} "
                  f"{n['broken_corner']:6d} {sum(c.heldout for c in rows):9d}")
    print(f"model: {model.nbody} bodies, {model.ngeom} geoms, {model.ncam} cameras, dt={model.opt.timestep:.5f}s")


def cmd_view(args):
    import mujoco.viewer

    env = QCSortSim(part_ids=args.parts or DEFAULT_PARTS, save_scene=True)
    mujoco.viewer.launch(env.model, env.data)


def cmd_render(args):
    from PIL import Image

    env = QCSortSim(part_ids=args.parts or DEFAULT_PARTS)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.hstack([env.render("top"), env.render("wrist")])).save(out)
    print(out)


def cmd_episode(args):
    parts = args.parts or DEFAULT_PARTS
    pid = args.part or parts[0]
    if pid not in parts:
        parts = [pid] + [p for p in parts if p != pid][: len(parts) - 1]
    env = QCSortSim(part_ids=parts, seed=args.seed, order=args.order)
    run_id = _run_id()
    out = Path(args.out) / run_id
    meta = run_episode(env, pid, args.bin, out, run_id, video=not args.no_video)
    print(json.dumps(meta, indent=1))
    print(f"recording: {out}")
    return 0 if meta["success"] else 1


def _plan(args):
    build_scene()
    catalog = load_catalog(CATALOG_JSON)
    return catalog, build_plan(catalog, order=args.order, seed=args.seed, heldout=args.heldout)


def cmd_plan(args):
    """Writes the balanced recording (or held-out eval) plan the operator will follow."""
    catalog, trays = _plan(args)
    name = f"{'eval-heldout' if args.heldout else 'record'}-{args.order}-seed{args.seed}"
    title = ("Held-out eval plan" if args.heldout else "Recording plan") + f" ({name})"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.md").write_text(to_markdown(trays, catalog, args.order, title))
    (out / f"{name}.json").write_text(json.dumps({"order": args.order, "seed": args.seed, "trays": trays}, indent=1))
    n = sum(len(t["picks"]) for t in trays)
    print(f"{out / (name + '.md')}: {len(trays)} trays, {n} episodes ({n // 2} correct command + {n // 2} opposite command)")


def cmd_batch(args):
    """Runs the balanced plan with the scripted policy in sim; reports normal and opposite-command trials separately."""
    catalog, trays = _plan(args)
    types = {c.id: c.type for c in catalog}
    run_id = _run_id()
    out = Path(args.out) / run_id
    results = []
    total = sum(len(t["picks"]) for t in trays)
    for t in trays:
        env = QCSortSim(part_ids=t["pockets"], seed=args.seed + t["tray"], order=args.order)
        for k, pick in enumerate(t["picks"], start=1):
            meta = run_episode(env, pick["part"], pick["command"], out, run_id, video=args.video,
                               probe=pick["probe"], name=f"t{t['tray']:02d}_{k}_{pick['part']}")
            results.append(meta)
            if not meta["success"]:
                print(f"{len(results):3d}/{total}  {pick['part']:8s} {'probe ' if pick['probe'] else ''}{meta['failure_type']}")

    def row(label, rows, key="success"):
        k = sum(r[key] for r in rows)
        lo, hi = wilson(k, len(rows))
        print(f"{label:22s} {k:3d}/{len(rows):<3d}  [{lo:.2f}, {hi:.2f}]")

    print(f"\n{'':22s} {'success':>7s}  Wilson 95%")
    row("correct command", [r for r in results if not r["probe"]])
    row("opposite (followed)", [r for r in results if r["probe"]], key="followed_command")
    for ptype in dict.fromkeys(types.values()):
        row(f"  {TYPE_NAMES[ptype]}", [r for r in results if types[r["part_id"]] == ptype], key="followed_command")
    failures = Counter(r["failure_type"] for r in results if not r["success"])
    print("failure_type:", dict(failures) or "none")
    summary = {"run_id": run_id, "policy_version": POLICY_VERSION, "order": args.order, "seed": args.seed,
               "heldout": args.heldout, "n": len(results), "successes": sum(r["success"] for r in results),
               "followed_command": sum(r["followed_command"] for r in results), "failures": dict(failures)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"recording: {out}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="flywheel-sim", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    default_out = str(REPO_ROOT / "outputs" / "sim")

    def add(name, fn, help_):
        sp = sub.add_parser(name, help=help_)
        sp.set_defaults(fn=fn)
        sp.add_argument("--parts", nargs="+", metavar="ID", help="part IDs to lay in the tray (at most 6)")
        return sp

    add("build", cmd_build, "build the scene and part catalog from params.yaml")
    add("view", cmd_view, "interactive MuJoCo viewer")
    sp = add("render", cmd_render, "save a single frame from the top + wrist camera")
    sp.add_argument("--out", default=str(Path(default_out) / "scene.png"))
    sp = add("episode", cmd_episode, "scripted look -> pick -> drop episode for one part")
    sp.add_argument("--part", help="part ID (default: first part in the tray)")
    sp.add_argument("--bin", choices=["ok", "reject"], help="target bin (default: correct bin per the work order)")
    sp.add_argument("--order", choices=list(VARIANTS), default="W", help="variant in the work order")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--no-video", action="store_true")
    sp.add_argument("--out", default=default_out)
    for name, fn, help_ in (("plan", cmd_plan, "write the balanced recording / eval plan (each part goes to both bins)"),
                            ("batch", cmd_batch, "run the plan with the scripted policy in sim, print a success table")):
        sp = sub.add_parser(name, help=help_)
        sp.set_defaults(fn=fn)
        sp.add_argument("--heldout", action="store_true", help="held-out eval set instead of the training set")
        sp.add_argument("--order", choices=list(VARIANTS), default="W", help="variant in the work order")
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--out", default=str(REPO_ROOT / "plans") if name == "plan" else default_out)
        if name == "batch":
            sp.add_argument("--video", action="store_true")

    args = parser.parse_args(argv)
    return args.fn(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
