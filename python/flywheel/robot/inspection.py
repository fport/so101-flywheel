"""Inspection with Claude: one wrist-camera frame from the look pose -> part attributes -> decide() -> bin.

    uv run flywheel-robot inspect --sim 24                  # sim parts with known ground truth
    uv run flywheel-robot inspect --camera 0                # the real wrist camera, now (arm above the part)
    uv run flywheel-robot inspect --image a.jpg b.jpg
    uv run flywheel-robot inspect --sim 8 --model claude-haiku-5-5 --model claude-sonnet-5-5

Claude only reports what it sees (type, colour, missing hole, broken); OK/REJECT comes from decide() and the work
order, like it will with any other classifier. Prints the bin and how long the request took, and keeps every frame,
answer and timing in outputs/inspect/<time>/.
"""

import base64
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from ..sim.params import REPO_ROOT
from ..sim.parts import decide

B, DIM, GREEN, ORANGE, RST = "\033[1m", "\033[2m", "\033[38;5;78m", "\033[38;5;208m", "\033[0m"
OUT_DIR = REPO_ROOT / "outputs" / "inspect"
# in the look pose the part sits slightly above the centre of the 640x480 wrist image (sim: u ~307, v ~190)
CROP_CENTER, CROP_SIZE = (320, 200), 288

PROMPT = """You are the inspection camera of an end-of-line quality control cell. The image comes from a camera on the \
robot's wrist, looking straight down at one 3D-printed part lying in a tray pocket; the gripper jaws may show at the \
bottom edge. Inspect only the part in the middle of the image, ignore neighbouring pockets.

Part types and their defects:
- rim (small wheel rim, round with spokes): missing_hole = no hub hole in the centre; broken = one spoke missing.
- gear (spur gear with a raised hub): missing_hole = no shaft bore in the hub; broken = two adjacent teeth missing.
- pad (brake pad: rectangular friction block on a backing plate with an ear at each end): missing_hole = the ears \
have no mounting holes; broken = a chipped corner.
- bracket (L-shaped mounting bracket): missing_hole = no hole in the flat base; broken = a chipped corner.

variant is the colour: W = white, T = orange. Report what you see with the report_part tool."""

TOOL = {
    "name": "report_part",
    "description": "Attributes of the inspected part.",
    "input_schema": {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": ["rim", "gear", "pad", "bracket"]},
            "variant": {"type": "string", "enum": ["W", "T"], "description": "W = white, T = orange"},
            "missing_hole": {"type": "boolean"},
            "broken": {"type": "boolean"},
            "note": {"type": "string", "description": "one short sentence: what decided it"},
        },
        "required": ["type", "variant", "missing_hole", "broken", "note"],
    },
}


def center_crop(frame):
    """Same fixed crop for sim and real frames: the part's place in the wrist image in the look pose."""
    h, w = frame.shape[:2]
    s = min(CROP_SIZE, h, w)
    u0 = int(np.clip(CROP_CENTER[0] - s / 2, 0, w - s))
    v0 = int(np.clip(CROP_CENTER[1] - s / 2, 0, h - s))
    return frame[v0 : v0 + s, u0 : u0 + s]


def ask(client, model, rgb):
    """-> (attributes, seconds, usage). rgb: HxWx3 uint8."""
    import cv2

    ok, jpg = cv2.imencode(".jpg", rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
    t0 = time.perf_counter()
    msg = client.messages.create(
        model=model, max_tokens=400, tools=[TOOL], tool_choice={"type": "tool", "name": "report_part"},
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": base64.b64encode(jpg.tobytes()).decode()}},
            {"type": "text", "text": PROMPT},
        ]}],
    )
    seconds = time.perf_counter() - t0
    attrs = next(b.input for b in msg.content if b.type == "tool_use")
    return attrs, seconds, {"input": msg.usage.input_tokens, "output": msg.usage.output_tokens}


def sim_frames(n, seed):
    """n parts spread over type x colour x defect (training parts only), wrist frame in the look pose."""
    from ..sim.env import QCSortSim
    from ..sim.scripted import ScriptedSorter

    catalog = json.loads((REPO_ROOT / "python/flywheel/sim/assets/generated/catalog.json").read_text())
    groups = {}
    for p in catalog:
        if not p["heldout"]:
            groups.setdefault((p["type"], p["variant"], p["defect"]), []).append(p)
    rng = np.random.default_rng(seed)
    keys = list(groups)
    rng.shuffle(keys)
    for k in range(n):
        g = groups[keys[k % len(keys)]]
        p = g[int(rng.integers(len(g)))]
        env = QCSortSim(part_ids=[p["id"], None, None, None, None, None])
        s = ScriptedSorter(env)
        target, frame, width = s.plan_grasp(p["id"])
        look, look_frame = env.look_pose(target, frame)
        s.move_joints(np.r_[s.ik.solve(look, look_frame, q0=s.q[:5]).q, s.jaw_angle(width + 0.012)])
        s.hold(0.4)
        truth = {"type": p["type"], "variant": p["variant"],
                 "missing_hole": p["defect"] == "missing_hole", "broken": p["defect"] == "broken_corner"}
        yield p["id"], env.render("wrist"), truth


def camera_frame(index):
    import cv2

    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    for _ in range(15):  # let exposure settle
        ok, f = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"camera {index} gave no frame")
    yield f"camera{index}", f[..., ::-1].copy(), None


def image_frames(paths):
    import cv2

    for path in paths:
        f = cv2.imread(path)
        if f is None:
            sys.exit(f"cannot read {path}")
        yield Path(path).stem, f[..., ::-1].copy(), None


def run(args):
    import anthropic
    import cv2

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. In this terminal: export ANTHROPIC_API_KEY=... (console.anthropic.com)")
    client = anthropic.Anthropic()
    models = args.model or ["claude-sonnet-5-5"]
    frames = (sim_frames(args.sim, args.seed) if args.sim else camera_frame(args.camera) if args.camera is not None
              else image_frames(args.image))
    out = OUT_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"{B}work order {args.order}{RST} {DIM}({'white' if args.order == 'W' else 'orange'} is the product) · "
          f"{', '.join(models)} · {out}{RST}\n")
    print(f"   {'part':<9}{'model':<20}{'bin':<8}{'truth':<8}{'seconds':>8}  seen")
    for name, frame, truth in frames:
        crop = center_crop(frame)
        cv2.imwrite(str(out / f"{name}.jpg"), crop[..., ::-1])
        for model in models:
            try:
                attrs, sec, usage = ask(client, model, crop)
            except anthropic.APIError as e:
                print(f"   {name:<9}{model:<20}{ORANGE}{type(e).__name__}: {e}{RST}")
                continue
            bin_ = decide(attrs, args.order)
            want = decide(truth, args.order) if truth else None
            mark = "" if want is None else f"{GREEN}✓{RST}" if bin_ == want else f"{ORANGE}✗{RST}"
            seen = f"{attrs['type']} {attrs['variant']}{' no-hole' if attrs['missing_hole'] else ''}" \
                   f"{' broken' if attrs['broken'] else ''}"
            print(f"   {name:<9}{model:<20}{bin_.upper():<8}{(want or '-').upper():<8}{sec:8.2f}  {mark} {seen} "
                  f"{DIM}{attrs.get('note', '')}{RST}")
            rows.append({"part": name, "model": model, "bin": bin_, "truth_bin": want, "attributes": attrs,
                         "truth": truth, "seconds": round(sec, 3), "usage": usage})
    (out / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    print()
    for model in models:
        rs = [r for r in rows if r["model"] == model]
        if not rs:
            continue
        secs = np.array([r["seconds"] for r in rs])
        line = f"   {B}{model}{RST}: {len(rs)} requests · median {np.median(secs):.2f} s · max {secs.max():.2f} s"
        scored = [r for r in rs if r["truth_bin"]]
        if scored:
            right = sum(r["bin"] == r["truth_bin"] for r in scored)
            attrs_right = sum(r["attributes"][k] == r["truth"][k] for r in scored for k in r["truth"])
            line += f" · bin {right}/{len(scored)} · attributes {attrs_right}/{4 * len(scored)}"
        print(line)
