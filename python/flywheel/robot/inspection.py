"""Inspection: one top-camera frame of the tray -> 6 pocket crops -> part attributes -> decide() -> bin per pocket.

    uv run flywheel-robot inspect --tray --parts J-W-03 B-W-07 - L-T-01 D-W-02 J-T-04   # top camera, now
    uv run flywheel-robot inspect --tray --image .../trays/record-W-seed0-tray01.png      # a frame the recorder kept
    uv run flywheel-robot inspect --live                    # teleop; → inspects the tray in front of the arm
    uv run flywheel-robot inspect --image pocket2_B-W-07.png  # Claude on saved crops; a part ID in the name scores it
    uv run flywheel-robot inspect --sim 24 --model claude-haiku-5-5 --model claude-sonnet-5-5  # sim crops, compare models

The arm is not involved: inspection happens once per tray, before the first pick, and the policy is then told "put the
part from pocket N in the X bin". Holes, rim spokes, gear teeth, broken corners and colour are measured from the
silhouette (robot.tray); Claude reports what it sees (type, colour, missing hole, broken) and covers what the
measurement does not. OK/REJECT comes from decide() and the work order, like it will with any other classifier. Every
frame, answer and timing is kept in outputs/inspect/<time>/.
"""

import base64
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from ..sim.params import REPO_ROOT, load_params
from ..sim.parts import decide

B, DIM, GREEN, ORANGE, RST = "\033[1m", "\033[2m", "\033[38;5;78m", "\033[38;5;208m", "\033[0m"
OUT_DIR = REPO_ROOT / "outputs" / "inspect"
_DIMS = load_params()["parts"]  # sound-part counts come from the same file as the printed parts
PROMPT = f"""You are the inspection camera of an end-of-line quality control cell. The image looks straight down at one \
3D-printed part lying in a pocket of a matte black tray (cut out of the top camera's frame of the whole tray; the crop is \
small and enlarged, so edges are soft). Inspect only the part in the middle of the image, ignore neighbouring pockets and markers.

How to see a hole: every hole goes all the way through the part, so the black tray shows through it as a dark, round \
spot with a crisp edge. A light or grey surface with a shaded ring around it is a solid face, not a hole. Look for the \
black spot itself; if the place where the hole belongs is the same light colour as the part, the hole is missing.

Part types, what a sound part looks like, and the defects:
- rim (small wheel rim, {_DIMS["rim"]["outer_d"]} mm): a sound rim has exactly {_DIMS["rim"]["spokes"]} spokes from a \
hub to the outer ring, and a {_DIMS["rim"]["bore_d"]} mm hub hole (about a fifth of the rim's width) in the centre of \
the hub. The windows between the spokes have a solid floor, so they look grey and shaded, not black; they are not \
holes. missing_hole = the hub centre is solid, no black spot; broken = one spoke missing, so only \
{_DIMS["rim"]["spokes"] - 1} spokes and one window twice as wide. Count the spokes.
- gear (spur gear, {_DIMS["gear"]["outer_d"]} mm, with a raised round hub in the middle): a sound gear has \
{_DIMS["gear"]["teeth"]} evenly spaced teeth and a {_DIMS["gear"]["bore_d"]} mm shaft bore, a black spot in the \
centre of the hub's flat top. missing_hole = the hub top is one solid light disc (its rim may look shaded); broken = \
two adjacent teeth missing, a gap in the ring of teeth about three teeth wide.
- pad (brake pad, {_DIMS["pad"]["length"]} x {_DIMS["pad"]["width"]} mm: a raised rectangular friction block on a \
slightly larger backing plate whose two short ends stick out as ears): a sound pad has a small \
{_DIMS["pad"]["ear_hole_d"]} mm mounting hole in each ear and a block with four square corners. One straight groove \
across the middle of the block is normal, not a crack. missing_hole = neither ear has a black spot; broken = one \
corner cut off through the whole pad, so the outline has a slanted edge there instead of a square corner.
- bracket (L-shaped mounting bracket): from above you see a flat base and, at one end, the top of the upright as a \
narrow strip. A sound bracket has a {_DIMS["bracket"]["hole_d"]} mm hole in the base and square corners. \
missing_hole = no hole in the base; broken = one corner at the free end of the base (away from the upright) cut off \
through the whole part, so the outline has a slanted edge there instead of a square corner.

variant is the colour: W = white, T = orange. Fill look first: what you see at each place that decides the part \
(the hole: black spot or solid; the spoke or tooth count; each corner), then the fields that follow from it. note is \
one short sentence on what decided it."""

SCHEMA = {  # structured output (output_config.format); forced tool_choice is not accepted by current models
    "type": "object",
    "properties": {  # look comes first: the observations are written before the verdicts that depend on them
        "look": {"type": "string"},
        "type": {"type": "string", "enum": ["rim", "gear", "pad", "bracket"]},
        "variant": {"type": "string", "enum": ["W", "T"]},
        "missing_hole": {"type": "boolean"},
        "broken": {"type": "boolean"},
        "note": {"type": "string"},
    },
    "required": ["look", "type", "variant", "missing_hole", "broken", "note"],
    "additionalProperties": False,
}
UPSCALE_TO = 768  # a 224 px crop is ~70 image tokens; enlarged, the model gets more patches on a 7 mm hole


def ask(client, model, rgb, effort=None):
    """-> (attributes, seconds, usage). rgb: HxWx3 uint8; effort: low … max, None = the model's default."""
    import cv2

    if max(rgb.shape[:2]) < UPSCALE_TO:
        f = UPSCALE_TO / max(rgb.shape[:2])
        rgb = cv2.resize(rgb, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    ok, jpg = cv2.imencode(".jpg", rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
    t0 = time.perf_counter()
    config = {"format": {"type": "json_schema", "schema": SCHEMA}}
    if effort:
        config["effort"] = effort
    msg = client.messages.create(
        model=model, max_tokens=4000, output_config=config,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": base64.b64encode(jpg.tobytes()).decode()}},
            {"type": "text", "text": PROMPT},
        ]}],
    )
    seconds = time.perf_counter() - t0
    attrs = json.loads(next(b.text for b in msg.content if b.type == "text"))
    return attrs, seconds, {"input": msg.usage.input_tokens, "output": msg.usage.output_tokens}


def sim_frames(n, seed):
    """n parts spread over type x colour x defect (training parts only), laid 6 to a tray; pocket crops from the sim top
    camera at the inspection resolution, cut out by the tags like a real frame."""
    from ..sim.env import QCSortSim
    from .tray import find_tags, pocket_crops, tray_homography, truth_for

    catalog = json.loads((REPO_ROOT / "python/flywheel/sim/assets/generated/catalog.json").read_text())
    groups = {}
    for p in catalog:
        if not p["heldout"]:
            groups.setdefault((p["type"], p["variant"], p["defect"]), []).append(p)
    rng = np.random.default_rng(seed)
    keys = list(groups)
    rng.shuffle(keys)
    ids = []
    for k in range(n):
        g = [p["id"] for p in groups[keys[k % len(keys)]] if p["id"] not in ids]
        ids.append(g[int(rng.integers(len(g)))])
    for start in range(0, n, 6):
        tray = ids[start : start + 6]
        frame = QCSortSim(part_ids=tray + [None] * (6 - len(tray)), seed=seed + start).tray_frame()
        for (k, crop), pid in zip(pocket_crops(frame, tray_homography(find_tags(frame))), tray):
            yield f"pocket{k}_{pid}", crop, truth_for(pid)


def image_frames(paths):
    import cv2

    from .tray import truth_for

    for path in paths:
        f = cv2.imread(path)
        if f is None:
            sys.exit(f"cannot read {path}")
        yield Path(path).stem, f[..., ::-1].copy(), truth_for(Path(path).stem)


def run(args):
    import anthropic
    import cv2

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. In this terminal: export ANTHROPIC_API_KEY=... (console.anthropic.com)")
    client = anthropic.Anthropic()
    models = args.model or ["claude-sonnet-5-5"]
    frames = sim_frames(args.sim, args.seed) if args.sim else image_frames(args.image)
    out = OUT_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"{B}work order {args.order}{RST} {DIM}({'white' if args.order == 'W' else 'orange'} is the product) · "
          f"{', '.join(models)} · effort {args.effort or 'default'} · {out}{RST}\n")
    print(f"   {'part':<9}{'model':<20}{'bin':<8}{'truth':<8}{'seconds':>8}  seen")
    for name, crop, truth in frames:
        name = name[:20]
        cv2.imwrite(str(out / f"{name}.jpg"), crop[..., ::-1])
        for model in models:
            try:
                attrs, sec, usage = ask(client, model, crop, args.effort)
            except (anthropic.APIError, StopIteration, json.JSONDecodeError) as e:
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


def live(args):
    """Teleop with tray inspection on demand: lay out the tray, park the arm out of the top camera's way, → inspects
    the 6 pockets from the top camera frame and prints a bin for each while the arm keeps following the leader; the
    operator then puts each part in its bin. Nothing is recorded to a dataset; frames and answers go to
    outputs/inspect/<time>/."""
    from concurrent.futures import ThreadPoolExecutor

    from lerobot.utils.robot_utils import precise_sleep
    from lerobot.utils.visualization_utils import init_visualization, log_visualization_data, shutdown_visualization

    from .recorder import CONTROL_MULT, Keys, Watchdog, connect_robot, make_rig, say

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. In this terminal: export ANTHROPIC_API_KEY=... (console.anthropic.com)")
    model = (args.model or ["claude-sonnet-5-5"])[0]
    out = OUT_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    pool = ThreadPoolExecutor(max_workers=1)
    if args.display:
        init_visualization("rerun", session_name="flywheel_inspect")
    robot, teleop = make_rig(args)
    try:
        connect_robot(robot)
    except (TimeoutError, ConnectionError) as e:
        sys.exit(f"{ORANGE}{e}{RST}\nCheck the cameras (--top/--wrist) and the follower's 12 V adapter.")
    teleop.connect()
    wd = Watchdog(out / "loop.log")
    print(f"\n{B}live inspection{RST} · work order {args.order} ({'white' if args.order == 'W' else 'orange'} is the "
          f"product) · {model} · effort {args.effort or 'default'}\n"
          f"   lay out the tray, bring the arm back to start (out of the top camera's view), then\n"
          f"   {B}→{RST} (works in any window, Rerun too) or ENTER (only while this terminal has focus)\n"
          f"   a bin for each pocket is printed; put the parts there yourself · Esc / q to quit · {DIM}{out}{RST}\n"
          f"   {DIM}loop log: {wd.path} (pid {os.getpid()}){RST}\n")
    hz = args.fps * CONTROL_MULT
    pending, n, done = None, 0, 0
    try:
        with Keys("→ / ENTER inspect the tray · Esc / q quit") as keys:
            tick, worst, beat = 0, 0.0, time.perf_counter()
            while True:
                t0 = time.perf_counter()
                wd.tick()
                tick += 1
                if t0 - beat >= 1:  # once a second: how fast the arm is really being driven
                    wd.f.write(f"{datetime.now():%H:%M:%S} {tick / (t0 - beat):5.1f} Hz · slowest step "
                               f"{worst * 1000:.0f} ms{' · inspecting' if pending else ''}\n")
                    wd.f.flush()
                    tick, worst, beat = 0, 0.0, t0
                obs = robot.get_observation()
                act = teleop.get_action()
                robot.send_action(act)
                if args.display and tick % CONTROL_MULT == 0:
                    log_visualization_data("rerun", observation=obs, action=act)

                key = keys.poll()
                if key is not None:
                    print(f"   {DIM}key {keys.last}{RST}")
                if key == "q":
                    break
                if key == "go":
                    if pending is not None:
                        print(f"   {DIM}still inspecting the previous tray{RST}")
                    else:
                        n += 1
                        frame = np.asarray(obs["top"]).copy()
                        pending = (n, pool.submit(inspect_tray, frame, args.parts or ["?"] * 6, args, out / f"tray{n:02d}"))
                        print(f"   tray #{n}: measuring and asking Claude …", flush=True)

                if pending is not None and pending[1].done():
                    k, fut = pending
                    pending = None
                    try:
                        rows = fut.result()
                    except Exception as e:  # tags not found, API error: tell, keep driving
                        print(f"   tray #{k} {ORANGE}{type(e).__name__}: {e}{RST}")
                    else:
                        done += 1
                        say(", ".join(f"{r['pocket']} {r['bin']}" for r in rows), args.voice)
                worst = max(worst, time.perf_counter() - t0)
                precise_sleep(max(1 / hz - (time.perf_counter() - t0), 0.0))
    except KeyboardInterrupt:
        pass
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        for dev in (robot, teleop):
            if dev.is_connected:
                dev.disconnect()
        if args.display:
            shutdown_visualization("rerun")
    print(f"\n   {done} trays inspected · {out}" if done else "\n   no tray was inspected: press → with the tray laid out")


def _cell(attrs):
    """rim W · no hole · broken, for one table cell."""
    if attrs is None:
        return "-"
    return (f"{attrs.get('type', '?')} {attrs.get('variant', '?')}{' · no hole' if attrs.get('missing_hole') else ''}"
            f"{' · broken' if attrs.get('broken') else ''}")


def tray_run(args):
    """One top-camera frame of the tray (--top now, or a saved frame with --image) -> inspect_tray()."""
    import cv2

    from .tray import capture

    parts = args.parts or ["?"] * 6
    if len(parts) != 6:
        sys.exit(f"--parts takes 6 IDs in pocket order (1 = near-left), '-' for an empty pocket; got {len(parts)}")
    if args.image:
        frame = cv2.imread(args.image[0])
        if frame is None:
            sys.exit(f"cannot read {args.image[0]}")
        frame = frame[..., ::-1].copy()
    else:
        try:
            frame = capture(args.top)
        except RuntimeError as e:
            sys.exit(str(e))
    out = OUT_DIR / "tray" / datetime.now().strftime("%Y%m%d-%H%M%S")
    try:
        inspect_tray(frame, parts, args, out)
    except RuntimeError as e:
        sys.exit(f"{ORANGE}{e}{RST}  frame: {out / 'frame.png'}")


def inspect_tray(frame, parts, args, out):
    """Tray frame -> 6 pocket crops -> measured attributes and Claude's, side by side; prints the table, returns the rows.

    The bin uses the measurement where it has one (holes, rim spokes, gear teeth, broken corners, colour) and Claude
    for the rest (the part type, and anything the measurement could not read). parts: 6 IDs in pocket order, '?' when
    unknown, '-' for an empty pocket; with IDs the crops are named after them, so the catalog gives the truth and
    `inspect --image` can score them again later. Raises RuntimeError when the tags are not found."""
    import cv2
    from concurrent.futures import ThreadPoolExecutor

    from .tray import find_tags, overlay, pocket_crops, measure, tray_homography, truth_for

    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "frame.png"), frame[..., ::-1])
    tags = find_tags(frame)
    H = tray_homography(tags)
    cv2.imwrite(str(out / "overlay.jpg"), overlay(frame, H, tags))

    crops = []
    for (k, crop), pid in zip(pocket_crops(frame, H), parts):
        if pid == "-":
            continue
        name = f"pocket{k}" + (f"_{pid}" if pid != "?" else "")
        cv2.imwrite(str(out / f"{name}.png"), crop[..., ::-1])
        crops.append((k, name, crop, truth_for(pid)))

    model = (args.model or ["claude-sonnet-5-5"])[0]
    asked = {}
    if os.environ.get("ANTHROPIC_API_KEY") and not args.no_claude:
        import anthropic

        client = anthropic.Anthropic()
        with ThreadPoolExecutor(max_workers=6) as pool:  # all six at once: the tray waits for the slowest
            futs = {k: pool.submit(ask, client, model, crop, args.effort) for k, _, crop, _ in crops}
            for k, f in futs.items():
                try:
                    asked[k] = f.result()
                except Exception as e:  # API error or unreadable answer: the measurement still stands
                    print(f"   pocket {k}: {ORANGE}{type(e).__name__}: {e}{RST}")
    elif not args.no_claude:
        print(f"   {DIM}ANTHROPIC_API_KEY is not set: measurement only, part types from --parts{RST}")

    print(f"{B}tray{RST} · work order {args.order} · tags {sorted(tags)} · {model if asked else 'no Claude'} · {out}\n")
    print(f"   {'pocket':<22}{'truth':<26}{'measured':<26}{'Claude':<26}{'bin':<8}")
    rows, score = [], {"measured": [0, 0], "claude": [0, 0], "bin": [0, 0]}
    for k, name, crop, truth in crops:
        claude, sec = (asked[k][0], asked[k][1]) if k in asked else (None, None)
        ptype = (truth or claude or {}).get("type")
        m = measure(crop, ptype) if ptype else None
        if m is None or m.get("empty"):
            print(f"   {name:<22}{ORANGE}{'empty pocket' if m else 'no part type: give --parts or an API key'}{RST}")
            continue
        seen = {"type": ptype, "variant": m["variant"]}
        for key in ("missing_hole", "broken"):
            seen[key] = m[key] if m[key] is not None else (claude or {}).get(key)
        measured = {"type": ptype, "variant": m["variant"], "missing_hole": m["missing_hole"], "broken": m["broken"]}
        # an unknown attribute only matters while everything known still says OK
        known_reject = seen["variant"] != args.order or seen["missing_hole"] or seen["broken"]
        bin_ = decide(seen, args.order) if None not in seen.values() else "reject" if known_reject else "?"
        want = decide(truth, args.order) if truth else None
        mark = "" if want is None else f" {GREEN}✓{RST}" if bin_ == want else f" {ORANGE}✗{RST}"
        print(f"   {name:<22}{_cell(truth):<26}{_cell(measured):<26}{_cell(claude):<26}{bin_.upper():<6}{mark}")
        nums = {key: m[key] for key in ("spokes", "teeth", "max_gap_deg", "ear_holes") if key in m}
        print(f"   {'':<22}{DIM}{nums or ''} {(claude or {}).get('note', '')}{RST}")
        if truth:
            for key, src in (("measured", measured), ("claude", claude)):
                if src:
                    checks = [src.get(a) == truth[a] for a in ("variant", "missing_hole", "broken") if src.get(a) is not None]
                    score[key][0] += sum(checks)
                    score[key][1] += len(checks)
            score["bin"][0] += bin_ == want
            score["bin"][1] += 1
        rows.append({"pocket": k, "name": name, "truth": truth, "measured": m, "claude": claude, "seconds": sec,
                     "bin": bin_, "truth_bin": want})
    (out / "results.jsonl").write_text("".join(json.dumps(r, default=float) + "\n" for r in rows))
    if any(n for _, n in score.values()):
        print("\n   " + " · ".join(f"{key} {r}/{n}" for key, (r, n) in score.items() if n) +
              f"  {DIM}(attributes: colour, hole, broken where each one gave an answer){RST}")
    print(f"\n   overlay (are the orange squares on the pockets?): {out / 'overlay.jpg'}")
    return rows
