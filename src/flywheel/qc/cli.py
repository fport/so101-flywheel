"""flywheel-inspect: one top-camera frame of the tray -> 6 pocket crops -> part attributes -> a bin per pocket.

    uv run flywheel-inspect                                   # top camera, now
    uv run flywheel-inspect --parts J-W-03 B-W-07 - L-T-01 D-W-02 J-T-04   # score against the known parts
    uv run flywheel-inspect --image outputs/inspect/<time>/frame.png      # a saved frame

The arm is not involved: inspection happens once per tray, before the first pick, and the policy is then told "put the
part from pocket N in the X bin". Holes, rim spokes, gear teeth, broken corners and colour are measured from the
silhouette (qc.tray). With ANTHROPIC_API_KEY set, Claude names the part type and covers what the measurement could
not read; without it, give --parts so the type is known. OK/REJECT comes from decide() and the work order. Every
frame, crop and answer is kept in outputs/inspect/<time>/.
"""

import argparse
import base64
import json
import os
import sys
import time
from datetime import datetime

from ..params import REPO_ROOT, load_params
from ..parts import decide
from ..task import task

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


def _cell(attrs):
    """rim W · no hole · broken, for one table cell."""
    if attrs is None:
        return "-"
    return (f"{attrs.get('type', '?')} {attrs.get('variant', '?')}{' · no hole' if attrs.get('missing_hole') else ''}"
            f"{' · broken' if attrs.get('broken') else ''}")


def inspect_tray(frame, parts, args, out):
    """Tray frame -> 6 pocket crops -> measured attributes and Claude's, side by side; prints the table, returns the rows.

    The bin uses the measurement where it has one (holes, rim spokes, gear teeth, broken corners, colour) and Claude
    for the rest (the part type, and anything the measurement could not read). parts: 6 IDs in pocket order, '?' when
    unknown, '-' for an empty pocket; with IDs the crops are named after them and the catalog gives the truth.
    Raises RuntimeError when the tags are not found."""
    from concurrent.futures import ThreadPoolExecutor

    import cv2

    from .tray import find_tags, measure, overlay, pocket_crops, tray_homography, truth_for

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

    model = args.model
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
            print(f"   {name:<22}{ORANGE}{'empty pocket' if m else 'no part type: give --parts or set ANTHROPIC_API_KEY'}{RST}")
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
    known = [r for r in rows if r["bin"] != "?"]
    if known:
        print(f"\n{B}tell the policy{RST}")
        for r in known:
            print(f"   {task(r['pocket'], r['bin'])}")
    print(f"\n   overlay (are the orange squares on the pockets?): {out / 'overlay.jpg'}")
    return rows


def main():
    parser = argparse.ArgumentParser(prog="flywheel-inspect", description=__doc__.split("\n")[0])
    parser.add_argument("--top", type=int, default=1, help="top camera OpenCV index")
    parser.add_argument("--image", metavar="FILE", help="inspect a saved tray frame instead of the camera")
    parser.add_argument("--parts", nargs=6, metavar="ID",
                        help="the 6 part IDs in pocket order (1 = near-left), '-' for empty; scores the result")
    parser.add_argument("--order", choices=["W", "T"], default="W", help="variant in the work order")
    parser.add_argument("--model", default="claude-sonnet-5-5")
    parser.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"], help="lower is faster")
    parser.add_argument("--no-claude", action="store_true", help="measurement only")
    args = parser.parse_args()

    import cv2

    from .tray import capture

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            sys.exit(f"cannot read {args.image}")
        frame = frame[..., ::-1].copy()
    else:
        try:
            frame = capture(args.top)
        except RuntimeError as e:
            sys.exit(str(e))
    out = OUT_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    try:
        inspect_tray(frame, args.parts or ["?"] * 6, args, out)
    except RuntimeError as e:
        sys.exit(f"{ORANGE}{e}{RST}  frame: {out / 'frame.png'}")


if __name__ == "__main__":
    main()
