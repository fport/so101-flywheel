"""Builds the tracking site's data from the repo: public/data.js.

    python/.venv/bin/python site/export_scene.py   # scene and sim playback (only when the sim changes)
    python/.venv/bin/python site/build.py && (cd site && npx wrangler deploy)

The only hand-written file is progress.json (step statuses, en/tr log); parts catalog, plans, sessions, recording progress,
cell dimensions, tag patterns, calibration and versions are read from the repo so the site stays in sync with the code.
"""

import json
import re
import sys
from pathlib import Path

import cv2
import yaml

SITE = Path(__file__).resolve().parent
REPO = SITE.parent
sys.path.insert(0, str(REPO / "python"))
from flywheel.robot.recorder import DATASETS_DIR, recorded  # noqa: E402
from flywheel.robot.session import sessions  # noqa: E402

RECORD_PLANS = {"record-W-seed0": "fport/so101-qc", "record-T-seed0": "fport/so101-qc"}
PUBLIC = SITE / "public"
LOCKED = ["lerobot", "mujoco", "torch", "torchvision", "numpy", "opencv-python-headless", "draccus",
          "feetech-servo-sdk", "manifold3d", "huggingface-hub", "pytest"]


def locked_versions():
    lock = (REPO / "python" / "uv.lock").read_text()
    found = dict(re.findall(r'\[\[package\]\]\nname = "([^"]+)"\nversion = "([^"]+)"', lock))
    return {name: found[name] for name in LOCKED if name in found}


def progress():
    """progress.json; every text it shows on the site (log entries, now_text) is {"en": ..., "tr": ...}."""
    p = json.loads((SITE / "progress.json").read_text())
    for text in [p["now_text"], *(entry["text"] for entry in p["log"])]:
        if not (isinstance(text, dict) and text.get("en") and text.get("tr")):
            sys.exit(f"progress.json: text needs both en and tr: {text!r}")
    return p


def catalog():
    parts = json.loads((REPO / "python/flywheel/sim/assets/generated/catalog.json").read_text())
    return [{k: p[k] for k in ("id", "type", "variant", "defect", "heldout")} for p in parts]


def geometry():
    """Subset of params.yaml for the cell drawing (mm, arm frame: +x forward, +y to the arm's left)."""
    p = yaml.safe_load((REPO / "hardware/params.yaml").read_text())
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    tags = [{**t, "bits": (cv2.aruco.generateImageMarker(dictionary, t["id"], 8) // 255).tolist()}
            for t in p["tags"]["items"]]
    return {
        "grid": p["grid"], "tray": p["tray"], "bins": p["bins"], "inspection": p["inspection"],
        "mast": {"pos": p["mast"]["pos"], "size": p["mast"]["size"]}, "tags": tags,
        "camera": {"height": p["measured"]["cam_top_height"], "tilt": p["mast"]["head"]["tilt"]},
    }


def recording():
    """Sessions and actually recorded plan rows for each recording plan (outputs/datasets/.../episodes.jsonl)."""
    out = {}
    for plan, repo in RECORD_PLANS.items():
        rows = recorded(DATASETS_DIR / repo.split("/")[-1])
        out[plan] = {"repo": repo, "sessions": sessions(plan),
                     "recorded": sorted({r["index"] for r in rows if r["plan"] == plan})}
    return out


def main():
    data = {
        "progress": progress(),
        "parts": catalog(),
        "plans": {f.stem: json.loads(f.read_text()) for f in sorted((REPO / "plans").glob("*.json"))},
        "calibration": {f.stem: json.loads(f.read_text())
                        for f in sorted((REPO / "hardware/calibration").glob("*.json"))},
        "geom": geometry(),
        "recording": recording(),
        "versions": locked_versions(),
    }
    js = "window.FLYWHEEL = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n"
    (PUBLIC / "data.js").write_text(js)
    print(f"data.js {len(js) // 1024} KB · {len(data['parts'])} parts · {len(data['plans'])} plans · "
          f"{len(data['geom']['tags'])} tag")


if __name__ == "__main__":
    main()
