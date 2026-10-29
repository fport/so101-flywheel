"""hardware/params.yaml reader. The file holds mm/degrees, the sim wants metres/radians."""

from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
PARAMS_YAML = REPO_ROOT / "hardware" / "params.yaml"
ASSETS = Path(__file__).resolve().parent / "assets"
GENERATED = ASSETS / "generated"
SO101_XML = ASSETS / "so101" / "so101_new_calib.xml"
SCENE_XML = GENERATED / "scene.xml"
CATALOG_JSON = GENERATED / "catalog.json"


def load_params(path=PARAMS_YAML):
    return yaml.safe_load(Path(path).read_text())


def mm(v):
    """mm -> m; number or list."""
    return np.asarray(v, dtype=float) / 1000.0


def pocket_centers(p):
    """Tray pocket centers (m), in pocket-number order. Pocket 1 = row nearest the arm, left column."""
    t = p["tray"]
    cx, cy = mm(t["center"])
    sx, sy = mm(t["size"])
    px, py = sx / t["rows"], sy / t["cols"]
    out = []
    for r in range(t["rows"]):
        for c in range(t["cols"]):
            out.append((cx - sx / 2 + (r + 0.5) * px, cy + sy / 2 - (c + 0.5) * py))
    return out


def pocket_floor_z(p):
    return float(mm(p["tray"]["floor"]))


MAST_BOSS = 50.0  # height of the mast socket on the baseplate tile (mm)
CRADLE_GAP = 0.4  # gap between camera body and cradle wall (mm)


def mast_top(p):
    return MAST_BOSS + p["mast"]["segments"] * p["mast"]["segment"]


def head_geometry(p):
    """Camera head hinge dimensions (mm). Sim and printed parts use the same numbers.

    sleeve: half outer size of the sleeve that slides over the mast. axle: hinge axis position relative to the mast
    axis (x) and mast top (z). hub: hinge hub position (x, z) in the cradle frame (lens at origin, camera face down).
    index_r / index_phi: lock pin distance from the axis and direction of the fixed hole in the head (x to z, degrees).
    """
    body = p["cameras"]["top"]["webcam"]["body"]
    sleeve = p["mast"]["size"] / 2 + p["fit"]["clearance"] + 3
    return {"sleeve": sleeve, "axle": (-(sleeve + 6), 30.0), "hub": (body[1] / 2 + CRADLE_GAP + 14, body[2] + 16),
            "index_r": 40.0, "index_phi": 50.0}


def top_camera_pose(p, tilt=None):
    """Top camera (lens position, unit view direction), mm, in the scene frame."""
    m, g = p["mast"], head_geometry(p)
    t = np.radians(m["head"]["tilt"] if tilt is None else tilt)
    hx, hz = g["hub"]
    x = -hx * np.cos(t) - hz * np.sin(t)  # hub-to-lens vector, tilted
    z = hx * np.sin(t) - hz * np.cos(t)
    pos = np.array([m["pos"][0] + g["axle"][0] + x, m["pos"][1], mast_top(p) + g["axle"][1] + z])
    return pos, np.array([-np.sin(t), 0.0, -np.cos(t)])
