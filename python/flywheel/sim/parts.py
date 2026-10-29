"""QC parts catalog: 4 part types and a held-out eval set for automotive supplier end-of-line inspection.

Types: rim (J), gear (D), brake pad (B), mounting bracket (L).

The set is factorial so the model can't memorize a shortcut:
- Variant (color) x structure crossed: white (W) and orange (T) finishes are both valid products; missing-hole and
  broken defects exist in both colors. Color alone doesn't tell the bin.
- Whether a part is OK or REJECT depends on the work order, not the part (`Part.expected_bin(order)`): OK if the
  structure is sound AND the variant matches the order. With "this batch is white", a sound orange part is REJECT (mixed part).
- The ID carries no label (`J-W-03`): type, variant, index.

Each part's visual mesh is built with manifold3d, collision from primitive geoms; breaks are visual only.
Part frame: origin at the base center, +z up, part in its resting pose on the table.
"""

import json
import struct
from dataclasses import asdict, dataclass, field

import numpy as np
from manifold3d import CrossSection, Manifold, OpType

TYPES = {"rim": "J", "gear": "D", "pad": "B", "bracket": "L"}
TYPE_NAMES = {"rim": "rim", "gear": "gear", "pad": "pad", "bracket": "bracket"}
DEFECTS = ("missing_hole", "broken_corner")  # structural defects; "wrong variant" comes from the work order
VARIANTS = {"W": "white", "T": "orange"}  # order is the rng draw order in generation: don't change
SEGMENTS = 48
TIP_REACH = 12.2  # pad center to jaw tip 9.4 + extra overhang of the moving jaw tip 1.8 + margin 1.0


@dataclass
class Part:
    id: str  # e.g. J-W-03: type, variant, index
    type: str
    variant: str  # "W" | "T"
    defect: str | None  # structural defect; None = sound
    heldout: bool
    color: list[float]
    mesh: str  # STL path relative to generated/
    # collision geoms: {"type": "box"|"cylinder", "size": [...], "pos": [...]}, metres
    collision: list[dict] = field(default_factory=list)
    # grasp recipe for the scripted policy, in the part frame:
    # center = midpoint between the jaws, width = width the jaws close to, axis = closing direction,
    # symmetry = number of equivalent grasps about z (0 = rotationally symmetric, any direction works)
    grasp: dict = field(default_factory=dict)
    height: float = 0.0
    mass: float = 0.0

    def expected_bin(self, order):
        """order: variant in the work order ("W" | "T")."""
        return "ok" if self.defect is None and self.variant == order else "reject"

    def attributes(self):
        """Attributes the inspection classifier will predict (ground truth)."""
        return {"type": self.type, "variant": self.variant,
                "missing_hole": self.defect == "missing_hole", "broken": self.defect == "broken_corner"}


def decide(attributes, order):
    """Inspection decision: attributes + work order -> "ok" | "reject". The decision lives in this rule, not the model."""
    sound = not attributes["missing_hole"] and not attributes["broken"]
    return "ok" if sound and attributes["variant"] == order else "reject"


def _cyl(d, h, z0=0.0):
    return Manifold.cylinder(h, d / 2, d / 2, SEGMENTS).translate([0, 0, z0])


def _box(sx, sy, sz, x0, y0, z0):
    return Manifold.cube([sx, sy, sz]).translate([x0, y0, z0])


def _chip(rng, corner, size):
    """Randomly rotated cube seated on a corner: boolean subtraction leaves an irregular chip."""
    s = size * rng.uniform(0.8, 1.3)
    rot = rng.uniform(15, 75, size=3)
    return Manifold.cube([2 * s, 2 * s, 2 * s], True).rotate(list(rot)).translate(list(corner))


def _gear_profile(d, missing=()):
    """Root circle + trapezoid teeth; missing: tooth indices to skip."""
    r0, r1, n = d["root_d"] / 2, d["outer_d"] / 2, d["teeth"]
    w0, w1 = np.pi * d["root_d"] / n * 0.52, np.pi * d["outer_d"] / n * 0.30
    tooth = CrossSection([[(r0 - 0.5, -w0 / 2), (r1, -w1 / 2), (r1, w1 / 2), (r0 - 0.5, w0 / 2)]])
    teeth = [tooth.rotate(360 * i / n) for i in range(n) if i not in missing]
    return CrossSection.batch_boolean([CrossSection.circle(r0, SEGMENTS)] + teeth, OpType.Add)


def _shape(ptype, dims, defect, rng):
    """Returns (mesh, collision, grasp, height); lengths in mm."""
    hole = defect != "missing_hole"
    broken = defect == "broken_corner"
    corners, chip_size = [], 0.0
    if ptype == "rim":
        od, h, floor, n = dims["outer_d"], dims["height"], dims["floor"], dims["spokes"]
        hub_r, lip = dims["bore_d"] / 2 + 3.0, 2.6
        well = _cyl(od - 2 * lip, h, floor) - _cyl(2 * hub_r, h + 1, floor - 0.5)  # rim well
        gone = int(rng.integers(n)) if broken else -1  # defect: broken spoke
        spokes = [
            _box(od / 2 - lip + 0.4 - hub_r + 0.4, dims["spoke_w"], h - floor, hub_r - 0.4, -dims["spoke_w"] / 2, floor)
            .rotate([0, 0, 360 * i / n + 18])
            for i in range(n) if i != gone
        ]
        body = _cyl(od, h) - well
        for spoke in spokes:
            body += spoke
        if hole:
            body -= _cyl(dims["bore_d"], h + 2, -1)  # hub bore
        col = [{"type": "cylinder", "size": [od / 2, h / 2], "pos": [0, 0, h / 2]}]
        grasp = {"center": [0, 0, h / 2], "width": od, "axis": [1, 0, 0], "symmetry": 0}
        height = h
    elif ptype == "gear":
        t, hub, h = dims["thickness"], dims["hub_d"], dims["height"]
        missing = ()
        if broken:  # defect: two adjacent missing teeth
            k = int(rng.integers(dims["teeth"]))
            missing = (k, (k + 1) % dims["teeth"])
        body = _gear_profile(dims, missing).extrude(t) + _cyl(hub, h)
        if hole:
            body -= _cyl(dims["bore_d"], h + 2, -1)
        mean_d = (dims["outer_d"] + dims["root_d"]) / 2
        col = [
            {"type": "cylinder", "size": [mean_d / 2, t / 2], "pos": [0, 0, t / 2]},
            {"type": "cylinder", "size": [hub / 2, (h - t) / 2], "pos": [0, 0, t + (h - t) / 2]},
        ]
        # gripped by the hub; jaw tips must stay above the teeth
        grasp = {"center": [0, 0, t + TIP_REACH], "width": hub, "axis": [1, 0, 0], "symmetry": 0}
        height = h
    elif ptype == "pad":
        ln, w, plate, h = dims["length"], dims["width"], dims["plate"], dims["height"]
        bl, bw = dims["block"]
        body = _box(ln, w, plate, -ln / 2, -w / 2, 0) + _box(bl, bw, h - plate, -bl / 2, -bw / 2, plate)
        body -= _box(dims["groove"], bw + 2, (h - plate) * 0.45, -dims["groove"] / 2, -bw / 2 - 1, h - (h - plate) * 0.45)
        if hole:  # mounting holes in the backing plate ears
            ex = ln / 2 - dims["ear_hole_d"] / 2 - 0.9  # 0.9 mm wall between hole and plate edge
            for sx in (-1, 1):
                body -= _cyl(dims["ear_hole_d"], plate + 2, -1).translate([sx * ex, 0, 0])
        corners = [(sx * bl / 2, sy * bw / 2, h) for sx in (-1, 1) for sy in (-1, 1)]
        chip_size = bw * 0.24
        col = [
            {"type": "box", "size": [ln / 2, w / 2, plate / 2], "pos": [0, 0, plate / 2]},
            {"type": "box", "size": [bl / 2, bw / 2, (h - plate) / 2], "pos": [0, 0, plate + (h - plate) / 2]},
        ]
        # gripped by the friction block; jaw tips must stay above the plate (plate is wider than the block)
        grasp = {"center": [0, 0, plate + TIP_REACH], "width": bw, "axis": [0, 1, 0], "symmetry": 2}
        height = h
    elif ptype == "bracket":
        ln, w, t, h, hd = dims["length"], dims["width"], dims["thickness"], dims["height"], dims["hole_d"]
        base = _box(ln, w, t, -ln / 2, -w / 2, 0)
        up = _box(t, w, h, ln / 2 - t, -w / 2, 0)
        holes = [
            _cyl(hd, t + 2, -1).translate([-ln / 2 + (ln - t) / 2, 0, 0]),
            _cyl(hd, t + 2, -1).rotate([0, 90, 0]).translate([ln / 2 - t, 0, t + (h - t) / 2]),
        ]
        if defect == "missing_hole":
            rng.integers(2)  # old choice; still drawn so other parts' defects don't change
            holes.pop(0)  # hole in the base: the one in the upright faces sideways, hidden from the top camera and between the jaws
        body = base + up
        for hcut in holes:
            body -= hcut
        corners = [(ln / 2 - t / 2, sy * w / 2, h) for sy in (-1, 1)] + [(-ln / 2, sy * w / 2, t) for sy in (-1, 1)]
        chip_size = w * 0.2
        col = [
            {"type": "box", "size": [ln / 2, w / 2, t / 2], "pos": [0, 0, t / 2]},
            {"type": "box", "size": [t / 2, w / 2, (h - t) / 2], "pos": [ln / 2 - t / 2, 0, t + (h - t) / 2]},
        ]
        grasp = {"center": [ln / 2 - t / 2, 0, t + (h - t) * 0.6], "width": t, "axis": [1, 0, 0], "symmetry": 2}
        height = h
    else:
        raise ValueError(ptype)
    if broken and corners:
        body -= _chip(rng, corners[int(rng.integers(len(corners)))], chip_size)
    return body, col, grasp, height


def write_stl(path, manifold, scale=1.0):
    mesh = manifold.to_mesh()
    v = np.asarray(mesh.vert_properties, dtype=np.float32)[:, :3] * scale
    tri = v[np.asarray(mesh.tri_verts)]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    data = np.zeros(len(tri), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    data["n"], data["v"] = n, tri
    with open(path, "wb") as f:
        f.write(b"so101-flywheel".ljust(80, b"\0"))
        f.write(struct.pack("<I", len(tri)))
        f.write(data.tobytes())


def generate(p):
    """Generates (Part, Manifold in mm) for each catalog part. Sim and print use the same geometry."""
    cfg = p["parts"]
    rng = np.random.default_rng(cfg["seed"])
    density = 1240 * 0.35  # PLA, ~35% infill print (kg/m3)
    out = []
    for t_index, (ptype, prefix) in enumerate(TYPES.items()):
        for variant in VARIANTS:
            counts = cfg["train"][variant]
            plan = [(None, False)] * counts["good"]
            for defect in DEFECTS:
                plan += [(defect, False)] * counts[defect]
            # held-out: one sound part per color per type + defects alternated across colors
            held_defect = DEFECTS[(t_index + list(VARIANTS).index(variant)) % len(DEFECTS)]
            plan += [(None, True), (held_defect, True)]
            for i, (defect, heldout) in enumerate(plan):
                pid = f"{prefix}-{variant}-{i + 1:02d}"
                body, col, grasp, height = _shape(ptype, cfg[ptype], defect, rng)
                # slight tint variation within a color too: so individual parts aren't memorized
                base = np.array(cfg["colors"][variant])
                color = np.clip(base * rng.uniform(0.92, 1.08) + rng.uniform(-0.02, 0.02, size=3), 0, 1)
                part = Part(
                    id=pid,
                    type=ptype,
                    variant=variant,
                    defect=defect,
                    heldout=heldout,
                    color=[round(float(c), 3) for c in color],
                    mesh=f"parts/{pid}.stl",
                    collision=[
                        {"type": c["type"], "size": [s / 1000 for s in c["size"]], "pos": [s / 1000 for s in c["pos"]]}
                        for c in col
                    ],
                    grasp={
                        "center": [s / 1000 for s in grasp["center"]],
                        "width": grasp["width"] / 1000,
                        "axis": grasp["axis"],
                        "symmetry": grasp["symmetry"],
                    },
                    height=height / 1000,
                    mass=round(float(body.volume()) * 1e-9 * density, 5),
                )
                out.append((part, body))
    return out


def build_catalog(p, out_dir):
    """Writes sim STLs (metres) for all parts under out_dir/parts, returns the catalog list."""
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    catalog = []
    for part, body in generate(p):
        write_stl(parts_dir / f"{part.id}.stl", body, scale=1e-3)
        catalog.append(part)
    return catalog


def save_catalog(catalog, path):
    path.write_text(json.dumps([asdict(c) for c in catalog], indent=1))


def load_catalog(path):
    return [Part(**d) for d in json.loads(path.read_text())]
