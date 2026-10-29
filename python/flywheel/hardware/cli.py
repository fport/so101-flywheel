"""flywheel-cad: build the print plates from hardware/params.yaml.

Output is numbered 3MF files under `hardware/print/`: each file is a single print job (one filament, laid out on the
plate), the number is the print order. `contents.md` lists what is in each file.
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
from manifold3d import Error

from ..sim.params import REPO_ROOT, load_params
from ..sim.parts import generate, write_stl
from .catalog_svg import build_catalog_svg
from .diagram import build_svg
from .fixtures import build_all
from .geom import union, write_3mf

OUT = REPO_ROOT / "hardware"
MARGIN, GAP = 8.0, 5.0

# (file name, filament, part names) in print order. QC plates are appended at the end.
JOBS = [
    ("test", "black", ["cam_cradle_fit_test", "mast_joint_test_socket", "mast_joint_test_plug"]),
    ("camera_head", "black", ["mast_head", "cam_cradle_c310", "head_axle_pin", "head_lock_pin"]),
    ("baseplate_A2_arm_seat", "black", ["baseplate_A2", "baseplate_key"]),
    ("baseplate_B2_mast_socket", "black", ["baseplate_B2"]),
    ("mast", "metal grey", ["mast_segment"]),
    ("tray_backdrop_single_color", "black", ["tray", "inspection_backdrop"]),
    ("baseplate_A1_and_liners", "black", ["baseplate_A1", "bin_liner"]),
    ("baseplate_A3", "black", ["baseplate_A3"]),
    ("bin_ok", "metal grey", ["bin_ok"]),
    ("tag_tiles", "black + white (two nozzles)", ["tag_0", "tag_1", "tag_2", "tag_3"]),
]
# QC parts that failed and were skipped: added to the last job of the same color, to come off the same spool
REPRINTS = ["D-T-04"]  # first layer didn't stick on plate 13, skipped
EXTRA_JOBS = [  # come after the QC plates
    # the orange spool goes to the parts first: all parts of one color must come off the same spool
    ("bin_reject_and_D-T-04", "orange", ["bin_reject", *REPRINTS]),
    ("optional_mast_foot", "metal grey", ["mast_foot"]),
    ("optional_cable_clips", "any", ["cable_clip", "usb_strain_relief"]),
]
QC_JOBS = {
    ("train", "W"): ("parts_white", "white"),
    ("train", "T"): ("parts_orange", "orange"),
    ("heldout", "W"): ("heldout_white", "white"),
    ("heldout", "T"): ("heldout_orange", "orange"),
}


def _bbox(m):
    b = m.bounding_box()
    return np.array(b[:3]), np.array(b[3:])


def to_bed(item, turn=False):
    """Drop the part onto the bed: centered in xy, lowest point at z=0. turn: rotate 90 degrees about z."""
    parts = [part.rotate([0, 0, 90]) for part in item.parts] if turn else item.parts
    lo, hi = _bbox(union(parts) if turn else item.body)
    shift = [-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]]
    return [part.translate(shift) for part in parts], hi - lo


def check_fit(items):
    """Do parts intersect each other in their assembly positions? Returns the intersection volumes (mm3)."""
    placed = [(i.name, i.body.translate([*i.at, 0]), i.group) for i in items if i.at and "z" not in i.meta]
    plate = union(m for _, m, g in placed if g == "baseplate")
    fixtures = [(n, m) for n, m, g in placed if g != "baseplate"]
    clashes = {}
    for k, (name, m) in enumerate(fixtures):
        v = (m ^ plate).volume()
        if v > 1.0:
            clashes[f"{name} x baseplate"] = round(v, 1)
        for other, mo in fixtures[k + 1 :]:
            v = (m ^ mo).volume()
            if v > 1.0:
                clashes[f"{name} x {other}"] = round(v, 1)
    return clashes


def pack(members, bed, xr):
    """Shelf packing: [(name, parts, (x, y, 0))]. x uses only the xr range. ValueError if it doesn't fit."""
    entries = []
    for item in members:
        parts, size = to_bed(item)
        turned = to_bed(item, turn=True)
        if size[0] > xr[1] - xr[0] >= size[1]:  # doesn't fit the shared area crosswise, fits lengthwise
            parts, size = turned
            turned = None
        entries += [(item.name if item.qty == 1 else f"{item.name}_{k + 1}", parts, size, turned) for k in range(item.qty)]
    entries.sort(key=lambda e: (-e[2][0] * e[2][1], -e[2][1]))  # large parts first
    x, y = xr[0], MARGIN
    row = 0.0
    out = []
    for name, parts, size, turned in entries:
        if x + size[0] > xr[1] and x > xr[0]:
            x, y, row = xr[0], y + row + GAP, 0.0
        if turned is not None and y + size[1] > bed[1] - MARGIN and y + turned[1][1] <= bed[1] - MARGIN:
            parts, size = turned  # doesn't fit the remaining depth lengthwise, fits when turned
            if x + size[0] > xr[1] and x > xr[0]:
                x, y, row = xr[0], y + row + GAP, 0.0
        if x + size[0] > xr[1] or y + size[1] > bed[1] - MARGIN or size[2] > bed[2]:
            raise ValueError(f"{name} doesn't fit the plate ({np.round(size, 1)})")
        out.append((name, parts, (x + size[0] / 2, y + size[1] / 2, 0.0)))
        x += size[0] + GAP
        row = max(row, size[1])
    return out


def qc_plates(items, bed, xr, pitch=(42.0, 38.0)):
    """Lays QC parts out on a grid by filament and train / held-out split: [(name, filament, objects)]."""
    groups = {}
    for item in (i for i in items if i.group == "qc"):
        groups.setdefault(("heldout" if item.meta["heldout"] else "train", item.meta["variant"]), []).append(item)
    cols, rows = int((xr[1] - xr[0]) // pitch[0]), int((bed[1] - 20) // pitch[1])
    out = []
    for key, (stem, filament) in QC_JOBS.items():
        members = groups.get(key, [])
        n_plates = -(-len(members) // (cols * rows))
        for n in range(n_plates):
            # distribute round-robin by number: every plate gets every type and both good and defective parts
            chunk = [m for m in members if int(m.name.rsplit("-", 1)[1]) % n_plates == (n + 1) % n_plates]
            objs = []
            for k, item in enumerate(chunk):
                parts, _ = to_bed(item)
                x = (k % cols - (cols - 1) / 2) * pitch[0] + (xr[0] + xr[1]) / 2
                y = (k // cols - (rows - 1) / 2) * pitch[1] + bed[1] / 2
                objs.append((item.name, parts, (x, y, 0.0)))
            out.append((stem + (f"_{n + 1}" if n_plates > 1 else ""), filament, objs))
    return out


def print_jobs(items, printer):
    """[(file name, filament, objects)] in print order. Every part is in exactly one job."""
    bed = printer["bed"]
    # area shared by both nozzles: the job prints whichever nozzle the filament is on
    xr = (printer["shared_x"][0] + 1.5, printer["shared_x"][1] - 1.5)
    by_name = {i.name: i for i in items}
    jobs = [(stem, filament, pack([by_name[n] for n in names], bed, xr)) for stem, filament, names in JOBS]
    jobs += qc_plates(items, bed, xr)
    jobs += [(stem, filament, pack([by_name[n] for n in names], bed, xr)) for stem, filament, names in EXTRA_JOBS]
    return [(f"{k:02d}_{stem}", filament, objs) for k, (stem, filament, objs) in enumerate(jobs, start=1)]


def render_preview(items, p, path):
    """Renders the parts in their assembly positions with MuJoCo."""
    import mujoco
    from PIL import Image

    spec = mujoco.MjSpec()
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1600, 1200
    spec.visual.headlight.ambient = [0.45, 0.45, 0.45]
    spec.visual.headlight.diffuse = [0.55, 0.55, 0.55]
    spec.worldbody.add_light(pos=[0.5, 0.4, 0.9], dir=[-0.4, -0.3, -0.85], diffuse=[0.5, 0.5, 0.5])
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.1], pos=[0, 0, -p["grid"]["thickness"] / 1000],
                            rgba=[0.62, 0.63, 0.65, 1])
    with tempfile.TemporaryDirectory() as tmp:
        for item in items:
            if not item.at:
                continue
            z = item.meta.get("z", 0.0)
            for k, part in enumerate(item.preview or item.parts):
                name = f"{item.name}_{k}"
                write_stl(Path(tmp) / f"{name}.stl", part, scale=1e-3)
                spec.add_mesh(name=name, file=str(Path(tmp) / f"{name}.stl"))
                rgba = [0.05, 0.05, 0.05, 1] if k and not item.preview and not item.part_colors else [*item.part_color(k), 1]
                spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, rgba=rgba,
                                        pos=[item.at[0] / 1000, item.at[1] / 1000, z / 1000])
        model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, 1200, 1600)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.12, 0, 0.22]
    cam.distance, cam.azimuth, cam.elevation = 1.45, 215, -28
    renderer.update_scene(data, camera=cam)
    Image.fromarray(renderer.render()).save(path)


def cmd_build(args):
    p = load_params()
    bed = p["printer"]["bed"]
    items = build_all(p)
    problems = []
    for item in items:
        for part in item.parts:
            if part.status() != Error.NoError or part.is_empty() or part.volume() <= 0:
                problems.append(f"{item.name}: invalid geometry ({part.status()})")
    jobs = print_jobs(items, p["printer"])
    placed = {name.rsplit("_", 1)[0] if name not in {i.name for i in items} else name for _, _, objs in jobs for name, _, _ in objs}
    problems += [f"{i.name}: not on any plate" for i in items if i.name not in placed]
    problems += [f"clash: {k} = {v} mm3" for k, v in check_fit(items).items()]

    out = OUT / "print"
    for stale in ("stl", "plates", "print"):
        shutil.rmtree(OUT / stale, ignore_errors=True)
    out.mkdir(parents=True)
    notes = {i.name: i.note for i in items}
    lines = ["# Print plates", "", "Each file is a single print job; the number is the print order.", "",
             "| File | Filament | Contents |", "| --- | --- | --- |"]
    print(f"{'file':34s} {'filament':40s} parts")
    for name, filament, objs in jobs:
        write_3mf(out / f"{name}.3mf", objs)
        counts = {}
        for obj_name, _, _ in objs:
            base = obj_name if obj_name in notes else obj_name.rsplit("_", 1)[0]
            counts[base] = counts.get(base, 0) + 1
        if all(n in notes and "-" in n for n in counts) and len(counts) > 6:
            inside = f"{len(counts)} QC parts"
        else:
            inside = ", ".join(f"{c} × `{n}`" if c > 1 else f"`{n}`" for n, c in counts.items())
        lines.append(f"| `{name}.3mf` | {filament} | {inside} |")
        print(f"{name + '.3mf':34s} {filament:40s} {sum(counts.values())}")
    (out / "contents.md").write_text("\n".join(lines) + "\n")
    render_preview(items, p, OUT / "preview.png")
    (OUT / "cell.svg").write_text(build_svg(p, jobs))
    (OUT / "parts.svg").write_text(build_catalog_svg(items, jobs, [c for c, _ in generate(p)]))
    print(f"\n{len(jobs)} plates -> {out}")
    print("no problems" if not problems else "PROBLEMS:\n  " + "\n  ".join(problems))
    return 1 if problems else 0


def cmd_showcase(args):
    from .assembly import render_assembly
    from .showcase import render_showcase

    for path in render_showcase(REPO_ROOT / "docs" / "media") + render_assembly(REPO_ROOT / "docs" / "media"):
        print(path)
    return 0


def cmd_slicecheck(args):
    from .slicecheck import report

    for path in args.files or sorted((OUT / "print").glob("*.3mf")):
        print(report(path))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="flywheel-cad", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="build the numbered plates under hardware/print and the preview").set_defaults(fn=cmd_build)
    sc = sub.add_parser("slicecheck", help="slice the print files with the Bambu Studio CLI, summarize warnings and layers")
    sc.add_argument("files", nargs="*", type=Path)
    sc.set_defaults(fn=cmd_slicecheck)
    sub.add_parser("showcase", help="render the finished cell and the assembly diagrams").set_defaults(fn=cmd_showcase)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
