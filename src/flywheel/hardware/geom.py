"""Shared geometry for printed parts: Gridfinity profiles, stroke font, 3MF writer. Units mm."""

import json
import re
import zipfile
from xml.sax.saxutils import quoteattr

import numpy as np
from manifold3d import CrossSection, JoinType, Manifold, OpType

SEG = 48
PITCH = 42.0
# Gridfinity profiles (bottom to top): chamfer, vertical, chamfer
FOOT = (0.8, 1.8, 2.15)  # bin foot, 4.75 total
CELL = (0.7, 1.8, 2.15)  # baseplate socket, 4.65 total
FOOT_SIZE, FOOT_R = 41.5, 3.75
CELL_SIZE, CELL_R = 42.0, 4.0


def rrect(sx, sy, r=0.0):
    if r <= 0:
        return CrossSection.square([sx, sy], center=True)
    return CrossSection.square([sx - 2 * r, sy - 2 * r], center=True).offset(r, JoinType.Round, 2.0, SEG)


def slab(cs, z0, z1):
    return cs.extrude(z1 - z0).translate([0, 0, z0])


def box(x0, x1, y0, y1, z0, z1):
    return Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


def taper(cs0, z0, cs1, z1, eps=1e-3):
    """Straight transition between two cross sections (for convex sections)."""
    return Manifold.batch_hull([slab(cs0, z0, z0 + eps), slab(cs1, z1 - eps, z1)])


def cyl(d, z0, z1):
    return Manifold.cylinder(z1 - z0, d / 2, d / 2, SEG).translate([0, 0, z0])


def cyl_x(d, x0, x1):
    """Cylinder along the x axis."""
    return Manifold.cylinder(x1 - x0, d / 2, d / 2, SEG).rotate([0, 90, 0]).translate([x0, 0, 0])


def cyl_y(d, y0, y1):
    return Manifold.cylinder(y1 - y0, d / 2, d / 2, SEG).rotate([-90, 0, 0]).translate([0, y0, 0])


def union(items):
    items = list(items)
    return Manifold.batch_boolean(items, OpType.Add) if items else Manifold()


def foot():
    """Single Gridfinity foot: top face at z=0, 4.75 mm downward."""
    c, v, t = FOOT
    mid = FOOT_SIZE - 2 * t
    low = mid - 2 * c
    h = c + v + t
    return union([
        taper(rrect(low, low, FOOT_R - t - c), -h, rrect(mid, mid, FOOT_R - t), -h + c),
        slab(rrect(mid, mid, FOOT_R - t), -h + c, -t),
        taper(rrect(mid, mid, FOOT_R - t), -t, rrect(FOOT_SIZE, FOOT_SIZE, FOOT_R), 0),
    ])


def feet(nx, ny):
    """A foot in every cell. With feet only at the corners the body floor hangs 4.75 mm in the air in the middle and
    prints as a bridge over 80 mm on the bed; a full foot grid carries the floor end to end."""
    one = foot()
    spots = {(i, j) for i in range(nx) for j in range(ny)}
    return union(one.translate([(i - (nx - 1) / 2) * PITCH, (j - (ny - 1) / 2) * PITCH, 0]) for i, j in sorted(spots))


def footed_body(nx, ny, height):
    """Flat footed body: feet at z<0, body 0..height. Edges 0.25 mm inside the tile footprint."""
    body = slab(rrect(nx * PITCH - 0.5, ny * PITCH - 0.5, FOOT_R), 0, height)
    return body + feet(nx, ny)


CELL_EXTRA = 0.4  # socket deeper than the foot (4.75): the part rests on the chamfers, not the foot tip


def cell_cut():
    """Volume to subtract for a baseplate socket: top at z=0, 4.65 + CELL_EXTRA mm deep."""
    c, v, t = CELL
    mid = CELL_SIZE - 2 * t
    low = mid - 2 * c
    h = c + v + t
    return union([
        taper(rrect(mid, mid, CELL_R - t), -t, rrect(CELL_SIZE, CELL_SIZE, CELL_R), 0.01),
        slab(rrect(mid, mid, CELL_R - t), -h + c, -t),
        taper(rrect(low, low, CELL_R - t - c), -h, rrect(mid, mid, CELL_R - t), -h + c),
        slab(rrect(low, low, CELL_R - t - c), -h - CELL_EXTRA, -h),
    ])


# --- stroke font: line segments on a 4 x 6 unit grid ----------------------------------------------
_O = [(0, 0), (4, 0), (4, 6), (0, 6), (0, 0)]
STROKES = {
    "0": [_O],
    "1": [[(2, 0), (2, 6)], [(1, 5), (2, 6)]],
    "2": [[(0, 6), (4, 6), (4, 3), (0, 3), (0, 0), (4, 0)]],
    "3": [[(0, 6), (4, 6), (4, 0), (0, 0)], [(1, 3), (4, 3)]],
    "4": [[(0, 6), (0, 3), (4, 3)], [(4, 6), (4, 0)]],
    "5": [[(4, 6), (0, 6), (0, 3), (4, 3), (4, 0), (0, 0)]],
    "6": [[(4, 6), (0, 6), (0, 0), (4, 0), (4, 3), (0, 3)]],
    "7": [[(0, 6), (4, 6), (2, 0)]],
    "8": [_O, [(0, 3), (4, 3)]],
    "9": [[(4, 3), (0, 3), (0, 6), (4, 6), (4, 0), (0, 0)]],
    "A": [[(0, 0), (0, 6), (4, 6), (4, 0)], [(0, 3), (4, 3)]],
    "B": [[(0, 0), (0, 6), (3, 6), (4, 5), (4, 4), (3, 3), (0, 3)], [(3, 3), (4, 2), (4, 1), (3, 0), (0, 0)]],
    "C": [[(4, 6), (0, 6), (0, 0), (4, 0)]],
    "D": [[(0, 0), (0, 6), (3, 6), (4, 5), (4, 1), (3, 0), (0, 0)]],
    "E": [[(4, 6), (0, 6), (0, 0), (4, 0)], [(0, 3), (3, 3)]],
    "G": [[(4, 5), (3, 6), (1, 6), (0, 5), (0, 1), (1, 0), (3, 0), (4, 1), (4, 3), (2, 3)]],
    "H": [[(0, 0), (0, 6)], [(4, 0), (4, 6)], [(0, 3), (4, 3)]],
    "J": [[(1, 6), (4, 6)], [(4, 6), (4, 1), (3, 0), (1, 0), (0, 1)]],
    "K": [[(0, 0), (0, 6)], [(4, 6), (0, 3), (4, 0)]],
    "L": [[(0, 6), (0, 0), (4, 0)]],
    "M": [[(0, 0), (0, 6), (2, 3), (4, 6), (4, 0)]],
    "O": [_O],
    "R": [[(0, 0), (0, 6), (4, 6), (4, 3), (0, 3)], [(2, 3), (4, 0)]],
    "S": [[(4, 6), (0, 6), (0, 3), (4, 3), (4, 0), (0, 0)]],
    "T": [[(0, 6), (4, 6)], [(2, 6), (2, 0)]],
    "W": [[(0, 6), (0, 0), (2, 3), (4, 0), (4, 6)]],
    "Y": [[(0, 6), (2, 3), (4, 6)], [(2, 3), (2, 0)]],
    "-": [[(1, 3), (3, 3)]],
    ".": [[(2, 0), (2, 0.3)]],
}


def text(s, height, stroke=0.7, center=True):
    """Text cross section in the stroke font. At small sizes (3-6 mm) a stroke font prints cleaner than an outline font."""
    u = height / 6
    advance = 4 * u + max(1.4 * u, 1.6 * stroke)
    dot = CrossSection.circle(stroke / 2, 12)
    parts = []
    for i, ch in enumerate(s):
        for line in STROKES[ch]:
            for (ax, ay), (bx, by) in zip(line, line[1:]):
                a = dot.translate([i * advance + ax * u, ay * u])
                b = dot.translate([i * advance + bx * u, by * u])
                parts.append(CrossSection.batch_hull([a, b]))
    out = CrossSection.batch_boolean(parts, OpType.Add)
    if center:
        width = (len(s) - 1) * advance + 4 * u
        out = out.translate([-width / 2, -height / 2])
    return out


def engrave_bottom(body, lines, height=4.0, depth=0.4, stroke=0.7):
    """Engraves mirrored text into the bottom face (z=0); reads correctly from below. lines: [(text, x, y, angle)]."""
    cuts = []
    for s, x, y, angle in lines:
        cs = text(s, height, stroke).mirror([1, 0]).rotate(angle).translate([x, y])
        cuts.append(slab(cs, -0.01, depth))
    return body - union(cuts)


# --- output ---------------------------------------------------------------------------------------
def mesh_arrays(m):
    mesh = m.to_mesh()
    return np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts)


def _mesh_xml(m):
    v, t = mesh_arrays(m)
    vs = "".join(f'<vertex x="{x:.4f}" y="{y:.4f}" z="{z:.4f}"/>' for x, y, z in v)
    ts = "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in t)
    return f"<mesh><vertices>{vs}</vertices><triangles>{ts}</triangles></mesh>"


# For Bambu to read the per-part filament assignment the file must be a Bambu project. Below is the minimum project
# config both Bambu Studio and the command line slicer need to open it and see two filaments (the filament count comes
# from the filament vectors; if one is missing it crashes or prints everything with filament 1). Anything more overrides
# the choices in Bambu Studio. Written only for two-color jobs; filament 1 black, 2 white.
BAMBU_PROJECT = {
    "printer_settings_id": "Bambu Lab X2D 0.4 nozzle",
    "print_settings_id": "0.20mm Standard @BBL X2D",
    "printable_height": "260",
    "nozzle_diameter": ["0.4", "0.4"],
    "filament_settings_id": ["Bambu PLA Basic @BBL X2D 0.4 nozzle"] * 2,
    "filament_colour": ["#161616", "#FFFFFF"],
    "filament_density": ["1.26", "1.26"],
    "filament_diameter": ["1.75", "1.75"],
    "filament_is_support": ["0", "0"],
    "filament_shrink": ["100%", "100%"],
    # both nozzles installed (otherwise the command line ignores the right nozzle, puts both filaments on one nozzle
    # and swaps color every layer); this way auto mapping gives each filament its own nozzle
    "extruder_nozzle_stats": ["Standard#1", "Standard#1"],
}


def write_3mf(path, objects):
    """objects: [(name, [part Manifolds], (tx, ty, tz))]. An object with several parts opens as a single object in the
    slicer; part k prints with filament k+1. If there is such an object the file is written as a
    Bambu project: on the X2D the two filaments go to two separate nozzles, no color swaps and no prime tower."""
    res, build, next_id = [], [], 1
    settings = []
    for name, parts, (tx, ty, tz) in objects:
        ids = []
        for k, part in enumerate(parts):
            pname = name if len(parts) == 1 else f"{name}_{k + 1}"
            res.append(f'<object id="{next_id}" name={quoteattr(pname)} type="model">{_mesh_xml(part)}</object>')
            ids.append(next_id)
            next_id += 1
        target = ids[0]
        if len(ids) > 1:
            comps = "".join(f'<component objectid="{i}"/>' for i in ids)
            res.append(f'<object id="{next_id}" name={quoteattr(name)} type="model"><components>{comps}</components></object>')
            target = next_id
            next_id += 1
            vols = "".join(
                f'<part id="{i}" subtype="normal_part"><metadata key="name" value={quoteattr(f"{name}_{k + 1}")}/>'
                f'<metadata key="extruder" value="{k + 1}"/></part>'
                for k, i in enumerate(ids)
            )
            settings.append(f'<object id="{target}"><metadata key="name" value={quoteattr(name)}/>'
                            f'<metadata key="extruder" value="1"/>{vols}</object>')
        build.append(f'<item objectid="{target}" transform="1 0 0 0 1 0 0 0 1 {tx:.3f} {ty:.3f} {tz:.3f}"/>')
    model = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">'
        + ('<metadata name="Application">BambuStudio-02.08.02.61</metadata>' if settings else "")
        + f"<resources>{''.join(res)}</resources><build>{''.join(build)}</build></model>"
    )
    types = (
        '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>'
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>'
        "</Relationships>"
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", types)
        z.writestr("_rels/.rels", rels)
        z.writestr("3D/3dmodel.model", model)
        if settings:
            items = re.findall(r'<item objectid="(\d+)"', model)
            plate = "".join(f'<model_instance><metadata key="object_id" value="{i}"/>'
                            '<metadata key="instance_id" value="0"/></model_instance>' for i in items)
            z.writestr("Metadata/model_settings.config",
                       f'<?xml version="1.0" encoding="UTF-8"?><config>{"".join(settings)}'
                       f'<plate><metadata key="plater_id" value="1"/>{plate}</plate></config>')
            z.writestr("Metadata/project_settings.config", json.dumps(BAMBU_PROJECT, indent=1))
