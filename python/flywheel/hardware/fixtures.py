"""Parts in the print plan: hardware/params.yaml -> Manifold. Units mm.

Each part is built in the assembly frame (z=0 is the baseplate top, part at its own center); `at` is its position
in the scene. On export for printing it is dropped onto the bed.
"""

from dataclasses import dataclass, field

import numpy as np
from manifold3d import CrossSection, Manifold, OpType

from ..sim.params import MAST_BOSS, head_geometry, mast_top
from ..sim.parts import generate
from .geom import (
    FOOT_R,
    PITCH,
    box,
    cell_cut,
    cyl_x,
    cyl_y,
    engrave_bottom,
    footed_body,
    rrect,
    slab,
    taper,
    text,
    union,
)
from .robot import Robot

KEY = [(-8, -5), (0, -3), (8, -5), (8, 5), (0, 3), (-8, 5)]  # bowtie key, crosses the seam along x
KEY_DEPTH = 2.4
TONGUE_FIT = 0.2  # per-side clearance of the tongue in the pocket
_ADD = OpType.Add


@dataclass
class Item:
    name: str
    parts: list  # single Manifold; for two-color prints [body, second color]
    material: str
    phase: str  # day1 | phase0 | phase1
    qty: int = 1
    note: str = ""
    at: tuple | None = None  # position in the scene (x, y); None means not in the assembly preview
    color: tuple = (0.8, 0.8, 0.8)
    group: str = "fixtures"
    meta: dict = field(default_factory=dict)
    preview: list | None = None  # seated body drawn instead of parts in the assembly preview
    part_colors: list | None = None  # color of each part in a two-color print; part k is filament k+1 in the 3MF

    def part_color(self, k):
        return self.part_colors[k] if self.part_colors else self.color

    @property
    def body(self):
        return union(self.parts) if len(self.parts) > 1 else self.parts[0]


# --- baseplate -------------------------------------------------------------------------------------
def _cells(x0, x1, y0, y1):
    """Tile indices that intersect a rectangle."""
    return {
        (i, j)
        for i in range(int(np.floor(x0 / PITCH)), int(np.ceil(x1 / PITCH)))
        for j in range(int(np.floor(y0 / PITCH)), int(np.ceil(y1 / PITCH)))
    }


def _arm_seat(p, robot):
    """Locating lips hugging the arm base front and back, and the solid tiles under it.

    The base footprint is nearly hexagonal: clamp ears in the middle, edges narrowing toward front and back. The lips
    rest against the slanted edges (locking x and y together); the middle strip with the ears stays open for the clamp.
    """
    c = p["fit"]["clearance"] + 0.1
    outline = CrossSection.hull_points(robot.base_outline())
    ring = outline.offset(c + 3.0) - outline.offset(c)
    x0, y0, x1, y1 = outline.bounds()
    pts = np.array(outline.to_polygons()[0])
    ears = pts[np.abs(pts[:, 1]) > y1 - 1.0][:, 0]  # x range of the ears
    open_strip = CrossSection.square([ears.max() - ears.min() + 24, 2 * (y1 - y0)], center=True).translate(
        [(ears.max() + ears.min()) / 2, 0]
    )
    return slab(ring - open_strip, 0, 5.0), _cells(x0 - 6, x1 + 6, y0 - 6, y1 + 6)


MAST_R = 4.0


def _mast_dims(p):
    m = p["mast"]
    return m["size"], m["wall"], m["size"] - 2 * m["wall"]


def _socket_boss(p, ribs, rib=16.0):
    """Socket the first mast segment seats in with its tapered spigot; ribs: gusset directions [(dx, dy)]. At the origin."""
    size, wall, bore = _mast_dims(p)
    half = size / 2 + 4
    boss = slab(rrect(2 * half, 2 * half, MAST_R + 2), 0, MAST_BOSS)
    boss -= slab(rrect(bore, bore, MAST_R - wall), 3, MAST_BOSS + 1)
    out = [boss]
    for dx, dy in ribs:
        if dx:
            near = box(dx * half - 0.01, dx * half + 0.01, -2, 2, 0, 36)
            far = box(min(dx * half, dx * (half + rib)), max(dx * half, dx * (half + rib)), -2, 2, 0, 0.01)
        else:
            near = box(-2, 2, dy * half - 0.01, dy * half + 0.01, 0, 36)
            far = box(-2, 2, min(dy * half, dy * (half + rib)), max(dy * half, dy * (half + rib)), 0, 0.01)
        out.append(Manifold.batch_hull([near, far]))
    return union(out), half


def _mast_socket(p):
    """Mast socket embedded in a baseplate tile. The mast is pressed in and pulls out."""
    mx, my = p["mast"]["pos"]
    rib = 16.0
    body, half = _socket_boss(p, [(-1, 0), (0, 1), (0, -1)], rib)
    return body.translate([mx, my, 0]), _cells(mx - half - rib, mx + half, my - half - rib, my + half + rib)


def baseplate(p, robot):
    g = p["grid"]
    thick = g["thickness"]
    bx, by = g["tiles_x"], g["tiles_y"]
    limit = p["printer"]["max_tile_units"]
    assert all(b - a <= limit for a, b in zip(bx, bx[1:])) and all(b - a <= limit for a, b in zip(by, by[1:]))
    seat, seat_cells = _arm_seat(p, robot)
    socket, socket_cells = _mast_socket(p)
    solid = seat_cells | socket_cells
    cut = cell_cut()

    skip = set(g.get("skip", []))
    cols, rows = list(zip(bx, bx[1:])), list(zip(by, by[1:]))
    label = lambda ci, ri: f"{'ABCDE'[ci]}{ri + 1}"  # noqa: E731
    exists = lambda ci, ri: 0 <= ci < len(cols) and 0 <= ri < len(rows) and label(ci, ri) not in skip  # noqa: E731

    key = CrossSection([KEY])
    seams = []  # (female tile, male tile, x, y, angle): only on seams with a tile on both sides
    for ci in range(len(cols) - 1):
        for ri, (y0, y1) in enumerate(rows):
            if exists(ci, ri) and exists(ci + 1, ri):
                seams += [(label(ci, ri), label(ci + 1, ri), cols[ci][1] * PITCH, (j + 0.5) * PITCH, 0) for j in range(y0, y1)]
    for ri in range(len(rows) - 1):
        for ci, (x0, x1) in enumerate(cols):
            if exists(ci, ri) and exists(ci, ri + 1):
                seams += [(label(ci, ri), label(ci, ri + 1), (i + 0.5) * PITCH, rows[ri][1] * PITCH, 90) for i in range(x0, x1)]
    # The joint is built into the tiles: a pocket from below on one side of the seam, a tongue fitting it on the other.
    # Pockets go on the tile with the most neighbors (the arm's tile); it goes down last, onto the tongues, holding all.
    degree = {}
    for a, b, *_ in seams:
        degree[a], degree[b] = degree.get(a, 0) + 1, degree.get(b, 0) + 1
    pockets, tongues = {}, {}
    for a, b, x, y, angle in seams:
        female, male = (a, b) if degree[a] >= degree[b] else (b, a)
        shape = key.rotate(angle).translate([x, y])
        pockets.setdefault(female, []).append(slab(shape, -thick - 0.01, -thick + KEY_DEPTH))
        tongues.setdefault(male, []).append(slab(shape.offset(-TONGUE_FIT), -thick, -thick + KEY_DEPTH - 0.2))

    gap = 0.1  # per-side gap between tiles
    items = []
    for ci, (x0, x1) in enumerate(cols):
        for ri, (y0, y1) in enumerate(rows):
            if not exists(ci, ri):
                continue
            name = label(ci, ri)
            plate = box(x0 * PITCH + gap, x1 * PITCH - gap, y0 * PITCH + gap, y1 * PITCH - gap, -thick, 0)
            cells = union(
                cut.translate([(i + 0.5) * PITCH, (j + 0.5) * PITCH, 0])
                for i in range(x0, x1) for j in range(y0, y1) if (i, j) not in solid
            )
            body = plate - cells - union(pockets.get(name, [])) + union(tongues.get(name, []))
            cx, cy = (x0 + x1) / 2 * PITCH, (y0 + y1) / 2 * PITCH
            region = box(x0 * PITCH, x1 * PITCH, y0 * PITCH, y1 * PITCH, -1, 100)
            note = f"{x1 - x0} x {y1 - y0} tiles"
            for extra, what in ((seat, "arm seat"), (socket, "mast socket")):
                piece = extra ^ region
                if not piece.is_empty():
                    body += piece
                    note += f" + {what}"
            body = body.translate([-cx, -cy, thick])
            body = engrave_bottom(body, [(name, 0, 0, 0)], height=10, depth=0.4, stroke=1.2)
            items.append(Item(f"baseplate_{name}", [body.translate([0, 0, -thick])], "PLA", "phase0", note=note,
                              at=(cx, cy), color=(0.23, 0.24, 0.26), group="baseplate",
                              meta={"cells": [x0, x1, y0, y1]}))
    gauge = slab(key.offset(-0.15), 0, KEY_DEPTH - 0.2)
    items.append(Item("baseplate_key", [gauge], "PLA", "phase0", qty=2,
                      note="joint gauge: if it fits the pocket, the neighboring tiles' tongues fit too", group="baseplate"))
    return items


# --- tray, bins, backdrop -------------------------------------------------------------------------
def tray(p):
    t = p["tray"]
    sx, sy = t["size"]
    nx, ny = round(sx / PITCH), round(sy / PITCH)
    floor, depth = t["floor"], t["pocket_depth"]
    psx, psy = t["pocket_size"]
    px, py = sx / t["rows"], sy / t["cols"]
    top = floor + depth
    body = footed_body(nx, ny, top)
    cuts, labels = [], []
    for r in range(t["rows"]):
        for c in range(t["cols"]):
            x, y = -sx / 2 + (r + 0.5) * px, sy / 2 - (c + 0.5) * py
            # chamfered pocket: 1 mm wider on each side at the top
            cuts.append(taper(rrect(psx, psy, 1.0), floor, rrect(psx + 2, psy + 2, 2.0), top + 0.01).translate([x, y, 0]))
            # finger slot: the wall on both sides of the pocket is cut so the jaws reach the floor
            slot = t["finger_slot"]
            cuts.append(box(x - slot / 2, x + slot / 2, max(y - py / 2, -sy / 2 - 1), min(y + py / 2, sy / 2 + 1),
                            floor, top + 1))
            seg, wy = (psx - slot) / 2, (py - psy) / 2
            digit = text(str(r * t["cols"] + c + 1), min(seg, wy) * 0.62, stroke=0.9).rotate(90)
            labels.append(slab(digit, top, top + 0.6).translate([x - slot / 2 - seg / 2, y + psy / 2 + wy / 2, 0]))
    body = body - union(cuts)
    # numbers raised 0.6 mm, one piece and one color with the tray (a second color needs a prime tower; plate is full)
    body = body + (union(labels) ^ slab(rrect(sx - 1.5, sy - 1.5, 3), 0, top + 1))
    return Item("tray", [body], "PLA", "phase0", at=tuple(t["center"]), color=(0.12, 0.12, 0.13),
                note=f"{t['rows'] * t['cols']} numbered pockets with finger slots; matte black")


def bins(p):
    b = p["bins"]
    n = round(b["size"][0] / PITCH)
    th, h = b["wall_thickness"], b["wall_height"]
    inset = h * np.tan(np.radians(b["wall_tilt"]))
    mouth = n * PITCH - 0.5  # outer mouth size
    base = mouth - 2 * inset  # outer wall size at the floor
    shell = taper(rrect(base, base, 3), th, rrect(mouth, mouth, 3.75), th + h)
    cavity = taper(rrect(base - 2 * th, base - 2 * th, 2), th, rrect(mouth - 2 * th, mouth - 2 * th, 3), th + h + 0.01)
    # outer ribs like a KLT bin: fill the gap between the slanted wall and the tile boundary, stiffen the wall
    fin = box(-1.5, 1.5, base / 2 - 1, mouth / 2, th, th + h) - shell
    fins = union(fin.translate([d, 0, 0]).rotate([0, 0, a]) for a in (0, 90, 180, 270) for d in (-36, 0, 36))
    step = 3 * np.tan(np.radians(b["wall_tilt"])) + 0.05  # overhang of the mouth past the wall below it
    rim = (taper(rrect(mouth - 2 * step, mouth - 2 * step, 3.75 - step), th + h - 3 - step, rrect(mouth, mouth, 3.75), th + h - 3)
           + slab(rrect(mouth, mouth, 3.75), th + h - 3, th + h))
    rim -= slab(rrect(mouth - 2 * th, mouth - 2 * th, 3), 0, th + h + 1)
    body = footed_body(n, n, th) + shell + fins + rim - cavity
    # liner: PLA is hard so it doesn't absorb bounce; the low grid stops round parts from rolling and climbing
    # the wall. Sits loose on the bin floor.
    side, sheet, total, pitch = base - 2 * th - 0.8, 1.0, float(b["liner_thickness"]), 14.0
    ribs = [box(k * pitch - 1, k * pitch + 1, -side / 2, side / 2, sheet, total) for k in range(-3, 4)]
    ribs += [box(-side / 2, side / 2, k * pitch - 1, k * pitch + 1, sheet, total) for k in range(-3, 4)]
    liner = slab(rrect(side, side, 2), 0, sheet) + (union(ribs) ^ slab(rrect(side, side, 2), 0, total))
    out = []
    for name in ("ok", "reject"):
        cfg = b[name]
        out.append(Item(f"bin_{name}", [body], "PLA", "phase0", at=tuple(cfg["center"]), color=tuple(cfg["color"]),
                        note=f"KLT-style bin; {'metal grey' if name == 'ok' else 'orange'} filament"))
    out.append(Item("bin_liner", [liner], "PLA", "phase0", qty=2, note="grid liner laid on the bin floor",
                    color=(0.13, 0.13, 0.14)))
    return out


def backdrop(p):
    i = p["inspection"]
    sx, sy = i["backdrop_size"]
    body = footed_body(round(sx / PITCH), round(sy / PITCH), 2.0)
    return Item("inspection_backdrop", [body], "PLA", "phase1", at=tuple(i["backdrop_center"]),
                color=tuple(i["backdrop_color"]), note="matte black filament")


# --- AprilTag tiles -------------------------------------------------------------------------------
TAG_INLAY = 1.0  # white inlay thickness: 5 layers; thinner lets the black body show through as grey
TAG_TOP = 2.0  # top face of the tile body (above the feet)
WHITE = (0.93, 0.93, 0.91)


def tag_sticker_side(t):
    """Side of the white area (mm): 8-cell tag + a 1-cell white quiet zone on each side."""
    return t["size"] * 10 / 8


def tag_bits(tag_id):
    """8 x 8 cells, 1 = white. Row 0 is the top of the pattern (+y), column 0 the left (-x): reads correctly from above."""
    import cv2

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    return cv2.aruco.generateImageMarker(dictionary, tag_id, 8) // 255


def tag_white(t):
    """Cross section of the white area on top of the tile (relative to the tile center): quiet zone + white tag cells."""
    ox, oy = t["center"][0] - t["tile"][0], t["center"][1] - t["tile"][1]
    cell, side, tile = t["size"] / 8, tag_sticker_side(t), t["units"] * PITCH - 0.5
    if (tile - side) / 2 - max(abs(ox), abs(oy)) < 0.8:
        # a black frame thinner than 0.8 mm would remain outside the white: the white extends to the tile edge
        white = rrect(tile, tile, FOOT_R).translate([-ox, -oy])
    else:
        white = rrect(side, side)
    black = [
        CrossSection.square([cell, cell]).translate([(c - 4) * cell, (3 - r) * cell])
        for r, row in enumerate(tag_bits(t["id"]))
        for c, bit in enumerate(row)
        if not bit
    ]
    return (white - CrossSection.batch_boolean(black, _ADD)).translate([ox, oy])


def tags(p):
    """Tag tiles print in two colors: black body, the tag's white cells and quiet zone as a white inlay on top.

    The inlay is flush with the top face, TAG_INLAY thick; the black cells are the body itself.
    """
    out = []
    for t in p["tags"]["items"]:
        ox, oy = t["center"][0] - t["tile"][0], t["center"][1] - t["tile"][1]
        side = tag_sticker_side(t)
        body = footed_body(t["units"], t["units"], TAG_TOP)
        white = slab(tag_white(t), TAG_TOP - TAG_INLAY, TAG_TOP)
        note = f"white inlay tag on top ({t['size']} mm)"
        if abs(ox) > 1 or abs(oy) > 1:
            # tag is off-center: if the tile goes in reversed the tag ends up behind the bin. White tag number in the
            # middle of the empty strip; the number must be on the side facing the bin
            edge = side / 2 - np.hypot(ox, oy)  # edge of the white area facing the empty strip, from the tile center
            free = (t["units"] * PITCH - 0.5) / 2 - edge
            mark = text(str(t["id"]), 10.0, stroke=1.4).translate([-np.sign(ox) * (edge + free / 2), 0])
            white += slab(mark, TAG_TOP - TAG_INLAY, TAG_TOP)
            note += "; white number faces the bin"
        else:
            # no room for a number on top (white reaches the edge): engrave it below so the two small tiles can be told apart
            body = engrave_bottom(body.translate([0, 0, 4.75]), [(str(t["id"]), 0, 0, 0)], height=10.0, stroke=1.2)
            body = body.translate([0, 0, -4.75])
            note += "; number engraved underneath"
        out.append(Item(f"tag_{t['id']}", [body - white, white], "PLA", "phase0", at=tuple(t["tile"]),
                        color=(0.17, 0.17, 0.19), part_colors=[(0.17, 0.17, 0.19), WHITE],
                        note=f"tag {t['id']} tile; {note}"))
    return out


# --- top camera mast ------------------------------------------------------------------------------
def _mast_clip(p, z):
    """Clamp on the mast's +x face that holds the cable on its way down. z: lower edge of the clamp (print orientation).

    Its section is a vertical C: the cable is pressed in through the narrow front mouth. The 45 degree ramp below it
    lets it print without supports; the channel runs through the ramp too, so the cable path stays open.
    """
    size = p["mast"]["size"]
    c = p["clips"]["mast"]
    out, along = c["channel"]
    wall, height = 2.0, 8.0
    x0 = size / 2 - 0.4  # starts slightly inside the tube face so it fuses fully
    x1 = size / 2 + out + wall
    half = along / 2 + wall
    block = box(x0, x1, -half, half, z, z + height)
    ramp = Manifold.batch_hull([block, box(x0, x0 + 0.4, -half, half, z - (x1 - size / 2), z - (x1 - size / 2) + 0.1)])
    channel = slab(rrect(out, along, 2.0), z - 30, z + height + 1).translate([size / 2 + out / 2, 0, 0])
    mouth = box(size / 2 + out - 1, x1 + 1, -c["mouth"] / 2, c["mouth"] / 2, z - 30, z + height + 1)
    return ramp - channel - mouth


def _mast_segment(p, tube, spigot=True, clips=False):
    """Square tube; in print orientation the female end is on the bed, the tapered male spigot on top. Flipped for assembly."""
    size, wall, bore = _mast_dims(p)
    m = p["mast"]
    r_in = MAST_R - wall
    neck = bore - m["fit"] - 0.7 - 2 * max(wall, 1.6)  # spigot inner void: spigot wall at least 1.6 mm even at the tip
    body = slab(rrect(size, size, MAST_R), 0, tube)
    if clips:
        # the first 50 mm goes into the head's sleeve; clamps beyond that, evenly spaced
        n = p["clips"]["mast"]["per_segment"]
        body += union(_mast_clip(p, 60 + k * (tube - 75) / max(n - 1, 1)) for k in range(n))
    body -= slab(rrect(bore, bore, r_in), -1, tube - 6)
    # female mouth on the bed: 0.4 mm chamfer so first layer squish doesn't narrow it
    body -= taper(rrect(bore + 0.8, bore + 0.8, r_in + 0.4), -0.01, rrect(bore, bore, r_in), 0.4)
    if not spigot:
        return body - slab(rrect(bore, bore, r_in), tube - 7, tube + 1)
    # the bore chamfers down to the spigot's inner size: the spigot wall doesn't start in mid-air, no supports
    body -= taper(rrect(bore, bore, r_in), tube - 6.01, rrect(neck, neck, 0.6), tube + 0.01)
    # Straight along the spigot: the female end is straight too, so a tapered spigot would touch only at the mouth and
    # wobble like a hinge (centimeters at the top of a 66 cm mast). The short chamfer at the tip only guides insertion.
    s0, s1, lead = bore - m["fit"], bore - m["fit"] - 0.8, 5.0
    plug = slab(rrect(s0, s0, r_in), tube, tube + m["spigot"] - lead)
    plug += taper(rrect(s0, s0, r_in), tube + m["spigot"] - lead, rrect(s1, s1, r_in), tube + m["spigot"])
    # Crush ribs: two rows on all four faces, m["crush"] mm interference per side into the female bore. Print
    # tolerance goes into crushing the ribs, not into play or binding; two rows take out angular play.
    reach = bore / 2 + m["crush"]
    for z0 in (tube + 3, tube + m["spigot"] - lead - 9):
        rib = box(s0 / 2 - 0.6, reach, -2, 2, z0, z0 + 6)
        plug += union(rib.rotate([0, 0, a]) for a in (0, 90, 180, 270))
    return body + (plug - slab(rrect(neck, neck, 0.6), tube - 1, tube + m["spigot"] + 1))


def _flat_pin(d, y0, y1):
    """Pin along y; flattened underneath, printed lying on its side without supports."""
    return cyl_y(d, y0, y1) - box(-d, d, y0 - 1, y1 + 1, -d, -d / 2 + 0.5)


def mast_parts(p):
    """Mast segments, articulated camera head, cradle and pins. Everything press-fits together; no screws."""
    f, m, top = p["fit"], p["mast"], p["cameras"]["top"]
    size, _, _ = _mast_dims(p)
    mx, my = m["pos"]
    top_z = mast_top(p)
    segment = _mast_segment(p, m["segment"], clips=True)
    upright = segment.rotate([180, 0, 0]).translate([0, 0, m["segment"]])  # assembly orientation: spigot down
    stack = [upright.translate([0, 0, MAST_BOSS + k * m["segment"]]) for k in range(m["segments"])]

    g = head_geometry(p)
    sh, (ax, az), radius = g["sleeve"], g["axle"], g["index_r"]
    phi0 = np.radians(g["index_phi"])
    ix, iz = ax + radius * np.cos(phi0), az + radius * np.sin(phi0)  # fixed lock hole in the head
    tilt = m["head"]["tilt"]

    # head: sleeve over the mast top + the two cheeks of the joint. The cradle hub turns between the cheeks;
    # the axle pin stays in, the lock pin goes through the hole of the chosen detent.
    inner = size / 2 + f["clearance"]
    depth = 45.0
    head = box(-sh, sh, -sh, sh, -depth, 3) - box(-inner, inner, -inner, inner, -depth - 1, 0)
    for sy in (-1, 1):
        y0, y1 = sorted((sy * 7.2, sy * 12.2))
        head += Manifold.batch_hull([
            cyl_y(20, y0, y1).translate([ax, 0, az]),
            cyl_y(16, y0, y1).translate([ix, 0, iz]),
            box(-sh, 12, y0, y1, 0, 3),
        ])
    head -= cyl_y(7.15, -13, 13).translate([ax, 0, az]) + cyl_y(5.25, -13, 13).translate([ix, 0, iz])
    # thin crush ribs inside the sleeve for a tight fit
    crush = [box(s * inner - (0.25 if s > 0 else 0), s * inner + (0.25 if s < 0 else 0), -2, 2, -depth, -4) for s in (-1, 1)]
    crush += [box(-2, 2, s * inner - (0.25 if s > 0 else 0), s * inner + (0.25 if s < 0 else 0), -depth, -4) for s in (-1, 1)]
    head += union(crush)

    cradle, fit_test = _webcam_cradle(top["webcam"]["body"], g, m["head"]["tilts"])
    hx, hz = g["hub"]
    place = lambda body: body.translate([-hx, 0, -hz]).rotate([0, tilt, 0]).translate([ax, 0, az])  # noqa: E731
    axle_pin = _flat_pin(7.0, -12.5, 12.5) + (_flat_pin(11.0, -14.5, -12.5) - box(-8, 8, -15, -12, -8, -3.0))
    lock_pin = _flat_pin(5.0, -12.5, 12.5) + box(-5, 5, -15.5, -12.5, -2.0, 11)
    foot_boss, _ = _socket_boss(p, [(1, 0), (-1, 0), (0, 1), (0, -1)], rib=34.0)
    foot = slab(rrect(170, 170, 12), -4, 0) + foot_boss

    dark, model = (0.13, 0.13, 0.14), top["webcam"]["model"]
    steps = ", ".join(f"{t:g}" for t in m["head"]["tilts"])
    return [
        Item("mast_segment", [segment], "PLA", "phase0", qty=m["segments"], at=(mx, my), color=(0.55, 0.57, 0.60),
             note=f"mast segment, {m['spigot']} mm tapered spigot, 2 cable clamps on its face; female end on the bed, no supports",
             meta={"z": 0.0},
             preview=stack),
        Item("mast_head", [head], "PLA", "phase0", at=(mx, my), color=dark,
             note="head that fits over the mast top: sleeve + joint cheeks; no supports", meta={"z": top_z}),
        Item("cam_cradle_c310", [cradle], "PLA", "phase0", at=(mx, my), color=dark,
             note=f"{model} snap-fit cradle with joint hub; detents {steps} degrees", meta={"z": top_z},
             preview=[place(cradle)]),
        Item("head_axle_pin", [axle_pin], "PLA", "phase0", note="joint axle; inserted once, stays in"),
        Item("head_lock_pin", [lock_pin], "PLA", "phase0", qty=2, note="pin that locks the angle (one spare)"),
        Item("mast_foot", [foot], "PLA", "phase0",
             note="optional: standalone foot for using the mast outside this cell, on another table"),
        Item("cam_cradle_fit_test", [fit_test], "PLA", "day1", group="tests",
             note=f"only the floor and latches of the {model} cradle; does the camera fit"),
        Item("mast_joint_test_socket", [_mast_segment(p, 50.0, spigot=False)], "PLA", "day1", group="tests",
             note="female end of the mast joint"),
        Item("mast_joint_test_plug", [_mast_segment(p, 15.0)], "PLA", "day1", group="tests",
             note="male end of the mast joint; pressed into the socket it should seat without play"),
    ]


def _stadium(length, width, grow=0.0):
    """Stadium cross section with semicircular ends: x short axis, y long axis."""
    r, half = width / 2 + grow, (length - width) / 2
    disc = CrossSection.circle(r, 64)
    return CrossSection.batch_hull([disc.translate([0, -half]), disc.translate([0, half])])


def _webcam_cradle(body, g, tilts):
    """(cradle, fit test piece).

    The C310 body is a stadium (rounded ends per Logitech's product images), lens near one end, the clip coming out of
    the middle of the lower long edge. The cradle is a seat for that stadium: the camera lies face down, the rim of its
    front face rests on a 3.5 mm ledge, the middle is fully open (the lens sees through whichever end it is at). A low
    wall wraps the body, the -x edge where the clip exits is open, spring latches at both ends lock the body. The tall
    wall at +x connects up to the joint hub and the fan carrying the detent holes.
    """
    length, width, depth = body
    c, floor, wall, rim, low = 0.4, 2.5, 2.0, 3.5, 6.0
    r, half = width / 2, (length - width) / 2
    hx = r + c
    pocket, outer = _stadium(length, width, c), _stadium(length, width, c + wall)
    back = CrossSection.square([wall + 8, 2 * half + 8]).translate([r - 6, -half - 4])  # strip the tall wall stands on
    base = slab(outer + back, -floor, 0) - slab(_stadium(length, width, -rim), -floor - 1, 1)
    ring = slab(outer - pocket, 0, low)
    ring -= box(-hx - wall - 1, -r + 6, -16, 16, -0.01, low + 1)  # opening for the clip neck
    fingers = []
    for s in (-1, 1):
        ring -= box(-7, 7, min(s * (half + r - 5), s * (half + hx + wall + 1)), max(s * (half + r - 5), s * (half + hx + wall + 1)),
                    -0.01, low + 1)  # wall is cut here so the latch can flex freely
        y0 = s * (length / 2 - 0.2)  # 0.2 mm preload
        y1 = y0 + s * 1.5
        fingers.append(box(-6, 6, min(y0, y1), max(y0, y1), -0.01, 17))
        # latch ridge: rises and falls at 45 degrees, prints without supports; sits over the widest point of the body
        thin = box(-6, 6, min(y0, y0 + s * 0.02), max(y0, y0 + s * 0.02), 8.5, 8.52)
        peak = box(-6, 6, min(y0 - s * 1.0, y0), max(y0 - s * 1.0, y0), 10.0, 10.02)
        fingers.append(Manifold.batch_hull([thin, peak, thin.translate([0, 0, 3.0])]))
    nest = base + ring + union(fingers)
    top = depth + 10
    span = half + 4
    tall = box(hx, hx + 4, -span, span, -floor, top)
    # joint: the hub goes on the axle pin; each hole in the fan is a tilt detent
    gx, gz = g["hub"]
    radius, phi0 = g["index_r"], g["index_phi"]
    hub = cyl_y(20, -7, 7).translate([gx, 0, gz])
    web = Manifold.batch_hull([hub, box(hx + 3.9, hx + 4, -7, 7, 14, top)])
    spots = [(gx + radius * np.cos(np.radians(phi0 + t)), gz + radius * np.sin(np.radians(phi0 + t))) for t in tilts]
    fan = Manifold.batch_hull([hub] + [cyl_y(13, -7, 7).translate([x, 0, z]) for x, z in spots])
    holes = cyl_y(7.4, -8, 8).translate([gx, 0, gz]) + union(cyl_y(5.3, -8, 8).translate([x, 0, z]) for x, z in spots)
    labels = []
    for t in tilts:  # angle written on the side face, inward of each hole
        a = np.radians(phi0 + t)
        mark = slab(text(f"{int(t)}", 4.6, 0.7), 0, 0.5).rotate([90, 0, 0])
        labels.append(mark.translate([gx + (radius - 12.5) * np.cos(a), -6.5, gz + (radius - 12.5) * np.sin(a)]))
    cradle = nest + tall + web + fan - holes - union(labels)
    fit_test = nest + box(hx, hx + wall, -span, span, -floor, low)
    return cradle, fit_test


# --- cable ----------------------------------------------------------------------------------------
def cable_parts(p):
    d = p["clips"]["cable_d"]
    base = footed_body(1, 1, 2.0)
    zc = 2.0 + d / 2 + 1.6
    ring = cyl_x(d + 3.2, -5, 5) - cyl_x(d, -6, 6)
    mouth = box(-6, 6, -d * 0.36, d * 0.36, 0, d)  # cable is pressed in from above
    stem = box(-5, 5, -2.5, 2.5, 2.0, zc - d / 2)
    clip = base + stem + (ring - mouth).translate([0, 0, zc])
    # USB strain relief: the cable runs along x over two bridges, a zip tie goes through the tunnel under each
    # bridge (along y) and wraps the cable
    bridge = box(-5, 5, -7, 7, 2.0, 9.0) - box(-3.2, 3.2, -8, 8, 4.0, 6.6)
    relief = base + bridge.translate([-11, 0, 0]) + bridge.translate([11, 0, 0])
    return [
        Item("cable_clip", [clip], "PLA", "phase0", qty=6, note=f"{d} mm cable; sits in empty tiles"),
        Item("usb_strain_relief", [relief], "PLA", "phase0", qty=2, note="cable is tied down with a 2.5 mm zip tie"),
    ]


# --- QC part set ----------------------------------------------------------------------------------
def qc_parts(p):
    cfg = p["parts"]
    out = []
    for part, body in generate(p):
        prefix, bin_ = part.id.rsplit("-", 1)
        d = cfg[part.type]
        if part.type == "bracket":
            x = -d["thickness"] / 2
            lines = [(prefix, x, 7.6, 0), (bin_, x, -7.6, 0)]
        elif part.type == "pad":
            lines = [(prefix, 0, 6.6, 0), (bin_, 0, -6.6, 0)]
        else:  # rim and gear: on both sides of the hub bore
            r = d["outer_d"] / 2 * 0.62
            lines = [(prefix, 0, r, 0), (bin_, 0, -r, 0)]
        body = engrave_bottom(body, lines, height=3.6, depth=0.4, stroke=0.65)
        defect = {None: "good", "missing_hole": "missing hole", "broken_corner": "broken corner"}[part.defect]
        out.append(Item(part.id, [body], "PLA", "phase1", group="qc", color=tuple(part.color), note=defect,
                        meta={"heldout": part.heldout, "variant": part.variant}))
    return out


def build_all(p):
    robot = Robot()
    items = baseplate(p, robot) + [tray(p)] + bins(p) + tags(p) + mast_parts(p) + cable_parts(p) + [backdrop(p)]
    return items + qc_parts(p)
