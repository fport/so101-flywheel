"""Assembly diagrams: parts floating above their places (exploded view) and a top plan, with numbered labels.

The number on a label is the print file number. Everything in the image is the same geometry as the printed files.
"""

import tempfile
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from ..sim.env import QCSortSim
from ..sim.params import REPO_ROOT, load_params, mm
from ..sim.parts import write_stl
from ..sim.scene import look_at_quat
from .fixtures import MAST_BOSS, build_all
from .showcase import HIDDEN, _font

SIZE = (1800, 1300)
# how far above its place each part is drawn (mm)
LIFT = {"tray": 70, "inspection_backdrop": 70, "bin_ok": 95, "bin_reject": 95, "bin_liner": 175, "tag": 45,
        "mast_segment": 70, "robot": 90}
ROBOT_HIDE = ("cam_", "pocket_")


def _lift(name):
    for key, dz in LIFT.items():
        if name.startswith(key):
            return dz
    return 0


def decorate_factory(items, tmp, exploded):
    def decorate(spec, p):
        spec.visual.global_.offwidth, spec.visual.global_.offheight = SIZE
        spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                         rgb1=[0.96, 0.96, 0.97], rgb2=[0.86, 0.87, 0.89], width=256, height=1536)
        table = spec.geom("table")
        table.size, table.rgba = [4, 4, 0.05], [0.9, 0.9, 0.91, 1]
        spec.visual.headlight.ambient = [0.55, 0.55, 0.57]
        spec.visual.headlight.diffuse = [0.4, 0.4, 0.4]
        for g in spec.geoms:
            if g.name.startswith(HIDDEN) or g.name.startswith(("cam_body", "tag_")):
                g.group = 4
        stack = [spec.body("base")]  # the arm isn't printed; show only its seat
        while stack:
            body = stack.pop()
            for g in body.geoms:
                g.group = 4
            stack += list(body.bodies)
        bins = {i.name: i for i in items if i.name.startswith("bin_")}
        placed = []
        for item in items:
            if item.at and item.name not in ("mast_head", "cam_cradle_c310"):
                bodies = item.preview or item.parts
                if item.name == "mast_segment":
                    bodies = bodies[:1]  # only the bottom segment; three stacked plus the head
                placed += [(item, k, b, item.at) for k, b in enumerate(bodies)]
            if item.name == "bin_liner":  # liners go inside the bins
                for side in ("bin_ok", "bin_reject"):
                    placed.append((item, side, item.parts[0], bins[side].at))
        for item, k, body, at in placed:
            name = f"cad_{item.name}_{k}"
            write_stl(Path(tmp) / f"{name}.stl", body, scale=1e-3)
            spec.add_mesh(name=name, file=str(Path(tmp) / f"{name}.stl"))
            z = item.meta.get("z", 0.0) + (_lift(item.name) if exploded else 0)
            if item.name == "bin_liner" and not exploded:
                z = p["bins"]["wall_thickness"]
            spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, rgba=[*item.part_color(k), 1],
                                    pos=[at[0] / 1000, at[1] / 1000, z / 1000], contype=0, conaffinity=0, group=2)
        for name, pos, target, up, fovy in CAMERAS:
            pos, target = np.array(pos) / 1000, np.array(target) / 1000
            spec.worldbody.add_camera(name=name, pos=pos, quat=look_at_quat(pos, target, up=up), fovy=fovy)

    return decorate


CAMERAS = [
    ("exploded", (-470, -600, 720), (130, 0, 30), (0, 0, 1), 40),
    ("plan", (126, 0, 760), (126, 0, 0), (1, 0, 0), 38),
]

# (text, world point mm)
EXPLODED_LABELS = [
    ("03  A2 tile: the arm sits here", (0, 0, 5)),
    ("04  B2 tile: mast socket", (320, 75, 0)),
    ("07  A1 tile", (-60, -110, 0)),
    ("08  A3 tile", (-60, 110, 0)),
    ("06  Tray", (168, 50, 70 + 6)),
    ("06  Backdrop", (84, 50, 70 + 2)),
    ("09  OK bin (grey)", (10, -200, 95 + 30)),
    ("16  Reject bin (orange)", (5, 147, 95 + 25)),
    ("07  Liner ×2, inside the bins", (63, 147, 175)),
    ("10  Tag tile (4 pcs)", (-42, -168, 45 + 2)),
    ("05  Mast: 3 segments stacked, 02 head on top", (306, 0, MAST_BOSS + 70 + 205)),
]
PLAN_LABELS = [
    ("A2", (-40, -55, 0)), ("B2", (290, -55, 0)), ("A1", (-60, -95, 0)), ("A3", (-60, 95, 0)),
    ("tray", (168, 0, 8)), ("backdrop", (84, 0, 4)), ("OK", (63, -147, 40)), ("REJECT", (63, 147, 40)),
    ("tag 0", (-42, -168, 4)), ("tag 1", (-42, 168, 4)), ("tag 2", (231, -63, 4)), ("tag 3", (231, 63, 4)),
    ("mast", (306, 0, 60)), ("arm seat", (-10, 0, 6)),
]


def _box(draw, xy, text, font, pad=9):
    l, t, r, b = draw.textbbox(xy, text, font=font)
    return (l - pad, t - pad, r + pad, b + pad)


def _hits(box, boxes):
    return any(not (box[2] < o[0] or o[2] < box[0] or box[3] < o[1] or o[3] < box[1]) for o in boxes)


def render(env, camera, labels, plan=False):
    w, h = SIZE
    renderer = mujoco.Renderer(env.model, h, w)
    renderer.update_scene(env.data, camera=camera)
    img = Image.fromarray(renderer.render())
    draw = ImageDraw.Draw(img)
    font = _font(30 if plan else 28)
    anchors = [np.array(env.project(np.array(pt, float) / 1000, camera=camera, width=w, height=h)) for _, pt in labels]
    center = np.mean(anchors, axis=0)
    boxes, placed = [], []
    for (text, _), a in zip(labels, anchors):
        tw = draw.textlength(text, font=font)
        if plan:
            xy = (a[0] - tw / 2, a[1] - 17)
        else:
            # push the label out from the center; a bit further if it hits another label
            d = a - center
            d = d / (np.linalg.norm(d) + 1e-6)
            for step in range(40):
                c = a + d * (130 + 25 * step)
                xy = (min(max(c[0] - tw / 2, 10), w - tw - 20), min(max(c[1] - 17, 10), h - 50))
                if not _hits(_box(draw, xy, text, font), boxes):
                    break
        box = _box(draw, xy, text, font)
        boxes.append(box)
        placed.append((text, xy, box, a))
    for text, xy, box, a in placed:
        if not plan:
            cx, cy = min(max(a[0], box[0]), box[2]), min(max(a[1], box[1]), box[3])
            draw.line([tuple(a), (cx, cy)], fill=(40, 42, 48), width=3)
            draw.ellipse([a[0] - 7, a[1] - 7, a[0] + 7, a[1] + 7], fill=(226, 112, 26), outline=(255, 255, 255), width=2)
    for text, xy, box, a in placed:
        draw.rounded_rectangle(box, radius=8, fill=(30, 32, 36) if plan else (255, 255, 255), outline=(40, 42, 48), width=2)
        draw.text(xy, text, fill=(255, 255, 255) if plan else (24, 25, 28), font=font)
    return img


def render_assembly(out_dir):
    p = load_params()
    out_dir = Path(out_dir)
    items = build_all(p)
    paths = []
    for exploded, camera, labels, stem in ((True, "exploded", EXPLODED_LABELS, "assembly-exploded"),
                                           (False, "plan", PLAN_LABELS, "assembly-top")):
        with tempfile.TemporaryDirectory() as tmp:
            env = QCSortSim([None] * 6, params=p, decorate=decorate_factory(items, tmp, exploded))
        img = render(env, camera, labels, plan=not exploded)
        path = out_dir / f"{stem}.png"
        img.save(path)
        paths.append(path)
    paths.append(build_plan(p, out_dir / "assembly-plan.png"))
    return paths


# --- scaled assembly plan (2D) ---------------------------------------------------------------------
INK, MUTED, TILE, CELL = (24, 25, 28), (102, 112, 133), (233, 234, 236), (190, 194, 201)
ORANGE, GREY, BLACK = (226, 112, 26), (141, 145, 153), (43, 44, 48)


def build_plan(p, path):
    """Scaled top plan: which part goes on which tile, in which orientation. Arm at the bottom, mast at the top."""
    from .fixtures import KEY, _arm_seat, _mast_socket
    from .robot import Robot
    from ..sim.params import pocket_centers

    g, u = p["grid"], p["grid"]["pitch"]
    W, H, s = 2250, 1250, 2.45
    x_hi, y_hi = g["tiles_x"][-1] * u, g["tiles_y"][-1] * u
    ox, oy = 90, 115

    def pt(x, y):  # world mm -> pixels: +x up (mast), +y left (reject bin)
        return ox + (y_hi - y) * s, oy + (x_hi - x) * s

    def rect(x0, x1, y0, y1):
        (a, b), (c, d) = pt(x1, y1), pt(x0, y0)
        return [a, b, c, d]

    img = Image.new("RGB", (W, H), (251, 251, 250))
    d = ImageDraw.Draw(img, "RGBA")
    f_title, f, f_small, f_big = _font(34), _font(24), _font(19), _font(30)
    d.text((ox, 26), "Assembly plan: top view, to scale (1 square = 42 mm)", fill=INK, font=f_title)
    d.text((ox, 70), "You are standing behind the arm, looking toward the mast. The number on a label is the print file number.",
           fill=MUTED, font=f_small)

    seat, seat_cells = _arm_seat(p, Robot())
    _, socket_cells = _mast_socket(p)
    solid = seat_cells | socket_cells
    skip = set(g.get("skip", []))
    tiles = []
    for ci, (x0, x1) in enumerate(zip(g["tiles_x"], g["tiles_x"][1:])):
        for ri, (y0, y1) in enumerate(zip(g["tiles_y"], g["tiles_y"][1:])):
            name = f"{'ABCDE'[ci]}{ri + 1}"
            if name not in skip:
                tiles.append((name, x0 * u, x1 * u, y0 * u, y1 * u))
    files = {"A1": "07", "A2": "03", "A3": "08", "B2": "04"}
    for name, x0, x1, y0, y1 in tiles:
        d.rectangle(rect(x0, x1, y0, y1), fill=TILE)
        for i in range(round(x0 / u), round(x1 / u)):
            for j in range(round(y0 / u), round(y1 / u)):
                if (i, j) not in solid:
                    a, b, c, e = rect(i * u + 3, (i + 1) * u - 3, j * u + 3, (j + 1) * u - 3)
                    d.rounded_rectangle([a, b, c, e], radius=8, outline=CELL, width=2)
    for name, x0, x1, y0, y1 in tiles:
        d.rectangle(rect(x0, x1, y0, y1), outline=(58, 61, 66), width=4)

    def label(x, y, text, fill=(255, 255, 255, 235), ink=INK, font=f, anchor="mm"):
        cx, cy = pt(x, y)
        l, t, r, b = d.textbbox((cx, cy), text, font=font, anchor=anchor)
        d.rounded_rectangle([l - 8, t - 6, r + 8, b + 6], radius=7, fill=fill, outline=(58, 61, 66, 200), width=2)
        d.text((cx, cy), text, fill=ink, font=font, anchor=anchor)

    # arm seat and the direction the arm faces
    for poly in seat.slice(2.5).to_polygons():
        d.line([pt(x, y) for x, y in poly] + [pt(*poly[0])], fill=ORANGE, width=5)
    a, b = pt(-30, 0), pt(30, 0)
    d.line([a, b], fill=ORANGE, width=6)
    d.polygon([pt(42, 0), pt(26, 9), pt(26, -9)], fill=ORANGE)
    label(-55, 0, "ARM SEAT  (arm faces the mast)", font=f_small)

    # backdrop
    i = p["inspection"]
    (bx, by), (bsx, bsy) = i["backdrop_center"], i["backdrop_size"]
    d.rectangle(rect(bx - bsx / 2, bx + bsx / 2, by - bsy / 2, by + bsy / 2), fill=(*BLACK, 215))
    label(bx, by, "06  BACKDROP", fill=(30, 32, 36, 240), ink=(255, 255, 255))

    # tray and pocket numbers
    t = p["tray"]
    (tx, ty), (tsx, tsy) = t["center"], t["size"]
    d.rectangle(rect(tx - tsx / 2, tx + tsx / 2, ty - tsy / 2, ty + tsy / 2), fill=(31, 31, 33, 230))
    psx, psy = t["pocket_size"]
    for k, (cx, cy) in enumerate(pocket_centers(p), start=1):
        cx, cy = cx * 1000, cy * 1000
        d.rounded_rectangle(rect(cx - psx / 2, cx + psx / 2, cy - psy / 2, cy + psy / 2), radius=6,
                            outline=(150, 154, 162), width=2)
        d.text(pt(cx, cy), str(k), fill=(255, 255, 255), font=f_big, anchor="mm")
    label(tx, ty - tsy / 2 - 8, "06  TRAY (1 on the reject side)", font=f_small, anchor="lm")

    # bins
    b = p["bins"]
    for key, title, color, file in (("ok", "09  OK BIN (grey)", GREY, "09"), ("reject", "16  REJECT BIN (orange)", ORANGE, "16")):
        (cx, cy), (sx, sy) = b[key]["center"], b["size"]
        d.rectangle(rect(cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2), fill=(*color, 200), outline=(58, 61, 66), width=3)
        inner = sx / 2 - 22
        d.rectangle(rect(cx - inner, cx + inner, cy - inner, cy + inner), outline=(255, 255, 255, 170), width=2)
        label(cx + 12, cy, title, font=f_small)
        label(cx - 30, cy, "liner inside (07)", font=f_small)

    # tag tiles: the white tag area is off-center on the two near tiles
    for tg in p["tags"]["items"]:
        (cx, cy), side = tg["tile"], tg["units"] * u - 0.5
        d.rectangle(rect(cx - side / 2, cx + side / 2, cy - side / 2, cy + side / 2), fill=(*BLACK, 235))
        (px_, py_), paper = tg["center"], tg["size"] * 10 / 8  # white inlay
        d.rectangle(rect(px_ - paper / 2, px_ + paper / 2, py_ - paper / 2, py_ + paper / 2), fill=(255, 255, 255))
        d.text(pt(px_, py_), f"tag {tg['id']}", fill=INK, font=f_small, anchor="mm")

    # mast socket
    (mx, my), half = p["mast"]["pos"], p["mast"]["size"] / 2 + 4
    d.rectangle(rect(mx - half, mx + half, my - half, my + half), fill=(*GREY, 255), outline=(58, 61, 66), width=3)
    label(mx, my - tsy / 2 - 8, "05  MAST SOCKET (+02 head on top)", font=f_small, anchor="lm")

    # tile names
    for name, x0, x1, y0, y1 in tiles:
        x = x0 + 14 if name != "B2" else x1 - 14
        y = (y0 + y1) / 2 if name == "A2" else (y1 - 34 if name == "A3" else y0 + 34)
        if name == "B2":
            y = y0 + 40
        label(x, y, f"{name}  ({files.get(name, '')})", fill=(58, 61, 66, 235), ink=(255, 255, 255), font=f_small)

    # tongue / pocket marks: tongues on the neighboring tiles, pockets under A2
    key = np.array(KEY, float)
    for xs, y in [((-63, -21, 21, 63, 105), -84), ((-63, -21, 21, 63, 105), 84)]:
        for x in xs:
            poly = key[:, ::-1] * [1, 1] * 0.6 + [x, y]
            d.polygon([pt(a_, b_) for a_, b_ in poly], fill=(58, 61, 66, 160))
    for y in (-63, -21, 21, 63):
        poly = key * 0.6 + [126, y]
        d.polygon([pt(a_, b_) for a_, b_ in poly], fill=(58, 61, 66, 160))

    # orientation text
    left, top = pt(x_hi, y_hi)
    right, bottom = pt(g["tiles_x"][0] * u, g["tiles_y"][0] * u)
    d.text(((left + right) / 2, bottom + 30), "↓ behind the arm (you are here)", fill=MUTED, font=f_small, anchor="mm")

    # steps
    steps = [
        ("Tiles", "A1 (07), A3 (08) and B2 (04) side by side on the table. Their tongue edges face where A2 will go."),
        ("", "Lower A2 (03) last, onto the tongues: they seat in the pockets underneath."),
        ("", "Seat the arm in the orange-outlined seat on A2 and clamp it. Clamp the mast end of B2 to the table."),
        ("Backdrop", "The flat plate from 06: on the 2×4 squares in front of the arm seat on A2. Either way round."),
        ("Tray", "The pocketed part from 06: on the 2×4 squares of B2 next to A2. 1-2-3 near the arm, 1 on the reject (left) side."),
        ("Bins", "OK (09, grey) on the 3×3 squares of A1, REJECT (16, orange) on the 3×3 squares of A3. Liners (07) inside."),
        ("Tags", "10: the two large tiles on the outer corners of A1 and A3, the two small ones on B2 between the tray and the mast."),
        ("", "On the large tiles the tag is off-center: the white number faces the bin, the tag faces behind the arm (down)."),
        ("", "The tile number must match its place on the plan (tag 0-3); for the small tiles see the list for 10."),
        ("Mast", "05: three segments stacked, the first in the socket on B2. 02 head on top, camera in the cradle."),
    ]
    sx0, sy0, wrap = 1420, 140, 40
    d.text((sx0, sy0), "Setup order", fill=INK, font=_font(30))
    y = sy0 + 56
    n = 0
    for head, text in steps:
        if head:
            n += 1
            d.ellipse([sx0, y + 2, sx0 + 34, y + 36], fill=ORANGE)
            d.text((sx0 + 17, y + 19), str(n), fill=(255, 255, 255), font=f_small, anchor="mm")
            d.text((sx0 + 48, y + 4), head, fill=INK, font=f)
            y += 38
        words, line, lines = text.split(), "", []
        for w_ in words:
            if d.textlength(line + " " + w_, font=f_small) > 740:
                lines.append(line)
                line = w_
            else:
                line = (line + " " + w_).strip()
        lines.append(line)
        for ln in lines:
            d.text((sx0 + 48, y), ln, fill=(52, 56, 64), font=f_small)
            y += 27
        y += 10
    img.save(path)
    return path
