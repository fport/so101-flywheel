"""Scaled diagram of the cell (SVG): top view, side view and print order. All from params.yaml."""

import numpy as np

from ..sim.env import QCSortSim
from ..sim.ik import IK
from ..sim.params import MAST_BOSS, mast_top, pocket_centers, top_camera_pose

INK, MUTED, LINE = "#1f2328", "#667085", "#c9ced6"
PLATE, TRAY, BACK = "#3b3d42", "#1f1f21", "#17171a"
FILAMENT = {"black": "#2b2c30", "metal grey": "#8d9199", "orange": "#e2701a", "white": "#eeeee8"}


def _rgb(c):
    return "#" + "".join(f"{int(round(255 * v)):02x}" for v in c)


def _camera_footprint(top, fovy):
    """Points where the four corners of the top camera image land on the baseplate (mm)."""
    pos, direction = top
    z = -direction
    x = np.cross([-1.0, 0, 0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    tv = np.tan(np.radians(fovy) / 2)
    th = tv * 4 / 3
    out = []
    for sx, sy in ((-1, 1), (1, 1), (1, -1), (-1, -1)):
        d = -z + sx * th * x + sy * tv * y
        out.append((pos - d * pos[2] / d[2])[:2])
    return np.array(out)


def _arm_side(p):
    """Arm joint points (x, z) in the inspection pose, mm."""
    env = QCSortSim()
    show, frame = env.inspection_pose()
    q = IK(env.model).solve(show, frame).q
    env.data.qpos[:5] = q
    import mujoco

    mujoco.mj_forward(env.model, env.data)
    pts = [env.data.xanchor[j] for j in (1, 2, 3)] + [env.data.xpos[env.model.body("gripper").id],
                                                         env.data.site_xpos[env.model.site("pinch").id]]
    return [(float(a[0]) * 1000, float(a[2]) * 1000) for a in pts]


def build_svg(p, jobs):
    g, u = p["grid"], p["grid"]["pitch"]
    top, mast, tray, ins = p["cameras"]["top"], p["mast"], p["tray"], p["inspection"]
    o = []
    add = o.append
    W, H = 1280, 930
    add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'font-family="-apple-system, Helvetica, Arial, sans-serif" font-size="12">')
    add(f'<rect width="{W}" height="{H}" fill="#fbfbfa"/>')
    add(f'<text x="28" y="36" font-size="20" font-weight="700" fill="{INK}">SO-101 quality control cell</text>')
    add(f'<text x="28" y="56" fill="{MUTED}">Scaled diagram, generated from hardware/params.yaml. Dimensions in mm.</text>')

    # --- top view: arm at the top, +x down, +y right (same orientation as the top camera image) ---
    s = 1.18
    ox, oy = 300, 200  # page position of the origin (the arm's pan axis)
    P = lambda x, y: (ox + y * s, oy + x * s)  # noqa: E731

    def rect(x0, x1, y0, y1, **kw):
        (px, py), (qx, qy) = P(x0, y0), P(x1, y1)
        attrs = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in kw.items())
        add(f'<rect x="{min(px, qx):.1f}" y="{min(py, qy):.1f}" width="{abs(qx - px):.1f}" height="{abs(qy - py):.1f}" {attrs}/>')

    def label(x, y, text, **kw):
        px, py = P(x, y)
        kw = {"text_anchor": "middle", "fill": INK, **kw}
        attrs = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in kw.items())
        add(f'<text x="{px:.1f}" y="{py:.1f}" {attrs}>{text}</text>')

    add(f'<text x="28" y="92" font-size="14" font-weight="600" fill="{INK}">Top view</text>')
    skip = set(g.get("skip", []))
    for ci, (x0, x1) in enumerate(zip(g["tiles_x"], g["tiles_x"][1:])):
        for ri, (y0, y1) in enumerate(zip(g["tiles_y"], g["tiles_y"][1:])):
            name = f"{'ABCDE'[ci]}{ri + 1}"
            if name in skip:
                rect(x0 * u, x1 * u, y0 * u, y1 * u, fill="none", stroke=LINE, stroke_dasharray="5 4")
                label((x0 + x1) / 2 * u, (y0 + y1) / 2 * u, "not printed", fill=MUTED, font_size=11)
                continue
            rect(x0 * u, x1 * u, y0 * u, y1 * u, fill=PLATE, stroke="#fbfbfa", stroke_width=2)
            for i in range(x0 + 1, x1):
                (ax, ay), (bx, by) = P(i * u, y0 * u), P(i * u, y1 * u)
                add(f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx:.1f}" y2="{by:.1f}" stroke="#4a4d53" stroke-width="0.6"/>')
            for j in range(y0 + 1, y1):
                (ax, ay), (bx, by) = P(x0 * u, j * u), P(x1 * u, j * u)
                add(f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx:.1f}" y2="{by:.1f}" stroke="#4a4d53" stroke-width="0.6"/>')
            label(x0 * u + 12, y0 * u + 14, name, fill="#9aa0a9", font_size=11, text_anchor="start")
    # area the camera sees
    cam, cam_dir = top_camera_pose(p)
    fp = _camera_footprint((cam, cam_dir), top["fovy"])
    add('<clipPath id="top-panel"><rect x="20" y="100" width="580" height="520"/></clipPath>')
    add('<polygon clip-path="url(#top-panel)" points="' + " ".join(f"{P(x, y)[0]:.1f},{P(x, y)[1]:.1f}" for x, y in fp) +
        '" fill="#f2c94c" fill-opacity="0.10" stroke="#d9a400" stroke-width="1.2" stroke-dasharray="6 4"/>')
    # reach band
    m = p["measured"]
    for r, dash in ((m["reach_min"], "3 3"), (m["reach_comfort"], "3 3")):
        (ax, ay), (bx, by) = P(0, -r), P(0, r)
        add(f'<path d="M{ax:.1f} {ay:.1f} A{r * s:.1f} {r * s:.1f} 0 0 0 {bx:.1f} {by:.1f}" fill="none" stroke="#7fb3ff" '
            f'stroke-width="1" stroke-dasharray="{dash}"/>')
    # fixtures
    bx_, by_ = ins["backdrop_center"]
    bw, bh = ins["backdrop_size"]
    rect(bx_ - bw / 2, bx_ + bw / 2, by_ - bh / 2, by_ + bh / 2, fill=BACK, stroke="#55585e")
    label(bx_ + 4, by_, "inspection backdrop", fill="#d0d3d8", font_size=11)
    tx, ty = tray["center"]
    tw, th = tray["size"]
    rect(tx - tw / 2, tx + tw / 2, ty - th / 2, ty + th / 2, fill=TRAY, stroke="#6b6e75")
    ps = tray["pocket_size"][0]
    for k, (cx, cy) in enumerate(pocket_centers(p), start=1):
        cx, cy = cx * 1000, cy * 1000
        rect(cx - ps / 2, cx + ps / 2, cy - ps / 2, cy + ps / 2, fill="#2b2c2f", stroke="#6b6e75", rx=3)
        label(cx + 4, cy, str(k), fill="#e6e7ea", font_size=12, font_weight=600)
    for name, text in (("ok", "OK bin"), ("reject", "REJECT bin")):
        b = p["bins"][name]
        cx, cy = b["center"]
        half = p["bins"]["size"][0] / 2
        rect(cx - half, cx + half, cy - half, cy + half, fill=_rgb(b["color"]), fill_opacity=0.9, stroke="#fbfbfa", rx=5)
        rect(cx - half + 11, cx + half - 11, cy - half + 11, cy + half - 11, fill="#111214", fill_opacity=0.55, rx=3)
        label(cx + 4, cy, text, fill="#ffffff", font_weight=600)
    for t in p["tags"]["items"]:
        cx, cy = t["center"]
        h = t["size"] * 10 / 8 / 2
        rect(cx - h, cx + h, cy - h, cy + h, fill="#f4f4f4", stroke="#9aa0a9")
        rect(cx - t["size"] / 2, cx + t["size"] / 2, cy - t["size"] / 2, cy + t["size"] / 2, fill="#111")
        label(cx + 4, cy, f"tag {t['id']}", fill="#fff", font_size=10)
    # arm
    px, py = P(0, 0)
    add(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{34 * s:.1f}" fill="#f1f1ee" stroke="#8a8f98" stroke-width="1.5"/>')
    label(4, 0, "SO-101", font_weight=700)
    # inspection pose
    ix, _ = ins["pose"]
    px, py = P(ix, 0)
    add(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="13" fill="none" stroke="#f2c94c" stroke-width="1.6" stroke-dasharray="3 3"/>')
    # mast and camera
    mx, my = mast["pos"]
    half = mast["size"] / 2
    rect(mx - half, mx + half, my - half, my + half, fill="#e2701a", stroke="#fbfbfa", rx=2)
    cx, cy = cam[:2]
    px, py = P(cx, cy)
    add(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="7" fill="#111" stroke="#f2c94c" stroke-width="2"/>')
    label(mx + 4, my + 94, "camera mast", font_size=11, text_anchor="start")
    label(cx - 16, cy, "camera", font_size=11, fill="#e6e7ea")
    # legend
    lx, ly = 28, 650
    notes = [("#d9a400", "area the camera sees (C310, narrow-end assumption)"), ("#7fb3ff", f"vertical grasp band: {m['reach_min']}–{m['reach_comfort']} mm"),
             ("#f2c94c", "inspection pose: the part is shown to the camera here, 12 cm up")]
    for k, (color, text) in enumerate(notes):
        add(f'<line x1="{lx}" y1="{ly + k * 18}" x2="{lx + 22}" y2="{ly + k * 18}" stroke="{color}" stroke-width="2" stroke-dasharray="5 3"/>')
        add(f'<text x="{lx + 30}" y="{ly + k * 18 + 4}" fill="{MUTED}">{text}</text>')
    x0, x1 = g["x_units"][0] * u, g["x_units"][1] * u
    y0, y1 = g["y_units"][0] * u, g["y_units"][1] * u
    add(f'<text x="{lx}" y="{ly + 62}" fill="{MUTED}">Baseplate {x1 - x0} × {y1 - y0} mm, T-shaped, 4 tiles (A1, A2, A3, B2).</text>')
    add(f'<text x="{lx}" y="{ly + 80}" fill="{MUTED}">Arm at the top; same orientation as the top camera image. Pocket 1 is on the arm\'s left.</text>')

    # --- side view (x-z) ---
    s2 = 0.62
    sx0, sz0 = 700, 600  # page position of x=0, z=0
    S = lambda x, z: (sx0 + x * s2, sz0 - z * s2)  # noqa: E731
    add(f'<text x="640" y="92" font-size="14" font-weight="600" fill="{INK}">Side view</text>')

    def srect(x0, x1, z0, z1, **kw):
        (px, py), (qx, qy) = S(x0, z1), S(x1, z0)
        attrs = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in kw.items())
        add(f'<rect x="{px:.1f}" y="{py:.1f}" width="{qx - px:.1f}" height="{qy - py:.1f}" {attrs}/>')

    def stext(x, z, text, **kw):
        px, py = S(x, z)
        kw = {"fill": MUTED, "font_size": 11, **kw}
        attrs = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in kw.items())
        add(f'<text x="{px:.1f}" y="{py:.1f}" {attrs}>{text}</text>')

    near, far = fp[:, 0].min(), fp[:, 0].max()
    (cxp, czp) = S(cam[0], cam[2])
    add(f'<polygon points="{cxp:.1f},{czp:.1f} {S(near, 0)[0]:.1f},{S(near, 0)[1]:.1f} {S(far, 0)[0]:.1f},{S(far, 0)[1]:.1f}" '
        f'fill="#f2c94c" fill-opacity="0.12" stroke="#d9a400" stroke-dasharray="6 4"/>')
    srect(g["x_units"][0] * u, 3 * u, -g["thickness"], 0, fill=PLATE)
    srect(3 * u, g["x_units"][1] * u, -g["thickness"], 0, fill=PLATE)
    srect(tx - tw / 2, tx + tw / 2, 0, tray["floor"] + tray["pocket_depth"], fill=TRAY)
    srect(bx_ - bw / 2, bx_ + bw / 2, 0, 2, fill=BACK)
    # mast segments
    top_z = mast_top(p)
    srect(mx - half - 4, mx + half + 4, 0, MAST_BOSS, fill="#2c2d31")
    for k in range(mast["segments"]):
        z0 = MAST_BOSS + k * mast["segment"]
        srect(mx - half, mx + half, z0, z0 + mast["segment"], fill="#232427", stroke="#fbfbfa", stroke_width=1.5)
        stext(mx + half + 8, z0 + mast["segment"] / 2, f"segment {k + 1}")
    srect(mx - half - 3, mx + half + 3, top_z - 45, top_z + 3, fill="#e2701a")
    add(f'<line x1="{S(mx - half - 3, 0)[0]:.1f}" y1="{S(0, top_z - 22)[1]:.1f}" x2="{cxp + 12:.1f}" y2="{czp - 16:.1f}" stroke="#e2701a" stroke-width="7"/>')
    tilt = mast["head"]["tilt"]
    add(f'<g transform="translate({cxp:.1f} {czp:.1f}) rotate({tilt:.1f})"><rect x="-13" y="-17" width="26" height="17" rx="5" fill="#111"/>'
        f'<circle cx="0" cy="0" r="4" fill="#f2c94c"/></g>')
    stext(cam[0] - 250, cam[2] + 34, f"Logitech C310, tilted {tilt:g}° (articulated)", fill=INK, font_size=12)
    # arm
    pts = _arm_side(p)
    srect(-58, 28, 0, 95, fill="#f1f1ee", stroke="#8a8f98")
    add('<polyline points="' + " ".join(f"{S(x, z)[0]:.1f},{S(x, z)[1]:.1f}" for x, z in pts) +
        '" fill="none" stroke="#8a8f98" stroke-width="9" stroke-linecap="round" stroke-linejoin="round"/>')
    add('<polyline points="' + " ".join(f"{S(x, z)[0]:.1f},{S(x, z)[1]:.1f}" for x, z in pts) +
        '" fill="none" stroke="#f1f1ee" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/>')
    ix, iz = ins["pose"]
    hit = cam[0] - (cam[0] - ix) * cam[2] / (cam[2] - iz)
    add(f'<line x1="{cxp:.1f}" y1="{czp:.1f}" x2="{S(hit, 0)[0]:.1f}" y2="{S(hit, 0)[1]:.1f}" stroke="#d9a400" stroke-width="1.2"/>')
    px, py = S(ix + 6, iz - 14)
    add(f'<rect x="{px - 9:.1f}" y="{py - 9:.1f}" width="18" height="18" rx="3" fill="#a9adb5" stroke="{INK}"/>')
    stext(ix + 30, iz + 2, "inspection pose", fill=INK)
    stext(tx - 30, 26, "tray")
    stext(bx_ - 40, -24, "backdrop")
    stext(-150, 60, "SO-101", fill=INK, font_weight=700, font_size=12)
    # dimension lines
    xd = g["x_units"][1] * u + 62
    (ax, ay), (bx2, by2) = S(xd, 0), S(xd, top_z)
    add(f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx2:.1f}" y2="{by2:.1f}" stroke="{MUTED}"/>')
    for z in (0, top_z):
        add(f'<line x1="{ax - 5:.1f}" y1="{S(0, z)[1]:.1f}" x2="{ax + 5:.1f}" y2="{S(0, z)[1]:.1f}" stroke="{MUTED}"/>')
    stext(xd + 8, top_z / 2, f"mast {top_z:.0f} mm", fill=INK)
    stext(xd + 8, top_z / 2 - 16, f"lens {cam[2]:.0f} mm")
    stext(g["x_units"][0] * u - 60, -42, f"baseplate {g['thickness']:g} mm", font_size=11)

    # --- print order ---
    bx0, by0 = 640, 690
    add(f'<text x="{bx0}" y="{by0}" font-size="14" font-weight="600" fill="{INK}">Print order: hardware/print/</text>')
    per_col = -(-len(jobs) // 2)
    for k, (name, filament, objs) in enumerate(jobs):
        cx = bx0 + (k // per_col) * 318
        cy = by0 + 24 + (k % per_col) * 25
        color = next((v for key, v in FILAMENT.items() if key in filament), "#55585e")
        add(f'<rect x="{cx}" y="{cy - 12}" width="14" height="14" rx="3" fill="{color}" stroke="#9aa0a9"/>')
        add(f'<text x="{cx + 22}" y="{cy}" fill="{INK}"><tspan font-weight="700">{name[:2]}</tspan> {name[3:].replace("_", " ")}'
            f'<tspan fill="{MUTED}"> · {len(objs)} parts</tspan></text>')
    add("</svg>")
    return "\n".join(o)
