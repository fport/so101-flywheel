"""Visual list of the parts to print (SVG): top and side silhouette, quantity, size and plate of each part."""

from collections import Counter

import numpy as np

from ..sim.parts import TYPE_NAMES
from .geom import union

INK, MUTED, CARD = "#1f2328", "#667085", "#ffffff"
NAMES = {
    "cam_cradle_fit_test": "Cradle fit test",
    "mast_joint_test_socket": "Joint test, female",
    "mast_joint_test_plug": "Joint test, male",
    "mast_head": "Mast head",
    "cam_cradle_c310": "Camera cradle (articulated)",
    "head_axle_pin": "Axle pin",
    "head_lock_pin": "Lock pin",
    "baseplate_A1": "Baseplate A1",
    "baseplate_A2": "Baseplate A2 (arm seat)",
    "baseplate_A3": "Baseplate A3",
    "baseplate_B2": "Baseplate B2 (mast socket)",
    "baseplate_key": "Joint gauge",
    "mast_segment": "Mast segment",
    "cable_clip": "Cable clip",
    "usb_strain_relief": "USB strain relief",
    "tray": "Input tray",
    "inspection_backdrop": "Inspection backdrop",
    "bin_liner": "Bin liner",
    "bin_ok": "OK bin",
    "bin_reject": "Reject bin",
    "tag_0": "AprilTag 0",
    "tag_1": "AprilTag 1",
    "tag_2": "AprilTag 2",
    "tag_3": "AprilTag 3",
    "mast_foot": "Mast foot (optional)",
}
FILL = {"black": "#2b2c30", "metal grey": "#8d9199", "orange": "#e2701a", "white": "#eeeee8"}
DEFECT = {None: "good", "missing_hole": "missing hole", "broken_corner": "broken corner"}


def _fill(filament):
    return next((v for k, v in FILL.items() if k in filament), "#6b6e75")


def _paths(m, side, scale, ox, oy, cut=None):
    """Silhouette: side=False from the top (x, y), True from the side (x, z). If cut is given, the section at that
    height is drawn (so pockets, sockets and holes show). Returns (path d, width, height)."""
    body = m.rotate([-90, 0, 0]) if side else m
    x0, y0, x1, y1 = body.project().bounds()
    cs = body.project() if cut is None else body.slice(cut)
    d = []
    for poly in cs.to_polygons():
        pts = [(ox + (x - x0) * scale, oy + (y1 - y) * scale) for x, y in np.asarray(poly)]
        d.append("M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in pts) + " Z")
    return " ".join(d), (x1 - x0) * scale, (y1 - y0) * scale


def _card(add, x, y, w, h, badge, title, qty, size, parts, fill):
    """parts: [(Manifold, color)] drawn on top of each other in the same frame (for two-color parts)."""
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{CARD}" stroke="#e1e4e8"/>')
    add(f'<rect x="{x + 10}" y="{y + 10}" width="30" height="20" rx="5" fill="{INK}"/>')
    add(f'<text x="{x + 25}" y="{y + 24}" text-anchor="middle" fill="#fff" font-weight="700" font-size="11">{badge}</text>')
    add(f'<text x="{x + w - 12}" y="{y + 26}" text-anchor="end" fill="{INK}" font-weight="700" font-size="15">× {qty}</text>')
    whole = union([m for m, _ in parts])
    lo, hi = np.array(whole.bounding_box()[:3]), np.array(whole.bounding_box()[3:])
    ext = hi - lo
    area_w, area_h = w - 24, h - 96
    # top and side views side by side, same scale
    scale = min((area_w - 12) / (2 * ext[0]), area_h / max(ext[1], ext[2]))
    for k, side in enumerate((False, True)):
        vw, vh = ext[0] * scale, (ext[2] if side else ext[1]) * scale
        ox = x + 12 + k * (area_w / 2 + 6) + (area_w / 2 - 6 - vw) / 2
        oy = y + 40 + (area_h - vh) / 2
        for m, color in parts:
            mlo = np.array(m.bounding_box()[:3])
            dx = (mlo[0] - lo[0]) * scale
            dy = ((hi[2] - m.bounding_box()[5]) if side else (hi[1] - m.bounding_box()[4])) * scale
            d, _, _ = _paths(m, side, scale, ox + dx, oy + dy)
            if side:
                add(f'<path d="{d}" fill="{color}" fill-rule="evenodd" stroke="{INK}" stroke-width="0.6"/>')
                continue
            # top: a section near the top over a faded silhouette; pockets and sockets stay light
            add(f'<path d="{d}" fill="{color}" fill-opacity="0.32" fill-rule="evenodd" stroke="{INK}" stroke-width="0.6"/>')
            z0, z1 = m.bounding_box()[2], m.bounding_box()[5]
            d, _, _ = _paths(m, side, scale, ox + dx, oy + dy, cut=z0 + 0.82 * (z1 - z0))
            if d:
                add(f'<path d="{d}" fill="{color}" fill-rule="evenodd"/>')
    add(f'<text x="{x + 12 + area_w / 4}" y="{y + h - 44}" text-anchor="middle" fill="{MUTED}" font-size="9.5">top</text>')
    add(f'<text x="{x + 12 + 3 * area_w / 4}" y="{y + h - 44}" text-anchor="middle" fill="{MUTED}" font-size="9.5">side</text>')
    add(f'<text x="{x + 12}" y="{y + h - 26}" fill="{INK}" font-weight="600" font-size="12">{title}</text>')
    add(f'<text x="{x + 12}" y="{y + h - 11}" fill="{MUTED}" font-size="10.5">{size}</text>')


def build_catalog_svg(items, jobs, catalog):
    by_name = {i.name: i for i in items}
    job_of, filament_of = {}, {}
    for name, filament, objs in jobs:
        for obj, _, _ in objs:
            base = obj if obj in by_name else obj.rsplit("_", 1)[0]
            job_of[base], filament_of[base] = name[:2], filament
    fixtures = [i for i in items if i.group != "qc"]
    fixtures.sort(key=lambda i: job_of[i.name])
    cols, cw, ch, gap = 5, 236, 236, 12
    left, top = 28, 96
    rows = -(-len(fixtures) // cols)
    qc_types = list(dict.fromkeys(c.type for c in catalog))
    qc_cards = [(t, d) for t in qc_types for d in DEFECT]
    qc_rows = -(-len(qc_cards) // 6)
    qc_top = top + rows * (ch + gap) + 64
    qw = (cols * (cw + gap) - gap - 5 * gap) / 6
    W = left * 2 + cols * (cw + gap) - gap
    H = qc_top + qc_rows * (ch + gap) + 40
    o = []
    add = o.append
    add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}" width="{W:.0f}" height="{H:.0f}" '
        f'font-family="-apple-system, Helvetica, Arial, sans-serif" font-size="12">')
    add(f'<rect width="{W:.0f}" height="{H:.0f}" fill="#f6f7f8"/>')
    total = sum(i.qty for i in items)
    qc_total = sum(1 for i in items if i.group == "qc")
    add(f'<text x="{left}" y="38" font-size="20" font-weight="700" fill="{INK}">Parts to print</text>')
    add(f'<text x="{left}" y="60" fill="{MUTED}">{len(jobs)} plates, {total} parts: {total - qc_total} cell parts '
        f'({len(fixtures)} distinct models) + {qc_total} quality control parts. Black badge is the plate number; each card '
        f'shows top and side views at the same scale.</text>')
    add(f'<text x="{left}" y="84" font-size="14" font-weight="600" fill="{INK}">Cell</text>')
    for k, item in enumerate(fixtures):
        x, y = left + (k % cols) * (cw + gap), top + (k // cols) * (ch + gap)
        color = _fill(filament_of[item.name])
        parts = [(item.parts[0], color)]
        lo, hi = np.array(item.body.bounding_box()[:3]), np.array(item.body.bounding_box()[3:])
        size = " × ".join(f"{v:.0f}" for v in hi - lo) + " mm"
        _card(add, x, y, cw, ch, job_of[item.name], NAMES[item.name], item.qty, size, parts, color)

    add(f'<text x="{left}" y="{qc_top - 30}" font-size="14" font-weight="600" fill="{INK}">Quality control parts</text>')
    add(f'<text x="{left}" y="{qc_top - 12}" fill="{MUTED}">Each card is one type and condition; quantity is white + orange, held-out '
        f'extra in parentheses. Plates 11-16.</text>')
    count = Counter((c.type, c.defect, c.variant, c.heldout) for c in catalog)
    sample = {}
    for c in catalog:
        sample.setdefault((c.type, c.defect), by_name[c.id])
    for k, (t, d) in enumerate(qc_cards):
        x, y = left + (k % 6) * (qw + gap), qc_top + (k // 6) * (ch + gap)
        s, yv = count[(t, d, "W", False)], count[(t, d, "T", False)]
        held = count[(t, d, "W", True)] + count[(t, d, "T", True)]
        item = sample[(t, d)]
        lo, hi = np.array(item.body.bounding_box()[:3]), np.array(item.body.bounding_box()[3:])
        size = f"{s} white + {yv} orange" + (f" (+{held} held-out)" if held else "")
        _card(add, x, y, qw, ch, "QC", f"{TYPE_NAMES[t].capitalize()}, {DEFECT[d]}", s + yv + held, size,
              [(item.parts[0], "#8d9199")], "#8d9199")
    add("</svg>")
    return "\n".join(o)
