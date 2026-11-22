"""Tray inspection from the top camera: one frame -> AprilTags -> six rectified pocket crops -> measured attributes.

The four baseplate tags have known positions (params.yaml), so a homography maps the tray plane in mm to the image.
Each pocket is warped to a square at a fixed scale (PX_PER_MM), the same for every frame, so the measurements below
are in mm. measure() checks what can be counted on the silhouette of a light part on the black tray: holes (the tray
shows through), rim spokes, the gap of missing gear teeth, and the corner bitten off a broken pad or bracket (cut
through the whole part since 2026-10-31, so it shows in the outline).
"""

import json
import re

import cv2
import numpy as np

from ..params import CATALOG_JSON, load_params, pocket_centers

PX_PER_MM = 4.0  # rectified crop scale; the 1280x960 frame has ~2.2 px/mm at the tray, so crops are enlarged ~2x
WINDOW_MM = 56  # crop side; a 38 mm pocket with room for a part that is off its pocket centre
RECENTER_MM = 12  # largest shift towards the part found near the pocket centre
BRIDGE_MM = 1.5  # dark lines thinner than this inside a part don't split it
DARK = 50  # lowest tray/part split; the split itself is Otsu's per crop (real tray < 20, sim tray ~ 60, parts > 90)
ID_RE = re.compile(r"[JDBL]-[WT]-\d\d")


def capture(index, size=(1280, 960), warmup=20):
    """One frame (RGB) from the top camera at the resolution used for inspection (recording runs at 640x480)."""
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    ok, f = False, None
    for _ in range(warmup):  # let exposure settle
        ok, f = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"camera {index} gave no frame")
    return f[..., ::-1].copy()


def find_tags(rgb):
    """-> {tag id: (u, v) centre in the image}."""
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11))
    corners, ids, _ = det.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    return {} if ids is None else {int(i): c[0].mean(axis=0) for i, c in zip(ids.ravel(), corners)}


def tray_homography(tags, p=None):
    """Baseplate (x, y) in mm -> image (u, v), from the tag centres; needs at least 3 of the 4 tags."""
    p = p or load_params()
    world = {t["id"]: t["center"] for t in p["tags"]["items"]}
    seen = [i for i in sorted(tags) if i in world]
    if len(seen) < 3:
        raise RuntimeError(f"found tags {sorted(tags)}, need 3 of {sorted(world)}: is the top camera on the cell?")
    src = np.float32([world[i] for i in seen])
    dst = np.float32([tags[i] for i in seen])
    if len(seen) == 3:
        return np.vstack([cv2.getAffineTransform(src, dst), [0, 0, 1]])
    return cv2.findHomography(src, dst)[0]


def _warp(rgb, H, center):
    """Square crop around center (mm). Up in the crop is towards the arm (-x), right is +y: the top camera's view."""
    n = int(WINDOW_MM * PX_PER_MM)
    half = (n - 1) / 2
    s = 1 / PX_PER_MM
    # crop (col, row) -> baseplate mm: x grows down the crop, y grows to the right
    T = np.array([[0, s, center[0] - half * s],
                  [s, 0, center[1] - half * s],
                  [0, 0, 1]])
    return cv2.warpPerspective(rgb, H @ T, (n, n), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)


def pocket_crops(rgb, H, p=None):
    """-> [(pocket no, crop RGB)] for the 6 pockets; each crop is re-centred on the light part nearest its centre."""
    p = p or load_params()
    out = []
    for k, c in enumerate(pocket_centers(p)):
        c = np.array(c) * 1000  # pocket_centers is in m
        crop = _warp(rgb, H, c)
        found = _central_part(crop)
        if found is not None:
            n = crop.shape[0]
            dy, dx = (found[1] - (n - 1) / 2) / PX_PER_MM, (found[2] - (n - 1) / 2) / PX_PER_MM  # col -> y, row -> x
            shift = np.clip([dx, dy], -RECENTER_MM, RECENTER_MM)
            crop = _warp(rgb, H, c + shift)
        out.append((k + 1, crop))
    return out


def _central_part(rgb, min_mm2=150):
    """Light component nearest the crop centre -> (mask, centre col, centre row) or None for an empty pocket."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    split = max(DARK, cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])
    light = (g > split).astype(np.uint8)
    # pieces split by a thin dark line (a shaded groove across a pad) still belong to one part
    n, lab, st, cen = cv2.connectedComponentsWithStats(cv2.dilate(light, _disc(BRIDGE_MM)))
    h, w = g.shape
    big = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= min_mm2 * PX_PER_MM ** 2]
    if not big:
        return None
    k = min(big, key=lambda i: np.hypot(cen[i][0] - w / 2, cen[i][1] - h / 2))
    mask = ((lab == k) & (light > 0)).astype(np.uint8)
    m = cv2.moments(mask, binaryImage=True)
    return mask, m["m10"] / m["m00"], m["m01"] / m["m00"]


def _disc(mm):
    n = max(3, int(round(mm * PX_PER_MM)) | 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))


def _ring(gray, cx, cy, r_mm, n=360, min_deg=5):
    """Light stretches around a circle of r_mm about (cx, cy) -> (count, widest dark gap in degrees). Stretches
    narrower than min_deg are noise and count as dark."""
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs = np.clip((cx + r_mm * PX_PER_MM * np.cos(ang)).astype(int), 0, gray.shape[1] - 1)
    ys = np.clip((cy + r_mm * PX_PER_MM * np.sin(ang)).astype(int), 0, gray.shape[0] - 1)
    v = gray[ys, xs]
    light = v > (np.percentile(v, 90) + np.percentile(v, 10)) / 2
    if light.all() or not light.any():
        return int(light.all()), 0 if light.all() else 360
    light = np.roll(light, -int(np.argmin(light)))  # start on a dark sample, so no stretch wraps around
    runs, k = [], 0
    while k < n:
        j = k
        while j < n and light[j] == light[k]:
            j += 1
        runs.append((bool(light[k]), (j - k) * 360 / n))
        k = j
    lit = [w >= min_deg for on, w in runs if on]
    gaps, gap = [], 0.0
    for on, w in runs:  # a dropped narrow stretch joins the gaps on both sides
        if on and w >= min_deg:
            gaps.append(gap)
            gap = 0.0
        else:
            gap += w
    gaps[0] = gaps[0] + gap if gaps else gap  # the last gap runs on into the first
    return sum(lit), int(max(gaps) if gaps else 360)


CORNER_MM = 12  # side of the square checked at each corner of the outline
# a broken corner (18 x 9 mm on a pad, 10 x 16 mm on a bracket) leaves 60-130 mm2 of its square empty; a sound corner
# under 30 seen from the top camera. In the sim's top view a bracket upright widens the outline more (up to ~75 at
# every corner), so for brackets one corner must also stand out from the others.
CORNER_GAP_MM2, CORNER_LEAD_MM2 = 45, 25


def _corner_gaps(mask, outline):
    """Empty area (mm2) in a CORNER_MM square at each corner of the smallest rectangle around the part."""
    box = cv2.boxPoints(cv2.minAreaRect(outline))
    gaps = []
    for corner in box:
        # square spanned by the two rectangle edges that meet at this corner, CORNER_MM along each
        others = sorted(box, key=lambda q: np.hypot(*(q - corner)))[1:3]
        u, v = [(q - corner) / np.hypot(*(q - corner)) * CORNER_MM * PX_PER_MM for q in others]
        square = np.int32([corner, corner + u, corner + u + v, corner + v])
        region = np.zeros_like(mask)
        cv2.fillPoly(region, [square], 1)
        empty = np.sum((region > 0) & (mask == 0)) / PX_PER_MM ** 2
        gaps.append(round(float(empty), 1))
    return gaps


def measure(rgb, ptype):
    """What the silhouette shows, in mm. ptype picks the checks. -> dict with missing_hole / broken as True, False
    or None (not measurable here), plus the numbers they came from."""
    dims = load_params()["parts"]
    found = _central_part(rgb)
    if found is None:
        return {"empty": True, "missing_hole": None, "broken": None}
    mask, cx, cy = found
    g = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (3, 3), 0)
    # the outline comes from the mask with thin dark lines closed; holes from the mask as it is
    solid = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _disc(BRIDGE_MM))
    outline = max(cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
    cs, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    holes = []
    for j in (i for i in range(len(cs)) if hier[0][i][3] != -1):
        a = cv2.contourArea(cs[j]) / PX_PER_MM ** 2
        if a >= 1.0:
            m = cv2.moments(cs[j])
            perim = cv2.arcLength(cs[j], True) / PX_PER_MM
            if m["m00"]:  # round: 1.0; a shadow crescent next to a raised hub is well under 0.5
                holes.append({"mm2": round(a, 1), "round": round(4 * np.pi * a / perim ** 2, 2),
                              "dx": round(float(m["m10"] / m["m00"] - cx) / PX_PER_MM, 1),
                              "dy": round(float(m["m01"] / m["m00"] - cy) / PX_PER_MM, 1)})
    r = {"holes": holes, "missing_hole": None, "broken": None}

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    r["saturation"] = int(np.median(hsv[..., 1][mask > 0]))
    r["variant"] = "T" if r["saturation"] > 90 else "W"

    near = [h for h in holes if np.hypot(h["dx"], h["dy"]) <= 6]  # parallax moves a tall hub's bore a few mm
    if ptype in ("rim", "gear"):
        hole_mm2 = np.pi * (dims[ptype]["bore_d"] / 2) ** 2
        r["missing_hole"] = not any(h["mm2"] >= 0.2 * hole_mm2 and h["round"] >= 0.5 for h in near)
    elif ptype == "bracket":
        r["missing_hole"] = not any(h["mm2"] >= 8 and h["round"] >= 0.5 for h in holes)
    elif ptype == "pad":
        (_, _), (w, h), ang = cv2.minAreaRect(outline)
        axis = np.radians(ang if w >= h else ang + 90)
        ears = [x for x in holes if x["mm2"] >= 2.5
                and abs(x["dx"] * np.cos(axis) + x["dy"] * np.sin(axis)) >= dims["pad"]["block"][0] / 2 - 2]
        r["ear_holes"] = len(ears)
        r["missing_hole"] = len(ears) == 0

    if ptype in ("pad", "bracket"):
        gaps = _corner_gaps(solid, outline)
        r["corner_gap_mm2"] = gaps
        g = sorted(gaps)
        if ptype == "pad":
            r["broken"] = g[-1] >= CORNER_GAP_MM2
        else:  # a bracket upright seen at an angle can widen every corner a little: one must also stand out
            r["broken"] = g[-1] >= CORNER_GAP_MM2 and g[-1] - g[-2] >= CORNER_LEAD_MM2
    if ptype in ("rim", "gear"):
        (ex, ey), _ = cv2.minEnclosingCircle(outline)
    if ptype == "rim":  # between the hub (6.5 mm) and the inner lip (14.4 mm), clear of both
        counts = [_ring(g, ex, ey, rr)[0] for rr in (8.0, 9.0, 10.0)]
        r["spokes"] = int(np.median(counts))
        n = dims["rim"]["spokes"]
        r["broken"] = None if r["spokes"] not in (n - 1, n) else r["spokes"] == n - 1
    elif ptype == "gear":  # just outside the root circle: the teeth stand out there all round
        root = dims["gear"]["root_d"] / 2
        r["teeth"], r["max_gap_deg"] = max(_ring(g, ex, ey, root + d) for d in (1.0, 1.5, 2.0))
        n, pitch = dims["gear"]["teeth"], 360 / dims["gear"]["teeth"]
        # two missing teeth leave a gap of ~2.7 pitches; between sound teeth it is under one pitch
        r["broken"] = None if r["teeth"] not in range(n - 3, n + 1) else r["max_gap_deg"] > 1.8 * pitch
    return r


def truth_for(name):
    """Ground truth from a part ID in the name (pocket2_B-W-07 -> pad, white, broken corner), else None."""
    m = ID_RE.search(name)
    if not m:
        return None
    part = next((p for p in json.loads(CATALOG_JSON.read_text()) if p["id"] == m.group()), None)
    if part is None:
        return None
    return {"type": part["type"], "variant": part["variant"], "missing_hole": part["defect"] == "missing_hole",
            "broken": part["defect"] == "broken_corner"}


def overlay(rgb, H, tags, p=None):
    """The frame with the found tags and the pocket outlines, to see that the crops sit on the pockets."""
    p = p or load_params()
    img = rgb[..., ::-1].copy()
    for i, (u, v) in tags.items():
        cv2.circle(img, (int(u), int(v)), 6, (0, 200, 0), -1)
        cv2.putText(img, f"tag {i}", (int(u) + 8, int(v)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    s = p["tray"]["pocket_size"][0] / 2
    for k, c in enumerate(pocket_centers(p)):
        c = np.array(c) * 1000
        sq = np.float32([[c[0] - s, c[1] - s], [c[0] - s, c[1] + s], [c[0] + s, c[1] + s], [c[0] + s, c[1] - s]])
        uv = cv2.perspectiveTransform(sq[None], H)[0].astype(int)
        cv2.polylines(img, [uv], True, (26, 122, 255), 2)
        cv2.putText(img, str(k + 1), tuple(uv.mean(axis=0).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (26, 122, 255), 2)
    return img
