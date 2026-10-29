"""hardware/params.yaml -> MuJoCo scene.

Scene: SO-101 on the baseplate, input tray with numbered pockets, OK/REJECT bins, inspection backdrop,
top + wrist camera, AprilTags, fixed lamp and QC parts laid in the tray.
"""

from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .params import (ASSETS, CATALOG_JSON, GENERATED, SCENE_XML, SO101_XML, load_params, mast_top, mm, pocket_centers,
                     top_camera_pose)
from .parts import build_catalog, save_catalog

G = mujoco.mjtGeom
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
DEFAULT_PARTS = ["J-W-01", "D-T-04", "L-W-07", "B-T-01", "J-T-06", "D-W-05"]

# Jaw collision boxes in the gripper body frame (from the mesh's inner/outer faces measured slice by slice):
# (center, half size). Jaws extend along -z; the fixed jaw's inner face is at x = -7.9 mm, the moving jaw is on
# the +x side. Fingers taper toward the tip, so the body is represented with stepped boxes.
def _span(x0, x1, y, z0, z1):
    return ((x0 + x1) / 2, 0.0, (z0 + z1) / 2), ((x1 - x0) / 2, y, (z1 - z0) / 2)


FIXED_PADS = [
    _span(-0.0180, -0.0079, 0.0060, -0.1044, -0.0850),  # tip pad
    _span(-0.0245, -0.0116, 0.0080, -0.0850, -0.0720),  # finger body, tapers toward the tip
    _span(-0.0295, -0.0116, 0.0090, -0.0720, -0.0580),
    _span(-0.0345, -0.0133, 0.0110, -0.0580, -0.0450),
    _span(-0.0352, 0.0300, 0.0240, -0.0350, 0.0000),  # palm / motor housing
]
# Moving jaw in its body frame (x same, y = gripper z, z = -gripper y)
MOVING_PADS = [
    ((-0.0075, -0.0718, 0.0189), (0.0048, 0.0102, 0.0050)),  # tip pad
    ((-0.0013, -0.0441, 0.0189), (0.0070, 0.0175, 0.0085)),  # finger body
    ((0.0000, -0.0083, 0.0189), (0.0100, 0.0183, 0.0240)),  # hub
]
PINCH_SITE = (-0.0079, 0.0, -0.0950)  # center of the fixed jaw pad face
BASE_BOTTOM_Z = 0.0301  # bottom face of the base mesh, in the base frame
PAN_AXIS_XY = (0.0207909, -0.0230745)  # shoulder_pan axis, in the base frame


def look_at_quat(pos, target, up=(0, 0, 1)):
    """MuJoCo camera looks along -z, +y up."""
    z = np.asarray(pos, float) - np.asarray(target, float)
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([0.0, 1.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.column_stack([x, y, z]).ravel())
    return quat


def _axis_quat(axis, deg):
    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, np.asarray(axis, float), np.radians(deg))
    return quat


def _texture_material(spec, name, png):
    spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(png))
    mat = spec.add_material(name=name, specular=0.0, shininess=0.0)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = name
    return name


def _decal(body, name, material, pos, half):
    """Visual plate: textured plane with no collision."""
    body.add_geom(name=name, type=G.mjGEOM_PLANE, size=[half[0], half[1], 0.01], pos=pos, material=material,
                  contype=0, conaffinity=0, group=2)


def _write_textures(p, out):
    tex = out / "textures"
    tex.mkdir(parents=True, exist_ok=True)
    pitch_px = 64
    grid = Image.new("RGB", (pitch_px, pitch_px), (58, 60, 64))
    d = ImageDraw.Draw(grid)
    d.rectangle([0, 0, pitch_px - 1, pitch_px - 1], outline=(40, 42, 45), width=2)
    d.rounded_rectangle([6, 6, pitch_px - 7, pitch_px - 7], radius=8, outline=(70, 72, 77), width=1)
    grid.save(tex / "baseplate.png")
    n = p["tray"]["rows"] * p["tray"]["cols"]
    font = ImageFont.load_default(size=52)
    for i in range(1, n + 1):
        im = Image.new("RGB", (64, 64), (31, 31, 33))
        ImageDraw.Draw(im).text((32, 32), str(i), fill=(235, 235, 235), font=font, anchor="mm")
        # as on the printed tray, the top of the digit faces the arm (-x); reads upside down from behind (fixtures.tray)
        im.rotate(90).save(tex / f"pocket_{i}.png")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    for tag_id in (t["id"] for t in p["tags"]["items"]):
        marker = Image.fromarray(cv2.aruco.generateImageMarker(dictionary, tag_id, 256)).convert("RGB")
        canvas = Image.new("RGB", (320, 320), (255, 255, 255))  # 1-cell white quiet zone
        canvas.paste(marker, (32, 32))
        canvas.save(tex / f"tag_{tag_id}.png")
    return tex


def _add_robot_extras(spec, p):
    for i, name in enumerate(JOINTS, start=1):
        spec.joint(str(i)).name = name
        act = spec.actuator(str(i))
        act.name = name
        act.target = name
        act.inheritrange = 1.0  # ctrlrange = joint limit

    base = spec.body("base")
    base.quat = _axis_quat([0, 0, 1], 90)  # model faces -y; make it face +x in the scene
    # Rz(90): (x, y) -> (-y, x); pan axis to the origin, base bottom face onto the baseplate
    base.pos = [PAN_AXIS_XY[1], -PAN_AXIS_XY[0], -BASE_BOTTOM_Z]

    gripper, jaw = spec.body("gripper"), spec.body("moving_jaw_so101_v1")
    jaw.name = "moving_jaw"
    # Mesh convex hulls fill the gap between the jaws; replace with box pads on the fingers.
    for body in (gripper, jaw):
        for g in body.geoms:
            if g.group == 3:
                g.contype = 0
                g.conaffinity = 0
    for body, pads, prefix in ((gripper, FIXED_PADS, "fixed"), (jaw, MOVING_PADS, "moving")):
        for k, (pos, half) in enumerate(pads):
            body.add_geom(name=f"{prefix}_pad_{k}", type=G.mjGEOM_BOX, pos=pos, size=half, group=3,
                          rgba=[0.1, 0.1, 0.1, 1], friction=[1.5, 0.02, 0.0005], condim=4,
                          solref=[0.004, 1])
    gripper.add_site(name="pinch", pos=PINCH_SITE, size=[0.003, 0.003, 0.003], group=4)

    w = p["cameras"]["wrist"]
    pos, target = mm(w["position"]), mm(w["look_at"])
    # image top points toward the jaw tips
    gripper.add_camera(name="wrist", pos=pos, quat=look_at_quat(pos, target, up=(0, 0, -1)), fovy=w["fovy"])
    gripper.add_geom(name="wrist_cam_body", type=G.mjGEOM_BOX, pos=pos + np.array([0, 0.006, 0.004]),
                     size=[0.016, 0.006, 0.016], rgba=[0.12, 0.12, 0.13, 1], contype=0, conaffinity=0,
                     group=2)


def _add_baseplate(spec, p, tex):
    g = p["grid"]
    u = g["pitch"] / 1000
    world = spec.worldbody
    spec.add_texture(name="baseplate", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(tex / "baseplate.png"))
    mat = spec.add_material(name="baseplate", texrepeat=[1 / u, 1 / u], texuniform=True, specular=0.05, shininess=0.05)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "baseplate"
    t = g["thickness"] / 1000
    world.add_geom(name="table", type=G.mjGEOM_PLANE, size=[1.0, 1.0, 0.05], pos=[0.15, 0, -t],
                   rgba=[0.62, 0.63, 0.65, 1], friction=[0.8, 0.005, 0.0001])
    skip = set(g.get("skip", []))
    for ci, (x0, x1) in enumerate(zip(g["tiles_x"], g["tiles_x"][1:])):
        for ri, (y0, y1) in enumerate(zip(g["tiles_y"], g["tiles_y"][1:])):
            name = f"{'ABCDE'[ci]}{ri + 1}"
            if name in skip:
                continue
            world.add_geom(name=f"baseplate_{name}", type=G.mjGEOM_BOX,
                           size=[(x1 - x0) * u / 2, (y1 - y0) * u / 2, t / 2],
                           pos=[(x1 + x0) * u / 2, (y1 + y0) * u / 2, -t / 2], material="baseplate",
                           friction=[0.8, 0.005, 0.0001])


def _add_tray(spec, p, tex):
    t = p["tray"]
    body = spec.worldbody.add_body(name="tray", pos=[*mm(t["center"]), 0])
    sx, sy = mm(t["size"])
    floor, depth = mm(t["floor"]), mm(t["pocket_depth"])
    psx, psy = mm(t["pocket_size"])
    px, py = sx / t["rows"], sy / t["cols"]
    rgba = [0.12, 0.12, 0.13, 1]  # black ESD tray: parts stand out both in the pocket and in the inspection crop
    kw = dict(type=G.mjGEOM_BOX, rgba=rgba, friction=[0.9, 0.005, 0.0001])
    body.add_geom(name="tray_floor", size=[sx / 2, sy / 2, floor / 2], pos=[0, 0, floor / 2], **kw)
    wx, wy = (px - psx) / 2, (py - psy) / 2  # wall thicknesses on either side of the pocket
    slot = float(mm(t["finger_slot"]))
    seg = (psx - slot) / 2
    zc = floor + depth / 2
    centers = pocket_centers(p)
    cx, cy = mm(t["center"])
    for i, (gx, gy) in enumerate(centers, start=1):
        lx, ly = gx - cx, gy - cy
        for s in (-1, 1):
            body.add_geom(name=f"pocket_{i}_x{s:+d}", size=[wx / 2, py / 2, depth / 2],
                          pos=[lx + s * (psx / 2 + wx / 2), ly, zc], **kw)
            # y wall is split in the middle by the finger slot
            for e in (-1, 1):
                body.add_geom(name=f"pocket_{i}_y{s:+d}_{'ab'[(e + 1) // 2]}", size=[seg / 2, wy / 2, depth / 2],
                              pos=[lx + e * (slot / 2 + seg / 2), ly + s * (psy / 2 + wy / 2), zc], **kw)
        body.add_site(name=f"pocket_{i}", pos=[lx, ly, floor], size=[0.002, 0.002, 0.002], group=4)
        # pocket number: on top of the wall left of the pocket
        d = min(wy, seg) * 0.85
        _decal(body, f"pocket_{i}_label", _texture_material(spec, f"pocket_{i}", tex / f"pocket_{i}.png"),
               [lx - slot / 2 - seg / 2, ly + psy / 2 + wy / 2, floor + depth + 0.0002], [d / 2, d / 2])


def _add_bin(spec, p, name):
    b = p["bins"]
    cfg = b[name]
    body = spec.worldbody.add_body(name=f"bin_{name}", pos=[*mm(cfg["center"]), 0])
    h, th, liner = mm(b["wall_height"]), mm(b["wall_thickness"]), mm(b["liner_thickness"])
    mx, my = mm(b["size"]) / 2 - 0.00025 - th  # inner half size of the opening: tile footprint - gridfinity clearance - wall
    tilt = b["wall_tilt"]
    inset = h * np.tan(np.radians(tilt))
    bx, by = mx - inset, my - inset  # floor half size
    rgba = [*cfg["color"], 1]
    kw = dict(type=G.mjGEOM_BOX, rgba=rgba, friction=[0.6, 0.005, 0.0001])
    body.add_geom(name=f"bin_{name}_floor", size=[bx + th, by + th, th / 2], pos=[0, 0, th / 2], **kw)
    # ribbed liner: keeps round parts from rolling. Modeled in sim as a flat high-friction surface
    body.add_geom(name=f"bin_{name}_liner", type=G.mjGEOM_BOX, size=[bx, by, liner / 2],
                  pos=[0, 0, th + liner / 2], rgba=[0.13, 0.13, 0.14, 1], friction=[1.2, 0.01, 0.001],
                  solref=[0.02, 1.2])
    slant = h / np.cos(np.radians(tilt)) / 2
    zc = th + h / 2
    for s in (-1, 1):
        body.add_geom(name=f"bin_{name}_wall_x{s:+d}", size=[th / 2, my + th, slant],
                      pos=[s * (bx + inset / 2 + th / 2), 0, zc], quat=_axis_quat([0, 1, 0], s * tilt), **kw)
        body.add_geom(name=f"bin_{name}_wall_y{s:+d}", size=[mx + th, th / 2, slant],
                      pos=[0, s * (by + inset / 2 + th / 2), zc], quat=_axis_quat([1, 0, 0], -s * tilt), **kw)
    # ground truth: is the part inside this bin? (floor area x inner height)
    body.add_site(name=f"bin_{name}_zone", type=G.mjGEOM_BOX, pos=[0, 0, th + h / 2], size=[mx, my, h / 2],
                  rgba=[*cfg["color"], 0.15], group=4)


def _add_inspection(spec, p):
    i = p["inspection"]
    hx, hy = mm(i["backdrop_size"]) / 2
    cx, cy = mm(i["backdrop_center"])
    spec.add_material(name="backdrop", rgba=[*i["backdrop_color"], 1], specular=0.0, shininess=0.0)
    spec.worldbody.add_geom(name="inspection_backdrop", type=G.mjGEOM_BOX, size=[hx, hy, 0.001],
                            pos=[cx, cy, 0.001], material="backdrop", friction=[0.8, 0.005, 0.0001])
    spec.worldbody.add_site(name="inspection_pose", pos=[mm(i["pose"][0]), 0, mm(i["pose"][1])],
                            size=[0.004, 0.004, 0.004], rgba=[1, 0.8, 0, 0.6], group=4)


def _add_cameras_and_light(spec, p):
    """Top camera, the printed mast carrying it, and the fixed light."""
    world = spec.worldbody
    top, mast = p["cameras"]["top"], p["mast"]
    cam_pos, cam_dir = top_camera_pose(p)
    pos, target = mm(cam_pos), mm(cam_pos + cam_dir * 1000)
    world.add_camera(name="top", pos=pos, quat=look_at_quat(pos, target, up=(-1, 0, 0)), fovy=top["fovy"])
    mx, my = mm(mast["pos"])
    s = float(mm(mast["size"])) / 2
    height = mast_top(p) / 1000
    world.add_geom(name="cam_mast", type=G.mjGEOM_BOX, size=[s, s, height / 2], pos=[mx, my, height / 2],
                   rgba=[0.13, 0.13, 0.14, 1])
    # head + camera body (visual only)
    world.add_geom(name="cam_head", type=G.mjGEOM_BOX, size=[s + 0.003, s + 0.003, 0.024], pos=[mx, my, height - 0.021],
                   rgba=[0.13, 0.13, 0.14, 1], contype=0, conaffinity=0)
    world.add_geom(name="cam_body", type=G.mjGEOM_BOX, size=[0.016, 0.036, 0.013], pos=pos - np.asarray(cam_dir) * 0.014,
                   quat=look_at_quat(pos, target, up=(-1, 0, 0)), rgba=[0.08, 0.08, 0.09, 1], contype=0, conaffinity=0)
    lamp = p["light"]
    lpos, ltarget = mm(lamp["position"]), mm(lamp["look_at"])
    direction = (ltarget - lpos) / np.linalg.norm(ltarget - lpos)
    world.add_light(name="lamp", pos=lpos, dir=direction, type=mujoco.mjtLightType.mjLIGHT_SPOT,
                    cutoff=60, exponent=4, diffuse=[0.85, 0.85, 0.83], specular=[0.15, 0.15, 0.15],
                    castshadow=True)
    spec.visual.headlight.ambient = [0.38, 0.38, 0.4]
    spec.visual.headlight.diffuse = [0.22, 0.22, 0.22]
    spec.visual.headlight.specular = [0, 0, 0]


def _add_tags(spec, p, tex):
    for t in p["tags"]["items"]:
        half = float(mm(t["size"])) * 10 / 8 / 2  # 8x8-cell tag + 1-cell white border
        mat = _texture_material(spec, f"tag_{t['id']}", tex / f"tag_{t['id']}.png")
        _decal(spec.worldbody, f"tag_{t['id']}", mat, [*mm(t["center"]), 0.0004], [half, half])


def _add_parts(spec, p, catalog, part_ids, out):
    by_id = {c.id: c for c in catalog}
    centers = pocket_centers(p)
    if len(part_ids) > len(centers):
        raise ValueError(f"{len(part_ids)} parts, but the tray has {len(centers)} pockets")
    z = float(mm(p["tray"]["floor"])) + 0.0005
    density = 1240 * 0.35
    for slot, pid in enumerate(part_ids):
        if pid is None:
            continue
        part = by_id[pid]
        spec.add_mesh(name=pid, file=str(out / part.mesh))
        body = spec.worldbody.add_body(name=pid, pos=[*centers[slot], z])
        body.add_freejoint(name=f"{pid}_joint")
        body.add_geom(name=f"{pid}_visual", type=G.mjGEOM_MESH, meshname=pid, rgba=[*part.color, 1],
                      contype=0, conaffinity=0, group=2, density=0)
        for k, c in enumerate(part.collision):
            body.add_geom(name=f"{pid}_col_{k}", type=G.mjGEOM_BOX if c["type"] == "box" else G.mjGEOM_CYLINDER,
                          size=c["size"] + [0] * (3 - len(c["size"])), pos=c["pos"], group=3, density=density,
                          rgba=[*part.color, 1], friction=[1.0, 0.01, 0.0005], condim=4)


def build_spec(p=None, part_ids=DEFAULT_PARTS, out=GENERATED, decorate=None):
    p = p or load_params()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    catalog = build_catalog(p, out)
    save_catalog(catalog, out / CATALOG_JSON.name)
    tex = _write_textures(p, out)

    spec = mujoco.MjSpec.from_file(str(SO101_XML))
    spec.modelname = "so101_qc_sort"
    for mesh in spec.meshes:  # robot meshes: absolute path, made relative on save
        mesh.file = str(SO101_XML.parent / "assets" / mesh.file)
    spec.meshdir = ""
    spec.option.timestep = 1 / 600  # 30 fps control = 20 substeps
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.option.noslip_iterations = 3  # prevents the held part from slowly slipping (creep) on the pads
    w, h = p["cameras"]["resolution"]
    spec.visual.global_.offwidth = max(1280, w)
    spec.visual.global_.offheight = max(960, h)
    spec.visual.quality.shadowsize = 4096
    spec.stat.center = [0.15, 0, 0.1]
    spec.stat.extent = 0.6

    _add_robot_extras(spec, p)
    _add_baseplate(spec, p, tex)
    _add_tray(spec, p, tex)
    _add_bin(spec, p, "ok")
    _add_bin(spec, p, "reject")
    _add_inspection(spec, p)
    _add_cameras_and_light(spec, p)
    _add_tags(spec, p, tex)
    _add_parts(spec, p, catalog, list(part_ids), out)
    if decorate:  # visual-only additions to the scene (e.g. real print geometry)
        decorate(spec, p)
    return spec, catalog


def build_scene(p=None, part_ids=DEFAULT_PARTS, out=GENERATED, save=True, decorate=None):
    """Compiles the scene; if save=True writes out/scene.xml. Returns (model, catalog)."""
    spec, catalog = build_spec(p, part_ids, out, decorate)
    model = spec.compile()
    if save:
        xml = spec.to_xml()
        # make absolute paths relative to scene.xml so the repo stays portable
        xml = xml.replace(str(Path(out)) + "/", "").replace(str(ASSETS) + "/", "../")
        (Path(out) / SCENE_XML.name).write_text(xml)
    return model, catalog

