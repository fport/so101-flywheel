"""The finished cell: real print geometry + SO-101 + parts, from several angles.

The sim scene's collision geometry stays in place (the scripted policy runs unchanged); the simple boxes are hidden
and CAD meshes drawn in their place. So everything in the image has the same dimensions as the printed files.
"""

import tempfile
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..sim.env import QCSortSim
from ..sim.params import REPO_ROOT, load_params
from ..sim.parts import write_stl
from ..sim.scripted import ScriptedSorter
from .fixtures import build_all

SIZE = (1600, 1200)
HIDDEN = ("baseplate_", "tray_floor", "pocket_", "bin_", "inspection_backdrop", "cam_mast", "cam_head")
HEAD_TINT = {"cad_mast_head": [0.42, 0.43, 0.46, 1], "cad_cam_cradle": [0.68, 0.69, 0.72, 1], "cam_body": [0.2, 0.45, 0.8, 1]}
# (file, title, camera): camera is either the name of a scene camera or (lookat, distance, azimuth, elevation)
VIEWS = [
    ("overview", "Overview", ([0.15, 0.0, 0.27], 1.30, 52, -17)),
    ("workspace", "Workspace: the arm shows the part to the top camera", ([0.12, 0.0, 0.06], 0.66, 48, -34)),
    ("side", "Side: 66 cm mast, camera above the tray", ([0.15, 0.0, 0.31], 1.30, 90, -3)),
    ("head-closeup", "Camera head (light colored for visibility; blue: camera)", ([0.27, 0.0, 0.645], 0.36, 235, -6)),
    ("top-camera", "Top camera view", "top"),
    ("wrist-camera", "Wrist camera view", "wrist"),
]


def cad_visuals(items, tmp):
    """Returns a decorate function for build_spec."""

    def decorate(spec, p):
        spec.visual.global_.offwidth, spec.visual.global_.offheight = SIZE
        spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                         rgb1=[0.93, 0.94, 0.96], rgb2=[0.70, 0.72, 0.76], width=256, height=1536)
        spec.geom("table").size = [4, 4, 0.05]
        spec.geom("table").rgba = [0.78, 0.79, 0.81, 1]
        spec.visual.headlight.ambient = [0.5, 0.5, 0.52]
        spec.visual.headlight.diffuse = [0.38, 0.38, 0.38]
        for g in spec.geoms:
            if g.name.startswith(HIDDEN) and not g.name.endswith("_label"):
                g.group = 4  # kept for collision, not drawn
        for item in items:
            if not item.at:
                continue
            z = item.meta.get("z", 0.0)
            for k, part in enumerate(item.preview or item.parts):
                name = f"cad_{item.name}_{k}"
                write_stl(Path(tmp) / f"{name}.stl", part, scale=1e-3)
                spec.add_mesh(name=name, file=str(Path(tmp) / f"{name}.stl"))
                spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name,
                                        rgba=[*item.part_color(k), 1],
                                        pos=[item.at[0] / 1000, item.at[1] / 1000, z / 1000],
                                        contype=0, conaffinity=0, group=2)
            if item.name.startswith("tag_"):
                spec.geom(item.name).group = 4  # tag is a white inlay on the tile; the scene's textured tag isn't needed

    return decorate


def _camera(view):
    if isinstance(view, str):
        return view
    cam = mujoco.MjvCamera()
    cam.lookat[:] = view[0]
    cam.distance, cam.azimuth, cam.elevation = view[1:]
    return cam


def _font(size):
    for path in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def render_showcase(out_dir, tray=("J-W-01", "D-T-04", "L-W-07", "B-T-01", "J-T-06", "D-W-05")):
    """Writes the images to out_dir, returns the file paths."""
    p = load_params()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        env = QCSortSim(list(tray), params=p, seed=3, decorate=cad_visuals(build_all(p), tmp))
    # two parts go to the bins, the third is caught in the inspection pose
    for pid in tray[:2]:
        ScriptedSorter(env).run(pid)
    snap = {}

    def grab(e, phase, action):
        if phase == "inspect":
            snap["n"] = snap.get("n", 0) + 1
            if snap["n"] == 45:  # after reaching the pose and settling
                snap["state"] = (e.data.qpos.copy(), e.data.qvel.copy(), e.data.ctrl.copy())

    ScriptedSorter(env, on_frame=grab, ).run(tray[2], inspect_seconds=2.0)
    env.data.qpos[:], env.data.qvel[:], env.data.ctrl[:] = snap["state"]
    mujoco.mj_forward(env.model, env.data)

    renderer = mujoco.Renderer(env.model, SIZE[1], SIZE[0])
    paths, tiles = [], []
    m = env.model
    names = [m.geom(i).name for i in range(m.ngeom)]
    true_rgba = m.geom_rgba.copy()
    for stem, title, view in VIEWS:
        m.geom_rgba[:] = true_rgba
        if stem == "head-closeup":  # black head, black cradle and black camera blur into one blob in their real colors
            for i, name in enumerate(names):
                for prefix, rgba in HEAD_TINT.items():
                    if name.startswith(prefix):
                        m.geom_rgba[i] = rgba
        renderer.update_scene(env.data, camera=_camera(view))
        img = Image.fromarray(renderer.render())
        path = out_dir / f"final-{stem}.png"
        img.save(path)
        paths.append(path)
        tiles.append((title, img))
    w, h, bar, cols = SIZE[0] // 2, SIZE[1] // 2, 44, 3
    sheet = Image.new("RGB", (cols * w, -(-len(tiles) // cols) * (h + bar)), (24, 25, 28))
    draw, font = ImageDraw.Draw(sheet), _font(24)
    for k, (title, img) in enumerate(tiles):
        x, y = (k % cols) * w, (k // cols) * (h + bar)
        sheet.paste(img.resize((w, h), Image.LANCZOS), (x, y + bar))
        draw.text((x + 14, y + 9), title, fill=(235, 236, 240), font=font)
    path = out_dir / "final.png"
    sheet.save(path)
    return [path, *paths]


def main():
    for path in render_showcase(REPO_ROOT / "docs" / "media"):
        print(path)
