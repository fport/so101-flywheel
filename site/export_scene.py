"""Exports the MuJoCo scene and a one-tray playback of the scripted expert for the browser: public/scene/.

    python/.venv/bin/python site/export_scene.py

Output:
  scene.json   geoms (type, size, body, local pose, color, texture), cameras, episodes, parts showcase
  meshes.bin   arm and part meshes: int16 quantized vertices + uint16/uint32 triangles
  frames.bin   float32 [frame, body, 7]: world position and quaternion (w, x, y, z) of each body in each frame
  tex/*.png    tag, pocket number and baseplate textures

The browser runs no physics; it plays back frames produced by MuJoCo. Scene and motion are identical to the sim.
"""

import io
import json
import struct
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

SITE = Path(__file__).resolve().parent
REPO = SITE.parent
sys.path.insert(0, str(REPO / "python"))
from flywheel.robot.session import load_plan  # noqa: E402
from flywheel.sim.env import FPS, QCSortSim  # noqa: E402
from flywheel.sim.params import CATALOG_JSON  # noqa: E402
from flywheel.sim.parts import load_catalog  # noqa: E402
from flywheel.sim.scripted import ScriptedSorter  # noqa: E402

OUT = SITE / "public" / "scene"
PLAN, TRAY = "record-W-seed0", 1
GEOM_TYPES = {0: "plane", 2: "sphere", 3: "capsule", 4: "ellipsoid", 5: "cylinder", 6: "box", 7: "mesh"}
SHOWCASE = [(t, d) for t in ("rim", "gear", "pad", "bracket") for d in (None, "missing_hole", "broken_corner")]


class Blob:
    """Appends aligned chunks to a single binary file, returns (offset, length)."""

    def __init__(self):
        self.buf = io.BytesIO()

    def add(self, arr):
        pad = (-self.buf.tell()) % 4
        self.buf.write(b"\0" * pad)
        off = self.buf.tell()
        self.buf.write(np.ascontiguousarray(arr).tobytes())
        return off


def pack_mesh(blob, vert, face):
    lo, hi = vert.min(0), vert.max(0)
    scale = np.maximum(hi - lo, 1e-9) / 65535.0
    q = np.round((vert - lo) / scale - 32768).astype(np.int16)
    idx = face.astype(np.uint16 if len(vert) < 65536 else np.uint32)
    return {"vOff": blob.add(q), "vCount": len(vert), "min": lo.tolist(), "scale": scale.tolist(),
            "iOff": blob.add(idx), "iCount": idx.size, "i32": idx.dtype == np.uint32}


def read_stl(path):
    data = path.read_bytes()
    n = struct.unpack("<I", data[80:84])[0]
    tri = np.frombuffer(data[84:84 + n * 50], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    v = tri["v"].reshape(-1, 3)
    uniq, inv = np.unique(np.round(v, 7), axis=0, return_inverse=True)
    return uniq.astype(np.float32), inv.reshape(-1, 3)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tex").mkdir(exist_ok=True)
    plan = load_plan(PLAN)
    tray = plan["trays"][TRAY - 1]
    env = QCSortSim(part_ids=tray["pockets"], seed=TRAY, order=plan["order"])
    m, d = env.model, env.data
    blob = Blob()

    # textures
    textures = {}
    for i in range(m.ntex):
        w, h, c = m.tex_width[i], m.tex_height[i], m.tex_nchannel[i]
        px = m.tex_data[m.tex_adr[i]: m.tex_adr[i] + w * h * c].reshape(h, w, c)
        name = m.texture(i).name
        Image.fromarray(px[..., :3] if c >= 3 else px[..., 0]).save(OUT / "tex" / f"{name}.png")
        textures[i] = name

    # meshes
    meshes = []
    for i in range(m.nmesh):
        va, vn = m.mesh_vertadr[i], m.mesh_vertnum[i]
        fa, fn = m.mesh_faceadr[i], m.mesh_facenum[i]
        meshes.append({"name": m.mesh(i).name, **pack_mesh(blob, m.mesh_vert[va:va + vn], m.mesh_face[fa:fa + fn])})

    # visual geoms (group 0: fixture, group 2: visual mesh/decal; 3: collision, 4: helper)
    geoms = []
    for g in range(m.ngeom):
        if m.geom_group[g] not in (0, 2) or m.geom_type[g] not in GEOM_TYPES:
            continue
        if m.geom_bodyid[g] == 0 and m.geom_type[g] == 0 and m.geom_size[g][0] == 0:
            continue  # infinite ground plane; the scene has its own floor
        mat = m.geom_matid[g]
        rgba = (m.mat_rgba[mat] if mat >= 0 else m.geom_rgba[g]).tolist()
        tex = None
        if mat >= 0:
            tid = m.mat_texid[mat][mujoco.mjtTextureRole.mjTEXROLE_RGB]
            if tid >= 0:
                tex = {"name": textures[tid], "repeat": m.mat_texrepeat[mat].tolist(), "uniform": bool(m.mat_texuniform[mat])}
        geoms.append({
            "name": m.geom(g).name, "type": GEOM_TYPES[m.geom_type[g]], "body": int(m.geom_bodyid[g]),
            "size": m.geom_size[g].tolist(), "pos": m.geom_pos[g].tolist(), "quat": m.geom_quat[g].tolist(),
            "rgba": rgba, "tex": tex, "mesh": int(m.geom_dataid[g]) if m.geom_type[g] == 7 else None,
        })

    cameras = [{"name": m.camera(i).name, "body": int(m.cam_bodyid[i]), "pos": m.cam_pos[i].tolist(),
                "quat": m.cam_quat[i].tolist(), "fovy": float(m.cam_fovy[i])} for i in range(m.ncam)]

    # playback: the tray's 6 episodes back to back, pose of every body in every frame
    frames, phases, episodes = [], [], []

    def on_frame(env, phase, action):
        frames.append(np.hstack([d.xpos, d.xquat]).astype(np.float32))
        if not phases or phases[-1][1] != phase:
            phases.append([len(frames) - 1, phase])

    on_frame(env, "idle", None)
    for k, pick in enumerate(tray["picks"]):
        start = len(frames)
        env.frame = 0  # timeout counter per episode (same as run_episode)
        out = ScriptedSorter(env, on_frame=on_frame).run(pick["part"], pick["command"], probe=pick["probe"])
        episodes.append({"part": pick["part"], "pocket": pick["pocket"], "command": pick["command"],
                         "probe": pick["probe"], "start": start, "end": len(frames) - 1,
                         "success": bool(out["success"]), "failure": out.get("failure_type")})
        print(f"  episode {k + 1}: {pick['part']} → {pick['command']} · {len(frames) - start} frames · "
              f"{'success' if out['success'] else out.get('failure_type')}")
    arr = np.stack(frames)
    (OUT / "frames.bin").write_bytes(arr.tobytes())

    # parts showcase: sound, missing-hole, broken of each type (white training parts); orange is recolored
    catalog = {c.id: c for c in load_catalog(CATALOG_JSON)}
    show = []
    for ptype, defect in SHOWCASE:
        c = next(c for c in catalog.values() if c.type == ptype and c.defect == defect and c.variant == "W" and not c.heldout)
        v, f = read_stl(REPO / "python/flywheel/sim/assets/generated" / c.mesh)
        show.append({"id": c.id, "type": ptype, "defect": defect, **pack_mesh(blob, v, f)})

    (OUT / "meshes.bin").write_bytes(blob.buf.getvalue())
    scene = {
        "fps": FPS, "nbody": m.nbody, "frames": len(arr),
        "bodies": [m.body(i).name for i in range(m.nbody)],
        "geoms": geoms, "meshes": meshes, "cameras": cameras,
        "plan": PLAN, "order": plan["order"], "tray": tray, "episodes": episodes, "phases": phases,
        "showcase": show,
    }
    (OUT / "scene.json").write_text(json.dumps(scene, separators=(",", ":")))
    mb = lambda p: f"{p.stat().st_size / 1e6:.1f} MB"  # noqa: E731
    print(f"scene.json {mb(OUT / 'scene.json')} · meshes.bin {mb(OUT / 'meshes.bin')} · frames.bin "
          f"{mb(OUT / 'frames.bin')} · {len(geoms)} geoms · {len(arr)} frames")


if __name__ == "__main__":
    main()
