"""MuJoCo environment for the QC sorting task: reset, step, camera images and ground-truth outcome.

Ground truth produces the same labels as the failure taxonomy in the PRD; it is the exact sim counterpart of what
the VLM verifier will predict on the real arm and can be used to measure the verifier.
"""

import mujoco
import numpy as np

from .ik import grasp_frame
from .params import load_params, mm, pocket_centers
from .scene import DEFAULT_PARTS, JOINTS, build_scene

FPS = 30
HOME = np.array([0.0, -1.54, 0.9, 1.55, 0.0, 0.0])  # arm folded, gripper above the backdrop
FAILURE_TYPES = ("grasp_fail", "drop", "wrong_bin", "misclassification", "out_of_bin", "timeout")


class QCSortSim:
    def __init__(self, part_ids=DEFAULT_PARTS, params=None, seed=0, save_scene=False, order="W", decorate=None):
        self.p = params or load_params()
        self.order = order  # variant in the work order: which color is the correct product in this batch
        self.rng = np.random.default_rng(seed)
        self.part_ids = [pid for pid in part_ids if pid is not None]
        self.model, catalog = build_scene(self.p, part_ids, save=save_scene, decorate=decorate)
        self.catalog = {c.id: c for c in catalog}
        self.data = mujoco.MjData(self.model)
        self.substeps = round(1 / FPS / self.model.opt.timestep)
        self._renderer = None
        m = self.model
        self._body = {pid: m.body(pid).id for pid in self.part_ids}
        self._qadr = {pid: m.joint(f"{pid}_joint").qposadr[0] for pid in self.part_ids}
        self._vadr = {pid: m.joint(f"{pid}_joint").dofadr[0] for pid in self.part_ids}
        self._pockets = pocket_centers(self.p)
        self._slots = [i for i, pid in enumerate(part_ids) if pid is not None]
        self._floor = float(mm(self.p["tray"]["floor"]))
        self._bin_zone = {name: m.site(f"bin_{name}_zone").id for name in ("ok", "reject")}
        self.reset()

    # --- state -------------------------------------------------------------------------------------
    def reset(self, jitter=True):
        """Lays parts in their pockets, moves the arm to home. jitter: small position/yaw offset within the pocket."""
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[:6] = HOME
        d.ctrl[:6] = HOME
        for slot, pid in zip(self._slots, self.part_ids):
            x, y = self._pockets[slot]
            yaw = 0.0
            if jitter:
                x, y = x + self.rng.uniform(-0.0015, 0.0015), y + self.rng.uniform(-0.0015, 0.0015)
                yaw = self.rng.uniform(-0.12, 0.12)
                if self.catalog[pid].grasp["symmetry"] == 0:
                    yaw = self.rng.uniform(-np.pi, np.pi)
            if self.catalog[pid].type == "bracket":
                yaw += np.pi / 2  # grasp the thin edge of the upright along y
            a = self._qadr[pid]
            d.qpos[a : a + 3] = [x, y, self._floor + 0.001]
            d.qpos[a + 3 : a + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(m, d)
        for _ in range(150):  # let parts settle into pockets
            mujoco.mj_step(m, d)
        self._max_z = {pid: self.part_pos(pid)[2] for pid in self.part_ids}
        self._start_z = dict(self._max_z)
        self.frame = 0
        return self.observe()

    def step(self, action):
        """action: 6 joint position targets (rad), in JOINTS order. Advances 1/FPS seconds."""
        self.data.ctrl[:6] = np.clip(action, self.model.actuator_ctrlrange[:6, 0], self.model.actuator_ctrlrange[:6, 1])
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        for pid in self.part_ids:
            self._max_z[pid] = max(self._max_z[pid], self.part_pos(pid)[2])
        self.frame += 1
        return self.observe()

    def observe(self):
        return {"state": self.data.qpos[:6].copy(), "names": JOINTS}

    def render(self, camera="top", width=None, height=None):
        w, h = self.p["cameras"]["resolution"]
        width, height = width or w, height or h
        if self._renderer is None or (self._renderer.width, self._renderer.height) != (width, height):
            self._renderer = mujoco.Renderer(self.model, height, width)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render().copy()

    def project(self, point, camera="top", width=None, height=None):
        """Pixel position (u, v) of a world point in the camera image."""
        w, h = self.p["cameras"]["resolution"]
        width, height = width or w, height or h
        cam = self.model.camera(camera).id
        local = (np.asarray(point) - self.data.cam_xpos[cam]) @ self.data.cam_xmat[cam].reshape(3, 3)
        f = height / 2 / np.tan(np.radians(self.model.cam_fovy[cam]) / 2)
        return width / 2 + f * local[0] / -local[2], height / 2 - f * local[1] / -local[2]

    def inspection_pose(self):
        """(pinch position, gripper orientation): the part is held between the top camera and the backdrop plate.

        Jaws tilted forward-down, closing axis +y: the wrist camera stays on the arm side, the top camera sees the
        part's top and front faces without the palm in the way.
        """
        ins = self.p["inspection"]
        x, h = mm(ins["pose"])
        show = np.array([x, 0.0, h])
        tilt = np.radians(ins["pose_tilt"])
        return show, grasp_frame([np.sin(tilt), 0, -np.cos(tilt)], [0, 1, 0])

    def inspection_crop(self, size=128, scale=2):
        """Input to the inspection classifier: a square crop of the part between the jaws from the top camera
        image. Called while the arm is in the inspection pose. At the default size the crop's background
        falls entirely on the inspection backdrop plate."""
        w, h = self.p["cameras"]["resolution"]
        image = self.render("top", w * scale, h * scale)
        site = self.model.site("pinch").id
        r = self.data.site_xmat[site].reshape(3, 3)
        center = self.data.site_xpos[site] + 0.012 * r[:, 0] - 0.008 * r[:, 2]
        u, v = self.project(center, "top", w * scale, h * scale)
        u0 = int(np.clip(u - size / 2, 0, w * scale - size))
        v0 = int(np.clip(v - size / 2, 0, h * scale - size))
        return image[v0 : v0 + size, u0 : u0 + size]

    # --- parts and ground truth ------------------------------------------------------------------
    def part_pos(self, pid):
        return self.data.xpos[self._body[pid]].copy()

    def part_pose(self, pid):
        b = self._body[pid]
        return self.data.xpos[b].copy(), self.data.xmat[b].reshape(3, 3).copy()

    def part_speed(self, pid):
        a = self._vadr[pid]
        return float(np.linalg.norm(self.data.qvel[a : a + 3]))

    def _in_bin(self, pos, name, margin=0.0):
        """Inside the bin opening footprint and at most 8 cm above the rim (stacked parts included)?"""
        s = self._bin_zone[name]
        local = pos - self.data.site_xpos[s]
        half = self.model.site_size[s]
        inside = abs(local[0]) <= half[0] + margin and abs(local[1]) <= half[1] + margin
        return bool(inside and pos[2] <= self.data.site_xpos[s][2] + half[2] + 0.08)

    def location(self, pid):
        """'held' | 'ok' | 'reject' | 'tray' | 'outside'"""
        pos = self.part_pos(pid)
        pinch = self.data.site_xpos[self.model.site("pinch").id]
        if np.linalg.norm(pos - pinch) < 0.05 and pos[2] > self._floor + 0.03:
            return "held"
        for name in ("ok", "reject"):
            if self._in_bin(pos, name):
                return name
        t = self.p["tray"]
        (cx, cy), (sx, sy) = mm(t["center"]), mm(t["size"])
        if abs(pos[0] - cx) <= sx / 2 and abs(pos[1] - cy) <= sy / 2 and pos[2] < self._floor + 0.03:
            return "tray"
        return "outside"

    def outcome(self, pid, commanded_bin, timed_out=False, probe=False):
        """Episode outcome.

        commanded_bin: target given to the policy ("ok"/"reject"). In a normal trial this is the inspection decision
        and must match the bin expected from the work order; otherwise the trial counts as `misclassification`.
        probe=True is a command-following probe: the command is deliberately opposite to the part's true class and
        the only criterion is that the policy follows it (it fails here if it memorized the bin from the image).
        """
        part = self.catalog[pid]
        expected = part.expected_bin(self.order)
        loc = self.location(pid)
        lifted = self._max_z[pid] - self._start_z[pid] > 0.02
        if loc == commanded_bin:
            manipulation = None
        elif timed_out:
            manipulation = "timeout"
        elif loc in ("ok", "reject"):
            manipulation = "wrong_bin"
        elif not lifted:
            manipulation = "grasp_fail"
        elif any(self._in_bin(self.part_pos(pid), n, margin=0.04) for n in ("ok", "reject")):
            manipulation = "out_of_bin"
        else:
            manipulation = "drop"
        failure = manipulation
        if not probe and commanded_bin != expected:
            failure = "misclassification"  # wrong decision; trial fails even if manipulation was right
        return {
            "part_id": pid,
            "success": failure is None,
            "failure_type": failure,
            "followed_command": loc == commanded_bin,
            "probe": probe,
            "location": loc,
            "order": self.order,
            "expected_bin": expected,
            "commanded_bin": commanded_bin,
            "attributes": part.attributes(),
        }
