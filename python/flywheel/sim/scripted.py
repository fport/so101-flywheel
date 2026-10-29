"""Scripted expert: pick from the tray -> show to the top camera -> drop into the commanded bin.

Reads part poses from the sim state (privileged information); meant for generating demo data in sim and validating
the scene. On the real arm this role is played by teleop demos and the trained policy.
"""

import numpy as np

from .env import FPS, HOME
from .ik import IK, grasp_frame
from .params import mm

TIP_BELOW_PINCH = 0.0094  # from the pinch site to the jaw tip
MAX_JOINT_SPEED = 1.6  # rad/s
FIXED_CLEARANCE = 0.002  # gap between fixed jaw and part on approach
TIP_CLEARANCE = 0.0022  # fixed jaw tip to pocket floor; moving jaw tip is ~1.8 mm lower
OPEN_MARGIN = 0.012  # jaw opening added to part width on approach; must not spill into the neighbouring pocket


def _smooth(t):
    return t * t * (3 - 2 * t)


class ScriptedSorter:
    def __init__(self, env, on_frame=None, max_seconds=30.0):
        self.env = env
        self.ik = IK(env.model)
        self.on_frame = on_frame
        self.max_frames = int(max_seconds * FPS)
        self.q = env.data.ctrl[:6].copy()
        self.phase = "idle"
        self._gap_table = self._measure_gap()

    # --- gripper -----------------------------------------------------------------------------------
    def _measure_gap(self):
        """Jaw angle -> pad face gap table (via FK)."""
        m, d = self.ik.m, self.ik.d
        pad, gripper = m.geom("moving_pad_0").id, m.body("gripper").id
        angles = np.linspace(m.jnt_range[5, 0], 1.2, 40)
        gaps = []
        for a in angles:
            d.qpos[5] = a
            self.ik.fk(self.q[:5])
            r = d.xmat[gripper].reshape(3, 3)
            face = d.geom_xpos[pad] - d.geom_xmat[pad].reshape(3, 3)[:, 0] * m.geom_size[pad][0]
            gaps.append((face - d.site_xpos[self.ik.site]) @ r[:, 0])
        d.qpos[5] = 0
        return np.array(gaps), angles

    def jaw_angle(self, gap):
        gaps, angles = self._gap_table
        return float(np.interp(gap, gaps, angles))

    # --- motion primitives -------------------------------------------------------------------------
    def _tick(self):
        if self.env.frame >= self.max_frames:
            raise TimeoutError
        self.env.step(self.q)
        if self.on_frame:
            self.on_frame(self.env, self.phase, self.q.copy())

    def hold(self, seconds):
        for _ in range(int(seconds * FPS)):
            self._tick()

    def move_joints(self, target, min_seconds=0.4):
        start = self.q.copy()
        target = np.asarray(target, float)
        n = max(int(min_seconds * FPS), int(np.max(np.abs(target - start)) / MAX_JOINT_SPEED * FPS))
        for i in range(1, n + 1):
            self.q = start + (target - start) * _smooth(i / n)
            self._tick()

    def pinch_pos(self):
        """Actual (not commanded) pinch position."""
        return self.env.data.site_xpos[self.ik.site].copy()

    def move_line(self, pos, rot, seconds, settle=0.3):
        """Moves the pinch point along a straight line; orientation fixed.

        The servos are P-only, so the arm sags a few mm under gravity. The difference between actual and target
        position is integrated and added to the command (the counterpart of a teleoperator correcting by eye).
        """
        start, _ = self.ik.fk(self.q[:5])
        pos = np.asarray(pos, float)
        bias = start - self.pinch_pos()
        n = max(2, int(seconds * FPS))
        for i in range(1, n + int(settle * FPS) + 1):
            goal = start + (pos - start) * _smooth(min(i / n, 1.0))
            self.q[:5] = self.ik.solve(goal + bias, rot, q0=self.q[:5]).q
            self._tick()
            bias = np.clip(bias + 0.25 * (goal - self.pinch_pos()), -0.012, 0.012)

    def set_gripper(self, angle, seconds=0.5):
        start = self.q[5]
        n = max(2, int(seconds * FPS))
        for i in range(1, n + 1):
            self.q[5] = start + (angle - start) * _smooth(i / n)
            self._tick()

    # --- grasp planning -----------------------------------------------------------------------------
    def plan_grasp(self, pid):
        """Returns (pinch target, orientation, width). The closing axis is aligned to world y when possible:
        pockets are adjacent along x, there is room for the jaws along y."""
        part = self.env.catalog[pid]
        pos, rot = self.env.part_pose(pid)
        g = part.grasp
        center = pos + rot @ np.array(g["center"])
        axis = rot @ np.array(g["axis"])
        axis[2] = 0
        axis /= np.linalg.norm(axis)
        if g["symmetry"] == 0:
            candidates = [np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0])]
        else:
            perp = np.array([-axis[1], axis[0], 0.0])
            candidates = [axis, -axis] + ([perp, -perp] if g["symmetry"] == 4 else [])
            candidates.sort(key=lambda a: -abs(a[1]))
        # jaws drop into the tray's finger slots; the tip stops just above the pocket floor
        z = max(center[2], float(mm(self.env.p["tray"]["floor"])) + TIP_BELOW_PINCH + TIP_CLEARANCE)
        best = None
        for a in candidates:
            if best is not None and abs(a[1]) < 0.7:
                break  # if a y-aligned grasp was found, skip the x-aligned ones
            frame = grasp_frame([0, 0, -1], a)
            target = np.array([*(center[:2] - a[:2] * (g["width"] / 2 + FIXED_CLEARANCE)), z])
            res = self.ik.solve(target, frame, q0=self.q[:5])
            if res.ok and (best is None or abs(res.q[4]) < abs(best[2].q[4]) - 0.3):
                best = (target, frame, res)
        if best is None:
            raise RuntimeError(f"{pid}: no reachable grasp")
        return best[0], best[1], g["width"]

    # --- episode -----------------------------------------------------------------------------------
    def run(self, pid, commanded_bin=None, inspect_seconds=1.0, probe=False):
        """Full episode for one part. If commanded_bin is not given, the correct bin per the work order (perfect inspection)."""
        env, p = self.env, self.env.p
        commanded_bin = commanded_bin or env.catalog[pid].expected_bin(env.order)
        self.q = env.data.ctrl[:6].copy()
        timed_out = False
        try:
            target, frame, width = self.plan_grasp(pid)
            above = target + np.array([0, 0, 0.045])

            self.phase = "reach"
            self.move_joints(np.r_[self.ik.solve(above, frame, q0=self.q[:5]).q, self.jaw_angle(width + OPEN_MARGIN)])
            self.move_line(target, frame, 0.7)

            self.phase = "grasp"
            self.set_gripper(max(self.jaw_angle(width) - 0.16, env.model.jnt_range[5, 0]), 0.6)
            self.hold(0.2)

            self.phase = "lift"
            self.move_line(above, frame, 0.6)

            self.phase = "inspect"
            show, show_frame = self.env.inspection_pose()
            self.move_joints(np.r_[self.ik.solve(show, show_frame, q0=self.q[:5]).q, self.q[5]])
            self.hold(inspect_seconds)

            self.phase = "place"
            bx, by = mm(p["bins"][commanded_bin]["center"])
            az = np.arctan2(by, bx)
            wall = float(mm(p["bins"]["wall_height"]))
            # parts dropped at the same spot every time stack into a tower; spread them in the bin
            spread = float(mm(p["bins"]["size"][0])) * 0.15
            bx, by = np.array([bx, by]) + env.rng.uniform(-spread, spread, size=2)
            drop = np.array([bx, by, wall + 0.05])
            res = None
            for tilt_deg in (25, 35, 45, 55, 65):
                t = np.radians(tilt_deg)
                approach = [np.cos(az) * np.sin(t), np.sin(az) * np.sin(t), -np.cos(t)]
                tangent = np.array([-np.sin(az), np.cos(az), 0.0])
                for sign in (1, -1):
                    cand = self.ik.solve(drop, grasp_frame(approach, sign * tangent), q0=self.q[:5])
                    if cand.ok and (res is None or abs(cand.q[4] - self.q[4]) < abs(res.q[4] - self.q[4])):
                        res = cand
                if res is not None:
                    break
            if res is None:
                res = self.ik.solve(drop, q0=self.q[:5])
            self.move_joints(np.r_[res.q, self.q[5]])
            self.hold(0.2)
            self.set_gripper(self.jaw_angle(width + 0.03), 0.4)
            self.hold(0.3)

            self.phase = "home"
            self.move_joints(HOME)
            self.hold(0.5)
        except TimeoutError:
            timed_out = True
        self.phase = "done"
        return env.outcome(pid, commanded_bin, timed_out=timed_out, probe=probe)
