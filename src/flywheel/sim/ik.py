"""Damped least squares (DLS) inverse kinematics for the SO-101.

The target is the position of the `pinch` site (fixed jaw pad face) and the orientation of the gripper frame. The arm
has 5 DOF, so not every orientation is reachable; the solver finds the closest pose and reports the residual error.
Gripper frame: jaws extend along -z, +x is the side the moving jaw closes toward.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

N_ARM = 5


@dataclass
class IKResult:
    q: np.ndarray  # 5 arm joints (rad)
    pos_err: float  # m
    rot_err: float  # rad
    ok: bool


def grasp_frame(approach, closing):
    """Gripper orientation matrix from the approach direction (where the jaws point) and the closing direction."""
    z = -np.asarray(approach, float)
    z /= np.linalg.norm(z)
    x = np.asarray(closing, float)
    x = x - z * (x @ z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def top_down(yaw):
    """Vertical approach; yaw = angle of the closing axis about world z."""
    return grasp_frame([0, 0, -1], [np.cos(yaw), np.sin(yaw), 0])


class IK:
    def __init__(self, model, site="pinch"):
        self.m = model
        self.d = mujoco.MjData(model)
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        self.lo = model.jnt_range[:N_ARM, 0] + 0.02
        self.hi = model.jnt_range[:N_ARM, 1] - 0.02
        self._jp = np.zeros((3, model.nv))
        self._jr = np.zeros((3, model.nv))

    def fk(self, q):
        self.d.qpos[:N_ARM] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

    def _error(self, q, pos, rot, approach):
        p, r = self.fk(q)
        if rot is not None:
            e_r = 0.5 * sum(np.cross(r[:, i], rot[:, i]) for i in range(3))
        elif approach is not None:
            e_r = np.cross(r[:, 2], -approach)  # only the jaw axis is aligned, roll is free
        else:
            e_r = np.zeros(3)
        return pos - p, e_r

    def _solve_from(self, q0, pos, rot, approach, roll, rot_weight, iters, damping):
        q = np.clip(np.asarray(q0, float).copy(), self.lo, self.hi)
        n = N_ARM if roll is None else N_ARM - 1
        if roll is not None:
            q[N_ARM - 1] = roll
        for _ in range(iters):
            e_p, e_r = self._error(q, pos, rot, approach)
            if np.linalg.norm(e_p) < 2e-4 and np.linalg.norm(e_r) < 2e-3:
                break
            mujoco.mj_jacSite(self.m, self.d, self._jp, self._jr, self.site)
            jac = np.vstack([self._jp[:, :n], rot_weight * self._jr[:, :n]])
            err = np.concatenate([e_p, rot_weight * e_r])
            dq = jac.T @ np.linalg.solve(jac @ jac.T + damping * np.eye(6), err)
            step = np.max(np.abs(dq))
            if step > 0.2:
                dq *= 0.2 / step
            q[:n] = np.clip(q[:n] + dq, self.lo[:n], self.hi[:n])
        e_p, e_r = self._error(q, pos, rot, approach)
        return q, float(np.linalg.norm(e_p)), float(np.linalg.norm(e_r))

    def solve(self, pos, rot=None, approach=None, roll=None, q0=None, rot_weight=0.1, iters=200,
              damping=1e-4, pos_tol=1.5e-3, rot_tol=0.05):
        """Returns the result from the best starting seed.

        rot: 3x3 target orientation. approach: only the direction the jaws point (unit vector); if roll is given,
        wrist_roll is locked to that value. If neither is given, only position is solved.
        """
        pos = np.asarray(pos, float)
        if approach is not None:
            approach = np.asarray(approach, float) / np.linalg.norm(approach)
        pan = np.arctan2(pos[1], pos[0])
        seeds = [] if q0 is None else [np.asarray(q0, float)[:N_ARM]]
        # pan axis is -z: positive angle turns clockwise
        seeds += [np.array([-pan, -0.3, 0.6, 1.2, 0.0]), np.array([-pan, 0.3, -0.2, 1.4, 0.0])]
        best = None
        for seed in seeds:
            q, ep, er = self._solve_from(seed, pos, rot, approach, roll, rot_weight, iters, damping)
            score = ep + 0.02 * er
            if best is None or score < best[0]:
                best = (score, q, ep, er)
            if ep < pos_tol and er < rot_tol:
                break
        _, q, ep, er = best
        return IKResult(q=q, pos_err=ep, rot_err=er, ok=ep < pos_tol and er < rot_tol)
