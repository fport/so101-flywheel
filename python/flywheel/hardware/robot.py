"""Surfaces where printed parts meet the arm are measured from the SO-101 meshes (the real arm is printed from the same STLs)."""

import mujoco
import numpy as np

from ..sim.params import SO101_XML
from ..sim.scene import BASE_BOTTOM_Z, PAN_AXIS_XY


def _body_points(model, data, body, frame_body=None):
    """Visual mesh vertices of a body, in the frame_body frame (world if None), mm."""
    pts = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != body or model.geom_group[g] != 2:
            continue
        mid = model.geom_dataid[g]
        v = model.mesh_vert[model.mesh_vertadr[mid] : model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
        pts.append(v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])
    pts = np.vstack(pts)
    if frame_body is not None:
        pts = (pts - data.xpos[frame_body]) @ data.xmat[frame_body].reshape(3, 3)
    return pts * 1000


class Robot:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(SO101_XML))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

    def base_outline(self, band=4.0):
        """xy points, in scene coordinates, of the base's bottom band touching the table (first `band` mm)."""
        pts = _body_points(self.model, self.data, self.model.body("base").id)
        pts = pts[pts[:, 2] < BASE_BOTTOM_Z * 1000 + band]
        ax, ay = np.array(PAN_AXIS_XY) * 1000
        # in the scene the base is rotated +90 degrees about z and the pan axis sits at the origin
        return np.column_stack([-(pts[:, 1] - ay), pts[:, 0] - ax])
