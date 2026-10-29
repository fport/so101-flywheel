import cv2
import mujoco
import numpy as np
import pytest

from flywheel.sim.env import QCSortSim
from flywheel.sim.ik import IK, top_down
from flywheel.sim.params import SCENE_XML, load_params, mm, pocket_centers
from flywheel.sim.parts import decide
from flywheel.sim.plan import build_plan
from flywheel.sim.scene import JOINTS, build_scene
from flywheel.sim.scripted import ScriptedSorter


@pytest.fixture(scope="module")
def env():
    return QCSortSim(save_scene=True)


def test_catalog_is_factorial(env):
    parts = list(env.catalog.values())
    train = [c for c in parts if not c.heldout]
    assert len(train) == 60 and len(parts) - len(train) == 16
    for ptype in ("rim", "gear", "pad", "bracket"):
        for variant in ("W", "T"):
            cell = [c for c in train if c.type == ptype and c.variant == variant]
            # color alone must not tell the bin: each color has sound parts and both defects
            assert {c.defect for c in cell} == {None, "missing_hole", "broken_corner"}
    assert not any("OK" in c.id or "REJECT" in c.id for c in parts)


def test_label_depends_on_work_order(env):
    good_white = next(c for c in env.catalog.values() if c.variant == "W" and c.defect is None)
    good_orange = next(c for c in env.catalog.values() if c.variant == "T" and c.defect is None)
    broken = next(c for c in env.catalog.values() if c.defect == "broken_corner")
    assert (good_white.expected_bin("W"), good_white.expected_bin("T")) == ("ok", "reject")
    assert (good_orange.expected_bin("W"), good_orange.expected_bin("T")) == ("reject", "ok")
    assert (broken.expected_bin("W"), broken.expected_bin("T")) == ("reject", "reject")
    for c in env.catalog.values():
        assert decide(c.attributes(), "W") == c.expected_bin("W")


def test_plan_sends_every_part_to_both_bins(env):
    catalog = list(env.catalog.values())
    trays = build_plan(catalog, order="W", seed=3)
    picks = [pick for t in trays for pick in t["picks"]]
    assert len(picks) == 120 and sum(p["probe"] for p in picks) == 60
    for c in catalog:
        if c.heldout:
            continue
        mine = [p for p in picks if p["part"] == c.id]
        assert {p["command"] for p in mine} == {"ok", "reject"}
        assert len({p["pocket"] for p in mine}) == 2
        assert next(p for p in mine if not p["probe"])["command"] == c.expected_bin("W")
    # no link between look and command: both commands equally often for each type, color and defect
    for key in ("type", "variant", "defect"):
        for value in {getattr(c, key) for c in catalog}:
            ids = {c.id for c in catalog if getattr(c, key) == value}
            cmds = [p["command"] for p in picks if p["part"] in ids]
            assert cmds.count("ok") == cmds.count("reject")


def test_saved_scene_reloads(env):
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    assert [model.joint(i).name for i in range(6)] == JOINTS
    assert [model.actuator(i).name for i in range(6)] == JOINTS
    assert {model.camera(i).name for i in range(model.ncam)} == {"top", "wrist"}
    assert "/Users/" not in SCENE_XML.read_text()


def test_fixtures_inside_work_band():
    p = load_params()
    lo, hi = mm(p["measured"]["reach_min"]), mm(p["measured"]["reach_comfort"])
    points = pocket_centers(p) + [tuple(mm(p["bins"][b]["center"])) for b in ("ok", "reject")]
    for x, y in points:
        assert lo <= np.hypot(x, y) <= hi


def test_every_pocket_reachable_top_down(env):
    ik = IK(env.model)
    for x, y in pocket_centers(env.p):
        for yaw in (np.pi / 2, -np.pi / 2):
            assert ik.solve([x, y, 0.015], top_down(yaw)).ok


def test_top_camera_sees_all_tags(env):
    env.reset()
    gray = cv2.cvtColor(env.render("top"), cv2.COLOR_RGB2GRAY)
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11))
    corners, ids, _ = detector.detectMarkers(gray)
    by_id = {t["id"]: t for t in env.p["tags"]["items"]}
    assert sorted(ids.ravel().tolist()) == sorted(by_id)
    for c, tag_id in zip(corners, ids.ravel()):
        center = mm(by_id[int(tag_id)]["center"])
        assert np.allclose(c.reshape(4, 2).mean(0), env.project([*center, 0.0]), atol=2.0)


def test_untouched_part_is_grasp_fail(env):
    env.reset()
    out = env.outcome("J-W-01", "ok")
    assert (out["success"], out["failure_type"], out["location"]) == (False, "grasp_fail", "tray")


def test_scripted_episode_sorts_into_expected_bin(env):
    env.reset()
    for pid in ("J-W-01", "D-T-04"):
        env.frame = 0
        out = ScriptedSorter(env).run(pid)
        assert out["success"] and out["location"] == env.catalog[pid].expected_bin(env.order)


def test_wrong_command_is_misclassification(env):
    env.reset()
    env.frame = 0
    out = ScriptedSorter(env).run("L-W-07", commanded_bin="ok")
    assert (out["failure_type"], out["location"]) == ("misclassification", "ok")


def test_probe_trial_scores_command_following(env):
    env.reset()
    env.frame = 0
    out = ScriptedSorter(env).run("B-T-01", commanded_bin="ok", probe=True)  # sound orange: REJECT when the order is white
    assert out["expected_bin"] == "reject" and out["followed_command"] and out["success"]


def test_parts_fit_the_stock_gripper(env):
    """Each part is gripped with a small fraction of the standard jaw opening; the pad contacts at least 6 mm."""
    sorter = ScriptedSorter(env)
    jaw_max = env.model.jnt_range[5, 1]
    for part in {c.type: c for c in env.catalog.values()}.values():
        assert sorter.jaw_angle(part.grasp["width"] + 0.012) < 0.4 * jaw_max, part.type
        pinch = max(part.grasp["center"][2], 0.0094 + 0.0022)
        contact = min(pinch + 0.010, part.height) - (pinch - 0.0094)
        assert contact >= 0.006, (part.type, contact)
