import json

import pytest

from flywheel.robot.recorder import connect_robot, episode_log, reconcile, recorded, restore, snapshot_dir, writing_marker
from flywheel.robot.session import COMMAND_NAMES, command_vector, picks, session_parts, sessions, tray_layout
from flywheel.sim.plan import task


def test_sessions_cover_plan_once():
    for plan, n_sessions in (("record-W-seed0", 4), ("eval-heldout-W-seed0", 2)):
        split = sessions(plan)
        trays = [t for s in split for t in s]
        assert len(split) == n_sessions
        assert trays == sorted(set(trays))  # each tray once, in order
        assert max(map(len, split)) - min(map(len, split)) <= 1


def test_session_uses_each_part_once():
    # parts are set aside once at session start; the same part must not appear in two trays of a session
    plan = "record-W-seed0"
    for s, trays in enumerate(sessions(plan), 1):
        ids = [pid for t in trays for pid in tray_layout(plan, t)]
        assert len(ids) == len(set(ids)) == len(session_parts(plan, s))


def test_picks_follow_plan_order():
    ps = picks("record-W-seed0")
    assert [p.index for p in ps] == list(range(120))
    assert sum(p.probe for p in ps) == 60  # half are deliberately opposite
    for p in ps:
        assert tray_layout(p.plan, p.tray)[p.pocket - 1] == p.part


def test_command_names_pocket_and_bin():
    # the task text and the ACT command channel say the same thing: which pocket, which bin
    assert task(3, "reject") == "put the part from pocket 3 in the REJECT bin"
    v = command_vector(3, "reject")
    assert len(v) == len(COMMAND_NAMES) == 8
    assert [COMMAND_NAMES[k] for k, x in enumerate(v) if x] == ["pocket_3", "reject"]


def _write_log(root, n):
    f = episode_log(root)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("".join(json.dumps({"episode_index": i, "plan": "p", "index": i}) + "\n" for i in range(n)))


def test_reconcile_drops_lost_episodes(tmp_path):
    root = tmp_path / "ds"
    _write_log(root, 5)
    assert reconcile(root, 3) == 2
    assert [r["episode_index"] for r in recorded(root)] == [0, 1, 2]
    assert reconcile(root, 3) == 0


def test_restore_returns_to_checkpoint(tmp_path, capsys):
    root = tmp_path / "ds"
    snap = snapshot_dir(root)
    _write_log(snap, 6)
    (snap / "meta").mkdir()
    (snap / "meta" / "info.json").write_text("{}")
    _write_log(root, 8)  # 2 more episodes written after the checkpoint, then the process died
    writing_marker(root).touch()
    restore(root)
    assert len(recorded(root)) == 6
    assert not writing_marker(root).exists()
    assert len(list(tmp_path.glob("ds.broken-*"))) == 1
    assert "restored the checkpoint" in capsys.readouterr().out


class _Part:
    is_connected = False

    def disconnect(self, **_):
        self.is_connected = False


class _Robot:
    """Servo bus that misses a packet on the first `fails` connects, like Lock on id 2 with no status packet."""

    def __init__(self, fails):
        self.fails, self.bus, self.cameras = fails, _Part(), {"top": _Part()}

    def connect(self):
        self.bus.is_connected = True
        if self.fails:
            self.fails -= 1
            raise ConnectionError("Failed to write 'Lock' on id_=2")
        self.cameras["top"].is_connected = True


def test_connect_retries_then_gives_up(monkeypatch):
    monkeypatch.setattr("flywheel.robot.recorder.time.sleep", lambda _: None)
    robot = _Robot(fails=2)
    connect_robot(robot)
    assert robot.bus.is_connected and robot.cameras["top"].is_connected
    robot = _Robot(fails=3)
    with pytest.raises(ConnectionError):
        connect_robot(robot)
    assert not robot.bus.is_connected  # port closed again, nothing left half open


def test_tray_crops_and_measures_sim_frame():
    # sim top camera at the inspection resolution: tags -> 6 crops -> holes, teeth gap and broken corners from the silhouette
    from flywheel.robot.tray import find_tags, measure, pocket_crops, tray_homography, truth_for
    from flywheel.sim.env import QCSortSim

    ids = ["D-W-01", "D-T-07", "L-W-05", "L-T-07", "B-W-01", "B-W-07"]
    env = QCSortSim(part_ids=ids)
    assert [env.pocket_of(pid) for pid in ids] == [1, 2, 3, 4, 5, 6]
    frame = env.tray_frame()
    tags = find_tags(frame)
    assert sorted(tags) == [0, 1, 2, 3]
    for (_, crop), pid in zip(pocket_crops(frame, tray_homography(tags)), ids):
        truth, m = truth_for(pid), measure(crop, truth_for(pid)["type"])
        assert m["variant"] == truth["variant"], pid
        assert m["broken"] == truth["broken"], (pid, m)  # missing teeth, broken corner cut through the outline
        if truth["type"] != "pad":  # the sim draws a pad's ear hole as a notch in the plate edge, not a hole
            assert m["missing_hole"] == truth["missing_hole"], pid
