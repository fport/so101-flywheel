import json

from flywheel.robot.recorder import episode_log, reconcile, recorded, restore, snapshot_dir, writing_marker
from flywheel.robot.session import command_onehot, phase2_task, picks, session_parts, sessions, tray_layout


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


def test_phase_two_command():
    assert phase2_task("ok") == "put the part in the OK bin"
    assert command_onehot("ok") == [1.0, 0.0] and command_onehot("reject") == [0.0, 1.0]


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
