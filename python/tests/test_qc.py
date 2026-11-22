def test_tray_crops_and_measures_sim_frame():
    # sim top camera at the inspection resolution: tags -> 6 crops -> holes, teeth gap and broken corners from the silhouette
    from flywheel.qc.tray import find_tags, measure, pocket_crops, tray_homography, truth_for
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
