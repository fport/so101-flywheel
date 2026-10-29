import zipfile

import cv2
import numpy as np
import pytest
from manifold3d import Error

from flywheel.hardware.cli import REPRINTS, check_fit, print_jobs
from flywheel.hardware.fixtures import build_all
from flywheel.hardware.geom import write_3mf
from flywheel.sim.params import load_params, mast_top, top_camera_pose


@pytest.fixture(scope="module")
def built():
    p = load_params()
    items = build_all(p)
    return p, items, print_jobs(items, p["printer"])


def _extent(parts, at):
    lo = np.min([m.bounding_box()[:3] for m in parts], axis=0) + at
    hi = np.max([m.bounding_box()[3:] for m in parts], axis=0) + at
    return lo, hi


def test_every_part_is_valid(built):
    _, items, _ = built
    for item in items:
        assert all(m.status() == Error.NoError and m.volume() > 0 for m in item.parts), item.name


def test_fixtures_seat_on_baseplate_without_interference(built):
    _, items, _ = built
    assert check_fit(items) == {}


def test_jobs_are_numbered_and_cover_every_part_once(built):
    _, items, jobs = built
    assert [name[:2] for name, _, _ in jobs] == [f"{k:02d}" for k in range(1, len(jobs) + 1)]
    placed = [obj for _, _, objs in jobs for obj, _, _ in objs]
    assert len(placed) - len(REPRINTS) == len(set(placed)) == sum(i.qty for i in items)  # skipped-and-reprinted parts appear twice


def test_every_job_fits_the_bed_without_overlap(built):
    p, _, jobs = built
    bed, (x0, x1) = p["printer"]["bed"], p["printer"]["shared_x"]
    for name, _, objs in jobs:
        boxes = [_extent(parts, at) for _, parts, at in objs]
        for lo, hi in boxes:
            # x: area shared by both nozzles; a job spilling into the left-nozzle-only strip can't print on the right nozzle
            assert lo[0] >= x0 and lo[1] >= 0 and hi[0] <= x1 and hi[1] <= bed[1] and hi[2] <= bed[2], name
        for k, (lo, hi) in enumerate(boxes):
            for lo2, hi2 in boxes[k + 1 :]:
                assert lo[0] >= hi2[0] or lo2[0] >= hi[0] or lo[1] >= hi2[1] or lo2[1] >= hi[1], name


def test_heldout_parts_print_on_their_own_plates(built):
    _, items, jobs = built
    heldout = {i.name for i in items if i.group == "qc" and i.meta["heldout"]}
    assert len(heldout) == 16
    for name, _, objs in jobs:
        on_plate = {obj for obj, _, _ in objs}
        if on_plate & heldout:
            assert on_plate <= heldout and "heldout" in name


def test_mast_holds_the_camera_at_its_configured_position(built):
    p, items, _ = built
    cradle = next(i for i in items if i.name == "cam_cradle_c310")
    lo, hi = _extent(cradle.preview, [*p["mast"]["pos"], mast_top(p)])
    cam, _ = top_camera_pose(p)
    assert np.all(lo - 1 <= cam) and np.all(cam <= hi + 1)  # lens inside the cradle
    segment = next(i for i in items if i.name == "mast_segment")
    assert segment.qty == p["mast"]["segments"] and segment.parts[0].bounding_box()[5] <= p["printer"]["bed"][2]


def _top_view(cs, px_per_mm=10):
    """Draws the section as a grey image seen from above (white = section), with a 10 mm black margin."""
    x0, y0, x1, y1 = cs.bounds()
    w, h = int((x1 - x0 + 20) * px_per_mm), int((y1 - y0 + 20) * px_per_mm)
    img = np.zeros((h, w), np.uint8)
    polys = [np.round([((x - x0 + 10) * px_per_mm, (y1 + 10 - y) * px_per_mm) for x, y in poly]).astype(np.int32)
             for poly in cs.to_polygons()]
    cv2.fillPoly(img, polys, 255)
    return img


def test_tag_tiles_carry_readable_two_color_tags(built, tmp_path):
    p, items, jobs = built
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11))
    for t in p["tags"]["items"]:
        tile = next(i for i in items if i.name == f"tag_{t['id']}")
        black, white = tile.parts
        assert (black ^ white).volume() < 0.01  # inlay doesn't overlap the body
        lo, hi = np.array(white.bounding_box()).reshape(2, 3)
        assert hi[2] == pytest.approx(black.bounding_box()[5]) and hi[2] - lo[2] >= 1.0  # flush with the top, 5 layers
        img = _top_view(white.slice(hi[2] - 0.1))
        corners, ids, _ = detector.detectMarkers(img)
        assert ids is not None and ids.ravel().tolist() == [t["id"]], t["id"]
        c = corners[0][0]
        assert abs(np.linalg.norm(c[0] - c[1]) / 10 - t["size"]) < 0.3
        assert c[0][1] < c[3][1]  # corner 0 on top: pattern not mirrored, +y up
    name, _, objs = next(j for j in jobs if j[0].endswith("tag_tiles"))
    write_3mf(tmp_path / f"{name}.3mf", objs)
    with zipfile.ZipFile(tmp_path / f"{name}.3mf") as z:
        conf = z.read("Metadata/model_settings.config").decode()
    assert conf.count('key="extruder" value="2"') == len(p["tags"]["items"])  # white inlay is the second filament


def test_baseplate_tiles_interlock_without_loose_keys(built):
    _, items, _ = built
    tiles = {i.name: i.body.translate([*i.at, 0]) for i in items if i.group == "baseplate" and i.name != "baseplate_key"}
    hub = tiles.pop("baseplate_A2")
    for name, tile in tiles.items():
        assert (hub ^ tile).volume() < 0.01, name  # no clash in place
        away = np.array(hub.bounding_box()).reshape(2, 3).mean(0) - np.array(tile.bounding_box()).reshape(2, 3).mean(0)
        step = np.zeros(3)
        step[np.argmax(np.abs(away[:2]))] = 3 * np.sign(away[np.argmax(np.abs(away[:2]))])
        assert (hub.translate(step) ^ tile).volume() > 10, name  # pulled sideways, the tongue catches in the pocket


def test_footed_parts_print_without_long_bridges(built):
    # feet on the bed; the body floor (z=0) must sit almost entirely on top of the feet
    _, items, _ = built
    for item in items:
        body = item.body
        if item.group == "baseplate" or abs(body.bounding_box()[2] + 4.75) > 0.01:
            continue
        base, feet = body.slice(0.05).area(), body.slice(-0.05).area()
        assert feet > 0.9 * base, item.name
