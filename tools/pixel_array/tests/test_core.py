import numpy as np
import pytest

from pixel_array.core import (MACHINES, Part, Settings, apply, arrange, min_footprint_angle,
                              rot_z, export_stl, preview_labels)
from pixel_array.stlio import read_stl, write_stl


def box(sx, sy, sz):
    v = np.array([[x, y, z] for x in (0, sx) for y in (0, sy) for z in (0, sz)], float)
    f = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
         (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    return v[np.array(f)]


def cylinder(r, h, n=48):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.column_stack([r * np.cos(a), r * np.sin(a)])
    tris = []
    for k in range(n):
        p, q = ring[k], ring[(k + 1) % n]
        tris += [[(0, 0, 0), (*q, 0), (*p, 0)], [(0, 0, h), (*p, h), (*q, h)],
                 [(*p, 0), (*q, 0), (*q, h)], [(*p, 0), (*q, h), (*p, h)]]
    return np.array(tris, float)


def placed_boxes(parts, layout):
    out = []
    for i, T in layout.transforms(parts):
        v = apply(parts[i].tris, T).reshape(-1, 3)
        out.append((v.min(axis=0), v.max(axis=0)))
    return out


def on_grid(x, px):
    k = x / px
    return abs(k - round(k)) < 1e-6


def test_stl_roundtrip(tmp_path):
    t = box(1, 2, 3)
    write_stl(tmp_path / "a.stl", t)
    assert np.allclose(read_stl(tmp_path / "a.stl"), t)
    with open(tmp_path / "b.stl", "w") as f:
        f.write("solid x\n")
        for tri in t:
            f.write("facet normal 0 0 0\nouter loop\n")
            for v in tri:
                f.write("vertex %f %f %f\n" % tuple(v))
            f.write("endloop\nendfacet\n")
        f.write("endsolid x\n")
    assert np.allclose(read_stl(tmp_path / "b.stl"), t)


def test_min_footprint_unrotates_box():
    t = apply(box(10, 4, 2), rot_z(30))
    ang = min_footprint_angle(t)
    v = apply(t, rot_z(ang)).reshape(-1, 3)
    ext = v.max(axis=0) - v.min(axis=0)
    assert np.allclose(ext[:2], [10, 4], atol=1e-6)


def test_axis_aligned_part_not_rotated():
    assert min_footprint_angle(box(10, 4, 2)) == 0.0


@pytest.mark.parametrize("mi,nest", [(0, False), (1, False), (0, True), (1, True)])
def test_array_copies_pixel_locked_no_overlap(mi, nest):
    m = MACHINES[mi]
    s = Settings(machine=m, spacing=1.0, margin=1.0, nest=nest)
    parts = [Part("b", apply(box(5, 3, 2), rot_z(17)) + [3, 4, 5], count=40)]
    lay = arrange(parts, s)
    assert lay.unplaced == {} and len(lay.instances) == 40
    boxes = placed_boxes(parts, lay)
    for lo, hi in boxes:
        assert on_grid(lo[0] + m.width / 2, m.px) and on_grid(lo[1] + m.height / 2, m.py)
        assert lo[2] == pytest.approx(0)
        assert lo[0] >= -m.width / 2 + 1 - 1e-6 and hi[0] <= m.width / 2 - 1 + 1e-6
        assert lo[1] >= -m.height / 2 + 1 - 1e-6 and hi[1] <= m.height / 2 - 1 + 1e-6
        assert np.allclose(sorted(hi[:2] - lo[:2]), [3, 5], atol=1e-6)
    for a in range(len(boxes)):
        for b in range(a + 1, len(boxes)):
            (l1, h1), (l2, h2) = boxes[a], boxes[b]
            gap = max(l2[0] - h1[0], l1[0] - h2[0], l2[1] - h1[1], l1[1] - h2[1])
            assert gap >= 1.0 - 1e-6
    # 紧密：同一行相邻两件的节距不超过 尺寸 + 1mm + 几个像素
    xs = sorted({round(lo[0], 6) for lo, _ in boxes})
    assert min(np.diff(xs)) <= 5 + 1.0 + 4 * m.px


def test_auto_array_is_regular_grid():
    m = MACHINES[0]
    parts = [Part("b", box(5, 3, 2), count=30)]
    lay = arrange(parts, Settings(machine=m, allow_rot90=False))
    lows = np.array([lo[:2] for lo, _ in placed_boxes(parts, lay)])
    xs, ys = np.unique(lows[:, 0].round(6)), np.unique(lows[:, 1].round(6))
    # 第一行放满平台宽度，节距 = ceil((5+1)/px) 像素
    assert np.allclose(np.diff(xs), np.ceil(6 / m.px - 1e-6) * m.px)
    assert np.allclose(np.diff(ys), np.ceil(4 / m.py - 1e-6) * m.py)
    assert len(xs) * (len(ys) - 1) < 30 <= len(xs) * len(ys)


def test_too_many_reports_unplaced():
    parts = [Part("b", box(20, 20, 2), count=50)]
    lay = arrange(parts, Settings())
    assert len(lay.instances) == 8 and lay.unplaced == {0: 42}


def test_grid_mode_pitch():
    m = MACHINES[0]
    parts = [Part("b", box(5, 3, 2), count=6, grid_x=3, grid_y=2, gap_x=1.0, gap_y=0.3)]
    lay = arrange(parts, Settings(machine=m))
    xs = sorted({round(lo[0], 6) for lo, _ in placed_boxes(parts, lay)})
    ys = sorted({round(lo[1], 6) for lo, _ in placed_boxes(parts, lay)})
    assert len(xs) == 3 and len(ys) == 2
    assert np.allclose(np.diff(xs), np.ceil(6 / m.px - 1e-6) * m.px)
    assert np.allclose(np.diff(ys), np.ceil(3.3 / m.py - 1e-6) * m.py)


def test_mixed_parts_and_export(tmp_path):
    parts = [Part("box", box(8, 6, 3), count=3), Part("cyl", cylinder(3, 4), count=5),
             Part("big", box(30, 20, 5))]
    s = Settings(machine=MACHINES[1])
    lay = arrange(parts, s)
    assert lay.unplaced == {} and len(lay.instances) == 9
    labels = preview_labels(lay)
    assert set(np.unique(labels)) == {0, 1, 2, 3}
    n = export_stl(tmp_path / "out.stl", parts, lay)
    t = read_stl(tmp_path / "out.stl")
    assert n == 9 and len(t) == 3 * 12 + 5 * len(parts[1].tris) + 12
