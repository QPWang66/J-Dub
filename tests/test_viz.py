import math

from jdub.viz import spread_overlaps


def test_spread_overlaps_separates_close_points():
    pts = [(10.0, 10.0), (10.0, 10.0), (10.5, 10.2), (50.0, 25.0)]
    out = spread_overlaps(pts, min_sep=2.0)
    for i in range(len(out)):
        for j in range(i + 1, len(out)):
            assert math.dist(out[i], out[j]) >= 2.0 - 1e-6
    assert out[3] == (50.0, 25.0)  # far point untouched


def test_spread_overlaps_noop_when_apart():
    pts = [(10.0, 10.0), (20.0, 20.0)]
    assert spread_overlaps(pts) == pts
