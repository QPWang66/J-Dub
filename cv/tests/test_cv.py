import numpy as np

from jdub_cv.calib import to_court
from jdub_cv.pipeline import _interp, merge_tracks
from jdub_cv.teams import assign_teams


def test_to_court_roundtrip():
    import cv2

    img = np.float32([[100, 500], [900, 480], [700, 200], [250, 210]])
    court = np.float32([[75, 17], [94, 17], [94, 33], [75, 33]])
    h, _ = cv2.findHomography(img, court)
    out = to_court(h, img)
    assert np.allclose(out, court, atol=1e-3)


def test_interp_fills_small_gaps_only():
    s = {0: (0.0, 0.0), 4: (4.0, 8.0), 20: (0.0, 0.0)}
    out = _interp(s, max_gap=5)
    assert out[2] == (2.0, 4.0)
    assert 10 not in out  # 16-frame hole stays a hole


def test_merge_tracks_sews_fragments_not_concurrent():
    fps = 30.0
    red = [np.array([120.0, 180, 160])]
    # fragment B resumes where A ended after a short hole, same color -> merged
    obs = {
        1: {i: (50.0 + i * 0.1, 25.0) for i in range(0, 60)},
        2: {i: (56.5 + (i - 75) * 0.1, 25.2) for i in range(75, 120)},
        # concurrent with 1 -> different person, never merged
        3: {i: (80.0, 40.0) for i in range(0, 120)},
    }
    colors = {1: red, 2: red, 3: [np.array([240.0, 128, 128])]}
    merged, _, _ = merge_tracks(obs, colors, fps)
    assert set(merged) == {1, 3}
    assert max(merged[1]) == 119  # fragment 2 folded into 1


def test_assign_teams_two_biggest_clusters():
    dark, light, gray = [10.0, 120, 120], [240.0, 130, 130], [128.0, 128, 128]
    colors = {
        1: [np.array(dark)] * 3,
        2: [np.array(dark)] * 3,
        3: [np.array(light)] * 3,
        4: [np.array(light)] * 3,
        5: [np.array(gray)],
    }
    team = assign_teams(colors)
    assert team[1] == team[2] and team[3] == team[4] and team[1] != team[3]
    assert team[5] is None
