import numpy as np

from jdub_cv.calib import to_court
from jdub_cv.pipeline import _interp, _pick_ball, merge_tracks, pack_slots
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


def test_pick_ball_locked_never_teleports_and_reacquires_confident():
    fps, w = 30.0, 1280
    track = {10: (600.0, 400.0)}
    near, far = (0.1, 620.0, 410.0), (0.99, 100.0, 100.0)
    # locked: nearest reachable wins even against a higher-score far candidate
    assert _pick_ball([far, near], track, 11, fps, w) == (620.0, 410.0)
    # locked but nothing reachable: stay lost instead of teleporting
    assert _pick_ball([far], track, 11, fps, w) is None
    # lock expired (> BALL_GAP_S): weak candidate rejected, confident accepted
    assert _pick_ball([(0.2, 100.0, 100.0)], track, 11 + int(2 * fps), fps, w) is None
    assert _pick_ball([far], track, 11 + int(2 * fps), fps, w) == (100.0, 100.0)
    # cold start: acquire only on confidence
    assert _pick_ball([near], {}, 0, fps, w) is None
    assert _pick_ball([far], {}, 0, fps, w) == (100.0, 100.0)


def test_pack_slots_stitches_fragments_and_caps_slots():
    fps = 30.0
    tracks = {
        # player A: two fragments, resumes nearby after a hole -> one slot
        1: {i: (50.0, 25.0) for i in range(0, 60)},
        2: {i: (52.0, 26.0) for i in range(90, 150)},
        # player B: concurrent with A -> its own slot
        3: {i: (80.0, 40.0) for i in range(0, 150)},
        # far teleport after A's fragment: not reachable -> new slot
        4: {i: (10.0, 5.0) for i in range(62, 80)},
    }
    packed = pack_slots(tracks, n_slots=2, fps=fps)
    assert len(packed) == 2
    by_members = {tuple(m): fs for fs, m in packed}
    assert (1, 2) in by_members  # A's fragments stitched
    assert max(by_members[(1, 2)]) == 149
    assert (3,) in by_members  # B untouched; track 4 dropped (no slot left)


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
