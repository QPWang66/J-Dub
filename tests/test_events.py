"""Synthetic-trajectory tests for M2/M3 detection. Court: hoop-left at (5.25, 25)."""

from jdub.events import classify_coverage, detect_actions, holders, matchups, offense

ATT = [1, 2, 3, 4, 5]
DEF = [11, 12, 13, 14, 15]
TEAM_OF = {p: 100 for p in ATT} | {p: 200 for p in DEF}
Z = 4.0  # held-ball height


def make_frames(n, ball_fn, pos_fn):
    return [(i, 720 - i / 25, ball_fn(i), pos_fn(i)) for i in range(n)]


def spread_5v5(att_xy):
    """Defenders 1.5 ft inside their attacker; returns full 10-player dict."""
    pos = {}
    for pid, (x, y) in zip(ATT, att_xy):
        pos[pid] = (x, y)
        pos[pid + 10] = (x - 1.5, y)
    return pos


def test_holder_sustained_and_flight():
    att_xy = [(30.0, 25.0), (20.0, 5.0), (20.0, 45.0), (15.0, 15.0), (15.0, 35.0)]
    # ball glued to player 1 for 20 frames, then flies far/high for 20
    frames = make_frames(
        40,
        lambda i: (30.0, 25.0, Z) if i < 20 else (60.0, 25.0, 15.0),
        lambda i: spread_5v5(att_xy),
    )
    hold = holders(frames)
    assert hold[:20] == [1] * 20
    assert hold[20:] == [None] * 20
    off = offense(hold, TEAM_OF)
    assert off == [100] * 40  # flight gap filled


def test_matchups_pair_nearest_defenders():
    att_xy = [(30.0, 25.0), (20.0, 5.0), (20.0, 45.0), (15.0, 15.0), (15.0, 35.0)]
    frames = make_frames(30, lambda i: (30.0, 25.0, Z), lambda i: spread_5v5(att_xy))
    hold = holders(frames)
    match = matchups(frames, offense(hold, TEAM_OF), TEAM_OF)
    assert match[0] == {p: p + 10 for p in ATT}
    assert match[29] is not None  # tail window covered


def test_screen_detected():
    def pos_fn(i):
        return {
            1: (30.0, 25.0),  # handler
            11: (28.5, 25.0),  # handler's defender (dd1 = 1.5)
            2: (27.5, 24.5),  # screener: da = 2.55, dd2 = 1.12 -> triangle fires
            12: (26.5, 23.5),
            3: (20.0, 5.0),
            13: (18.5, 5.0),
            4: (20.0, 45.0),
            14: (18.5, 45.0),
            5: (15.0, 15.0),
            15: (13.5, 15.0),
        }

    frames = make_frames(40, lambda i: (30.0, 25.0, Z), pos_fn)
    hold = holders(frames)
    off = offense(hold, TEAM_OF)
    match = matchups(frames, off, TEAM_OF)
    acts = detect_actions(frames, hold, off, match, TEAM_OF)
    screens = [a for a in acts if a["type"] == "screen"]
    assert screens and screens[0]["p1"] == 2 and screens[0]["p2"] == 1
    assert screens[0]["confidence"] == 1.0  # stationary screener = set screen


def test_drive_detected():
    def pos_fn(i):
        x = 25.0 - 17.0 * i / 49  # 25 -> 8 over 50 frames, toward hoop at 5.25
        base = {1: (x, 25.0), 11: (x - 1.5, 25.0)}
        for k, (pid, dpid) in enumerate(((2, 12), (3, 13), (4, 14), (5, 15))):
            base[pid] = (40.0, 8.0 + 10 * k)
            base[dpid] = (38.5, 8.0 + 10 * k)
        return base

    frames = make_frames(50, lambda i: (*pos_fn(i)[1], Z), pos_fn)
    hold = holders(frames)
    off = offense(hold, TEAM_OF)
    acts = detect_actions(frames, hold, off, [None] * 50, TEAM_OF)
    drives = [a for a in acts if a["type"] == "drive"]
    assert drives and drives[0]["p1"] == 1
    assert drives[0]["confidence"] >= 0.9  # straight line: distance proportion ~1


def test_cut_detected():
    def pos_fn(i):
        cx = 20.0 - 12.0 * min(i, 29) / 29  # cutter: 20 -> 8 over 30 frames
        return {
            1: (30.0, 25.0),
            11: (28.5, 25.0),  # static handler
            2: (cx, 25.0),
            12: (cx - 1.5, 25.0),  # cutter arrives at the rim
            3: (20.0, 45.0),
            13: (18.5, 45.0),
            4: (40.0, 8.0),
            14: (38.5, 8.0),
            5: (40.0, 42.0),
            15: (38.5, 42.0),
        }

    frames = make_frames(40, lambda i: (30.0, 25.0, Z), pos_fn)
    hold = holders(frames)
    off = offense(hold, TEAM_OF)
    acts = detect_actions(frames, hold, off, [None] * 40, TEAM_OF)
    cuts = [a for a in acts if a["type"] == "cut"]
    assert cuts and cuts[0]["p1"] == 2
    assert cuts[0]["confidence"] > 0.9


def _pnr_positions(d2_xy):
    return {
        1: (30.0, 25.0),  # handler
        11: (28.5, 25.0),  # on-ball defender
        2: (27.5, 24.5),  # screener
        12: d2_xy,  # screener's defender: position drives the coverage
        3: (20.0, 45.0),
        13: (18.5, 45.0),
        4: (40.0, 8.0),
        14: (38.5, 8.0),
        5: (40.0, 42.0),
        15: (38.5, 42.0),
    }


HOOP = (5.25, 25.0)
SCREEN = {"start": 5, "end": 15, "p1": 2, "p2": 1}
BASE_PAIRING = {1: 11, 2: 12, 3: 13, 4: 14, 5: 15}


def test_coverage_switch():
    frames = make_frames(60, lambda i: (30.0, 25.0, Z), lambda i: _pnr_positions((26.5, 23.5)))
    swapped = BASE_PAIRING | {1: 12, 2: 11}
    match = [BASE_PAIRING] * 20 + [swapped] * 40
    cov = classify_coverage(frames, match, SCREEN, HOOP)
    assert cov and cov["coverage"] == "switch" and cov["confidence"] >= 0.6


def test_coverage_drop():
    # screener's defender parked 12 ft from the hoop, on-ball defender far off the handler
    def pos_fn(i):
        p = _pnr_positions((12.0, 25.0))
        p[11] = (22.0, 25.0)  # d1 sagging 8 ft off: not "over"
        return p

    frames = make_frames(60, lambda i: (30.0, 25.0, Z), pos_fn)
    match = [BASE_PAIRING] * 60
    cov = classify_coverage(frames, match, SCREEN, HOOP)
    assert cov and cov["coverage"] == "drop"


def test_coverage_over():
    frames = make_frames(60, lambda i: (30.0, 25.0, Z), lambda i: _pnr_positions((26.5, 23.5)))
    match = [BASE_PAIRING] * 60  # no swap, d1 stays attached (1.5 ft), d2 high
    cov = classify_coverage(frames, match, SCREEN, HOOP)
    assert cov and cov["coverage"] in ("over", "blitz")  # d2 is also near: blitz acceptable


def test_handoff_detected():
    att_xy = [(30.0, 25.0), (33.0, 25.0), (20.0, 45.0), (15.0, 15.0), (15.0, 35.0)]

    def ball_fn(i):
        if i < 20:
            return (30.0, 25.0, Z)
        if i < 22:
            return (31.5, 25.0, Z)
        return (33.0, 25.0, Z)

    frames = make_frames(45, ball_fn, lambda i: spread_5v5(att_xy))
    hold = holders(frames)
    off = offense(hold, TEAM_OF)
    acts = detect_actions(frames, hold, off, [None] * 45, TEAM_OF)
    handoffs = [a for a in acts if a["type"] == "handoff"]
    assert handoffs and (handoffs[0]["p1"], handoffs[0]["p2"]) == (1, 2)
