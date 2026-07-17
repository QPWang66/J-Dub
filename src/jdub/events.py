"""M2: ball holder, offense inference, matchup assignment, atomic action detection.

Thresholds follow the literature (see docs/detection-research.md):
- screen: NETS triangle rule (Hauri & Vucetic, ECAI 2023; 82% precision on this
  same public SportVU dump) — handler-screener <= 6 ft, defender-handler <= 6 ft,
  defender-screener <= 3 ft.
- handoff: NETS — possession change between offensive players < 6.5 ft apart.
- matchup: cost to the Franks et al. (AoAS 2015) canonical defender spot
  0.62*offender + 0.11*ball + 0.27*hoop, optimal 5x5 assignment.
- holder: nearest player, ball within 5 ft and below 10 ft, sustained >= 5 frames.
- drive/cut: PMC9904462 thresholds (unverified by the research pass — marked).

All detection runs per SportVU event on "complete" frames (ball + exactly 10
players). Coordinates are raw court feet; the attacked hoop is inferred per
event from where the ball lives.
"""

from __future__ import annotations

import math
from itertools import permutations
from pathlib import Path

import polars as pl

from jdub.data import PARQUET_DIR, load_moments

FPS = 25
HOLD_DIST = 5.0  # ft: ball within this of the nearest player = candidate holder
HOLD_MAX_Z = 10.0  # ft: ball above this is a shot, not possession
HOLD_MIN_FRAMES = 5  # sustained ~0.2s
WINDOW = 25  # matchup window (1s)
STEP = 12  # matchup stride (~0.5s)
GAMMA_OFF, GAMMA_BALL, GAMMA_HOOP = 0.62, 0.11, 0.27  # Franks canonical spot
SCREEN_DA = 6.0  # ft: screener to handler
SCREEN_DD1 = 6.0  # ft: on-ball defender to handler
SCREEN_DD2 = 3.0  # ft: on-ball defender to screener
SCREEN_MIN_FRAMES = 3
SCREEN_SET_SPEED = 2.0  # ft/s: below this the screener is "set" (confidence signal)
HANDOFF_MAX_GAP = 5  # frames between holders: a real handoff is near-instant
HANDOFF_MAX_DIST = 6.5  # ft at exchange (NETS: average wingspan)
HANDOFF_MIN_RUN = 8  # frames: both giver and receiver must really hold the ball
# ponytail: drive/cut numbers are the unverified PMC9904462 set; spot-check the PDF
DRIVE_MIN_START = 8.5  # ft from hoop (excludes post moves)
DRIVE_MAX_START = 28.4
DRIVE_MIN_SPEED = 5.23  # ft/s mean along path
DRIVE_MIN_DP = 0.50  # distance proportion: basket-distance drop / path length
DRIVE_MIN_DECLINE = 5.0  # ft: a drive has to actually go somewhere
DRIVE_MIN_FRAMES = 12
CUT_MIN_SPEED = 7.5  # ft/s: PMC's 5.96 floor fires on jogs; a called cut is a burst
CUT_MIN_DECLINE = 10.0  # ft: perimeter-to-rim depth, not a short slide
CUT_MIN_DP = 0.77
CUT_MAX_START = 23.4
CUT_MAX_END = 8.5  # cut must arrive at the rim area
CUT_LOOKBACK = 75  # frames (3s)
CUT_MIN_FRAMES = 10
CUT_HELD_FRAC = 0.6  # a cut happens inside settled offense, not rebound/transition scrambles

# frame tuple: (moment_idx, game_clock, (bx, by, bz), {pid: (x, y)})


def complete_frames(m: pl.DataFrame, event_id: int) -> tuple[list[tuple], dict[int, int], int]:
    """One event's frames having ball + exactly 10 players. Returns (frames, pid->team, quarter)."""
    ev = m.filter(pl.col("event_id") == event_id).sort("moment_idx")
    frames: list[tuple] = []
    team_of: dict[int, int] = {}
    quarter = ev["quarter"][0] if ev.height else 0
    for (mi,), g in ev.group_by("moment_idx", maintain_order=True):
        ball, pos, gc = None, {}, None
        for r in g.iter_rows(named=True):
            gc = r["game_clock"]
            if r["entity"] == "ball":
                ball = (r["x"], r["y"], r["z"])
            else:
                pos[r["player_id"]] = (r["x"], r["y"])
                team_of[r["player_id"]] = r["team_id"]
        if ball is not None and len(pos) == 10:
            frames.append((mi, gc, ball, pos))
    return frames, team_of, quarter


def holders(frames: list[tuple]) -> list[int | None]:
    """Per-frame ball holder: nearest player, ball within HOLD_DIST and below
    HOLD_MAX_Z, sustained >= HOLD_MIN_FRAMES."""
    raw: list[int | None] = []
    for _, _, ball, pos in frames:
        best, bd = None, HOLD_DIST
        if ball[2] < HOLD_MAX_Z:
            for pid, (x, y) in pos.items():
                d = math.hypot(x - ball[0], y - ball[1])
                if d < bd:
                    best, bd = pid, d
        raw.append(best)
    out: list[int | None] = [None] * len(raw)
    i = 0
    while i < len(raw):
        j = i
        while j < len(raw) and raw[j] == raw[i]:
            j += 1
        if raw[i] is not None and j - i >= HOLD_MIN_FRAMES:
            out[i:j] = [raw[i]] * (j - i)
        i = j
    return out


def offense(hold: list[int | None], team_of: dict[int, int]) -> list[int | None]:
    """Offensive team per frame: holder's team, filled across no-holder gaps (ball in flight)."""
    # ponytail: gap frames around a turnover inherit the old offense until the new
    # holder appears; possession-boundary precision is an M3 concern
    off = [team_of[h] if h is not None else None for h in hold]
    last = None
    for i, v in enumerate(off):
        if v is None:
            off[i] = last
        else:
            last = v
    nxt = None
    for i in range(len(off) - 1, -1, -1):
        if off[i] is None:
            off[i] = nxt
        else:
            nxt = off[i]
    return off


def attacked_hoop(frames: list[tuple]) -> tuple[float, float]:
    """Hoop the offense attacks: the one on the half where the ball lives."""
    mean_x = sum(f[2][0] for f in frames) / len(frames)
    return (5.25, 25.0) if mean_x <= 47.0 else (88.75, 25.0)


def matchups(
    frames: list[tuple], off: list[int | None], team_of: dict[int, int]
) -> list[dict[int, int] | None]:
    """Per-frame {attacker_pid: defender_pid} via optimal assignment on sliding windows.

    Cost = summed distance from the defender to the Franks canonical spot
    (0.62*attacker + 0.11*ball + 0.27*hoop); 5x5 optimum by brute force over
    120 permutations (exact, no dependency).
    """
    n = len(frames)
    assign: list[dict[int, int] | None] = [None] * n
    if not n:
        return assign
    hx, hy = attacked_hoop(frames)
    for s in range(0, n, STEP):
        w = range(s, min(s + WINDOW, n))
        teams = {off[i] for i in w}
        if len(teams) != 1 or None in teams:
            continue
        off_team = next(iter(teams))
        pids = list(frames[s][3].keys())
        attackers = [p for p in pids if team_of[p] == off_team]
        defenders = [p for p in pids if team_of[p] != off_team]
        if len(attackers) != 5 or len(defenders) != 5:
            continue
        cost = [[0.0] * 5 for _ in range(5)]
        ok = True
        for i in w:
            _, _, ball, pos = frames[i]
            if any(p not in pos for p in pids):  # substitution mid-window
                ok = False
                break
            for ai, ap in enumerate(attackers):
                ax, ay = pos[ap]
                cx = GAMMA_OFF * ax + GAMMA_BALL * ball[0] + GAMMA_HOOP * hx
                cy = GAMMA_OFF * ay + GAMMA_BALL * ball[1] + GAMMA_HOOP * hy
                for di, dp in enumerate(defenders):
                    dx, dy = pos[dp]
                    cost[di][ai] += math.hypot(dx - cx, dy - cy)
        if not ok:
            continue
        best = min(permutations(range(5)), key=lambda p: sum(cost[d][p[d]] for d in range(5)))
        pairing = {attackers[best[d]]: defenders[d] for d in range(5)}
        for i in range(s, min(s + STEP, n)):
            assign[i] = pairing
    return assign


def _speed(frames: list[tuple], pid: int, i: int) -> float:
    j0, j1 = max(0, i - 2), min(len(frames) - 1, i + 2)
    a, b = frames[j0][3].get(pid), frames[j1][3].get(pid)
    if a is None or b is None or j0 == j1:
        return 0.0
    return math.hypot(b[0] - a[0], b[1] - a[1]) * FPS / (j1 - j0)


def _intervals(idxs: list[int], max_gap: int, min_len: int) -> list[tuple[int, int]]:
    """Group sorted frame indices into (start, end) runs, tolerating gaps."""
    out: list[tuple[int, int]] = []
    for i in idxs:
        if out and i - out[-1][1] <= max_gap:
            out[-1] = (out[-1][0], i)
        else:
            out.append((i, i))
    return [(a, b) for a, b in out if b - a + 1 >= min_len]


def _path_stats(frames: list[tuple], pid: int, a: int, b: int) -> tuple[float, float]:
    """(path length ft, mean speed ft/s) for pid over frames a..b."""
    path = 0.0
    for i in range(a + 1, b + 1):
        p, q = frames[i - 1][3].get(pid), frames[i][3].get(pid)
        if p and q:
            path += math.hypot(q[0] - p[0], q[1] - p[1])
    dur = (b - a) / FPS
    return path, (path / dur if dur > 0 else 0.0)


def detect_actions(
    frames: list[tuple],
    hold: list[int | None],
    off: list[int | None],
    match: list[dict[int, int] | None],
    team_of: dict[int, int],
) -> list[dict]:
    """Screens, drives, cuts, handoffs as dicts with frame-index spans and confidence."""
    # ponytail: post-up detection deferred; PnR coverage classification is M3
    n = len(frames)
    if not n:
        return []
    hoop = attacked_hoop(frames)
    d_hoop = [
        {pid: math.hypot(x - hoop[0], y - hoop[1]) for pid, (x, y) in f[3].items()} for f in frames
    ]
    actions: list[dict] = []

    def emit(kind: str, a: int, b: int, p1: int, p2: int | None, conf: float) -> None:
        actions.append(
            {
                "type": kind,
                "start": a,
                "end": b,
                "p1": p1,
                "p2": p2,
                "confidence": round(max(0.0, min(1.0, conf)), 2),
            }
        )

    # --- screens: NETS triangle rule; "set" fraction feeds confidence ---
    hits: dict[tuple[int, int], list[int]] = {}
    set_flags: dict[tuple[int, int], list[bool]] = {}
    for i in range(n):
        h, pairing = hold[i], match[i]
        if h is None or pairing is None:
            continue
        hd = pairing.get(h)
        pos = frames[i][3]
        if hd is None or hd not in pos:
            continue
        hx_, hy_ = pos[h]
        dx, dy = pos[hd]
        if math.hypot(dx - hx_, dy - hy_) > SCREEN_DD1:
            continue
        for pid, (x, y) in pos.items():
            if pid == h or team_of[pid] != off[i]:
                continue
            if (
                math.hypot(x - hx_, y - hy_) <= SCREEN_DA
                and math.hypot(x - dx, y - dy) <= SCREEN_DD2
            ):
                key = (pid, h)
                hits.setdefault(key, []).append(i)
                set_flags.setdefault(key, []).append(_speed(frames, pid, i) <= SCREEN_SET_SPEED)
    for (screener, handler), idxs in hits.items():
        flags = set_flags[(screener, handler)]
        for a, b in _intervals(idxs, max_gap=5, min_len=SCREEN_MIN_FRAMES):
            in_iv = [flags[k] for k, i in enumerate(idxs) if a <= i <= b]
            set_frac = sum(in_iv) / len(in_iv) if in_iv else 0.0
            emit("screen", a, b, screener, handler, 0.6 + 0.4 * set_frac)

    # --- holder runs (shared by drives + handoffs) ---
    runs: list[tuple[int, int, int]] = []  # (pid, start, end)
    i = 0
    while i < n:
        j = i
        while j < n and hold[j] == hold[i]:
            j += 1
        if hold[i] is not None:
            runs.append((hold[i], i, j - 1))
        i = j

    # --- drives: carrier attacks the hoop with speed and directness ---
    for pid, a, b in runs:
        if b - a < DRIVE_MIN_FRAMES:
            continue
        ds = [d_hoop[i].get(pid) for i in range(a, b + 1)]
        if any(d is None for d in ds):
            continue
        e = ds.index(min(ds))
        if e == 0:
            continue
        s_rel = max(range(e), key=lambda k: ds[k])  # deepest start before the arrival
        if not (DRIVE_MIN_START <= ds[s_rel] <= DRIVE_MAX_START):
            continue
        decline = ds[s_rel] - ds[e]
        path, speed = _path_stats(frames, pid, a + s_rel, a + e)
        if (
            path <= 0
            or decline < DRIVE_MIN_DECLINE
            or decline / path < DRIVE_MIN_DP
            or speed < DRIVE_MIN_SPEED
        ):
            continue
        emit("drive", a + s_rel, a + e, pid, None, decline / path)

    # --- cuts: off-ball arrival at the rim, direct and fast ---
    seen_arrivals: set[tuple[int, int]] = set()
    for i in range(n):
        pos = frames[i][3]
        for pid in pos:
            if team_of[pid] != off[i] or hold[i] == pid:
                continue
            d = d_hoop[i].get(pid)
            if d is None or d > CUT_MAX_END:
                continue
            j0 = max(0, i - CUT_LOOKBACK)
            cands = [
                (d_hoop[j].get(pid, 0.0), j)
                for j in range(j0, i)
                if d_hoop[j].get(pid) is not None and d_hoop[j][pid] <= CUT_MAX_START
            ]
            if not cands:
                continue
            start_d, j = max(cands)
            if i - j < CUT_MIN_FRAMES:
                continue
            if any((pid, k) in seen_arrivals for k in range(j, i + 1)):
                continue  # already emitted a cut ending in this stretch
            held = sum(1 for k in range(j, i + 1) if hold[k] is not None and hold[k] != pid)
            if held / (i - j + 1) < CUT_HELD_FRAC:
                continue  # rebound scramble or transition, not a cut
            decline = start_d - d
            path, speed = _path_stats(frames, pid, j, i)
            if (
                path <= 0
                or decline < CUT_MIN_DECLINE
                or decline / path < CUT_MIN_DP
                or speed < CUT_MIN_SPEED
            ):
                continue
            if hold[j] == pid:  # was the ball-handler at the start: that's a drive
                continue
            seen_arrivals.update((pid, k) for k in range(j, i + 1))
            emit("cut", j, i, pid, None, decline / path)

    # --- handoffs: holder passes directly to an adjacent teammate ---
    for (p1, a1, e1), (p2, s2, e2) in zip(runs, runs[1:]):
        if p1 == p2 or team_of[p1] != team_of[p2] or s2 - e1 > HANDOFF_MAX_GAP:
            continue
        if e1 - a1 < HANDOFF_MIN_RUN or e2 - s2 < HANDOFF_MIN_RUN:
            continue
        a, b = frames[e1][3].get(p1), frames[e1][3].get(p2)
        c, d = frames[s2][3].get(p1), frames[s2][3].get(p2)
        if not (a and b and c and d):
            continue
        # close at release AND at receipt: a pass separates by receipt, a handoff doesn't
        if (
            math.hypot(a[0] - b[0], a[1] - b[1]) <= HANDOFF_MAX_DIST
            and math.hypot(c[0] - d[0], c[1] - d[1]) <= HANDOFF_MAX_DIST
        ):
            emit("handoff", e1, s2, p1, p2, 1.0 - (s2 - e1 - 1) / HANDOFF_MAX_GAP)
    return actions


def detect_game(game_id: str, parquet_dir: Path = PARQUET_DIR) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Run M2 detection over every event of a game. Returns (matchups_df, actions_df)."""
    m = load_moments(game_id, parquet_dir)
    match_rows: list[dict] = []
    action_rows: list[dict] = []
    for event_id in m["event_id"].unique().sort().to_list():
        frames, team_of, quarter = complete_frames(m, event_id)
        if len(frames) < WINDOW:
            continue
        hold = holders(frames)
        off = offense(hold, team_of)
        match = matchups(frames, off, team_of)
        for i, pairing in enumerate(match):
            if pairing is None:
                continue
            mi = frames[i][0]
            for a, d in pairing.items():
                match_rows.append(
                    {
                        "game_id": game_id,
                        "event_id": event_id,
                        "moment_idx": mi,
                        "off_player_id": a,
                        "def_player_id": d,
                    }
                )
        for act in detect_actions(frames, hold, off, match, team_of):
            action_rows.append(
                {
                    "game_id": game_id,
                    "event_id": event_id,
                    "quarter": quarter,
                    "type": act["type"],
                    "start_idx": frames[act["start"]][0],
                    "end_idx": frames[act["end"]][0],
                    "gc_start": frames[act["start"]][1],
                    "gc_end": frames[act["end"]][1],
                    "p1": act["p1"],
                    "p2": act["p2"],
                    "confidence": act["confidence"],
                }
            )
    matchups_df = pl.DataFrame(
        match_rows,
        schema={
            "game_id": pl.String,
            "event_id": pl.Int32,
            "moment_idx": pl.Int32,
            "off_player_id": pl.Int64,
            "def_player_id": pl.Int64,
        },
    )
    actions_df = pl.DataFrame(
        action_rows,
        schema={
            "game_id": pl.String,
            "event_id": pl.Int32,
            "quarter": pl.Int8,
            "type": pl.String,
            "start_idx": pl.Int32,
            "end_idx": pl.Int32,
            "gc_start": pl.Float64,
            "gc_end": pl.Float64,
            "p1": pl.Int64,
            "p2": pl.Int64,
            "confidence": pl.Float64,
        },
    )
    return matchups_df, actions_df


def detect_to_parquet(game_id: str, parquet_dir: Path = PARQUET_DIR) -> dict[str, int]:
    matchups_df, actions_df = detect_game(game_id, parquet_dir)
    for name, df in (("matchups", matchups_df), ("actions", actions_df)):
        d = parquet_dir / name
        d.mkdir(parents=True, exist_ok=True)
        df.write_parquet(d / f"{game_id}.parquet")
    return {"matchups": len(matchups_df), "actions": len(actions_df)}
