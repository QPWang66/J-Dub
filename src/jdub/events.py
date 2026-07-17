"""M2: ball holder, offense inference, matchup assignment, atomic action detection.

All detection runs per SportVU event on "complete" frames (ball + exactly 10
players). Coordinates are raw court feet; the attacked hoop is inferred per
event from where the ball lives (same heuristic as normalize_offense_direction).
"""

from __future__ import annotations

import math
from itertools import permutations
from pathlib import Path

import polars as pl

from jdub.data import PARQUET_DIR, load_moments

FPS = 25
HOLD_DIST = 3.0  # ft: ball within this of a player = candidate holder
HOLD_MIN_FRAMES = 5  # holder must be sustained ~0.2s
WINDOW = 25  # matchup window (1s)
STEP = 12  # matchup stride (~0.5s)
SCREEN_MAX_SPEED = 2.0  # ft/s: screener is set
SCREEN_DEF_DIST = 6.0  # ft: screener near handler's defender
SCREEN_HANDLER_DIST = 12.0
SCREEN_MIN_FRAMES = 8  # ~0.3s
CUT_MIN_DECLINE = 10.0  # ft toward hoop over ~1s
DRIVE_MIN_DECLINE = 8.0
DRIVE_START_DIST = 15.0
DRIVE_MAX_START = 30.0  # ft: beyond this it's a transition push, not a drive
CUT_MAX_START = 35.0
HANDOFF_MAX_GAP = 10  # frames between holders
HANDOFF_MAX_DIST = 5.0  # ft between giver and receiver at exchange

# frame tuple: (moment_idx, game_clock, (bx, by), {pid: (x, y)})


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
                ball = (r["x"], r["y"])
            else:
                pos[r["player_id"]] = (r["x"], r["y"])
                team_of[r["player_id"]] = r["team_id"]
        if ball is not None and len(pos) == 10:
            frames.append((mi, gc, ball, pos))
    return frames, team_of, quarter


def holders(frames: list[tuple]) -> list[int | None]:
    """Per-frame ball holder: nearest player within HOLD_DIST, sustained >= HOLD_MIN_FRAMES."""
    raw: list[int | None] = []
    for _, _, ball, pos in frames:
        best, bd = None, HOLD_DIST
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


def matchups(
    frames: list[tuple], off: list[int | None], team_of: dict[int, int]
) -> list[dict[int, int] | None]:
    """Per-frame {attacker_pid: defender_pid} via optimal assignment on sliding windows.

    Cost = summed distance over the window; 5x5 optimum by brute force over 120
    permutations (exact, no dependency).
    """
    n = len(frames)
    assign: list[dict[int, int] | None] = [None] * n
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
            pos = frames[i][3]
            if any(p not in pos for p in pids):  # substitution mid-window
                ok = False
                break
            for di, dp in enumerate(defenders):
                dx, dy = pos[dp]
                for ai, ap in enumerate(attackers):
                    ax, ay = pos[ap]
                    cost[di][ai] += math.hypot(dx - ax, dy - ay)
        if not ok:
            continue
        best = min(permutations(range(5)), key=lambda p: sum(cost[d][p[d]] for d in range(5)))
        pairing = {attackers[best[d]]: defenders[d] for d in range(5)}
        for i in range(s, min(s + STEP, n)):
            assign[i] = pairing
    return assign


def attacked_hoop(frames: list[tuple]) -> tuple[float, float]:
    """Hoop the offense attacks: the one on the half where the ball lives."""
    mean_x = sum(f[2][0] for f in frames) / len(frames)
    return (5.25, 25.0) if mean_x <= 47.0 else (88.75, 25.0)


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


def detect_actions(
    frames: list[tuple],
    hold: list[int | None],
    off: list[int | None],
    match: list[dict[int, int] | None],
    team_of: dict[int, int],
) -> list[dict]:
    """Screens, drives, cuts, handoffs as dicts with frame-index spans and confidence."""
    # ponytail: post-up detection deferred; PnR classification (accept/reject,
    # coverage) is M3 on top of these primitives
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
                "confidence": round(min(1.0, conf), 2),
            }
        )

    # --- screens: off-ball attacker set (slow) next to the handler's defender ---
    hits: dict[tuple[int, int], list[int]] = {}
    for i in range(n):
        h, pairing = hold[i], match[i]
        if h is None or pairing is None:
            continue
        hd = pairing.get(h)
        pos = frames[i][3]
        if hd is None or hd not in pos:
            continue
        hx, hy = pos[h]
        dx, dy = pos[hd]
        for pid, (x, y) in pos.items():
            if pid == h or team_of[pid] != off[i]:
                continue
            if (
                _speed(frames, pid, i) <= SCREEN_MAX_SPEED
                and math.hypot(x - dx, y - dy) <= SCREEN_DEF_DIST
                and math.hypot(x - hx, y - hy) <= SCREEN_HANDLER_DIST
            ):
                hits.setdefault((pid, h), []).append(i)
    for (screener, handler), idxs in hits.items():
        for a, b in _intervals(idxs, max_gap=5, min_len=SCREEN_MIN_FRAMES):
            emit("screen", a, b, screener, handler, (b - a + 1) / 25)

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

    # --- drives: holder closes >= DRIVE_MIN_DECLINE ft toward the hoop ---
    for pid, a, b in runs:
        if b - a < 12:
            continue
        ds = [d_hoop[i][pid] for i in range(a, b + 1)]
        i_min = ds.index(min(ds))
        if (
            DRIVE_START_DIST <= ds[0] <= DRIVE_MAX_START
            and ds[0] - ds[i_min] >= DRIVE_MIN_DECLINE
            and i_min > 0
        ):
            emit("drive", a, a + i_min, pid, None, (ds[0] - ds[i_min]) / 15)

    # --- cuts: off-ball attacker closes >= CUT_MIN_DECLINE ft toward the hoop in ~1s ---
    cut_hits: dict[int, list[int]] = {}
    for s in range(0, n - WINDOW, 5):
        e = s + WINDOW
        for pid in frames[s][3]:
            if team_of[pid] != off[s] or pid not in frames[e][3]:
                continue
            if any(hold[i] == pid for i in range(s, e)):
                continue
            if (
                d_hoop[s][pid] <= CUT_MAX_START
                and d_hoop[s][pid] - d_hoop[e][pid] >= CUT_MIN_DECLINE
            ):
                cut_hits.setdefault(pid, []).extend(range(s, e))
    for pid, idxs in cut_hits.items():
        for a, b in _intervals(sorted(set(idxs)), max_gap=5, min_len=WINDOW):
            emit("cut", a, b, pid, None, (d_hoop[a][pid] - d_hoop[b].get(pid, d_hoop[a][pid])) / 15)

    # --- handoffs: holder passes directly to an adjacent teammate ---
    for (p1, _, e1), (p2, s2, _) in zip(runs, runs[1:]):
        if p1 == p2 or team_of[p1] != team_of[p2] or s2 - e1 > HANDOFF_MAX_GAP:
            continue
        a, b = frames[e1][3].get(p1), frames[e1][3].get(p2)
        if a and b and math.hypot(a[0] - b[0], a[1] - b[1]) <= HANDOFF_MAX_DIST:
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
