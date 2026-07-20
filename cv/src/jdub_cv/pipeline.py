"""Broadcast clip -> moments-schema Parquet that the main jdub pipeline accepts as-is.

    uv run python -m jdub_cv.pipeline <video> <calib.json> \
        --out ../data/parquet --game-id cv-okc-nyk [--overlay qc.mp4]

Then, from the repo root: `uv run jdub detect <game-id>`.

Stages: YOLO detect+track (persons + ball) -> per-frame homography (calib.py)
-> foot-point projection to court ft -> team clustering (teams.py) -> keep the
10 real players -> gap-fill -> resample to 25 Hz -> moments/players/games
Parquet. The overlay video is the eyeball-QC tool (same iron rule as studio).
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import polars as pl

from jdub_cv.calib import COURT_LENGTH, COURT_WIDTH, court_lines, make_calibrator, to_court
from jdub_cv.teams import assign_teams, torso_color

OUT_HZ = 25.0  # SportVU rate; events.py assumes it
PERSON_CONF = 0.35
BALL_CONF = 0.05  # ball candidates kept loose; picked by motion continuity below
BALL_MAX_FRAC = 0.08  # ball box no wider than this fraction of the frame
BALL_LOCK_CONF = 0.5  # min candidate score to (re)acquire the ball when lost
BALL_REACH = 0.02  # frame-widths: per-frame slack of the pixel continuity gate
BALL_SPEED = 0.6  # frame-widths/s: fastest believable image-space ball travel
MARGIN = 1.0  # ft of slack around the court when testing "on court"
MIN_TRACK_S = 1.0
BALL_GAP_S = 1.5
COLOR_EVERY = 5  # frames between torso-color samples
MERGE_GAP_S = 2.5  # sew tracklet fragments across holes up to this long
MERGE_SPEED = 18.0  # ft/s a player can plausibly cover inside the hole
MERGE_COLOR = 30.0  # max Lab distance between fragment torso colors
# ponytail: ball z is unknowable from one camera — emitted as 0.0, which always
# passes the holder z-gate; shot/air phases will look like floor possession


def _in_court(x: float, y: float) -> bool:
    return -MARGIN <= x <= COURT_LENGTH + MARGIN and -MARGIN <= y <= COURT_WIDTH + MARGIN


def _interp(series: dict[int, tuple[float, float]], max_gap: int) -> dict[int, tuple[float, float]]:
    """Fill index holes <= max_gap by linear interpolation (no extrapolation)."""
    if not series:
        return series
    out = dict(series)
    idxs = sorted(series)
    for a, b in zip(idxs, idxs[1:]):
        if 1 < b - a <= max_gap:
            (x0, y0), (x1, y1) = series[a], series[b]
            for i in range(a + 1, b):
                t = (i - a) / (b - a)
                out[i] = (x0 + t * (x1 - x0), y0 + t * (y1 - y0))
    return out


def merge_tracks(
    obs: dict[int, dict[int, tuple[float, float]]],
    colors: dict[int, list[np.ndarray]],
    fps: float,
    feats: dict[int, list[np.ndarray]] | None = None,
) -> tuple[
    dict[int, dict[int, tuple[float, float]]],
    dict[int, list[np.ndarray]],
    dict[int, list[np.ndarray]] | None,
]:
    """Sew tracker-ID fragments of the same player back together.

    A fragment may continue an earlier one when it starts after the other ends
    (never merge concurrent tracks — those are different people), the hole is
    short, the court-space jump is coverable at MERGE_SPEED, and torso colors
    agree. Greedy, earliest-start-first; cheapest candidate wins.
    """
    med = {t: (np.median(np.array(cs), axis=0) if cs else None) for t, cs in colors.items()}
    roots: list[int] = []  # merged track heads, keyed by their first fragment id
    fts: dict[int, list[np.ndarray]] = {}
    frames: dict[int, dict[int, tuple[float, float]]] = {}
    cols: dict[int, list[np.ndarray]] = {}
    tail: dict[int, tuple[int, float, float]] = {}  # root -> (end_frame, x, y)
    for t in sorted(obs, key=lambda t: min(obs[t])):
        fs = obs[t]
        s = min(fs)
        sx, sy = fs[s]
        best, best_cost = None, None
        for r in roots:
            end, ex, ey = tail[r]
            gap = s - end
            if gap <= 0 or gap > MERGE_GAP_S * fps:
                continue
            dist = np.hypot(sx - ex, sy - ey)
            if dist > 2.0 + MERGE_SPEED * gap / fps:
                continue
            if med.get(t) is not None and med.get(r) is not None:
                if np.linalg.norm(med[t] - med[r]) > MERGE_COLOR:
                    continue
            cost = dist + 5.0 * gap / fps
            if best_cost is None or cost < best_cost:
                best, best_cost = r, cost
        if best is None:
            roots.append(t)
            frames[t] = dict(fs)
            cols[t] = list(colors.get(t, []))
            if feats is not None:
                fts[t] = list(feats.get(t, []))
        else:
            frames[best].update(fs)
            cols[best].extend(colors.get(t, []))
            if feats is not None:
                fts[best].extend(feats.get(t, []))
            if med.get(best) is None:
                med[best] = med.get(t)
            t = best
        e = max(frames[t])
        tail[t] = (e, *frames[t][e])
    return frames, cols, (fts if feats is not None else None)


def _pick_ball(
    cands: list[tuple[float, float, float]],
    track: dict[int, tuple[float, float]],
    fi: int,
    fps: float,
    width: int,
) -> tuple[float, float] | None:
    """Pick this frame's ball from (score, x_px, y_px) candidates.

    Continuity is gated in image space — court space is corrupted by H jitter
    and by the floor projection of an airborne ball. While locked, take the
    nearest reachable candidate and never teleport; once the lock expires,
    re-acquire only on a confident candidate (one bad confident pick
    self-recovers after BALL_GAP_S).
    """
    if not cands:
        return None
    if track:
        last = max(track)
        dt = (fi - last) / fps
        if dt <= BALL_GAP_S:
            lx, ly = track[last]
            reach = (BALL_REACH + BALL_SPEED * dt) * width
            near = [c for c in cands if np.hypot(c[1] - lx, c[2] - ly) <= reach]
            if near:
                c = min(near, key=lambda c: np.hypot(c[1] - lx, c[2] - ly))
                return c[1], c[2]
            return None  # locked but nothing reachable: stay lost this frame
    c = max(cands)
    return (c[1], c[2]) if c[0] >= BALL_LOCK_CONF else None


def pack_slots(
    tracks: dict[int, dict[int, tuple[float, float]]], n_slots: int, fps: float
) -> list[tuple[dict[int, tuple[float, float]], list[int]]]:
    """Pack one team's tracklets into at most n_slots player timelines.

    A team has exactly five players, so every fragment must continue one of
    five timelines. Seeded longest-first: the most reliable tracks anchor the
    five slots, shorter fragments attach to a slot's free end (before its
    first frame or after its last) when reachable at MERGE_SPEED — no gap cap,
    a player can be off-frame for seconds. Else open a new slot, else drop
    (a sixth concurrent teammate is a ref/crowd leak). Earliest-start-first
    instead lets short early stubs squat in slots and block the long tracks
    (measured: okc-nyk 72%->95% complete frames when switched to longest).
    Returns (frames, member tracklet ids) per slot.
    """
    slots: list[dict[int, tuple[float, float]]] = []
    members: list[list[int]] = []
    for t in sorted(tracks, key=lambda t: -len(tracks[t])):
        fs = tracks[t]
        s, e = min(fs), max(fs)
        best, best_cost = None, None
        for i, sl in enumerate(slots):
            lo, hi = min(sl), max(sl)
            if e < lo:  # fragment ends before the slot starts: prepend
                gap, (sx, sy), (ex, ey) = (lo - e) / fps, fs[e], sl[lo]
            elif s > hi:  # fragment starts after the slot ends: append
                gap, (sx, sy), (ex, ey) = (s - hi) / fps, fs[s], sl[hi]
            else:
                continue  # spans overlap -> different player (interior gaps interp-filled)
            dist = float(np.hypot(sx - ex, sy - ey))
            if dist > 3.0 + MERGE_SPEED * gap:
                continue
            cost = dist + 5.0 * gap
            if best_cost is None or cost < best_cost:
                best, best_cost = i, cost
        if best is not None:
            slots[best].update(fs)
            members[best].append(t)
        elif len(slots) < n_slots:
            slots.append(dict(fs))
            members.append([t])
    return list(zip(slots, members))


def _team_colors(
    picked: dict[int, int], colors: dict[int, list[np.ndarray]]
) -> dict[int, tuple[int, int, int]]:
    """Each side's dominant jersey color (median Lab of its tracks -> BGR)."""
    out: dict[int, tuple[int, int, int]] = {}
    for side in (1, 2):
        labs = [c for t, s in picked.items() if s == side for c in colors.get(t, [])]
        if labs:
            lab = np.median(np.array(labs), axis=0).astype(np.uint8).reshape(1, 1, 3)
            b, g, r = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)[0, 0]
            out[side] = (int(b), int(g), int(r))
        else:
            out[side] = (200, 200, 200)
    return out


def run(
    video: Path,
    calib: Path,
    out_dir: Path,
    game_id: str,
    model_name: str = "yolo11m.pt",
    overlay: Path | None = None,
    limit: int | None = None,
    imgsz: int = 1280,
    tracker: str = "botsort.yaml",
    ball_weights: Path | None = None,
    teams: str = "color",
) -> dict[str, int]:
    from ultralytics import YOLO

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    calibrator = make_calibrator(calib)
    model = YOLO(model_name)
    if ball_weights is None:
        default = Path(__file__).resolve().parents[2] / "weights" / "wasb_basketball_best.pth.tar"
        ball_weights = default if default.exists() else None
    wasb = None
    if ball_weights is not None:
        from jdub_cv.ball import WasbBallDetector

        wasb = WasbBallDetector(ball_weights)
    embedder = None
    if teams == "siglip":
        from jdub_cv.teams import SiglipEmbedder

        embedder = SiglipEmbedder()
    feats: dict[int, list[np.ndarray]] = defaultdict(list)
    obs: dict[int, dict[int, tuple[float, float]]] = defaultdict(dict)  # tid -> frame -> ft
    colors: dict[int, list[np.ndarray]] = defaultdict(list)
    ball_px: dict[int, tuple[float, float]] = {}  # frame -> image px
    hs: list[np.ndarray] = []

    cap = cv2.VideoCapture(str(video))
    fi = -1
    while True:
        fi += 1
        ok, frame = cap.read()
        if not ok or (limit is not None and fi >= limit):
            break
        if calibrator.flip:
            frame = cv2.flip(frame, 1)
        r = model.track(
            frame,
            persist=True,
            classes=[0, 32],
            conf=BALL_CONF,
            imgsz=imgsz,
            tracker=tracker,
            verbose=False,
        )[0]
        h = calibrator.update(frame).copy()
        hs.append(h)
        has_boxes = r.boxes is not None and len(r.boxes)
        if has_boxes:
            cls = r.boxes.cls.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            xyxy = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy() if r.boxes.id is not None else np.full(len(cls), -1)
            persons = (cls == 0) & (conf >= PERSON_CONF) & (ids >= 0)
            if persons.any():
                feet = np.stack(
                    [(xyxy[persons, 0] + xyxy[persons, 2]) / 2, xyxy[persons, 3]], axis=1
                )
                court = to_court(h, feet)
                for tid, box, (cx, cy) in zip(ids[persons], xyxy[persons], court):
                    if not _in_court(cx, cy):
                        continue
                    obs[int(tid)][fi] = (float(cx), float(cy))
                    if fi % COLOR_EVERY == 0:
                        c = torso_color(frame, box)
                        if c is not None:
                            colors[int(tid)].append(c)
                        if embedder is not None:
                            e = embedder.embed(frame, box)
                            if e is not None:
                                feats[int(tid)].append(e)
        # ball candidates from both detectors; picked by the pixel-space gate
        cands: list[tuple[float, float, float]] = []  # (score, x_px, y_px)
        if wasb is not None:
            det = wasb.detect(frame)
            if det:
                cands.append((det[2], det[0], det[1]))
        if has_boxes:
            balls = (cls == 32) & ((xyxy[:, 2] - xyxy[:, 0]) <= BALL_MAX_FRAC * frame.shape[1])
            for cf, (x1, y1, x2, y2) in zip(conf[balls], xyxy[balls]):
                cands.append((float(cf), float((x1 + x2) / 2), float((y1 + y2) / 2)))
        if cands:
            court = to_court(h, np.array([[c[1], c[2]] for c in cands]))
            cands = [c for c, (cx, cy) in zip(cands, court) if _in_court(cx, cy)]
        pick = _pick_ball(cands, ball_px, fi, fps, frame.shape[1])
        if pick is not None:
            ball_px[fi] = pick

    n_frames = len(hs)
    n_fragments = len(obs)
    obs, colors, feats = merge_tracks(obs, colors, fps, feats if embedder else None)
    # the 10 real players: every long-enough non-referee fragment belongs to
    # one of each team's 5 slot timelines (whole-track top-5 selection leaves
    # per-frame holes wherever the chosen track has one)
    alive = {t: fs for t, fs in obs.items() if len(fs) >= MIN_TRACK_S * fps}
    clusterable = feats if embedder else colors
    team = assign_teams({t: clusterable[t] for t in alive})
    picked: dict[int, int] = {}  # slot id -> team
    tracks: dict[int, dict[int, tuple[float, float]]] = {}
    slot_colors: dict[int, list[np.ndarray]] = {}
    for side in (1, 2):
        packed = pack_slots({t: alive[t] for t in alive if team.get(t) == side}, 5, fps)
        for i, (fs, tids) in enumerate(packed):
            sid = side * 10 + i
            picked[sid] = side
            tracks[sid] = _interp(fs, int(MERGE_GAP_S * fps))
            slot_colors[sid] = [c for t in tids for c in colors.get(t, [])]
    # gap-fill the ball in pixel space, then project through each frame's own H
    ball_px = _interp(ball_px, int(BALL_GAP_S * fps))
    ball = {f: tuple(map(float, to_court(hs[f], np.array([p]))[0])) for f, p in ball_px.items()}
    team_bgr = _team_colors(picked, slot_colors)

    # resample to 25 Hz and emit moments long-format
    rows: list[dict] = []
    n_out = int(n_frames / fps * OUT_HZ)
    for k in range(n_out):
        fi = min(n_frames - 1, round(k / OUT_HZ * fps))
        gc = 720.0 - k / OUT_HZ
        common = {
            "game_id": game_id,
            "event_id": 1,
            "moment_idx": k,
            "quarter": 1,
            "game_clock": gc,
            "shot_clock": None,
        }
        if fi in ball:
            rows.append(
                common
                | {
                    "entity": "ball",
                    "team_id": -1,
                    "player_id": -1,
                    "x": ball[fi][0],
                    "y": ball[fi][1],
                    "z": 0.0,
                }
            )
        for t, side in picked.items():
            if fi in tracks[t]:
                rows.append(
                    common
                    | {
                        "entity": "player",
                        "team_id": side,
                        "player_id": t,
                        "x": tracks[t][fi][0],
                        "y": tracks[t][fi][1],
                        "z": 0.0,
                    }
                )
    moments = pl.DataFrame(
        rows,
        schema={
            "game_id": pl.String,
            "event_id": pl.Int32,
            "moment_idx": pl.Int32,
            "quarter": pl.Int8,
            "game_clock": pl.Float64,
            "shot_clock": pl.Float64,
            "entity": pl.String,
            "team_id": pl.Int64,
            "player_id": pl.Int64,
            "x": pl.Float64,
            "y": pl.Float64,
            "z": pl.Float64,
        },
    )
    players = pl.DataFrame(
        [
            {
                "game_id": game_id,
                "team_id": side,
                "player_id": t,
                "firstname": "Track",
                "lastname": f"T{t}",
                "jersey": "",
                "position": "",
            }
            for t, side in picked.items()
        ]
    )
    # sync the matchup name from the clip filename (e.g. "okc-nyk.mp4" -> OKC @ NYK);
    # which cluster is which team is unknown until jersey-number identity lands
    parts = re.split(r"[-_]", video.stem)
    visitor_abbr = parts[0].upper() if len(parts) >= 2 else "CVA"
    home_abbr = re.sub(r"\d+$", "", parts[1]).upper() if len(parts) >= 2 else "CVB"
    games = pl.DataFrame(
        {
            "game_id": [game_id],
            "date": ["CV"],
            "home_team_id": [1],
            "home_team": [home_abbr],
            "home_abbr": [home_abbr],
            "visitor_team_id": [2],
            "visitor_team": [visitor_abbr],
            "visitor_abbr": [visitor_abbr],
        }
    )
    for name, df in (("moments", moments), ("games", games), ("players", players)):
        d = out_dir / name
        d.mkdir(parents=True, exist_ok=True)
        df.write_parquet(d / f"{game_id}.parquet")

    if overlay is not None:
        _write_overlay(video, overlay, hs, tracks, picked, ball, fps, team_bgr, calibrator.flip)

    per_frame = moments.group_by("moment_idx").agg(
        ((pl.col("entity") == "player").sum().alias("np")),
        (pl.col("entity") == "ball").sum().alias("nb"),
    )
    complete = per_frame.filter((pl.col("np") == 10) & (pl.col("nb") == 1)).height
    return {
        "frames": n_frames,
        "fragments": n_fragments,
        "tracks_merged": len(obs),
        "tracks_kept": len(picked),
        "moments_rows": len(moments),
        "moments_25hz": n_out,
        "players_per_frame_median": float(per_frame["np"].median() or 0),
        "complete_frames": complete,
        "ball_frames": len(ball),
    }


def _write_overlay(
    video: Path,
    out: Path,
    hs: list[np.ndarray],
    tracks: dict[int, dict[int, tuple[float, float]]],
    picked: dict[int, int],
    ball: dict[int, tuple[float, float]],
    fps: float,
    team_color: dict[int, tuple[int, int, int]] | None = None,
    flip: bool = False,
) -> None:
    """QC video: court lines + kept players (team-colored) + ball, all re-projected."""
    cap = cv2.VideoCapture(str(video))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if team_color is None:
        team_color = {1: (60, 130, 255), 2: (255, 160, 40)}  # BGR fallback
    for fi in range(len(hs)):
        ok, frame = cap.read()
        if not ok:
            break
        if flip:
            frame = cv2.flip(frame, 1)
        hinv = np.linalg.inv(hs[fi])
        for line in court_lines():
            px = cv2.perspectiveTransform(np.float32(line).reshape(-1, 1, 2), hinv).reshape(-1, 2)
            if np.isfinite(px).all():
                cv2.polylines(frame, [px.astype(np.int32)], False, (255, 255, 255), 1)
        for t, side in picked.items():
            if fi in tracks[t]:
                ((px, py),) = cv2.perspectiveTransform(
                    np.float32([tracks[t][fi]]).reshape(-1, 1, 2), hinv
                ).reshape(-1, 2)
                cv2.circle(frame, (int(px), int(py)), 8, team_color[side], -1)
                cv2.circle(frame, (int(px), int(py)), 8, (255, 255, 255), 1)
                cv2.putText(
                    frame,
                    str(t),
                    (int(px) + 8, int(py)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    team_color[side],
                    1,
                )
        if fi in ball:
            ((px, py),) = cv2.perspectiveTransform(
                np.float32([ball[fi]]).reshape(-1, 1, 2), hinv
            ).reshape(-1, 2)
            cv2.circle(frame, (int(px), int(py)), 5, (0, 255, 255), -1)
        vw.write(frame)
    cap.release()
    vw.release()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", type=Path)
    ap.add_argument("calib", type=Path)
    ap.add_argument("--out", type=Path, default=Path("../data/parquet"))
    ap.add_argument("--game-id", default=None)
    ap.add_argument("--model", default="yolo11m.pt")
    ap.add_argument("--overlay", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--tracker", default="botsort.yaml", help="botsort.yaml | bytetrack.yaml")
    ap.add_argument("--teams", default="color", choices=["color", "siglip"])
    ap.add_argument(
        "--ball-weights",
        type=Path,
        default=None,
        help="WASB checkpoint; default weights/wasb_basketball_best.pth.tar if present",
    )
    args = ap.parse_args()
    game_id = args.game_id or f"cv-{args.video.stem}"
    print(
        run(
            args.video,
            args.calib,
            args.out,
            game_id,
            args.model,
            args.overlay,
            args.limit,
            args.imgsz,
            args.tracker,
            args.ball_weights,
            args.teams,
        )
    )


if __name__ == "__main__":
    main()
