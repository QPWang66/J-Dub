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
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import polars as pl

from jdub_cv.calib import COURT_LENGTH, COURT_WIDTH, Calibrator, court_lines, to_court
from jdub_cv.teams import assign_teams, torso_color

OUT_HZ = 25.0  # SportVU rate; events.py assumes it
PERSON_CONF = 0.35
BALL_CONF = 0.05  # ball candidates kept loose; picked by motion continuity below
BALL_MAX_FRAC = 0.08  # ball box no wider than this fraction of the frame
BALL_JUMP = 40.0  # ft/s: fastest believable ball travel when gating candidates
MARGIN = 1.0  # ft of slack around the court when testing "on court"
IN_COURT_FRAC = 0.6
MIN_TRACK_S = 1.0
PLAYER_GAP_S = 0.6  # interpolate track holes up to this long
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
) -> tuple[dict[int, dict[int, tuple[float, float]]], dict[int, list[np.ndarray]]]:
    """Sew tracker-ID fragments of the same player back together.

    A fragment may continue an earlier one when it starts after the other ends
    (never merge concurrent tracks — those are different people), the hole is
    short, the court-space jump is coverable at MERGE_SPEED, and torso colors
    agree. Greedy, earliest-start-first; cheapest candidate wins.
    """
    med = {t: (np.median(np.array(cs), axis=0) if cs else None) for t, cs in colors.items()}
    roots: list[int] = []  # merged track heads, keyed by their first fragment id
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
        else:
            frames[best].update(fs)
            cols[best].extend(colors.get(t, []))
            if med.get(best) is None:
                med[best] = med.get(t)
            t = best
        e = max(frames[t])
        tail[t] = (e, *frames[t][e])
    return frames, cols


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
) -> dict[str, int]:
    from ultralytics import YOLO

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    calibrator = Calibrator(calib)
    model = YOLO(model_name)
    if ball_weights is None:
        default = Path(__file__).resolve().parents[2] / "weights" / "wasb_basketball_best.pth.tar"
        ball_weights = default if default.exists() else None
    wasb = None
    if ball_weights is not None:
        from jdub_cv.ball import WasbBallDetector

        wasb = WasbBallDetector(ball_weights)
    obs: dict[int, dict[int, tuple[float, float]]] = defaultdict(dict)  # tid -> frame -> ft
    colors: dict[int, list[np.ndarray]] = defaultdict(list)
    ball: dict[int, tuple[float, float]] = {}
    hs: list[np.ndarray] = []

    for fi, r in enumerate(
        model.track(
            source=str(video),
            stream=True,
            classes=[0, 32],
            conf=BALL_CONF,
            imgsz=imgsz,
            tracker=tracker,
            verbose=False,
        )
    ):
        if limit is not None and fi >= limit:
            break
        h = calibrator.update(r.orig_img).copy()
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
                        c = torso_color(r.orig_img, box)
                        if c is not None:
                            colors[int(tid)].append(c)
        # ball candidates from both detectors; the motion-continuity gate picks
        pts: list[list[float]] = []
        scores: list[float] = []
        if wasb is not None:
            det = wasb.detect(r.orig_img)
            if det:
                pts.append([det[0], det[1]])
                scores.append(det[2])
        if has_boxes:
            balls = (cls == 32) & ((xyxy[:, 2] - xyxy[:, 0]) <= BALL_MAX_FRAC * r.orig_img.shape[1])
            if balls.any():
                pts.extend(
                    np.stack(
                        [
                            (xyxy[balls, 0] + xyxy[balls, 2]) / 2,
                            (xyxy[balls, 1] + xyxy[balls, 3]) / 2,
                        ],
                        axis=1,
                    ).tolist()
                )
                scores.extend(float(c) for c in conf[balls])
        raw = np.array(pts) if pts else np.empty((0, 2))
        if len(raw):
            cands = [
                (float(cf), float(cx), float(cy))
                for cf, (cx, cy) in zip(scores, to_court(h, raw))
                if _in_court(cx, cy)
            ]
            if cands:
                if ball:
                    last_fi = max(ball)
                    lx, ly = ball[last_fi]
                    dt = (fi - last_fi) / fps
                    reach = 3.0 + BALL_JUMP * dt
                    near = [c for c in cands if np.hypot(c[1] - lx, c[2] - ly) <= reach]
                    pick = (
                        min(near, key=lambda c: np.hypot(c[1] - lx, c[2] - ly))
                        if near and dt <= BALL_GAP_S
                        else max(cands, key=lambda c: c[0])
                    )
                else:
                    pick = max(cands, key=lambda c: c[0])
                ball[fi] = (pick[1], pick[2])

    n_frames = len(hs)
    n_fragments = len(obs)
    obs, colors = merge_tracks(obs, colors, fps)
    # the 10 real players: on-court, long-lived, non-referee, max 5 per team
    alive = {
        t: fs
        for t, fs in obs.items()
        if len(fs) >= MIN_TRACK_S * fps and len(fs) / (max(fs) - min(fs) + 1) >= IN_COURT_FRAC
    }
    team = assign_teams({t: colors[t] for t in alive})
    picked: dict[int, int] = {}  # tid -> team
    for side in (1, 2):
        tids = sorted((t for t in alive if team.get(t) == side), key=lambda t: -len(alive[t]))[:5]
        picked.update({t: side for t in tids})
    tracks = {t: _interp(alive[t], int(PLAYER_GAP_S * fps)) for t in picked}
    ball = _interp(ball, int(BALL_GAP_S * fps))

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
    games = pl.DataFrame(
        {
            "game_id": [game_id],
            "date": [""],
            "home_team_id": [1],
            "home_team": ["CV Team A"],
            "home_abbr": ["CVA"],
            "visitor_team_id": [2],
            "visitor_team": ["CV Team B"],
            "visitor_abbr": ["CVB"],
        }
    )
    for name, df in (("moments", moments), ("games", games), ("players", players)):
        d = out_dir / name
        d.mkdir(parents=True, exist_ok=True)
        df.write_parquet(d / f"{game_id}.parquet")

    if overlay is not None:
        _write_overlay(video, overlay, hs, tracks, picked, ball, fps)

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
) -> None:
    """QC video: court lines + kept players (team-colored) + ball, all re-projected."""
    cap = cv2.VideoCapture(str(video))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    team_color = {1: (60, 130, 255), 2: (255, 160, 40)}  # BGR
    for fi in range(len(hs)):
        ok, frame = cap.read()
        if not ok:
            break
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
                cv2.circle(frame, (int(px), int(py)), 7, team_color[side], 2)
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
        )
    )


if __name__ == "__main__":
    main()
