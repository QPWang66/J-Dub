"""Whole-clip calibration QA — every frame, not spot checks.

    uv run python -m jdub_cv.stability <video> <calib.json> [--dump-worst DIR]

Reports per-frame paint IoU and corner velocity across the entire clip and
dumps the worst frames as overlay images for eyeballing. A clip passes when
IoU stays high and corners move smoothly (no snap teleports).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from jdub_cv.calib import Calibrator, court_lines


def _overlay(frame: np.ndarray, h: np.ndarray) -> np.ndarray:
    vis = frame.copy()
    hinv = np.linalg.inv(h)
    for line in court_lines():
        px = cv2.perspectiveTransform(np.float32(line).reshape(-1, 1, 2), hinv).reshape(-1, 2)
        if np.isfinite(px).all():
            cv2.polylines(vis, [px.astype(np.int32)], False, (0, 0, 255), 2)
    return vis


def run(video: Path, calib: Path, dump_worst: Path | None = None) -> dict:
    cap = cv2.VideoCapture(str(video))
    c = Calibrator(calib)
    ious: list[float] = []
    vels: list[float] = []
    frames_kept: list[tuple[int, float, np.ndarray, np.ndarray]] = []
    prev_pts = None
    fi = -1
    while True:
        fi += 1
        ok, frame = cap.read()
        if not ok:
            break
        if c.flip:
            frame = cv2.flip(frame, 1)
        h = c.update(frame).copy()
        pts = c._paint_pts()
        ious.append(c.iou)
        if prev_pts is not None:
            vels.append(float(np.abs(pts - prev_pts).max()))
        prev_pts = pts
        frames_kept.append((fi, c.iou, h, frame if dump_worst else None))
    cap.release()

    iou = np.array(ious)
    vel = np.array(vels) if vels else np.zeros(1)
    report = {
        "frames": len(iou),
        "iou_p50": round(float(np.median(iou)), 3),
        "iou_p10": round(float(np.percentile(iou, 10)), 3),
        "frames_low_iou": int((iou < 0.4).sum()),
        "corner_vel_p95_px": round(float(np.percentile(vel, 95)), 1),
        "corner_vel_max_px": round(float(vel.max()), 1),
    }
    if dump_worst is not None:
        dump_worst.mkdir(parents=True, exist_ok=True)
        worst = sorted(frames_kept, key=lambda r: r[1])[:5]
        jumps = np.argsort(vel)[-3:] + 1
        picks = {r[0]: r for r in worst} | {int(j): frames_kept[int(j)] for j in jumps}
        for fi, iou_v, h, frame in picks.values():
            cv2.imwrite(str(dump_worst / f"worst_f{fi}_iou{iou_v:.2f}.png"), _overlay(frame, h))
        report["dumped"] = sorted(picks)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", type=Path)
    ap.add_argument("calib", type=Path)
    ap.add_argument("--dump-worst", type=Path, default=None)
    args = ap.parse_args()
    print(run(args.video, args.calib, args.dump_worst))


if __name__ == "__main__":
    main()
