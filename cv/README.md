# jdub-cv

Stage 1 of the jdub pipeline: **broadcast/gym video → moments-schema trajectories**
(C1 baseline of `../docs/cv-plan.md` — off-the-shelf models, no fine-tuning yet).

Separate uv project on purpose: torch/ultralytics stay out of the main jdub
package. The only interface to the main pipeline is Parquet in the moments
schema — after running this, `uv run jdub detect <game-id>` and
`uv run jdub studio` in the repo root work on the output as if it were SportVU.

## Run

```bash
cd cv
uv sync
uv run python -m jdub_cv.pipeline <video.mp4> calib/<video>.json \
    --out ../data/parquet --game-id cv-myclip --overlay qc.mp4
cd .. && uv run jdub detect cv-myclip
```

## How it works

| Stage | Method | Ceiling / upgrade |
|-------|--------|-------------------|
| Players: detect + track | YOLO11m (COCO pretrained) + BoT-SORT (`--tracker`), then court-space tracklet sewing (`merge_tracks`: temporal hole + reachable jump + torso-color agreement) | fine-tune detector on basketball broadcast data; PRTreID-grade ReID (C2) |
| Ball | [WASB](https://github.com/nttcom/WASB-SBDT) HRNet heatmap model (BMVC'23, trained on NBA broadcast, MIT) when `weights/wasb_basketball_best.pth.tar` exists — `uvx gdown 1nfECuSyJvPUmz3njZCdFERSQQbERt8FU -O weights/wasb_basketball_best.pth.tar`; else YOLO sports-ball. Either way candidates pass a motion-continuity gate (nearest reachable to last ball) | fine-tune WASB on own footage |
| Homography | manual anchor points on frame 0 (`calib/*.json`, `"static": true` for fixed cameras) + LK optical-flow propagation for panning cameras | trained court-keypoint model → absolute H per frame (C2) |
| Team split | per-track median torso color (Lab) + k-means(3), two biggest clusters = teams, third = refs | jersey/number ReID (PRTreID-style) |
| Output | in-court filter → top-10 tracks → gap interpolation → resample 25 Hz → moments/players/games Parquet | — |

The `--overlay` mp4 re-projects court lines + kept tracks onto the video —
that's the eyeball-QC tool (same iron rule as jdub studio: acceptance is
visual, not test-only).

## Calibration files

`calib/<name>.json` holds `image` (px) ↔ `court` (ft) correspondences for
frame 0. ≥4 points, no 3 collinear, spread as wide as possible. Conventions
are noted per file. Writing one by hand off a broadcast frame takes several
iterations of project-and-compare — which is exactly the argument for the C2
court-keypoint model.

## Known limits (deliberate, documented)

- Ball `z` is emitted as 0.0 (single camera can't give height); the holder
  z-gate always passes.
- Off-screen players simply don't exist in the output — frames without
  10 players + ball are dropped by the main pipeline's `complete_frames`.
- Flow-chained homography drifts on long clips; re-anchoring is C2's job.
- Team k-means needs actual jerseys; pickup games in mixed clothing (test.mp4)
  get near-random team splits.
