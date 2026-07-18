# jdub-cv

Stage 1 of the jdub pipeline: **NBA broadcast video → moments-schema trajectories**.

Separate uv project on purpose: torch/ultralytics stay out of the main jdub
package. The only interface to the main pipeline is Parquet in the moments
schema — after running this, `uv run jdub detect <game-id>` and
`uv run jdub studio` in the repo root work on the output as if it were SportVU.

## Layout

```
src/jdub_cv/
  pipeline.py    video -> detect/track -> project -> team split -> moments/players/games Parquet
  calib.py       homography: KeypointCalibrator (trained court-landmark model, primary)
                 + Calibrator (classical paint-snap, fallback/no-model courts)
  ball.py        WASB ball detector (HRNet heatmap, NBA-broadcast pretrained)
  teams.py       team split from jersey colors (chest crop, k-means)
  stability.py   whole-clip calibration QA: per-frame IoU / corner velocity + worst-frame dumps
  vendor/        vendored third-party model code (WASB HRNet, MIT)
calib/           per-clip config: rough frame-0 anchors, paint_hsv, flip/static, kp_model
clips/           evaluation clips (NBA broadcast, git-ignored; sourced from the BCT project)
weights/         model weights (git-ignored; download commands below)
datasets/        training data (git-ignored; pulled via Roboflow API)
```

## Run

```bash
cd cv
uv sync
./get-weights.sh   # pulls court_kp.pt + WASB ball weights from the GitHub release
uv run python -m jdub_cv.pipeline clips/dal-lac1.mp4 calib/dal-lac1.json \
    --out ../data/parquet --game-id cv-dal-lac1 --overlay qc.mp4
cd .. && uv run jdub detect cv-dal-lac1
```

`--overlay` writes the eyeball-QC video (court model + tracks re-projected onto
the frames). `python -m jdub_cv.stability <clip> <calib>` runs the whole-clip
calibration QA — every calibration change must pass it on all of `clips/`
(lesson learned: spot-checking two frames ships broken pans).

## Models

| Stage | Model | Weights |
|-------|-------|---------|
| Court calibration | **court-keypoint YOLO11s-pose** — 18 court landmarks per frame → RANSAC → absolute H; no flow, no drift, no per-court color tuning. Trained on the Roboflow `fyp-3bwmg/reloc2` broadcast dataset (1.4k images) | `weights/court_kp.pt` (train: see below) |
| Court calibration (fallback) | classical paint-IoU snap (`paint_hsv` color mask + greedy quad fit). Kept for courts the keypoint model hasn't seen | — |
| Players: detect | YOLO11m, COCO pretrained | auto-download |
| Players: track | BoT-SORT + court-space tracklet sewing (`merge_tracks`) | — |
| Ball | [WASB](https://github.com/nttcom/WASB-SBDT) (BMVC'23, NBA-broadcast trained) + YOLO hybrid behind a motion-continuity gate | `uvx gdown 1nfECuSyJvPUmz3njZCdFERSQQbERt8FU -O weights/wasb_basketball_best.pth.tar` |
| Team split | chest-crop dominant color (2-means) + k-means(3), two biggest clusters = teams, third = refs; QC dots painted with each side's extracted jersey color | — |

Planned upgrades (per the RF-DETR/SAM2/SigLIP/SmolVLM2 stack): SigLIP embeddings
for team clustering, SAM2 for occlusion-proof tracking, RF-DETR fine-tune for
players/ball/refs, jersey-number OCR for real player identity. On hold until the
court model lands.

### Train the court model

```bash
export ROBOFLOW_API_KEY=...   # roboflow.com free account
uv run python -c "from roboflow import Roboflow; import os; \
  Roboflow(api_key=os.environ['ROBOFLOW_API_KEY']).workspace('fyp-3bwmg') \
  .project('reloc2-den7l').version(1).download('yolov8', location='datasets/reloc2-1')"
uv run yolo pose train model=yolo11s-pose.pt data=datasets/reloc2-1/data.yaml \
    epochs=80 imgsz=640 batch=16 device=mps fliplr=0.0 patience=20
cp runs/.../weights/best.pt weights/court_kp.pt
```

`fliplr=0.0` is load-bearing: the dataset's `flip_idx` is identity, so
horizontal-flip augmentation would corrupt left/right landmark semantics.

## Calibration files

`calib/<name>.json`: `kp_model` (path to court-keypoint weights, preferred) or
rough `image`↔`court` anchors + `paint_hsv` for the classical fallback;
optional `flip` (mirror left-attack clips into the right-attack convention)
and `static` (fixed camera).

## Known limits (deliberate, documented)

- Ball `z` is emitted as 0.0 (single camera can't give height); the holder
  z-gate always passes.
- Off-screen players don't exist in the output — frames without 10 players +
  ball are dropped by the main pipeline's `complete_frames`.
- Front-row crowd occasionally passes the in-court filter at frame edges.
- Team k-means needs actual jerseys (mixed-clothing pickup games split randomly).
