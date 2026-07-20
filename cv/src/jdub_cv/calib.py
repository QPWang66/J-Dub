"""Per-frame homography for a panning broadcast camera.

Manual anchor correspondences (rough is fine) on frame 0 give an initial H;
each frame then (1) coarse-propagates H with LK optical flow on court-plane
features, and (2) SNAPS it to the court by maximizing the IoU between the
projected paint model (key rectangle + outer free-throw half-disc) and the
paint-colored pixel mask (`paint_hsv` in the calib json). Region IoU has no
collapse degeneracy and is robust to players standing on the paint; the snap
absorbs both sloppy manual anchors and flow drift.

Court frame: 94 x 50 ft, x toward the attacked (right) basket, y=0 at the
near (bottom-of-image) sideline.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

COURT_LENGTH, COURT_WIDTH = 94.0, 50.0
MAX_CORNERS = 400
MIN_FLOW_PTS = 12
REFRESH_EVERY = 20  # frames between feature re-harvests
# snapping
PAINT_CORNERS = np.float32([[75, 17], [94, 17], [94, 33], [75, 33]])  # H is parameterized on these
SNAP_SCALE = 0.5  # masks evaluated at half resolution
MIN_IOU = 0.25  # below this the paint isn't really in view: keep flow-only H
RESNAP_IOU = 0.45  # tracking snap under this triggers a wide re-search
TRUST_IOU = 0.55  # snap fit good enough to accept big corrections
MAX_CORRECTION = 10.0  # half-res px: max per-frame snap correction at mediocre fit
BLEND = 0.7  # snap weight when blending with the flow prediction (temporal smoothing)
MAX_AREA_STEP = 1.25  # max per-frame quad area ratio (real zooms change ~1-2%/frame)
INIT_STEPS = (32.0, 16.0, 8.0, 4.0, 2.0, 1.0)  # frame-0 snap: wide basin
TRACK_STEPS = (16.0, 8.0, 4.0, 2.0, 1.0)  # per-frame snap after flow propagation
# ponytail: needs a paint_hsv color per court; courts whose paint matches the
# floor won't snap (flow-only fallback). The C2 court-keypoint model replaces this.


def load_anchor(calib_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    spec = json.loads(Path(calib_path).read_text())
    return np.float32(spec["image"]), np.float32(spec["court"])


def _paint_model_poly(ft_disc: bool) -> np.ndarray:
    """Painted-key outline in court ft: lane rectangle, optionally + the outer
    free-throw half-disc (courts like Staples paint that too)."""
    if not ft_disc:
        return np.float32([[94, 17], [75, 17], [75, 33], [94, 33]])
    phi = np.linspace(np.pi / 2, 3 * np.pi / 2, 20)
    half_disc = (np.array([75.0, 25.0]) + 6 * np.stack([np.cos(phi), np.sin(phi)], axis=1))[::-1]
    return np.float32(np.vstack([[[94, 17]], [[75, 17]], half_disc, [[75, 33]], [[94, 33]]]))


_MODEL_POLY = _paint_model_poly(ft_disc=True)


def paint_mask(frame_bgr: np.ndarray, paint_hsv: list) -> np.ndarray:
    """Binary mask of paint-colored pixels at SNAP_SCALE resolution."""
    small = cv2.resize(frame_bgr, None, fx=SNAP_SCALE, fy=SNAP_SCALE)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mask = np.zeros(small.shape[:2], np.uint8)
    for lo, hi in paint_hsv:
        mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))


def _valid_quad(pts: np.ndarray) -> bool:
    """Convex, non-self-intersecting quad (a crossed 'bowtie' can fake high IoU)."""
    v = np.roll(pts, -1, axis=0) - pts
    w = np.roll(v, -1, axis=0)
    cross = v[:, 0] * w[:, 1] - v[:, 1] * w[:, 0]
    return bool((cross > 0).all() or (cross < 0).all())


def paint_iou(mask: np.ndarray, pts_img: np.ndarray) -> float:
    """IoU between the projected paint model and the paint-color mask.

    `pts_img` and `mask` share the same (SNAP_SCALE) pixel coordinates."""
    if not _valid_quad(np.asarray(pts_img)):
        return 0.0
    try:
        h = cv2.getPerspectiveTransform(np.float32(pts_img), PAINT_CORNERS)
        poly = cv2.perspectiveTransform(_MODEL_POLY.reshape(-1, 1, 2), np.linalg.inv(h))
    except (cv2.error, np.linalg.LinAlgError):
        return 0.0
    poly = poly.reshape(-1, 2)
    if not np.isfinite(poly).all():
        return 0.0
    canvas = np.zeros_like(mask)
    cv2.fillPoly(canvas, [poly.astype(np.int32)], 1)
    inter = int(np.count_nonzero(canvas & mask))
    union = int(np.count_nonzero(canvas | mask))
    return inter / union if union else 0.0


def _components_near(mask: np.ndarray, pts_img: np.ndarray) -> np.ndarray:
    """Keep only mask components overlapping the current model projection —
    drops the center circle, floor logos, ad boards and crowd reds."""
    canvas = np.zeros_like(mask)
    try:
        h = cv2.getPerspectiveTransform(np.float32(pts_img), PAINT_CORNERS)
        poly = cv2.perspectiveTransform(_MODEL_POLY.reshape(-1, 1, 2), np.linalg.inv(h))
        cv2.fillPoly(canvas, [poly.reshape(-1, 2).astype(np.int32)], 1)
    except (cv2.error, np.linalg.LinAlgError):
        return mask
    n, labels = cv2.connectedComponents((mask > 0).astype(np.uint8))
    hits = np.unique(labels[(canvas > 0) & (labels > 0)])
    return np.isin(labels, hits).astype(np.uint8) if len(hits) else np.zeros_like(mask)


def _candidates(pts: np.ndarray, step: float):
    """Per-corner nudges plus whole-quad moves (translate/rotate/scale) —
    single-corner descent alone gets stuck in sheared local optima."""
    for i in range(4):
        for dx, dy in ((step, 0), (-step, 0), (0, step), (0, -step)):
            cand = pts.copy()
            cand[i] += (dx, dy)
            yield cand
    for dx, dy in ((step, 0), (-step, 0), (0, step), (0, -step)):
        yield pts + np.float32([dx, dy])
    c = pts.mean(axis=0)
    for theta in (step / 120.0, -step / 120.0):  # radians, scaled with step
        rot = np.float32([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        yield (pts - c) @ rot.T + c
    for s in (1 + step / 150.0, 1 / (1 + step / 150.0)):
        yield (pts - c) * s + c


def snap(mask: np.ndarray, pts_img: np.ndarray, steps=INIT_STEPS) -> tuple[np.ndarray, float]:
    """Greedy descent on the 4 paint-corner image points, maximizing IoU."""
    pts = pts_img.astype(np.float32).copy()
    best = paint_iou(mask, pts)
    for step in steps:
        for _ in range(4):  # a few passes per scale
            moved = False
            for cand in _candidates(pts, step):
                s = paint_iou(mask, cand)
                if s > best:
                    best, pts, moved = s, np.float32(cand), True
            if not moved:
                break
    return pts, best


# the 18 court landmarks of the reloc2 keypoint dataset, in our court ft
# (verified against labeled frames: idx7 = halfcourt x far sideline, idx16/17 =
# right FT-line x far/near paint edge; dataset "top" = far sideline = y=50)
COURT_KP_FT = np.float32(
    [
        [0, 50],
        [0, 47],
        [0, 33],
        [0, 17],
        [0, 3],
        [0, 0],  # left baseline, far -> near
        [47, 0],
        [47, 50],  # halfcourt x near / far sideline
        [19, 33],
        [19, 17],  # left FT line x far / near paint edge
        [94, 0],
        [94, 3],
        [94, 17],
        [94, 33],
        [94, 47],
        [94, 50],  # right baseline, near -> far
        [75, 33],
        [75, 17],  # right FT line x far / near paint edge
    ]
)
KP_CONF = 0.5
KP_MIN_PTS = 5
# one-euro smoothing of the derived paint corners: per-frame keypoint noise is
# a few px (smooth hard at rest), pans move corners tens of px/frame (follow)
EURO_FREQ = 30.0  # nominal broadcast fps; exact value only shifts the tuning
EURO_MIN_CUTOFF = 1.0  # Hz
EURO_BETA = 0.02  # swept on okc-nyk: accel p95 10.9 -> 7.3px vs fixed blend, +3px pan lag


class _OneEuro:
    """One-euro filter (Casiez et al. 2012), elementwise over an array."""

    def __init__(self):
        self.x: np.ndarray | None = None
        self.dx: np.ndarray | None = None

    @staticmethod
    def _alpha(cutoff) -> np.ndarray:
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau * EURO_FREQ)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.x is None:
            self.x, self.dx = x, np.zeros_like(x)
            return x
        ad = self._alpha(1.0)
        self.dx = ad * (x - self.x) * EURO_FREQ + (1 - ad) * self.dx
        a = self._alpha(EURO_MIN_CUTOFF + EURO_BETA * np.abs(self.dx))
        self.x = a * x + (1 - a) * self.x
        return self.x


class KeypointCalibrator:
    """Absolute per-frame H from a trained court-keypoint model (YOLO-pose).

    No flow, no paint color, no drift: every frame is calibrated independently
    from detected landmarks, lightly smoothed. `iou` reports the fraction of
    confident keypoints (for stability.py)."""

    def __init__(self, weights: str | Path, flip: bool = False, static: bool = False):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.flip = flip
        self.static = static
        self.H: np.ndarray = np.eye(3)
        self.iou = 0.0
        self.filt = _OneEuro()

    def _paint_pts(self) -> np.ndarray:
        return cv2.perspectiveTransform(
            PAINT_CORNERS.reshape(-1, 1, 2), np.linalg.inv(self.H)
        ).reshape(-1, 2)

    def update(self, frame: np.ndarray) -> np.ndarray:
        r = self.model.predict(frame, imgsz=640, verbose=False)[0]
        k = r.keypoints
        self.iou = 0.0
        if k is None or k.conf is None or not len(k.xy):
            return self.H
        best = int(r.boxes.conf.argmax()) if r.boxes is not None and len(r.boxes) else 0
        xy = k.xy[best].cpu().numpy()
        cf = k.conf[best].cpu().numpy()
        ok = (cf >= KP_CONF) & (xy[:, 0] > 1) & (xy[:, 1] > 1)
        self.iou = round(float(ok.sum()) / len(COURT_KP_FT), 3)
        if ok.sum() < KP_MIN_PTS:
            return self.H
        h, _ = cv2.findHomography(xy[ok], COURT_KP_FT[ok], cv2.RANSAC, 4.0)
        if h is None or not np.isfinite(h).all():
            return self.H
        pts = cv2.perspectiveTransform(PAINT_CORNERS.reshape(-1, 1, 2), np.linalg.inv(h))
        pts = pts.reshape(-1, 2)
        if not (np.isfinite(pts).all() and _valid_quad(pts)):
            return self.H
        pts = self.filt(pts)
        self.H = cv2.getPerspectiveTransform(np.float32(pts), PAINT_CORNERS)
        return self.H


def make_calibrator(calib_path: str | Path):
    """Factory: keypoint-model calibrator when the calib file names one, else
    the classical paint-snap calibrator."""
    spec = json.loads(Path(calib_path).read_text())
    if spec.get("kp_model"):
        weights = Path(calib_path).resolve().parent.parent / spec["kp_model"]
        return KeypointCalibrator(
            weights, flip=bool(spec.get("flip", False)), static=bool(spec.get("static", False))
        )
    return Calibrator(calib_path)


class Calibrator:
    def __init__(self, calib_path: str | Path):
        spec = json.loads(Path(calib_path).read_text())
        self.img_pts = np.float32(spec["image"])
        self.court_pts = np.float32(spec["court"])
        self.static = bool(spec.get("static", False))  # fixed camera: H never changes
        self.flip = bool(
            spec.get("flip", False)
        )  # mirror frames: left-attack clip -> right-attack convention
        self.paint_hsv = spec.get("paint_hsv")  # [[lo,hi], ...] HSV ranges of the painted key
        h, _ = cv2.findHomography(self.img_pts, self.court_pts)
        if h is None:
            raise ValueError(f"degenerate anchor points in {calib_path}")
        self.H: np.ndarray = h
        self.prev_gray: np.ndarray | None = None
        self.feats: np.ndarray | None = None
        self.age = 0
        self.iou = 0.0

    def _mask(self, gray: np.ndarray) -> np.ndarray:
        """Court-floor region in image space: visible half-court through H^-1."""
        rect = np.float32(
            [
                [COURT_LENGTH / 2, 0],
                [COURT_LENGTH, 0],
                [COURT_LENGTH, COURT_WIDTH],
                [COURT_LENGTH / 2, COURT_WIDTH],
            ]
        ).reshape(-1, 1, 2)
        poly = cv2.perspectiveTransform(rect, np.linalg.inv(self.H)).reshape(-1, 2)
        mask = np.zeros_like(gray)
        if np.all(np.isfinite(poly)):
            hh, ww = gray.shape
            poly = np.clip(poly, [-ww, -hh], [2 * ww, 2 * hh])
            cv2.fillConvexPoly(mask, cv2.convexHull(poly.astype(np.int32)), 255)
        if not mask.any():  # degenerate projection: fall back to the anchor hull
            cv2.fillConvexPoly(mask, cv2.convexHull(self.img_pts.astype(np.int32)), 255)
        return mask

    def _harvest(self, gray: np.ndarray) -> None:
        self.feats = cv2.goodFeaturesToTrack(
            gray, MAX_CORNERS, qualityLevel=0.01, minDistance=12, mask=self._mask(gray)
        )

    def _paint_pts(self) -> np.ndarray:
        return cv2.perspectiveTransform(
            PAINT_CORNERS.reshape(-1, 1, 2), np.linalg.inv(self.H)
        ).reshape(-1, 2)

    def _snap_to_paint(self, frame: np.ndarray, steps) -> None:
        self.iou = 0.0
        if not self.paint_hsv:
            return  # no paint color known: flow-only
        mask = paint_mask(frame, self.paint_hsv)
        pts0 = self._paint_pts() * SNAP_SCALE  # flow-propagated prediction
        mask = _components_near(mask, pts0)
        if not mask.any():
            return
        pts, iou = snap(mask, pts0, steps)
        if iou < RESNAP_IOU and steps is not INIT_STEPS:  # lost the paint: search wider
            pts, iou = snap(mask, pts, INIT_STEPS)
        self.iou = iou
        if iou < MIN_IOU:
            return  # paint not really in view: keep flow-only H
        correction = float(np.abs(pts - pts0).max())
        if steps is not INIT_STEPS:
            if iou < TRUST_IOU and correction > MAX_CORRECTION:
                return  # mediocre fit demanding a big jump: distrust, keep flow
            ratio = cv2.contourArea(np.float32(pts)) / max(cv2.contourArea(np.float32(pts0)), 1.0)
            if not (1 / MAX_AREA_STEP <= ratio <= MAX_AREA_STEP):
                return  # quad inflating/deflating faster than any real zoom: motion blur
            pts = BLEND * pts + (1 - BLEND) * pts0  # temporal smoothing
        self.H = cv2.getPerspectiveTransform(np.float32(pts) / SNAP_SCALE, PAINT_CORNERS)

    def update(self, frame: np.ndarray) -> np.ndarray:
        """Feed the next frame; returns the current image->court homography."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:  # frame 0: deep snap from the manual anchors
            self._snap_to_paint(frame, INIT_STEPS)
            if self.static:
                self.prev_gray = gray
                return self.H
            self._harvest(gray)
        elif self.static:
            return self.H
        elif self.feats is not None and len(self.feats):
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.feats, None)
            ok = st.reshape(-1) == 1
            old, new = self.feats[ok], nxt[ok]
            if len(new) >= MIN_FLOW_PTS:
                m, _ = cv2.findHomography(old, new, cv2.RANSAC, 3.0)
                if m is not None and np.isfinite(m).all() and abs(np.linalg.det(m)) > 1e-6:
                    self.H = self.H @ np.linalg.inv(m)
            self._snap_to_paint(frame, TRACK_STEPS)
            self.feats = new.reshape(-1, 1, 2)
            self.age += 1
            if self.age % REFRESH_EVERY == 0 or len(self.feats) < MAX_CORNERS // 4:
                self._harvest(gray)
        else:
            self._harvest(gray)
        self.prev_gray = gray
        return self.H


def to_court(H: np.ndarray, pts_px: np.ndarray) -> np.ndarray:
    """(N,2) image px -> (N,2) court ft."""
    if not len(pts_px):
        return pts_px
    return cv2.perspectiveTransform(np.float32(pts_px).reshape(-1, 1, 2), H).reshape(-1, 2)


def court_lines() -> list[np.ndarray]:
    """Right-half court markings in court ft, as polylines (for overlay QC)."""

    def seg(a, b, n=2):
        return np.linspace(a, b, n)

    hoop = np.array([COURT_LENGTH - 5.25, COURT_WIDTH / 2])
    phi = np.linspace(-1.18, 1.18, 60)  # 3pt arc, cut where it meets the corner lines
    arc = hoop + 23.75 * np.stack([-np.cos(phi), np.sin(phi)], axis=1)
    arc = arc[(arc[:, 1] >= 3) & (arc[:, 1] <= COURT_WIDTH - 3)]
    ft_circle = np.array([COURT_LENGTH - 19, COURT_WIDTH / 2]) + 6 * np.stack(
        [np.cos(np.linspace(0, 2 * np.pi, 40)), np.sin(np.linspace(0, 2 * np.pi, 40))], axis=1
    )
    return [
        seg([47, 0], [94, 0]),  # near sideline
        seg([47, 50], [94, 50]),  # far sideline
        seg([94, 0], [94, 50]),  # baseline
        seg([47, 0], [47, 50]),  # halfcourt
        seg([75, 17], [94, 17]),  # paint near
        seg([75, 33], [94, 33]),  # paint far
        seg([75, 17], [75, 33]),  # free-throw line
        seg([80, 3], [94, 3]),  # corner 3 near
        seg([80, 47], [94, 47]),  # corner 3 far
        arc,
        ft_circle,
    ]
