"""Per-frame homography for a panning broadcast camera.

Manual anchor correspondences (image px -> court ft) on frame 0 give H0;
subsequent frames chain H via LK optical flow on court-plane features with
RANSAC (players/crowd rejected as outliers).

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
# ponytail: flow-chained homography drifts over long clips; the C2 upgrade is a
# trained court-keypoint model giving absolute H per frame (docs/cv-plan.md)


def load_anchor(calib_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    spec = json.loads(Path(calib_path).read_text())
    return np.float32(spec["image"]), np.float32(spec["court"])


class Calibrator:
    def __init__(self, calib_path: str | Path):
        spec = json.loads(Path(calib_path).read_text())
        self.img_pts = np.float32(spec["image"])
        self.court_pts = np.float32(spec["court"])
        self.static = bool(spec.get("static", False))  # fixed camera: H never changes
        h, _ = cv2.findHomography(self.img_pts, self.court_pts)
        if h is None:
            raise ValueError(f"degenerate anchor points in {calib_path}")
        self.H: np.ndarray = h
        self.prev_gray: np.ndarray | None = None
        self.feats: np.ndarray | None = None
        self.age = 0

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

    def update(self, frame: np.ndarray) -> np.ndarray:
        """Feed the next frame; returns the current image->court homography."""
        if self.static:
            return self.H
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self._harvest(gray)
        elif self.feats is not None and len(self.feats):
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.feats, None)
            ok = st.reshape(-1) == 1
            old, new = self.feats[ok], nxt[ok]
            if len(new) >= MIN_FLOW_PTS:
                m, _ = cv2.findHomography(old, new, cv2.RANSAC, 3.0)
                if m is not None and np.isfinite(m).all() and abs(np.linalg.det(m)) > 1e-6:
                    self.H = self.H @ np.linalg.inv(m)
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
