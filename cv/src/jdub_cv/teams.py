"""Team assignment from jersey color: per-track median torso color in Lab,
k-means with k=3 (two teams + officials), the two most populous clusters are
the teams. Returns {track_id: 1 | 2 | None}; None (referee cluster) is dropped
upstream."""

from __future__ import annotations

import cv2
import numpy as np


def torso_color(frame: np.ndarray, xyxy: np.ndarray) -> np.ndarray | None:
    """Median Lab color of the central upper-body crop of one detection."""
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    w, h = x2 - x1, y2 - y1
    if w < 8 or h < 16:
        return None
    crop = frame[y1 + h // 6 : y1 + h // 2, x1 + w // 4 : x2 - w // 4]
    if crop.size == 0:
        return None
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    return np.median(lab, axis=0)


def _kmeans(x: np.ndarray, k: int = 3, iters: int = 30) -> np.ndarray:
    # deterministic farthest-point init: avoids the empty-cluster degeneracy
    centers = [x[0]]
    for _ in range(k - 1):
        d = np.min([np.linalg.norm(x - c, axis=1) for c in centers], axis=0)
        centers.append(x[d.argmax()])
    centers = np.array(centers)
    labels = np.zeros(len(x), dtype=int)
    for _ in range(iters):
        d = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2)
        labels = d.argmin(axis=1)
        for j in range(k):
            if (labels == j).any():
                centers[j] = x[labels == j].mean(axis=0)
    return labels


def assign_teams(track_colors: dict[int, list[np.ndarray]]) -> dict[int, int | None]:
    tids = [t for t, cs in track_colors.items() if cs]
    if len(tids) < 3:
        return {t: 1 for t in tids}
    med = np.array([np.median(np.array(track_colors[t]), axis=0) for t in tids])
    labels = _kmeans(med, k=3)
    by_size = sorted(range(3), key=lambda j: -(labels == j).sum())
    team_of_cluster = {by_size[0]: 1, by_size[1]: 2, by_size[2]: None}
    return {t: team_of_cluster[c] for t, c in zip(tids, labels)}
