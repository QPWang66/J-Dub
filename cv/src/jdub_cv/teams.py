"""Team assignment: cluster player appearance into two teams + officials.

Two feature backends, same clustering:
- color (default): dominant chest color in Lab (fast, no extra model)
- siglip: SigLIP vision-tower embeddings of the torso crop (robust to
  lighting/texture; the RF-DETR/SAM2/SigLIP stack's approach)

k-means with k=3 (two teams + officials); the two most populous clusters are
the teams. Returns {track_id: 1 | 2 | None}; None (referee cluster) is dropped
upstream."""

from __future__ import annotations

import cv2
import numpy as np


def torso_color(frame: np.ndarray, xyxy: np.ndarray) -> np.ndarray | None:
    """Dominant Lab color of the chest crop of one detection.

    Tight crop (chest only) + 2-means, keeping the bigger cluster — separates
    the jersey from skin/background instead of blending them into mud."""
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    w, h = x2 - x1, y2 - y1
    if w < 10 or h < 24:
        return None
    crop = frame[y1 + int(0.22 * h) : y1 + int(0.45 * h), x1 + int(0.3 * w) : x2 - int(0.3 * w)]
    if crop.size == 0:
        return None
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    if len(lab) < 8:
        return lab.mean(axis=0)
    if len(lab) > 256:
        lab = lab[:: len(lab) // 256]
    labels = _kmeans(lab, k=2)
    big = np.bincount(labels).argmax()
    return lab[labels == big].mean(axis=0)


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


def assign_teams(track_feats: dict[int, list[np.ndarray]]) -> dict[int, int | None]:
    tids = [t for t, cs in track_feats.items() if cs]
    if len(tids) < 3:
        return {t: 1 for t in tids}
    med = np.array([np.median(np.array(track_feats[t]), axis=0) for t in tids])
    labels = _kmeans(med, k=3)
    by_size = sorted(range(3), key=lambda j: -(labels == j).sum())
    team_of_cluster = {by_size[0]: 1, by_size[1]: 2, by_size[2]: None}
    return {t: team_of_cluster[c] for t, c in zip(tids, labels)}


class SiglipEmbedder:
    """Torso-crop embeddings from SigLIP's vision tower (zero-shot, no labels).

    Drop-in alternative feature for assign_teams: embeddings capture color AND
    texture, so home whites with colored trim don't collapse into the ref
    cluster the way pure color sometimes does."""

    MODEL = "google/siglip-base-patch16-224"

    def __init__(self, device: str = "cpu"):
        from transformers import SiglipImageProcessor, SiglipVisionModel

        self.processor = SiglipImageProcessor.from_pretrained(self.MODEL)
        self.model = SiglipVisionModel.from_pretrained(self.MODEL).to(device).eval()
        self.device = device

    def embed(self, frame: np.ndarray, xyxy: np.ndarray) -> np.ndarray | None:
        import torch

        x1, y1, x2, y2 = (int(v) for v in xyxy)
        w, h = x2 - x1, y2 - y1
        if w < 10 or h < 24:
            return None
        crop = frame[y1 : y1 + int(0.55 * h), x1:x2]  # head+torso, skip legs
        if crop.size == 0:
            return None
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs).pooler_output[0]
        v = out.cpu().numpy()
        return v / (np.linalg.norm(v) + 1e-8)
