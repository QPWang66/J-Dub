"""WASB ball detector (Tarashima et al., BMVC 2023) — specialized heatmap model
for small fast balls, trained on NBA broadcast footage. MIT-licensed weights:
`uvx gdown 1nfECuSyJvPUmz3njZCdFERSQQbERt8FU -O weights/wasb_basketball_best.pth.tar`

The model eats 3 consecutive frames (512x288, ImageNet-normalized, 9 channels)
and emits one heatmap per frame; we read the current frame's heatmap and return
the strongest blob above threshold as (x_px, y_px, score) in original coords.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import cv2
import numpy as np

# wasb.yaml, inlined (hydra not needed for inference)
_CFG = {
    "frames_in": 3,
    "frames_out": 3,
    "out_scales": [0],
    "MODEL": {
        "EXTRA": {
            "FINAL_CONV_KERNEL": 1,
            "PRETRAINED_LAYERS": ["*"],
            "STEM": {"INPLANES": 64, "STRIDES": [1, 1]},
            "STAGE1": {
                "NUM_MODULES": 1,
                "NUM_BRANCHES": 1,
                "BLOCK": "BOTTLENECK",
                "NUM_BLOCKS": [1],
                "NUM_CHANNELS": [32],
                "FUSE_METHOD": "SUM",
            },
            "STAGE2": {
                "NUM_MODULES": 1,
                "NUM_BRANCHES": 2,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": [2, 2],
                "NUM_CHANNELS": [16, 32],
                "FUSE_METHOD": "SUM",
            },
            "STAGE3": {
                "NUM_MODULES": 1,
                "NUM_BRANCHES": 3,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": [2, 2, 2],
                "NUM_CHANNELS": [16, 32, 64],
                "FUSE_METHOD": "SUM",
            },
            "STAGE4": {
                "NUM_MODULES": 1,
                "NUM_BRANCHES": 4,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": [2, 2, 2, 2],
                "NUM_CHANNELS": [16, 32, 64, 128],
                "FUSE_METHOD": "SUM",
            },
            "DECONV": {"NUM_DECONVS": 0, "KERNEL_SIZE": [], "NUM_BASIC_BLOCKS": 2},
        },
        "INIT_WEIGHTS": False,
    },
}
INP_W, INP_H = 512, 288
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
SCORE_TH = 0.5


class _AttrDict(dict):
    """dict that also answers attribute access — WASB code mixes both styles."""

    __getattr__ = dict.__getitem__

    @classmethod
    def wrap(cls, obj):
        if isinstance(obj, dict):
            return cls({k: cls.wrap(v) for k, v in obj.items()})
        return obj


class WasbBallDetector:
    def __init__(self, weights: str | Path):
        import torch

        from jdub_cv.vendor.wasb_hrnet import HRNet

        self._torch = torch
        self.model = HRNet(_AttrDict.wrap(_CFG))
        ck = torch.load(str(weights), map_location="cpu", weights_only=False)
        self.model.load_state_dict(ck["model_state_dict"])
        self.model.eval()
        self.buf: deque[np.ndarray] = deque(maxlen=3)

    def _prep(self, frame_bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame_bgr, (INP_W, INP_H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return ((img - MEAN) / STD).transpose(2, 0, 1)  # CHW

    def detect(self, frame_bgr: np.ndarray) -> tuple[float, float, float] | None:
        """Feed the next frame; returns (x_px, y_px, score) or None."""
        torch = self._torch
        self.buf.append(self._prep(frame_bgr))
        if len(self.buf) < 3:
            return None
        x = torch.from_numpy(np.concatenate(list(self.buf), axis=0)[None])  # 1x9xHxW
        with torch.no_grad():
            hm = torch.sigmoid(self.model(x)[0])[0, -1].numpy()  # current frame's heatmap
        if hm.max() <= SCORE_TH:
            return None
        _, th = cv2.threshold(hm, SCORE_TH, 1, cv2.THRESH_BINARY)
        n, labels = cv2.connectedComponents(th.astype(np.uint8))
        best, best_mass, best_peak = None, 0.0, 0.0
        for m in range(1, n):
            ys, xs = np.where(labels == m)
            ws = hm[ys, xs]
            mass = float(ws.sum())
            if mass > best_mass:
                cx = float((xs * ws).sum() / ws.sum())
                cy = float((ys * ws).sum() / ws.sum())
                best, best_mass, best_peak = (cx, cy), mass, float(ws.max())
        if best is None:
            return None
        h, w = frame_bgr.shape[:2]
        # score is the blob's peak sigmoid (0-1) so it compares against YOLO conf
        return best[0] * w / INP_W, best[1] * h / INP_H, best_peak
