"""Gate 0 for the CV front-end: how much positional noise can the rule
detectors take before F1 collapses?

Clean-SportVU detections act as ground truth; the same detectors re-run on
jittered tracks give an F1-vs-noise curve. Whatever sigma the curve dies at is
the localization accuracy any video->trajectory model must beat before its
output is allowed into the pipeline (docs/cv-plan.md, milestone C0).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from jdub.data import load_moments
from jdub.events import detect_game

SIGMAS = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0]  # ft, per-axis gaussian RMS
IOU_BAR = 0.3  # temporal overlap for two detections to count as the same event
KEY_TYPES = ["screen", "drive", "cut", "handoff"]
# ponytail: jitter-only noise model; track breaks / ID swaps / dropped frames
# are the other CV failure modes — add a dropout axis when C1 shows real tracks


def perturb(m: pl.DataFrame, sigma: float, seed: int = 0) -> pl.DataFrame:
    """Add per-axis gaussian noise (ft) to every entity's x/y. Ball z untouched."""
    if sigma == 0.0:
        return m
    rng = np.random.default_rng(seed)
    return m.with_columns(
        (pl.col("x") + pl.Series(rng.normal(0.0, sigma, m.height))),
        (pl.col("y") + pl.Series(rng.normal(0.0, sigma, m.height))),
    )


def _iou(a0: int, a1: int, b0: int, b1: int) -> float:
    inter = min(a1, b1) - max(a0, b0)
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 and inter > 0 else 0.0


def _match(clean: list[dict], noisy: list[dict]) -> int:
    """Greedy 1:1 matches: same event/type/actors, temporal IoU >= IOU_BAR."""
    used: set[int] = set()
    hits = 0
    for c in clean:
        best, best_iou = None, IOU_BAR
        for j, x in enumerate(noisy):
            if j in used:
                continue
            if (x["event_id"], x["type"], x["p1"], x["p2"]) != (
                c["event_id"],
                c["type"],
                c["p1"],
                c["p2"],
            ):
                continue
            iou = _iou(c["start_idx"], c["end_idx"], x["start_idx"], x["end_idx"])
            if iou >= best_iou:
                best, best_iou = j, iou
        if best is not None:
            used.add(best)
            hits += 1
    return hits


def f1(clean: list[dict], noisy: list[dict]) -> float:
    if not clean and not noisy:
        return 1.0
    return 2 * _match(clean, noisy) / (len(clean) + len(noisy))


def _cov_rows(df: pl.DataFrame) -> list[dict]:
    """Coverages reshaped so f1() can treat them like actions."""
    return [
        {
            "event_id": r["event_id"],
            "type": "coverage",
            "p1": r["screener"],
            "p2": r["handler"],
            "start_idx": r["start_idx"],
            "end_idx": r["end_idx"],
            "label": r["coverage"],
        }
        for r in df.iter_rows(named=True)
    ]


def _label_agree(clean: list[dict], noisy: list[dict]) -> float:
    """Among coverage pairs that match temporally, how often the label survives."""
    used: set[int] = set()
    same = total = 0
    for c in clean:
        for j, x in enumerate(noisy):
            if j in used or (x["event_id"], x["p1"], x["p2"]) != (
                c["event_id"],
                c["p1"],
                c["p2"],
            ):
                continue
            if _iou(c["start_idx"], c["end_idx"], x["start_idx"], x["end_idx"]) >= IOU_BAR:
                used.add(j)
                total += 1
                same += c["label"] == x["label"]
                break
    return same / total if total else 1.0


def curve(game_id: str, sigmas: list[float] = SIGMAS, seed: int = 0) -> list[dict]:
    """One row per sigma: F1 per key action type + coverage detection/label survival."""
    m = load_moments(game_id)
    base_actions: list[dict] | None = None
    base_cov: list[dict] | None = None
    rows: list[dict] = []
    for sigma in sigmas:
        _, actions_df, cov_df = detect_game(game_id, moments=perturb(m, sigma, seed))
        actions = actions_df.to_dicts()
        cov = _cov_rows(cov_df)
        if base_actions is None:
            base_actions, base_cov = actions, cov
        row: dict = {"sigma_ft": sigma}
        for t in KEY_TYPES:
            row[f"{t}_f1"] = round(
                f1(
                    [a for a in base_actions if a["type"] == t],
                    [a for a in actions if a["type"] == t],
                ),
                3,
            )
        row["coverage_f1"] = round(f1(base_cov, cov), 3)
        row["cov_label_agree"] = round(_label_agree(base_cov, cov), 3)
        rows.append(row)
    return rows
