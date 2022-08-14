import numpy as np
from pathlib import Path
from typing import Dict, List, Union
from pdmetrics.metrics.base import pdMetrics


def recover_tp_fp_tn_fn(scores, labels, threshold):
    """Recovers idxs of true positives, false positives and false negatives from core metrics.
    NOTE: Assumes that all boxes are already thresholded by some threshold."""
    tp = np.where((scores >= threshold) & (labels.astype(np.uint8) == 1))[0]
    fp = np.where((scores >= threshold) & (labels.astype(np.uint8) == 0))[0]
    tn = np.where((scores <= threshold) & (labels.astype(np.uint8) == 0))[0]
    fn = np.where((scores <= threshold) & (labels.astype(np.uint8) == 1))[0]
    return tp, fp, tn, fn


def compute_f1(tp, fp, tn, fn, eps=1e-5):
    """Computes f1 related stats, from only total tp, fp, tn, fn."""
    _tp = tp.sum()
    _fp = fp.sum()
    _tn = tn.sum()
    _fn = fn.sum()
    tpr = _tp / (_tp + _fn + eps)
    fpr = _fp / (_tp + _fp + eps)
    recall = _tp / (_tp + _fn + eps)
    precision = _tp / (_tp + _fp + eps)

    # Handle edge cases
    # No ground truth
    if (_tp + _fn) == 0:
        recall = 1.0
        tpr = 1.0
    # No predictions
    if (_tp + _fp) == 0:
        precision = 1.0
        fpr = 1.0

    f1 = 2 * (precision * recall) / (precision + recall + eps)
    stats = {
        "tp": _tp,
        "fp": _fp,
        "tn": _tn,
        "fn": _fn,
        "tpr": tpr,
        "fpr": fpr,
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }
    return stats


def compute_over_example(example: Dict[str, np.ndarray], threshold: float):
    scores = example["preds"].flatten()
    labels = example["target"].flatten()
    tp, fp, tn, fn = recover_tp_fp_tn_fn(scores, labels, threshold)
    stats = compute_f1(tp, fp, tn, fn)
    return stats


class pdF1(pdMetrics):
    r"""

    Binary classification f1 metrics.

    preds: shape, np.ndarray, float \in [0, 1]
        represents the score predicted.

    target: shape, np.ndarray, int32 \in {0, 1}
        represents the positive label.
    """
    name: str = "f1"
    tracked: List[str] = ["f1", "precision", "recall", "tp", "fp", "fn"]

    def __init__(self, db_path: Union[Path, str], threshold: float = 0.5) -> None:
        super().__init__(self.name, self.tracked, db_path)
        self.threshold = threshold

    def compute_over_example(self, example: Dict[str, np.ndarray]):
        return compute_over_example(example, self.threshold)
