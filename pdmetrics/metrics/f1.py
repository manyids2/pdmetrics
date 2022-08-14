import numpy as np
from pathlib import Path
from typing import Dict, List, Union
from pdmetrics.metrics.base import pdMetrics


def recover_tp_fp_tn_fn(scores, labels, threshold):
    """Recovers boolean indicator of true positives, false positives and false
    negatives from core metrics.
    """
    tp = (scores >= threshold) & (labels == 1)
    fp = (scores >= threshold) & (labels == 0)
    tn = (scores < threshold) & (labels == 0)
    fn = (scores < threshold) & (labels == 1)
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
    scores = example["preds"].flatten().astype(np.int32)
    labels = example["target"].flatten().astype(np.int32)
    print(scores)
    print(labels)
    tp, fp, tn, fn = recover_tp_fp_tn_fn(scores, labels, threshold)
    stats = compute_f1(tp, fp, tn, fn)
    return stats


class pdF1(pdMetrics):
    r"""Classification f1 metrics.

    Binary case::

        preds: shape, np.ndarray, float32 \in [0, 1]
            represents the score predicted.

        target: shape, np.ndarray, int32 \in {0, 1}
            represents the positive label.

    Multiclass case::

        preds: shape, np.ndarray, int32 \in {0, num_classes}
            represents the class predicted.

        target: shape, np.ndarray, int32 \in {0, 1}
            represents the positive label.

    Tracks the following metrics::

        "f1", "precision", "recall", "tp", "fp", "fn"
    """
    name: str = "f1"
    tracked: List[str] = ["f1", "precision", "recall", "tp", "fp", "fn"]

    def __init__(
        self, db_path: Union[Path, str], labels: Dict[int, str], threshold: float = 0.5
    ) -> None:
        super().__init__(self.name, self.tracked, db_path)
        self.labels = labels
        self.num_classes = len(labels)
        self.threshold = threshold

    def compute_over_example(self, example: Dict[str, np.ndarray]):
        """Compute binary stats."""
        return compute_over_example(example, self.threshold)

    def compute_over_example_multiclass(self, example: Dict[str, np.ndarray]):
        """Compute multiclass stats in one vs all manner.
        TODO: Currently is O(n) in number of labels, can be made O(1)."""
        stats = {}
        for class_idx, label in self.labels.items():
            _example = {
                "preds": example["preds"] == class_idx,
                "target": example["target"] == class_idx,
            }
            _stats = compute_over_example(_example, self.threshold)
            stats[label] = _stats
        return stats
