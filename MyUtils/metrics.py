from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _binary_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    pos = probs[labels == 1]
    neg = probs[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    greater = 0.0
    for value in pos:
        greater += float((value > neg).sum())
        greater += 0.5 * float((value == neg).sum())
    return float(greater / (pos.size * neg.size))


def binary_classification_metrics(
    labels: Sequence[float],
    probs: Sequence[float],
    threshold: float = 0.5,
) -> dict[str, float]:
    labels_arr = np.asarray(labels, dtype=np.int64)
    probs_arr = np.asarray(probs, dtype=np.float64)
    preds = (probs_arr >= threshold).astype(np.int64)

    tp = int(((preds == 1) & (labels_arr == 1)).sum())
    tn = int(((preds == 0) & (labels_arr == 0)).sum())
    fp = int(((preds == 1) & (labels_arr == 0)).sum())
    fn = int(((preds == 0) & (labels_arr == 1)).sum())
    total = max(int(labels_arr.size), 1)

    precision = _safe_div(tp, tp + fp)
    sensitivity = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = _safe_div(2 * precision * sensitivity, precision + sensitivity)
    mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = _safe_div(tp * tn - fp * fn, mcc_den)

    return {
        "ACC": float((tp + tn) / total),
        "Precision": precision,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "F1": f1,
        "AUC": _binary_auc(labels_arr, probs_arr),
        "MCC": mcc,
        "TP": float(tp),
        "TN": float(tn),
        "FP": float(fp),
        "FN": float(fn),
    }
