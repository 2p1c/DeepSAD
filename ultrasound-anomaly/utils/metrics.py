"""Metrics helpers for anomaly detection evaluation."""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

import numpy as np


def roc_curve_binary(y_true: Iterable[int], y_score: Iterable[float]):
    """Compute a binary ROC curve without sklearn.

    Returns
    -------
    fpr : np.ndarray
        False positive rates sorted by threshold.
    tpr : np.ndarray
        True positive rates sorted by threshold.
    thresholds : np.ndarray
        Thresholds corresponding to each ROC point, descending.
    """

    y_true = np.asarray(list(y_true), dtype=np.int64).ravel()
    y_score = np.asarray(list(y_score), dtype=np.float64).ravel()

    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("y_true and y_score must have the same length")
    if y_true.size == 0:
        raise ValueError("roc_curve_binary requires at least one sample")

    y_true = (y_true > 0).astype(np.int64)

    order = np.argsort(-y_score, kind="mergesort")
    y_true = y_true[order]
    y_score = y_score[order]

    distinct_indices = np.where(np.diff(y_score))[0]
    threshold_indices = np.r_[distinct_indices, y_true.size - 1]

    tps = np.cumsum(y_true)[threshold_indices]
    fps = np.cumsum(1 - y_true)[threshold_indices]

    positives = y_true.sum()
    negatives = y_true.size - positives

    if positives == 0:
        tpr = np.zeros_like(tps, dtype=np.float64)
    else:
        tpr = tps / positives

    if negatives == 0:
        fpr = np.zeros_like(fps, dtype=np.float64)
    else:
        fpr = fps / negatives

    thresholds = y_score[threshold_indices]

    # Prepend the origin so plots form a closed ROC curve.
    fpr = np.r_[0.0, fpr]
    tpr = np.r_[0.0, tpr]
    thresholds = np.r_[np.inf, thresholds]
    return fpr, tpr, thresholds


def auc_from_roc(fpr: Iterable[float], tpr: Iterable[float]) -> float:
    """Compute ROC AUC via trapezoidal integration."""

    fpr = np.asarray(list(fpr), dtype=np.float64).ravel()
    tpr = np.asarray(list(tpr), dtype=np.float64).ravel()
    if fpr.shape[0] != tpr.shape[0]:
        raise ValueError("fpr and tpr must have the same length")
    if fpr.size < 2:
        return 0.0

    order = np.argsort(fpr, kind="mergesort")
    fpr = fpr[order]
    tpr = tpr[order]
    return float(np.trapz(tpr, fpr))


def aggregate_scores_by_path(scores, path_ids, method="mean"):
    """Aggregate window scores by path id.

    Parameters
    ----------
    scores:
        Window-level scores.
    path_ids:
        Per-window path identifiers.
    method:
        ``mean``, ``max`` or ``p95``.

    Returns
    -------
    np.ndarray
        Aggregated scores in order of first appearance of each path id.
    """

    scores = np.asarray(scores, dtype=np.float64).ravel()
    path_ids = np.asarray(path_ids).ravel()

    if scores.shape[0] != path_ids.shape[0]:
        raise ValueError("scores and path_ids must have the same length")
    if scores.size == 0:
        return np.asarray([], dtype=np.float64)

    groups = OrderedDict()
    for score, path_id in zip(scores, path_ids):
        key = path_id.item() if hasattr(path_id, "item") else path_id
        groups.setdefault(key, []).append(float(score))

    aggregated = []
    for values in groups.values():
        arr = np.asarray(values, dtype=np.float64)
        if method == "mean":
            aggregated.append(float(arr.mean()))
        elif method == "max":
            aggregated.append(float(arr.max()))
        elif method == "p95":
            aggregated.append(float(np.percentile(arr, 95)))
        else:
            raise ValueError("method must be one of: mean, max, p95")

    return np.asarray(aggregated, dtype=np.float64)
