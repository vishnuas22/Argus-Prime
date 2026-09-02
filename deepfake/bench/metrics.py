"""Metrics for P0 baseline: AUC, EER, Brier score, runtime stats."""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """labels: 1=fake, 0=real. scores: higher means 'more fake'."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def eer(scores: np.ndarray, labels: np.ndarray) -> float:
    """Equal error rate via roc_curve interpolation (robust to tied scores)."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    d = fpr - (1 - tpr)
    i = int(np.argmin(np.abs(d)))
    if 0 < i < len(fpr) and d[i] != d[i - 1]:
        w = -d[i - 1] / (d[i] - d[i - 1])
        return float(min(1.0, max(0.0, fpr[i - 1] + w * (fpr[i] - fpr[i - 1]))))
    return float(min(1.0, max(0.0, (fpr[i] + (1 - tpr[i])) / 2)))


def brier(p_real: np.ndarray, labels: np.ndarray) -> float:
    """Brier on P(real); labels 1=fake."""
    if len(p_real) == 0:
        return float("nan")
    return float(np.mean((p_real - (1 - labels)) ** 2))


def abstention_rate(flags: list) -> float:
    return float(np.mean(flags)) if flags else float("nan")


def mean_ms(values: list) -> float:
    v = [x for x in values if x is not None]
    return float(np.mean(v)) if v else float("nan")
