"""Classifier and attack metrics: accuracy, cross-entropy, ROC-AUC with bootstrap CI, advantage, TPR at low FPR,
lift-based severity, and a permutation-tested Spearman correlation."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score, roc_curve


def accuracy(P: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(P.argmax(axis=1) == y))


def mean_ce_loss(P: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1.0))))


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores))


def advantage(scores: np.ndarray, labels: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(np.max(tpr - fpr))


def tpr_at_fpr(scores: np.ndarray, labels: np.ndarray, alpha: float) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    ok = fpr <= alpha
    return float(tpr[ok].max()) if ok.any() else 0.0


def threshold_at_fpr(scores: np.ndarray, labels: np.ndarray, alpha: float = 0.01) -> float:
    """Score threshold (flag `score >= threshold`) of the highest-TPR ROC point with FPR <= alpha."""
    fpr, tpr, thr = roc_curve(labels, scores)
    ok = np.flatnonzero(fpr <= alpha)
    t = thr[ok[np.argmax(tpr[ok])]]
    return float(np.nextafter(np.max(scores), np.inf)) if np.isinf(t) else float(t)


def _auc_from_ranks(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney form of ROC-AUC (ties count 1/2); equals roc_auc_score, about 3x faster for the bootstrap."""
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def bootstrap_auc_ci(scores: np.ndarray, labels: np.ndarray, n: int = 1000, seed: int = 7) -> tuple[float, float]:
    """95% CI: resample members and non-members separately with replacement."""
    rng = np.random.default_rng(seed)
    pos, neg = scores[labels == 1], scores[labels == 0]
    aucs = [_auc_from_ranks(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(n)]
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)


def lift(tpr_at_01pct: float, alpha: float = 0.001) -> float:
    return tpr_at_01pct / alpha


def severity(lift_value: float | None, bounds: dict) -> str | None:
    """Low < moderate <= Moderate < high <= High < critical <= Critical (bounds on TPR/FPR at FPR = 0.1%)."""
    if lift_value is None:
        return None
    if lift_value >= bounds["critical"]:
        return "Critical"
    if lift_value >= bounds["high"]:
        return "High"
    return "Moderate" if lift_value >= bounds["moderate"] else "Low"


def spearman_with_permutation(x, y, n: int = 10000, seed: int = 7) -> tuple[float, float]:
    """Spearman rho and a two-sided permutation p-value ((#|rho_perm| >= |rho|) + 1) / (n + 1)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    rho = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    hits = sum(abs(spearmanr(x, rng.permutation(y)).statistic) >= abs(rho) - 1e-12 for _ in range(n))
    return rho, float((hits + 1) / (n + 1))
