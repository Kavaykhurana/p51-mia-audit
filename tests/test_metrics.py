import numpy as np
from sklearn.metrics import roc_auc_score

from mia.metrics import _auc_from_ranks, advantage, roc_auc, tpr_at_fpr

LABELS = np.array([1] * 5 + [0] * 5)


def test_perfect_and_reversed_scores():
    s = np.arange(10, 0, -1, dtype=float)  # members score highest
    assert roc_auc(s, LABELS) == 1.0 and advantage(s, LABELS) == 1.0
    assert roc_auc(-s, LABELS) == 0.0


def test_random_scores_near_chance():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 20000)
    assert abs(roc_auc(rng.random(20000), labels) - 0.5) < 0.02


def test_tpr_at_fpr_hand_built():
    # sorted by score: M M N M M N M N N N
    labels = np.array([1, 1, 0, 1, 1, 0, 1, 0, 0, 0])
    scores = np.arange(10, 0, -1, dtype=float)
    assert tpr_at_fpr(scores, labels, 0.0) == 0.4   # 2 of 5 members before the first non-member
    assert tpr_at_fpr(scores, labels, 0.2) == 0.8   # one non-member allowed -> 4 of 5
    assert tpr_at_fpr(scores, labels, 0.4) == 1.0


def test_rank_auc_matches_sklearn_with_ties():
    rng = np.random.default_rng(3)
    pos, neg = rng.integers(0, 5, 300).astype(float), rng.integers(0, 4, 200).astype(float)
    lab = np.r_[np.ones(300), np.zeros(200)]
    assert abs(_auc_from_ranks(pos, neg) - roc_auc_score(lab, np.r_[pos, neg])) < 1e-12
