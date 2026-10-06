import numpy as np

from mia.attacks import attack_features, confidence_scores, fcm_attack, loss_scores, mentr_scores, shokri_lda
from mia.metrics import roc_auc


def _probs(py, rng):
    """Probability rows whose true-class entry is py; the rest spread randomly over the other 9 classes."""
    n = len(py)
    y = rng.integers(0, 10, n)
    rest = rng.dirichlet(np.ones(9), n) * (1 - py)[:, None]
    P = np.empty((n, 10))
    for i in range(n):
        P[i, np.arange(10) != y[i]] = rest[i]
        P[i, y[i]] = py[i]
    return P, y


def _leaky(n, rng):
    Pm, ym = _probs(rng.beta(20, 1, n), rng)
    Pn, yn = _probs(rng.beta(5, 2, n), rng)
    return np.vstack([Pm, Pn]), np.r_[ym, yn], np.r_[np.ones(n), np.zeros(n)]


def test_threshold_and_fcm_attacks_separate_a_leaky_model():
    P, y, lab = _leaky(2000, np.random.default_rng(0))
    for scores in (loss_scores(P, y), confidence_scores(P, y), mentr_scores(P, y),
                   fcm_attack(attack_features(P, y), seed=0)):
        assert roc_auc(scores, lab) > 0.8


def test_attacks_fail_on_a_non_leaky_model():
    rng = np.random.default_rng(1)
    P, y = _probs(rng.beta(5, 2, 4000), rng)
    lab = np.r_[np.ones(2000), np.zeros(2000)]
    for scores in (loss_scores(P, y), confidence_scores(P, y), mentr_scores(P, y)):
        assert abs(roc_auc(scores, lab) - 0.5) < 0.03


def test_shokri_lda_transfers_from_shadows_to_target():
    rng = np.random.default_rng(2)
    Ps, ys, ins = _leaky(3000, rng)
    Pt, yt, lab = _leaky(2000, rng)
    scores = shokri_lda(attack_features(Ps, ys), ins, attack_features(Pt, yt))
    assert roc_auc(scores, lab) > 0.8
