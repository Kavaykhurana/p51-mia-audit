"""Membership-inference attacks A1-A6. Every score follows one convention: higher means "member"."""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from . import db
from .classifiers import Standardizer
from .config import Experiment
from .metrics import (advantage, bootstrap_auc_ci, lift, roc_auc, severity, threshold_at_fpr, tpr_at_fpr)
from .training import load_probs

ATTACKS = ("shokri_lda", "shokri_lda_perclass", "loss", "confidence", "mentr", "kmeans", "fcm")
SHADOW_ATTACKS = ("shokri_lda", "shokri_lda_perclass")
ATTACK_SEED = 0
_EPS = 1e-12


def attack_features(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """[top1, top2, top3 of the sorted probability vector, p_y] -> (n, 4)."""
    top = -np.sort(-P, axis=1)[:, :3]
    return np.column_stack([top, P[np.arange(len(y)), y]])


def loss_scores(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Yeom et al.: -loss = log p_y."""
    return np.log(np.clip(P[np.arange(len(y)), y], _EPS, 1.0))


def confidence_scores(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    return P[np.arange(len(y)), y].copy()


def mentr_scores(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Song & Mittal modified entropy, negated: -[-(1-p_y) log p_y - sum_{i!=y} p_i log(1-p_i)]."""
    Pc = np.clip(P, _EPS, 1 - _EPS)
    rows = np.arange(len(y))
    py = Pc[rows, y]
    other = Pc * np.log(1 - Pc)
    other[rows, y] = 0.0
    mentr = -(1 - py) * np.log(py) - other.sum(axis=1)
    return -mentr


def _lda() -> LinearDiscriminantAnalysis:
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")


def shokri_lda(shadow_feats: np.ndarray, shadow_in: np.ndarray, target_feats: np.ndarray) -> np.ndarray:
    """A1: one Fisher discriminant learns IN (1) vs OUT (0) from all shadow rows; score = decision function."""
    return _lda().fit(shadow_feats, shadow_in).decision_function(target_feats)


def shokri_lda_perclass(shadow_feats, shadow_in, shadow_y, target_feats, target_y) -> np.ndarray:
    """A1b: one discriminant per true class (Shokri et al.); global model for classes with < 50 rows of a label."""
    scores = shokri_lda(shadow_feats, shadow_in, target_feats)
    for c in np.unique(target_y):
        rows = shadow_y == c
        if min((shadow_in[rows] == 1).sum(), (shadow_in[rows] == 0).sum()) < 50:
            continue
        q = target_y == c
        scores[q] = _lda().fit(shadow_feats[rows], shadow_in[rows]).decision_function(target_feats[q])
    return scores


def kmeans_attack(target_feats: np.ndarray, seed: int, query_feats: np.ndarray | None = None) -> np.ndarray:
    """A5: 2-means on standardised attack features; the cluster with higher mean p_y is "member". Binary output."""
    sc = Standardizer().fit(target_feats)
    km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(sc.transform(target_feats))
    member = int(np.argmax(km.cluster_centers_[:, 3]))  # column 3 = p_y; standardising keeps the order
    q = target_feats if query_feats is None else query_feats
    return (km.predict(sc.transform(q)) == member).astype(np.float64)


def fcm_memberships(X: np.ndarray, centres: np.ndarray, m: float) -> np.ndarray:
    d = np.maximum(np.linalg.norm(X[:, None, :] - centres[None, :, :], axis=2), _EPS)
    ratio = (d[:, :, None] / d[:, None, :]) ** (2.0 / (m - 1.0))
    return 1.0 / ratio.sum(axis=2)


def fuzzy_cmeans(X: np.ndarray, c: int = 2, m: float = 2.0, max_iter: int = 300, tol: float = 1e-6,
                 seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Fuzzy C-means (Bezdek). Returns (centres (c, d), memberships U (n, c))."""
    U = np.random.default_rng(seed).random((len(X), c))
    U /= U.sum(axis=1, keepdims=True)
    for _ in range(max_iter):
        Um = U ** m
        centres = (Um.T @ X) / Um.sum(axis=0)[:, None]
        U_new = fcm_memberships(X, centres, m)
        done = np.max(np.abs(U_new - U)) < tol
        U = U_new
        if done:
            break
    return centres, U


def fcm_attack(target_feats: np.ndarray, seed: int, query_feats: np.ndarray | None = None) -> np.ndarray:
    """A6: membership degree in the fuzzy cluster whose centre has the higher p_y."""
    sc = Standardizer().fit(target_feats)
    centres, U = fuzzy_cmeans(sc.transform(target_feats), seed=seed)
    member = int(np.argmax(centres[:, 3]))
    if query_feats is None:
        return U[:, member]
    return fcm_memberships(sc.transform(query_feats), centres, 2.0)[:, member]


def score_attack(attack: str, query_P, query_y, target_P, target_y, shadow=None, seed: int = ATTACK_SEED) -> np.ndarray:
    """Score query outputs with one attack.

    target_P / target_y: the target's outputs on the audit set (the unsupervised attacks cluster these).
    shadow: dict(P, y, is_in) stacked over all shadows of the same recipe (needed by the Shokri attacks).
    """
    if attack == "loss":
        return loss_scores(query_P, query_y)
    if attack == "confidence":
        return confidence_scores(query_P, query_y)
    if attack == "mentr":
        return mentr_scores(query_P, query_y)
    qf, tf = attack_features(query_P, query_y), attack_features(target_P, target_y)
    if attack == "kmeans":
        return kmeans_attack(tf, seed, qf)
    if attack == "fcm":
        return fcm_attack(tf, seed, qf)
    sf = attack_features(shadow["P"], shadow["y"])
    if attack == "shokri_lda":
        return shokri_lda(sf, shadow["is_in"], qf)
    if attack == "shokri_lda_perclass":
        return shokri_lda_perclass(sf, shadow["is_in"], shadow["y"], qf, query_y)
    raise ValueError(f"unknown attack {attack!r}")


def target_audit_data(target_model_id: str, labels: np.ndarray) -> dict:
    """The target's outputs on the 20,000 audit images (members first, then non-members)."""
    p = load_probs(target_model_id)
    idx = np.concatenate([p["idx_in"], p["idx_out"]])
    return {"P": np.vstack([p["P_in"], p["P_out"]]), "idx": idx, "y": labels[idx],
            "is_member": np.concatenate([np.ones(len(p["idx_in"]), int), np.zeros(len(p["idx_out"]), int)])}


def shadow_model_ids(conn, exp: Experiment, target_model_id: str) -> list[str]:
    """Shadows of the target's recipe; for checkpointed configs (C2, C6) only those at the same epoch."""
    config_id, epoch = conn.execute("SELECT config_id, epoch FROM models WHERE model_id = ?", (target_model_id,)).fetchone()
    if exp.config(config_id).has_epoch_suffix:
        rows = conn.execute("SELECT model_id FROM models WHERE config_id = ? AND role = 'shadow' AND epoch = ? "
                            "ORDER BY shadow_index", (config_id, epoch))
    else:
        rows = conn.execute("SELECT model_id FROM models WHERE config_id = ? AND role = 'shadow' ORDER BY shadow_index",
                            (config_id,))
    return [r[0] for r in rows]


def shadow_data(shadow_ids: list[str], labels: np.ndarray) -> dict:
    P, y, is_in = [], [], []
    for sid in shadow_ids:
        p = load_probs(sid)
        P += [p["P_in"], p["P_out"]]
        y += [labels[p["idx_in"]], labels[p["idx_out"]]]
        is_in += [np.ones(len(p["idx_in"]), int), np.zeros(len(p["idx_out"]), int)]
    return {"P": np.vstack(P), "y": np.concatenate(y), "is_in": np.concatenate(is_in)}


def run_all_attacks(conn, exp: Experiment, target_model_id: str, labels: np.ndarray) -> list[dict]:
    """Run all seven attacks on one target, store attack_runs + 20,000 sample_scores each. Skips existing runs."""
    config_id = conn.execute("SELECT config_id FROM models WHERE model_id = ?", (target_model_id,)).fetchone()[0]
    tgt = target_audit_data(target_model_id, labels)
    sids = shadow_model_ids(conn, exp, target_model_id)
    shadow = shadow_data(sids, labels) if sids else None
    bounds, (fpr1, fpr01) = exp.severity_bounds, sorted(exp.fpr_points, reverse=True)
    results = []
    for attack in ATTACKS:
        uses_shadows = attack in SHADOW_ATTACKS
        shadow_cfg, n_sh = (config_id, len(sids)) if uses_shadows else (None, 0)
        if db.attack_run_exists(conn, target_model_id, attack, shadow_cfg, n_sh):
            continue
        if uses_shadows and not sids:
            print(f"{target_model_id}: no shadow models for {config_id}; {attack} skipped")
            continue
        s, lab = score_attack(attack, tgt["P"], tgt["y"], tgt["P"], tgt["y"], shadow), tgt["is_member"]
        row = dict(target_model_id=target_model_id, attack=attack, shadow_config_id=shadow_cfg, n_shadows=n_sh)
        if attack == "kmeans":  # one operating point only
            tpr, fpr = float(s[lab == 1].mean()), float(s[lab == 0].mean())
            row.update(auc=None, auc_ci_low=None, auc_ci_high=None, advantage=tpr - fpr, tpr_at_1pct_fpr=None,
                       tpr_at_01pct_fpr=None, lift_at_01pct_fpr=None, severity=None, threshold_1pct=None)
            extra = {"tpr": tpr, "fpr": fpr}
        else:
            lo, hi = bootstrap_auc_ci(s, lab, exp.bootstrap, exp.bootstrap_seed)
            t01 = tpr_at_fpr(s, lab, fpr01)
            lf = lift(t01, fpr01)
            row.update(auc=roc_auc(s, lab), auc_ci_low=lo, auc_ci_high=hi, advantage=advantage(s, lab),
                       tpr_at_1pct_fpr=tpr_at_fpr(s, lab, fpr1), tpr_at_01pct_fpr=t01, lift_at_01pct_fpr=lf,
                       severity=severity(lf, bounds), threshold_1pct=threshold_at_fpr(s, lab, fpr1))
            extra = {}
        with conn:
            run_id = db.insert_attack_run(conn, row)
            db.insert_sample_scores(conn, run_id, tgt["idx"], lab, s)
        results.append({**row, **extra, "attack_run_id": run_id})
    return results
