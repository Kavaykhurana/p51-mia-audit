"""Render the responsible-disclosure report from the database (no hard-coded numbers)."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import jinja2
import pandas as pd

from . import db
from .attacks import ATTACKS
from .config import REPORTS_DIR, TEMPLATES_DIR, Experiment
from .training import headline_model_id

SEVERITY_RANK = {"Low": 0, "Moderate": 1, "High": 2, "Critical": 3}
TARGET_CONFIGS = ("C1", "C2", "C3", "C4", "C5", "C6", "C7")


def _frontier_summary(df: pd.DataFrame, base: str) -> dict:
    """Most accurate vs most private (lowest AUC) point of one sweep on one base config, seeds averaged."""
    g = df[df["base_config_id"] == base].groupby("x_value", as_index=False)[["test_acc", "auc", "tpr_at_01pct_fpr"]].mean()
    if g.empty:
        raise ValueError(f"no sweep points found for base config {base}; run notebooks 06/07")
    acc, priv = g.loc[g["test_acc"].idxmax()], g.loc[g["auc"].idxmin()]
    return {"base": base, "points": g.to_dict("records"), "acc_x": acc["x_value"], "acc_acc": acc["test_acc"],
            "acc_auc": acc["auc"], "priv_x": priv["x_value"], "priv_acc": priv["test_acc"], "priv_auc": priv["auc"],
            "cost_pp": (acc["test_acc"] - priv["test_acc"]) * 100}


def build_context(conn, exp: Experiment) -> dict:
    models = db.models_df(conn, role="target").set_index("model_id")
    best = db.leakage_vs_gap_df(conn).drop_duplicates("model_id").set_index("model_id")
    runs = db.attack_runs_df(conn)
    assets, findings = [], []
    for cid in TARGET_CONFIGS:
        mid = headline_model_id(exp, cid)
        if mid not in models.index:
            raise ValueError(f"missing target model {mid} (config {cid}); train it before rendering the report")
        missing = set(ATTACKS) - set(runs.loc[runs["target_model_id"] == mid, "attack"])
        if missing:
            raise ValueError(f"missing attack run(s) {sorted(missing)} for model {mid}; run notebook 05")
        m, b = models.loc[mid], best.loc[mid]
        assets.append({"config_id": cid, "model_id": mid, "description": exp.config(cid).description, "family": m["family"],
                       "test_acc": m["test_acc"], "gap": m["gap_acc"], "n_params": int(m["n_params"])})
        findings.append({"config_id": cid, "model_id": mid, "attack": b["attack"], "auc": b["auc"], "lo": b["auc_ci_low"],
                         "hi": b["auc_ci_high"], "tpr1": b["tpr_at_1pct_fpr"], "tpr01": b["tpr_at_01pct_fpr"],
                         "lift": b["lift_at_01pct_fpr"], "severity": b["severity"]})
    worst = max(findings, key=lambda f: (SEVERITY_RANK[f["severity"]], f["auc"]))
    safest = min(findings, key=lambda f: (SEVERITY_RANK[f["severity"]], f["auc"]))

    es_df, ls_df = db.sweep_df(conn, "early_stopping"), db.sweep_df(conn, "label_smoothing")
    early = [_frontier_summary(es_df, b) for b in exp.early_stopping]
    smoothing = []
    for base in exp.ls_bases:
        s = _frontier_summary(ls_df, base)
        pts = pd.DataFrame(s["points"])
        auc0 = pts.loc[pts["x_value"] == 0, "auc"]
        if auc0.empty:
            raise ValueError(f"missing label-smoothing x=0 point for {base}")
        s["auc0"], s["auc_ls_mean"] = float(auc0.iloc[0]), float(pts.loc[pts["x_value"] > 0, "auc"].mean())
        s["verdict"] = "reduced" if s["auc_ls_mean"] < s["auc0"] else "increased"
        smoothing.append(s)
    return {"today": date.today().isoformat(), "assets": assets, "findings": findings, "worst": worst, "safest": safest,
            "early": early, "smoothing": smoothing, "bounds": exp.severity_bounds, "n_audit": 10 * (exp.per_class["member"] + exp.per_class["nonmember"]),
            "n_members": 10 * exp.per_class["member"], "classical_k": exp.classical_k, "yolo_k": exp.yolo_k}


def render_report(conn, exp: Experiment, out_path: Path = REPORTS_DIR / "disclosure_report.md") -> str:
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(TEMPLATES_DIR), undefined=jinja2.StrictUndefined,
                             trim_blocks=True, lstrip_blocks=True)
    text = env.get_template("disclosure_report.md.j2").render(**build_context(conn, exp))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return text
