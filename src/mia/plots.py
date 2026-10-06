"""Report figures F1-F7 in one Matplotlib style, saved as PDF and 300-dpi PNG."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from .config import FIGURES_DIR
from .data import CLASS_NAMES

FAMILY_COLORS = {"nn": "#1B3A6B", "softmax": "#C55A11", "lda": "#1E5631", "yolo": "#5C2D91"}
FAMILY_MARKERS = {"nn": "o", "softmax": "s", "lda": "D", "yolo": "^"}
FAMILY_LABELS = {"nn": "Soft 1-NN", "softmax": "Softmax discriminant", "lda": "Fisher LDA", "yolo": "YOLO26n-cls"}
CONFIG_COLORS = {"C1": "#1B3A6B", "C2": "#C55A11", "C3": "#E08A3C", "C4": "#8C3B0A", "C5": "#1E5631",
                 "C6": "#5C2D91", "C7": "#9B72C9"}


def apply_style() -> None:
    plt.style.use("default")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.bbox": "tight", "figure.dpi": 100})


def save_figure(fig, stem: str, out_dir: Path = FIGURES_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{stem}.pdf", out_dir / f"{stem}.png"]
    for p in paths:
        fig.savefig(p, dpi=300)
    return paths


def _legend_outside(ax, **kw) -> None:
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, **kw)


def _family_scatter(ax, df: pd.DataFrame, x: str, xscale: float = 1.0) -> None:
    for fam, g in df.groupby("family"):
        yerr = np.vstack([g["auc"] - g["auc_ci_low"], g["auc_ci_high"] - g["auc"]]).clip(min=0)
        ax.errorbar(g[x] * xscale, g["auc"], yerr=yerr, fmt=FAMILY_MARKERS[fam], color=FAMILY_COLORS[fam],
                    ecolor=FAMILY_COLORS[fam], elinewidth=0.8, capsize=2, markersize=5, label=FAMILY_LABELS[fam])
    ax.axhline(0.5, ls="--", lw=0.8, color="grey", label="Chance (AUC 0.5)")
    ax.set_ylim(0.45, 1.01)
    ax.set_ylabel("Best attack ROC-AUC")


def f1_auc_vs_gap(df: pd.DataFrame, rho: float, p: float):
    """df: v_leakage_vs_gap rows for every target point."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    _family_scatter(ax, df, "gap_acc", 100)
    ax.plot([], [], " ", label=f"Spearman ρ = {rho:.2f} (p = {p:.4f})")
    ax.set_xlabel("Train-test accuracy gap (percentage points)")
    ax.set_title("F1  Membership leakage vs generalisation gap")
    _legend_outside(ax)
    return fig


def f2_auc_vs_loss_ratio(df: pd.DataFrame, rho: float, p: float):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    _family_scatter(ax, df[df["loss_ratio"] > 0], "loss_ratio")
    ax.plot([], [], " ", label=f"Spearman ρ = {rho:.2f} (p = {p:.4f})")
    ax.set_xscale("log")
    ax.set_xlabel("Test loss / train loss (ratio, log scale)")
    ax.set_title("F2  Membership leakage vs loss ratio")
    _legend_outside(ax)
    return fig


def _frontier(df: pd.DataFrame, title: str, fmt):
    """Two panels: test accuracy vs AUC and vs TPR@0.1% FPR; one line per base config, seeds averaged."""
    agg = df.groupby(["base_config_id", "x_value"], as_index=False)[["test_acc", "auc", "tpr_at_01pct_fpr"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, ycol, ylab in ((axes[0], "auc", "Best attack ROC-AUC"), (axes[1], "tpr_at_01pct_fpr", "TPR at 0.1% FPR")):
        for base, g in agg.groupby("base_config_id"):
            g = g.sort_values("x_value")
            ax.plot(g["test_acc"] * 100, g[ycol], "-o", color=CONFIG_COLORS.get(base, "black"), markersize=4, label=base)
            for _, r in g.iterrows():
                ax.annotate(fmt(r["x_value"]), (r["test_acc"] * 100, r[ycol]), textcoords="offset points",
                            xytext=(4, 4), fontsize=7)
        ax.set_xlabel("Test (non-member) accuracy (%)")
        ax.set_ylabel(ylab)
    axes[1].set_yscale("symlog", linthresh=1e-3)
    _legend_outside(axes[1], title="Base config")
    fig.suptitle(title)
    return fig


def f3_early_stopping(df: pd.DataFrame):
    return _frontier(df, "F3  Early-stopping frontier (labels = epoch)", lambda x: f"{int(x)}")


def f4_label_smoothing(df: pd.DataFrame):
    return _frontier(df, "F4  Label-smoothing frontier (labels = ε)", lambda x: f"{x:.2f}")


def f5_roc_curves(curves: dict):
    """curves: {label: (scores, is_member, config_id)} for C1-C7 (best attack each)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    grid = np.logspace(-4, 0, 200)
    ax.plot(grid, grid, ls="--", lw=0.8, color="grey", label="Chance (y = x)")
    for label, (scores, lab, cid) in curves.items():
        fpr, tpr, _ = roc_curve(lab, scores)
        ax.plot(np.clip(fpr, 1e-4, 1), np.clip(tpr, 1e-4, 1), lw=1.4, color=CONFIG_COLORS.get(cid, "black"), label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-4, 1)
    ax.set_ylim(1e-4, 1)
    ax.set_xlabel("False-positive rate (non-members flagged)")
    ax.set_ylabel("True-positive rate (members flagged)")
    ax.set_title("F5  Log-log ROC, best attack per configuration")
    _legend_outside(ax, fontsize=8)
    return fig


def f6_exposed_members(images: np.ndarray, labels: np.ndarray, panels: dict):
    """panels: {model_id: DataFrame(sample_idx, is_member, score)} for two models -> 4x8 grid of top-16 members."""
    fig, axes = plt.subplots(4, 8, figsize=(12, 7))
    for p, (mid, scores) in enumerate(panels.items()):
        top = scores[scores["is_member"] == 1].nlargest(16, "score")
        for j, (_, r) in enumerate(top.iterrows()):
            ax = axes[2 * p + j // 8, j % 8]
            ax.imshow(images[int(r["sample_idx"])][..., ::-1], interpolation="nearest")
            ax.set_title(f"{CLASS_NAMES[labels[int(r['sample_idx'])]]}\n{r['score']:.2f}", fontsize=7)
        axes[2 * p, 0].text(-0.15, 1.35, f"{mid}: 16 most-exposed members (A1 score)", transform=axes[2 * p, 0].transAxes,
                            fontsize=9, fontweight="bold")
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("F6  Training images an attacker identifies with the highest confidence", y=1.0)
    fig.tight_layout()
    return fig


def f7_cost_vs_leakage(df: pd.DataFrame):
    """df: one row per headline target with infer_ms_per_img, model_bytes, auc, family, config_id."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    size = 30 + 400 * df["model_bytes"] / df["model_bytes"].max()
    for fam, g in df.groupby("family"):
        ax.scatter(g["infer_ms_per_img"], g["auc"], s=size[g.index], color=FAMILY_COLORS[fam],
                   marker=FAMILY_MARKERS[fam], alpha=0.75, label=FAMILY_LABELS[fam])
    for _, r in df.iterrows():
        ax.annotate(r["config_id"], (r["infer_ms_per_img"], r["auc"]), textcoords="offset points", xytext=(6, -3), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Inference time (ms per image, log scale; marker area ∝ model size in bytes)")
    ax.set_ylabel("Best attack ROC-AUC")
    ax.set_title("F7  Edge cost vs privacy leakage")
    _legend_outside(ax, markerscale=0.6)
    return fig
