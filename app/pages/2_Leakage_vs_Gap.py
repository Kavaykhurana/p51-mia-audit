"""Interactive headline plot: attack AUC against the generalisation gap (or the loss ratio)."""
import plotly.express as px
import streamlit as st

import common
from mia.attacks import ATTACKS

common.setup("Leakage vs generalisation gap")
st.caption("Each point is one target model (every configuration, seed and sweep checkpoint).")

c1, c2, c3 = st.columns(3)
attack = c1.selectbox("Attack", ["Best per model"] + [a for a in ATTACKS if a != "kmeans"],
                      help="K-means gives one operating point, not a ROC curve, so it has no AUC to plot.")
x_label = c3.radio("x-axis", ["Accuracy gap", "Loss ratio"], horizontal=True)

try:
    df = common.leakage_df() if attack == "Best per model" else common.attack_runs_df().query("attack == @attack").rename(
        columns={"target_model_id": "model_id"})
except Exception as e:
    common.fail(e, "load attack results")

families = sorted(df["family"].unique())
picked = c2.multiselect("Families", families, default=families)
df = df[df["family"].isin(picked) & df["auc"].notna()].copy()
if df.empty:
    st.warning("No attack runs match these filters yet.")
    st.stop()

x, x_title, log_x = ("gap_pp", "Train–test accuracy gap (percentage points)", False) if x_label == "Accuracy gap" else \
    ("loss_ratio", "Test loss / train loss (log scale)", True)
df["gap_pp"] = df["gap_acc"] * 100
df["AUC [CI]"] = df.apply(lambda r: f"{r['auc']:.3f} [{r['auc_ci_low']:.3f}, {r['auc_ci_high']:.3f}]", axis=1)
df["err_hi"], df["err_lo"] = df["auc_ci_high"] - df["auc"], df["auc"] - df["auc_ci_low"]

if len(df) >= 3:
    rho, p = common.spearman(tuple(df[x]), tuple(df["auc"]))
    st.markdown(f"**Spearman ρ = {rho:.3f}** (permutation p = {p:.4f}, n = {len(df)} models)")

fig = px.scatter(df, x=x, y="auc", color="family", symbol="family", error_y="err_hi", error_y_minus="err_lo",
                 color_discrete_map=common.FAMILY_COLORS, log_x=log_x,
                 hover_data={"model_id": True, "gap_pp": ":.1f", "loss_ratio": ":.2f", "AUC [CI]": True,
                             "tpr_at_01pct_fpr": ":.4f", "severity": True, "attack": True, x: False, "auc": False,
                             "err_hi": False, "err_lo": False},
                 labels={"auc": "Attack ROC-AUC", "gap_pp": x_title, "loss_ratio": x_title, "family": "Family",
                         "tpr_at_01pct_fpr": "TPR @ 0.1% FPR"})
fig.add_hline(y=0.5, line_dash="dash", line_color="grey", annotation_text="chance")
fig.update_layout(height=520, xaxis_title=x_title, yaxis_range=[0.45, 1.01])
st.plotly_chart(fig)

cols = ["model_id", "config_id", "family", "seed", "epoch", "train_acc", "test_acc", "gap_acc", "loss_ratio", "attack", "auc",
        "auc_ci_low", "auc_ci_high", "advantage", "tpr_at_1pct_fpr", "tpr_at_01pct_fpr", "lift_at_01pct_fpr", "severity"]
table = df[cols].sort_values("auc", ascending=False)
st.dataframe(table.style.map(common.severity_style, subset=["severity"]), hide_index=True)
st.download_button("Download CSV", table.to_csv(index=False).encode(), file_name="leakage_vs_gap.csv", mime="text/csv")
