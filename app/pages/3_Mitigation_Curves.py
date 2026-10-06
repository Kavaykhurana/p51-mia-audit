"""Accuracy-privacy frontiers for early stopping and label smoothing."""
import plotly.express as px
import streamlit as st

import common

common.setup("Mitigation curves")
sweep_label = st.radio("Mitigation", ["Early stopping", "Label smoothing"], horizontal=True)
sweep, x_name = ("early_stopping", "epoch") if sweep_label == "Early stopping" else ("label_smoothing", "ε")

try:
    df = common.sweep_df(sweep)
except Exception as e:
    common.fail(e, "load sweep points")
if df.empty:
    st.warning(f"No {sweep_label.lower()} sweep points yet. Run notebook {'06' if sweep == 'early_stopping' else '07'}.")
    st.stop()

base = st.selectbox("Base configuration", sorted(df["base_config_id"].unique()))
g = (df[df["base_config_id"] == base].groupby("x_value", as_index=False)[["test_acc", "auc", "tpr_at_01pct_fpr"]].mean()
     .sort_values("x_value"))
g["label"] = g["x_value"].map(lambda v: f"{int(v)}" if sweep == "early_stopping" else f"{v:.2f}")
g["test_acc_pct"] = g["test_acc"] * 100

c1, c2 = st.columns(2)
for col, y, title in ((c1, "auc", "Best attack ROC-AUC"), (c2, "tpr_at_01pct_fpr", "TPR at 0.1% FPR")):
    fig = px.line(g, x="test_acc_pct", y=y, text="label", markers=True,
                  labels={"test_acc_pct": "Test (non-member) accuracy (%)", y: title})
    fig.update_traces(textposition="top center", line_color="#C55A11")
    fig.update_layout(height=420, title=f"{base}: test accuracy vs {title} (labels = {x_name})")
    col.plotly_chart(fig)

acc, priv = g.loc[g["test_acc"].idxmax()], g.loc[g["auc"].idxmin()]
cost = (acc["test_acc"] - priv["test_acc"]) * 100
st.markdown(
    f"On **{base}**, the most private point ({x_name} = {priv['label']}, AUC {priv['auc']:.3f}) "
    + (f"is also the most accurate one, so this mitigation costs no accuracy."
       if acc["x_value"] == priv["x_value"] else
       f"costs **{cost:.1f} percentage points** of test accuracy relative to the most accurate point "
       f"({x_name} = {acc['label']}, {acc['test_acc']:.1%}, AUC {acc['auc']:.3f}).")
)
if df["seed"].nunique() > 1:
    st.caption("Points average the target seeds available at each value.")
