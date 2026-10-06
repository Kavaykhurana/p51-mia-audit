"""Landing page: the question, the pipeline and the headline numbers."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common

common.setup("Can a model reveal its training images?")

st.markdown(
    "We trained seven CIFAR-10 classifiers, from a pure memoriser to a well-regularised model, using hand-crafted "
    "descriptors and a YOLO Nano network. We then attacked each one: given a single image and the model's output, can an "
    "outsider tell whether that exact image was in the training set? The answer depends on how far each model's training "
    "accuracy runs ahead of its test accuracy, and this app lets you explore that link image by image."
)


def pipeline_figure() -> go.Figure:
    steps = ["Split", "Train target", "Train shadows", "Learn the tell", "Score & compare"]
    notes = ["members / non-members<br>/ shadow pool", "on 10,000 members", "same recipe,<br>known IN / OUT",
             "LDA on shadow outputs", "AUC, TPR @ low FPR"]
    fig = go.Figure()
    w, gap = 1.6, 0.55
    for i, (s, n) in enumerate(zip(steps, notes)):
        x0 = i * (w + gap)
        fig.add_shape(type="rect", x0=x0, x1=x0 + w, y0=0, y1=1, line=dict(color="#1B3A6B", width=2),
                      fillcolor="#EEF2F8", layer="below")
        fig.add_annotation(x=x0 + w / 2, y=0.66, text=f"<b>{s}</b>", showarrow=False, font=dict(size=14, color="#1B3A6B"))
        fig.add_annotation(x=x0 + w / 2, y=0.3, text=n, showarrow=False, font=dict(size=11, color="#444"))
        if i < len(steps) - 1:
            fig.add_annotation(x=x0 + w + gap - 0.05, y=0.5, ax=x0 + w + 0.05, ay=0.5, xref="x", yref="y", axref="x",
                               ayref="y", showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=2, arrowcolor="#C55A11",
                               text="")
    total = len(steps) * w + (len(steps) - 1) * gap
    fig.update_xaxes(visible=False, range=[-0.1, total + 0.1])
    fig.update_yaxes(visible=False, range=[-0.1, 1.1])
    fig.update_layout(height=170, margin=dict(l=0, r=0, t=0, b=0), plot_bgcolor="white", paper_bgcolor="white")
    return fig


st.plotly_chart(pipeline_figure(), config={"displayModeBar": False})

try:
    pts = common.leakage_df()
    if pts.empty:
        st.warning("No attack runs yet. Run notebook 05_attacks, then press Refresh data.")
        st.stop()
    rho, p = common.spearman(tuple(pts["gap_acc"]), tuple(pts["auc"]))
    heads = common.headline_ids()
    best = pts.set_index("model_id")
    rows = [{"Config": cid, "Model": mid, "Test accuracy": best.loc[mid, "test_acc"], "Gap (pp)": best.loc[mid, "gap_acc"] * 100,
             "Best attack": best.loc[mid, "attack"], "Best AUC": best.loc[mid, "auc"],
             "TPR @ 0.1% FPR": best.loc[mid, "tpr_at_01pct_fpr"], "Severity": best.loc[mid, "severity"]}
            for cid, mid in heads.items() if mid in best.index]
except Exception as e:
    common.fail(e, "load the headline results")

hi, lo = pts.loc[pts["auc"].idxmax()], pts.loc[pts["auc"].idxmin()]
rank = {"Low": 0, "Moderate": 1, "High": 2, "Critical": 3}
worst = max(rows, key=lambda r: rank.get(r["Severity"], -1)) if rows else None
c1, c2, c3, c4 = st.columns(4)
c1.metric("Highest attack AUC", f"{hi['auc']:.3f}", hi["config_id"], delta_color="off")
c2.metric("Lowest attack AUC", f"{lo['auc']:.3f}", lo["config_id"], delta_color="off")
c3.metric("Spearman ρ, AUC vs gap", f"{rho:.2f}", f"p = {p:.4f}", delta_color="off")
c4.metric("Worst severity", worst["Severity"] if worst else "—", worst["Config"] if worst else None, delta_color="off")

st.subheader("The seven audited configurations")
if rows:
    table = pd.DataFrame(rows)
    st.dataframe(table.style.map(common.severity_style, subset=["Severity"]).format(
        {"Test accuracy": "{:.1%}", "Gap (pp)": "{:.1f}", "Best AUC": "{:.3f}", "TPR @ 0.1% FPR": "{:.2%}"}),
        hide_index=True)
    missing = [c for c, m in heads.items() if m not in best.index]
    if missing:
        st.info(f"Not yet attacked: {', '.join(missing)} (train and attack them with notebooks 02–05).")
else:
    st.info("None of C1–C7 has been attacked yet. Run notebooks 02–05.")
st.caption("Severity uses the lift λ = TPR/FPR at FPR = 0.1%: Low < 2 ≤ Moderate < 10 ≤ High < 50 ≤ Critical.")
