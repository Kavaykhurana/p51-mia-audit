"""Pick (or upload) one image and watch the attack decide whether it was in the training set."""
import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st

import common
from mia.attacks import ATTACKS, score_attack
from mia.data import CLASS_NAMES
from mia.training import predict_images

common.setup("Live audit")

try:
    runs = common.attack_runs_df()
    images, labels = common.load_pool_images()
    parts = common.split()
except Exception as e:
    common.fail(e, "load attack runs and images")
if runs.empty:
    st.warning("No attack runs yet. Run notebook 05_attacks.")
    st.stop()

targets = sorted(runs["target_model_id"].unique())
default = common.headline_ids()["C2"]
c1, c2 = st.columns(2)
model_id = c1.selectbox("Target model", targets, index=targets.index(default) if default in targets else 0)
available = [a for a in ATTACKS if a in set(runs.loc[runs["target_model_id"] == model_id, "attack"])]
attack = c2.selectbox("Attack", available, index=available.index("shokri_lda") if "shokri_lda" in available else 0)
run = runs[(runs["target_model_id"] == model_id) & (runs["attack"] == attack)].sort_values("attack_run_id").iloc[-1]
threshold = 0.5 if attack == "kmeans" else run["threshold_1pct"]

members, nonmembers = parts["member"], parts["nonmember"]
if "audit_idx" not in st.session_state:
    st.session_state.update(audit_idx=int(members[0]), pick_seed=0)


def pick(pool):
    st.session_state.audit_idx = int(np.random.default_rng(st.session_state.pick_seed).choice(pool))
    st.session_state.last_seed = st.session_state.pick_seed
    st.session_state.pick_seed += 1


b1, b2, b3, b4 = st.columns([1, 1, 1, 2])
b1.number_input("Seed", min_value=0, step=1, key="pick_seed")
b2.button("Pick a member", on_click=pick, args=(members,))
b3.button("Pick a non-member", on_click=pick, args=(nonmembers,))
b4.number_input("…or a pool index from the audit set", 0, 59999, key="audit_idx")
if "last_seed" in st.session_state:
    st.caption(f"Last random pick used seed {st.session_state.last_seed}.")
uploaded = st.file_uploader("…or upload an image (png, jpg, jpeg; max 5 MB)", type=["png", "jpg", "jpeg"])

try:
    family, model = common.load_model(model_id)
    if uploaded is not None:
        if uploaded.size > 5 * 1024 * 1024:
            st.error("That file is larger than 5 MB.")
            st.stop()
        decoded = cv2.imdecode(np.frombuffer(uploaded.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            st.error("Could not read that file as an image.")
            st.stop()
        img = cv2.resize(decoded, (32, 32), interpolation=cv2.INTER_AREA)
        P = predict_images(model, family, img[None])
        y = st.selectbox("True class of the uploaded image (attacks use p of the true class)", range(10),
                         index=int(P.argmax()), format_func=lambda c: CLASS_NAMES[c])
        ctx = common.attack_context(model_id)
        t = ctx["target"]
        score = float(score_attack(attack, P, np.array([y]), t["P"], t["y"], ctx["shadow"])[0])
        truth = None
    else:
        idx = int(st.session_state.audit_idx)
        if idx not in set(members) | set(nonmembers):
            st.warning("That index is not in the audit set (members ∪ non-members). Pick another one.")
            st.stop()
        img, y = images[idx], int(labels[idx])
        P = predict_images(model, family, img[None])
        row = common.q("SELECT score, is_member FROM sample_scores WHERE attack_run_id = ? AND sample_idx = ?",
                       (int(run["attack_run_id"]), idx))
        score, truth = float(row["score"].iloc[0]), bool(row["is_member"].iloc[0])
except Exception as e:
    common.fail(e, "score this image")

flagged = score >= threshold
left, mid, right = st.columns([1, 2, 2])
left.image(cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST),
           caption=f"Class: {CLASS_NAMES[y]}")
bars = go.Figure(go.Bar(x=list(CLASS_NAMES), y=P[0], marker_color=["#C55A11" if c == y else "#1B3A6B" for c in range(10)]))
bars.update_layout(height=300, yaxis_title="Model probability", yaxis_range=[0, 1], margin=dict(t=30),
                   title=f"{model_id} output (true class in orange)")
mid.plotly_chart(bars)

with right:
    st.metric("Attack score", f"{score:.4f}", f"threshold at 1% FPR: {threshold:.4f}", delta_color="off")
    (st.error if flagged else st.success)("Flagged as MEMBER" if flagged else "Not flagged")
    if truth is None:
        st.info("Ground truth unknown: this image is outside the audit set.")
    else:
        correct = flagged == truth
        st.markdown(f"Ground truth: **{'member' if truth else 'non-member'}** "
                    f"{'✓ the attack is right' if correct else '✗ the attack is wrong'}")
    if attack == "kmeans":
        st.caption("K-means outputs a hard cluster label (1 = member cluster), so the threshold is 0.5.")

try:
    s = common.sample_scores(int(run["attack_run_id"]))
except Exception as e:
    common.fail(e, "load the score distribution")
hist = go.Figure()
for is_m, name, color in ((1, "Members", "#C55A11"), (0, "Non-members", "#1B3A6B")):
    hist.add_histogram(x=s.loc[s["is_member"] == is_m, "score"], name=name, marker_color=color, opacity=0.6, nbinsx=80)
hist.add_vline(x=score, line_color="black", line_width=2, annotation_text="this image")
hist.add_vline(x=threshold, line_dash="dash", line_color="#B00020", annotation_text="1% FPR threshold",
               annotation_position="bottom right")
auc_note = "" if np.isnan(run["auc"]) else f" (AUC {run['auc']:.3f})"
hist.update_layout(barmode="overlay", height=340, xaxis_title=f"{attack} score (higher = more likely member)",
                   yaxis_title="Images", title=f"Score distribution for this run{auc_note}")
st.plotly_chart(hist)
