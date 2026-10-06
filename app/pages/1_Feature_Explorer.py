"""Every preprocessing step and syllabus descriptor on one image."""
import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st

import common
from mia import visual
from mia.data import CLASS_NAMES
from mia.features import BLOCKS, describe
from mia.preprocess import enhance, to_gray

common.setup("Feature Explorer")
st.caption("Pick an image (or upload one) and see what each classical descriptor extracts from it.")

try:
    images, labels = common.load_pool_images()
    parts = common.split()
except Exception as e:
    common.fail(e, "load the CIFAR-10 pool and split")

col_a, col_b = st.columns([1, 2])
partition = col_a.selectbox("Partition", ["member", "nonmember", "val"])
pool = parts[partition]
if "fe_pos" not in st.session_state:
    st.session_state.fe_pos = 0
if col_a.button("Random image"):
    st.session_state.fe_pos = int(np.random.default_rng().integers(len(pool)))
pos = col_b.slider("Image within the partition", 0, len(pool) - 1, key="fe_pos")
uploaded = st.file_uploader("…or upload an image (png, jpg, jpeg; max 5 MB)", type=["png", "jpg", "jpeg"])

img, caption = images[pool[pos]], f"Pool index {int(pool[pos])} · {CLASS_NAMES[labels[pool[pos]]]} · {partition}"
if uploaded is not None:
    if uploaded.size > 5 * 1024 * 1024:
        st.error("That file is larger than 5 MB.")
        st.stop()
    decoded = cv2.imdecode(np.frombuffer(uploaded.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        st.error("Could not read that file as an image.")
        st.stop()
    img, caption = cv2.resize(decoded, (32, 32), interpolation=cv2.INTER_AREA), f"Uploaded: {uploaded.name}"
    st.info(f"Your image ({decoded.shape[1]}×{decoded.shape[0]}) was resized to 32×32 with INTER_AREA, the CIFAR-10 size the models use.")

enh = enhance(img)
gray = to_gray(enh)
rgb = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # noqa: E731
st.markdown(f"**{caption}**")
c1, c2, c3 = st.columns(3)
c1.image(visual.upscale(rgb(img)), caption="Original (×8, nearest)")
c2.image(visual.upscale(rgb(enh)), caption="Enhanced: CLAHE on Y + Gaussian σ=0.5")
c3.image(visual.upscale(gray), caption="Grayscale (input to HOG, LBP, Gabor, DWT)")

tabs = st.tabs(["HOG glyphs", "LBP map", "Gabor 2×4", "DWT mosaic", "FFT spectrum", "Edges", "Colour histogram"])
tabs[0].image(visual.upscale(visual.hog_glyphs(gray), 2), caption="9 orientation bins per 8×8 cell (324-d HOG)")
tabs[1].image(visual.upscale(visual.lbp_code_map(gray)), caption="Uniform LBP codes 0–9 (P=8, R=1), histogrammed on a 2×2 grid")
tabs[2].image(visual.upscale(visual.gabor_grid(gray), 4), caption="|response|; rows λ = 4, 8 px; columns θ = 0°, 45°, 90°, 135°")
tabs[3].image(visual.upscale(visual.dwt_subbands(gray)), caption="2-level Haar DWT: level-2 bands top-left, level-1 bands around them")
tabs[4].image(visual.upscale(visual.fft_spectrum(gray)), caption="log(1 + |FFT|), zero frequency at the centre (visualisation only)")
e1, e2, e3 = tabs[5].columns(3)
e1.image(visual.upscale(visual.canny(gray)), caption="Canny (50, 150)")
e2.image(visual.upscale(visual.log_edges(gray)), caption="Laplacian of Gaussian (σ=1)")
e3.image(visual.upscale(visual.dog(gray)), caption="Difference of Gaussians (σ 1 − σ 2)")
hist = go.Figure()
for name, h in visual.hsv_hist_data(enh).items():
    hist.add_bar(x=[f"{name[0]}{i}" for i in range(len(h))], y=h, name=name)
hist.update_layout(height=320, yaxis_title="Fraction of pixels", xaxis_title="Bin", margin=dict(t=20))
tabs[6].plotly_chart(hist)

st.subheader("The 426-dimensional descriptor")
vec = describe(img)
fig = go.Figure(go.Scatter(y=vec, mode="lines", line=dict(color="#1B3A6B", width=1)))
start = 0
for name, n in BLOCKS:
    fig.add_vrect(x0=start - 0.5, x1=start + n - 0.5, fillcolor="#C55A11" if len(fig.layout.shapes) % 2 else "#1B3A6B",
                  opacity=0.06, line_width=0)
    fig.add_annotation(x=start + n / 2, y=1.04, yref="paper", text=f"{name} ({n})", showarrow=False)
    start += n
fig.update_layout(height=300, xaxis_title="Dimension", yaxis_title="Value", margin=dict(t=40))
st.plotly_chart(fig)
