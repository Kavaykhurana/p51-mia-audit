"""Shared helpers for every Streamlit page: page setup, cached DB access, model and image loading."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import NoReturn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from mia import attacks, db  # noqa: E402
from mia.config import DB_PATH, RAW_DIR, SPLIT_PATH, load_experiment  # noqa: E402
from mia.data import load_pool, load_split  # noqa: E402
from mia.metrics import spearman_with_permutation  # noqa: E402
from mia.training import headline_model_id, load_model as _load_model  # noqa: E402

SEVERITY_COLORS = {"Low": "#1E5631", "Moderate": "#B58900", "High": "#C55A11", "Critical": "#B00020"}
FAMILY_COLORS = {"nn": "#1B3A6B", "softmax": "#C55A11", "lda": "#1E5631", "yolo": "#5C2D91"}
TARGET_CONFIGS = ("C1", "C2", "C3", "C4", "C5", "C6", "C7")
RUN_COMMANDS = """python -m venv .venv            # Windows: py -3.11 -m venv .venv
.venv\\Scripts\\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter lab                     # run notebooks/00 ... 05 (then 06-09) in order
streamlit run app/Home.py"""


def setup(page_title: str) -> None:
    """Page config, sidebar with data status, and a hard stop when the results are missing."""
    st.set_page_config(page_title="P51 Membership-Inference Audit", layout="wide")
    sidebar()
    if not DB_PATH.exists() or not SPLIT_PATH.exists():
        st.error("Results not found. Run notebooks 00–05 first.")
        st.code(RUN_COMMANDS, language="bash")
        st.stop()
    st.title(page_title)


def fail(e: Exception, what: str) -> NoReturn:
    """Show a one-line cause and the fix instead of a traceback, then stop the page."""
    st.error(f"Could not {what}: {type(e).__name__}: {e}. Fix: re-run the notebooks that produce this data "
             f"(00–09 in order), then press **Refresh data** in the sidebar.")
    st.stop()


def sidebar() -> None:
    with st.sidebar:
        st.markdown("### P51 · Membership-Inference Audit")
        st.caption("Can an attacker tell whether a specific image was in a model's training set?")
        if DB_PATH.exists():
            try:
                n_models, n_runs = q("SELECT (SELECT COUNT(*) FROM models), (SELECT COUNT(*) FROM attack_runs)").iloc[0]
                st.success(f"Database found · {n_models} models · {n_runs} attack runs")
            except Exception as e:
                st.warning(f"Database found but unreadable ({type(e).__name__}).")
        else:
            st.warning("Database not found (results/mia.sqlite).")
        if st.button("Refresh data"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()


@st.cache_resource
def get_db():
    return db.connect(DB_PATH, check_same_thread=False)


@st.cache_resource
def experiment():
    return load_experiment()


@st.cache_data(ttl=600)
def q(sql: str, params: tuple = ()) -> pd.DataFrame:
    return db.query_df(get_db(), sql, params)


@st.cache_data(ttl=600)
def leakage_df() -> pd.DataFrame:
    return db.leakage_vs_gap_df(get_db()).drop_duplicates("model_id").reset_index(drop=True)


@st.cache_data(ttl=600)
def attack_runs_df() -> pd.DataFrame:
    return db.attack_runs_df(get_db())


@st.cache_data(ttl=600)
def sweep_df(sweep: str) -> pd.DataFrame:
    return db.sweep_df(get_db(), sweep)


@st.cache_data(ttl=600)
def sample_scores(attack_run_id: int) -> pd.DataFrame:
    return db.sample_scores_df(get_db(), attack_run_id)


@st.cache_data(ttl=600)
def spearman(x: tuple, y: tuple) -> tuple[float, float]:
    exp = experiment()
    return spearman_with_permutation(np.array(x), np.array(y), exp.permutations, exp.bootstrap_seed)


@st.cache_resource
def load_pool_images() -> tuple[np.ndarray, np.ndarray]:
    images, labels, _ = load_pool(RAW_DIR)
    return images, labels


@st.cache_resource
def split() -> dict:
    return load_split(SPLIT_PATH)


@st.cache_resource
def load_model(model_id: str):
    row = q("SELECT c.family, m.artifact_path FROM models m JOIN configs c USING (config_id) WHERE m.model_id = ?",
            (model_id,))
    if row.empty:
        raise KeyError(f"model {model_id} is not in the database")
    return row["family"].iloc[0], _load_model(row["family"].iloc[0], row["artifact_path"].iloc[0], model_id)


@st.cache_resource
def attack_context(model_id: str) -> dict:
    """Target audit outputs and stacked shadow outputs, used to score images outside the audit set."""
    _, labels = load_pool_images()
    conn = get_db()
    sids = attacks.shadow_model_ids(conn, experiment(), model_id)
    return {"target": attacks.target_audit_data(model_id, labels),
            "shadow": attacks.shadow_data(sids, labels) if sids else None}


def headline_ids() -> dict:
    exp = experiment()
    return {cid: headline_model_id(exp, cid) for cid in TARGET_CONFIGS}


def severity_style(v) -> str:
    return f"background-color: {SEVERITY_COLORS[v]}; color: white; font-weight: 600" if v in SEVERITY_COLORS else ""
