import sqlite3

import pytest

from mia import db


@pytest.fixture
def conn():
    c = db.connect(":memory:")
    with c:
        db.insert_config(c, "C2", "softmax", "test", "{}")
        db.insert_sample_rows(c, [(0, 3, "cifar_train", "member")])
    return c


def _model(**kw):
    row = dict(model_id="m", config_id="C2", role="target", shadow_index=None, seed=0, epoch=1, artifact_path="x",
               n_train=10, train_acc=0.9, test_acc=0.6, train_loss=0.2, test_loss=0.8, n_params=1, model_bytes=1,
               train_seconds=0.1, infer_ms_per_img=0.1)
    return {**row, **kw}


def test_schema_creates(conn):
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
    assert {"samples", "configs", "models", "shadow_membership", "attack_runs", "sample_scores", "sweep_points",
            "v_leakage_vs_gap"} <= names


def test_bad_label_rejected(conn):
    with pytest.raises(sqlite3.IntegrityError):
        db.insert_sample_rows(conn, [(1, 10, "cifar_train", "member")])


def test_shadow_without_index_rejected(conn):
    with pytest.raises(sqlite3.IntegrityError):
        db.insert_model(conn, _model(model_id="s", role="shadow", shadow_index=None))


def test_generated_gap(conn):
    db.insert_model(conn, _model())
    gap, ratio = conn.execute("SELECT gap_acc, loss_ratio FROM models WHERE model_id='m'").fetchone()
    assert gap == pytest.approx(0.9 - 0.6) and ratio == pytest.approx(4.0)
    assert db.model_exists(conn, "m") and not db.model_exists(conn, "nope")
