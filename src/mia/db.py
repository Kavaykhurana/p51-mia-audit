"""SQLite experiment store: schema, typed inserts and DataFrame queries.

Insert helpers never commit on their own; callers wrap related writes in `with conn:` so a crash
never leaves half a model's rows behind.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .config import DB_PATH

SCHEMA = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS samples (
    idx            INTEGER PRIMARY KEY CHECK (idx BETWEEN 0 AND 59999),
    label          INTEGER NOT NULL CHECK (label BETWEEN 0 AND 9),
    source_split   TEXT    NOT NULL CHECK (source_split IN ('cifar_train', 'cifar_test')),
    partition      TEXT    NOT NULL CHECK (partition IN ('member', 'nonmember', 'val', 'shadow'))
);
CREATE INDEX IF NOT EXISTS ix_samples_partition ON samples(partition);
CREATE INDEX IF NOT EXISTS ix_samples_label     ON samples(label);

CREATE TABLE IF NOT EXISTS configs (
    config_id      TEXT PRIMARY KEY,
    family         TEXT NOT NULL CHECK (family IN ('nn', 'softmax', 'lda', 'yolo')),
    base_config_id TEXT REFERENCES configs(config_id),
    description    TEXT NOT NULL,
    params_json    TEXT NOT NULL,
    created_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS models (
    model_id          TEXT PRIMARY KEY,
    config_id         TEXT NOT NULL REFERENCES configs(config_id),
    role              TEXT NOT NULL CHECK (role IN ('target', 'shadow')),
    shadow_index      INTEGER CHECK (shadow_index IS NULL OR shadow_index >= 0),
    seed              INTEGER NOT NULL,
    epoch             INTEGER,
    artifact_path     TEXT NOT NULL,
    n_train           INTEGER NOT NULL CHECK (n_train > 0),
    train_acc         REAL CHECK (train_acc BETWEEN 0 AND 1),
    test_acc          REAL CHECK (test_acc  BETWEEN 0 AND 1),
    train_loss        REAL CHECK (train_loss >= 0),
    test_loss         REAL CHECK (test_loss  >= 0),
    gap_acc           REAL GENERATED ALWAYS AS (train_acc - test_acc) VIRTUAL,
    loss_ratio        REAL GENERATED ALWAYS AS (test_loss / NULLIF(train_loss, 0)) VIRTUAL,
    n_params          INTEGER CHECK (n_params >= 0),
    model_bytes       INTEGER CHECK (model_bytes >= 0),
    train_seconds     REAL CHECK (train_seconds >= 0),
    infer_ms_per_img  REAL CHECK (infer_ms_per_img >= 0),
    created_at        TEXT NOT NULL DEFAULT (datetime('now')),
    CHECK ((role = 'target' AND shadow_index IS NULL) OR (role = 'shadow' AND shadow_index IS NOT NULL))
);
CREATE INDEX IF NOT EXISTS ix_models_config_role ON models(config_id, role);

CREATE TABLE IF NOT EXISTS shadow_membership (
    model_id    TEXT    NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,
    sample_idx  INTEGER NOT NULL REFERENCES samples(idx),
    is_in       INTEGER NOT NULL CHECK (is_in IN (0, 1)),
    PRIMARY KEY (model_id, sample_idx)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS attack_runs (
    attack_run_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    target_model_id   TEXT NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,
    attack            TEXT NOT NULL CHECK (attack IN ('shokri_lda', 'shokri_lda_perclass', 'loss', 'confidence', 'mentr', 'kmeans', 'fcm')),
    shadow_config_id  TEXT REFERENCES configs(config_id),
    n_shadows         INTEGER NOT NULL DEFAULT 0 CHECK (n_shadows >= 0),
    auc               REAL CHECK (auc BETWEEN 0 AND 1),
    auc_ci_low        REAL,
    auc_ci_high       REAL,
    advantage         REAL CHECK (advantage BETWEEN -1 AND 1),
    tpr_at_1pct_fpr   REAL CHECK (tpr_at_1pct_fpr  BETWEEN 0 AND 1),
    tpr_at_01pct_fpr  REAL CHECK (tpr_at_01pct_fpr BETWEEN 0 AND 1),
    lift_at_01pct_fpr REAL CHECK (lift_at_01pct_fpr >= 0),
    severity          TEXT CHECK (severity IN ('Low', 'Moderate', 'High', 'Critical')),
    threshold_1pct    REAL,
    created_at        TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (target_model_id, attack, shadow_config_id, n_shadows)
);
CREATE INDEX IF NOT EXISTS ix_attack_runs_attack ON attack_runs(attack);

CREATE TABLE IF NOT EXISTS sample_scores (
    attack_run_id  INTEGER NOT NULL REFERENCES attack_runs(attack_run_id) ON DELETE CASCADE,
    sample_idx     INTEGER NOT NULL REFERENCES samples(idx),
    is_member      INTEGER NOT NULL CHECK (is_member IN (0, 1)),
    score          REAL    NOT NULL,
    PRIMARY KEY (attack_run_id, sample_idx)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS sweep_points (
    sweep           TEXT NOT NULL CHECK (sweep IN ('early_stopping', 'label_smoothing')),
    base_config_id  TEXT NOT NULL REFERENCES configs(config_id),
    model_id        TEXT NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,
    x_value         REAL NOT NULL,
    PRIMARY KEY (sweep, model_id)
);

CREATE VIEW IF NOT EXISTS v_leakage_vs_gap AS
SELECT m.model_id, m.config_id, c.family, m.seed, m.epoch,
       m.train_acc, m.test_acc, m.gap_acc, m.loss_ratio,
       m.n_params, m.infer_ms_per_img,
       a.attack, a.auc, a.auc_ci_low, a.auc_ci_high, a.advantage,
       a.tpr_at_1pct_fpr, a.tpr_at_01pct_fpr, a.lift_at_01pct_fpr, a.severity
FROM models m
JOIN configs c      ON c.config_id = m.config_id
JOIN attack_runs a  ON a.target_model_id = m.model_id
WHERE m.role = 'target'
  AND a.auc = (SELECT MAX(a2.auc) FROM attack_runs a2 WHERE a2.target_model_id = m.model_id);
"""

MODEL_COLUMNS = ("model_id", "config_id", "role", "shadow_index", "seed", "epoch", "artifact_path", "n_train",
                 "train_acc", "test_acc", "train_loss", "test_loss", "n_params", "model_bytes", "train_seconds",
                 "infer_ms_per_img")
ATTACK_COLUMNS = ("target_model_id", "attack", "shadow_config_id", "n_shadows", "auc", "auc_ci_low", "auc_ci_high",
                  "advantage", "tpr_at_1pct_fpr", "tpr_at_01pct_fpr", "lift_at_01pct_fpr", "severity", "threshold_1pct")


def connect(path: Path | str = DB_PATH, check_same_thread: bool = True) -> sqlite3.Connection:
    """Open (and create if needed) the experiment DB with foreign keys on and WAL journaling."""
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=check_same_thread)
    conn.executescript(SCHEMA)
    return conn


def insert_sample_rows(conn, rows) -> None:
    """rows: iterable of (idx, label, source_split, partition)."""
    conn.executemany("INSERT INTO samples(idx, label, source_split, partition) VALUES (?,?,?,?)", rows)


def insert_config(conn, config_id, family, description, params_json, base_config_id=None) -> None:
    conn.execute("INSERT OR IGNORE INTO configs(config_id, family, base_config_id, description, params_json) "
                 "VALUES (?,?,?,?,?)", (config_id, family, base_config_id, description, params_json))


def insert_model(conn, row: dict) -> None:
    conn.execute(f"INSERT INTO models({', '.join(MODEL_COLUMNS)}) VALUES ({', '.join('?' * len(MODEL_COLUMNS))})",
                 tuple(row[c] for c in MODEL_COLUMNS))


def insert_shadow_membership(conn, model_id, in_idx, out_idx) -> None:
    rows = [(model_id, int(i), 1) for i in in_idx] + [(model_id, int(i), 0) for i in out_idx]
    conn.executemany("INSERT INTO shadow_membership(model_id, sample_idx, is_in) VALUES (?,?,?)", rows)


def insert_attack_run(conn, row: dict) -> int:
    cur = conn.execute(f"INSERT INTO attack_runs({', '.join(ATTACK_COLUMNS)}) "
                       f"VALUES ({', '.join('?' * len(ATTACK_COLUMNS))})", tuple(row[c] for c in ATTACK_COLUMNS))
    return int(cur.lastrowid)


def insert_sample_scores(conn, attack_run_id, sample_idx, is_member, scores) -> None:
    conn.executemany("INSERT INTO sample_scores(attack_run_id, sample_idx, is_member, score) VALUES (?,?,?,?)",
                     zip([attack_run_id] * len(sample_idx), map(int, sample_idx), map(int, is_member), map(float, scores)))


def insert_sweep_point(conn, sweep, base_config_id, model_id, x_value) -> None:
    conn.execute("INSERT OR IGNORE INTO sweep_points(sweep, base_config_id, model_id, x_value) VALUES (?,?,?,?)",
                 (sweep, base_config_id, model_id, float(x_value)))


def model_exists(conn, model_id) -> bool:
    return conn.execute("SELECT 1 FROM models WHERE model_id = ?", (model_id,)).fetchone() is not None


def attack_run_exists(conn, target_model_id, attack, shadow_config_id, n_shadows) -> bool:
    # `IS` (not `=`) so a NULL shadow_config_id matches; SQLite's UNIQUE treats NULLs as distinct.
    return conn.execute("SELECT 1 FROM attack_runs WHERE target_model_id = ? AND attack = ? "
                        "AND shadow_config_id IS ? AND n_shadows = ?",
                        (target_model_id, attack, shadow_config_id, n_shadows)).fetchone() is not None


def query_df(conn, sql: str, params=()) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)


def models_df(conn, role: str | None = None) -> pd.DataFrame:
    sql = "SELECT m.*, c.family, c.base_config_id FROM models m JOIN configs c USING (config_id)"
    return query_df(conn, sql + (" WHERE m.role = ?" if role else "") + " ORDER BY m.model_id", (role,) if role else ())


def attack_runs_df(conn) -> pd.DataFrame:
    """Every attack run on a target, joined with that target's accuracy, gap and cost columns."""
    return query_df(conn, """
        SELECT a.*, m.config_id, c.family, c.base_config_id, m.seed, m.epoch, m.train_acc, m.test_acc, m.gap_acc,
               m.loss_ratio, m.n_params, m.model_bytes, m.infer_ms_per_img
        FROM attack_runs a JOIN models m ON m.model_id = a.target_model_id JOIN configs c ON c.config_id = m.config_id
        WHERE m.role = 'target' ORDER BY a.target_model_id, a.attack""")


def leakage_vs_gap_df(conn) -> pd.DataFrame:
    return query_df(conn, "SELECT * FROM v_leakage_vs_gap ORDER BY model_id")


def sweep_df(conn, sweep: str) -> pd.DataFrame:
    """Sweep points joined with the best attack (by AUC) of each point's model."""
    return query_df(conn, """
        SELECT s.sweep, s.base_config_id, s.x_value, v.*
        FROM sweep_points s JOIN v_leakage_vs_gap v ON v.model_id = s.model_id
        WHERE s.sweep = ? ORDER BY s.base_config_id, s.x_value, v.seed""", (sweep,))


def sample_scores_df(conn, attack_run_id: int) -> pd.DataFrame:
    return query_df(conn, "SELECT sample_idx, is_member, score FROM sample_scores WHERE attack_run_id = ?",
                    (int(attack_run_id),))
