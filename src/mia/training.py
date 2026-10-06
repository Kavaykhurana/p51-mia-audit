"""Train target and shadow models for any config, evaluate every checkpoint, and record it in the DB."""
from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from . import db
from .classifiers import LDAClassifier, SoftmaxDiscriminant, SoftNN, YoloClassifier
from .config import (AUG_DIR, DB_PATH, ERROR_LOG, FEATURES_PATH, MODELS_DIR, PROBS_DIR, RAW_DIR, ROOT, SPLIT_PATH,
                     YOLO_DS_DIR, Experiment, ModelConfig, load_experiment)
from .data import export_yolo_folder, load_pool, load_split
from .features import augmented_features, describe_batch
from .metrics import accuracy, mean_ce_loss


@dataclass
class Workspace:
    """Everything a training or attack notebook needs, loaded once."""
    exp: Experiment
    conn: object
    images: np.ndarray
    labels: np.ndarray
    split: dict
    features: np.ndarray | None

    @classmethod
    def open(cls, need_features: bool = True) -> Workspace:
        for p, nb in ((SPLIT_PATH, "00_setup_and_split"), (FEATURES_PATH, "01_preprocessing_and_features")):
            if (p is SPLIT_PATH or need_features) and not p.exists():
                raise FileNotFoundError(f"{p} not found; run notebooks/{nb}.ipynb first")
        images, labels, _ = load_pool(RAW_DIR)
        feats = np.load(FEATURES_PATH) if need_features else None
        return cls(load_experiment(), db.connect(DB_PATH), images, labels, load_split(SPLIT_PATH), feats)

    def shadow_set(self, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.split[f"shadow_{k}_in"], self.split[f"shadow_{k}_out"], self.split[f"shadow_{k}_val"]


def base_model_id(config_id: str, seed: int, shadow_index: int | None = None) -> str:
    role = "target" if shadow_index is None else f"shadow{shadow_index}"
    return f"{config_id}_{role}_s{seed}"


def model_ids(mcfg: ModelConfig, seed: int, shadow_index: int | None = None) -> list[str]:
    base = base_model_id(mcfg.config_id, seed, shadow_index)
    return [f"{base}_e{e}" for e in mcfg.checkpoint_epochs] if mcfg.has_epoch_suffix else [base]


def headline_model_id(exp: Experiment, config_id: str) -> str:
    """The representative target of a config: first seed, final checkpoint (e.g. C2_target_s0_e200)."""
    mcfg = exp.config(config_id)
    return model_ids(mcfg, exp.seeds_for(mcfg)[0])[-1]


def save_probs(model_id: str, P_in, P_out, idx_in, idx_out) -> None:
    PROBS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(PROBS_DIR / f"{model_id}.npz", P_in=P_in, P_out=P_out, idx_in=idx_in, idx_out=idx_out)


def load_probs(model_id: str) -> dict[str, np.ndarray]:
    """Probabilities a model assigns to its IN (members / shadow-IN) and OUT (non-members / shadow-OUT) images."""
    path = PROBS_DIR / f"{model_id}.npz"
    if not path.exists():
        raise FileNotFoundError(f"probability cache for model {model_id} not found at {path}; retrain it")
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def load_model(family: str, artifact_path: str | Path, model_id: str = ""):
    path = Path(artifact_path)
    path = path if path.is_absolute() else ROOT / path  # the DB stores repo-relative paths
    if not path.exists():
        raise FileNotFoundError(f"checkpoint for model {model_id} not found: {path}")
    return {"nn": SoftNN, "softmax": SoftmaxDiscriminant, "lda": LDAClassifier, "yolo": YoloClassifier}[family].load(path)


def predict_images(model, family: str, images: np.ndarray) -> np.ndarray:
    """Raw BGR images -> class probabilities (classical models go through the 426-d descriptor)."""
    return model.predict_proba(images) if family == "yolo" else model.predict_proba(describe_batch(images))


def _infer_ms(model, family: str, images: np.ndarray, batch: int = 100) -> float:
    """Mean ms/image over 1000 images (full pipeline from pixels), after 3 untimed warm-up batches."""
    imgs = images[:1000]
    for s in range(0, 3 * batch, batch):
        predict_images(model, family, imgs[s:s + batch])
    t0 = time.perf_counter()
    for s in range(0, len(imgs), batch):
        predict_images(model, family, imgs[s:s + batch])
    return (time.perf_counter() - t0) * 1000 / len(imgs)


def _fit(ws: Workspace, mcfg: ModelConfig, base_id: str, seed: int, shadow_index, train_idx, val_idx):
    """Fit one run. Returns (n_train, [(epoch_or_None, artifact_path, train_seconds), ...])."""
    fam, y = mcfg.family, ws.labels[train_idx]
    classical_dir = MODELS_DIR / "classical"
    classical_dir.mkdir(parents=True, exist_ok=True)
    if fam == "yolo":
        ds = export_yolo_folder(ws.images, ws.labels, train_idx, val_idx, YOLO_DS_DIR / base_id)
        saved = YoloClassifier().fit(ds, model_id=base_id, epochs=mcfg.epochs, seed=seed,
                                     train_kwargs=mcfg.yolo_train_kwargs, checkpoint_epochs=mcfg.checkpoint_epochs,
                                     project_dir=MODELS_DIR / "yolo")
        return len(train_idx), [(e, p, s) for e, (p, s) in sorted(saved.items())]
    X = ws.features[train_idx]
    t0 = time.perf_counter()
    if fam == "nn":
        path = classical_dir / f"{base_id}.npz"
        SoftNN().fit(X, y, X_val=ws.features[val_idx]).save(path)
        return len(X), [(None, path, time.perf_counter() - t0)]
    if fam == "lda":
        path = classical_dir / f"{base_id}.npz"
        LDAClassifier(pca=mcfg.pca, seed=seed).fit(X, y).save(path)
        return len(X), [(None, path, time.perf_counter() - t0)]
    if mcfg.aug_copies:
        aug_seed = seed if shadow_index is None else seed + 10_000 * (shadow_index + 1)
        A = augmented_features(ws.images, train_idx, mcfg.aug_copies, aug_seed, AUG_DIR / f"{base_id}.npy")
        X, y = np.vstack([X, A]), np.concatenate([y, np.repeat(y, mcfg.aug_copies)])
    model = SoftmaxDiscriminant(pca=mcfg.pca, l2=mcfg.l2, epsilon=mcfg.label_smoothing, epochs=mcfg.epochs, seed=seed)
    saved = model.fit(X, y, checkpoint_epochs=mcfg.checkpoint_epochs, checkpoint_dir=classical_dir / base_id)
    return len(X), [(e, p, s) for e, (p, s) in sorted(saved.items()) if e in mcfg.checkpoint_epochs]


def ensure_config(conn, mcfg: ModelConfig) -> None:
    db.insert_config(conn, mcfg.config_id, mcfg.family, mcfg.description, json.dumps(mcfg.params), mcfg.base_config_id)


def train_model(ws: Workspace, config_id: str, seed: int, shadow_index: int | None = None) -> list[str]:
    """Train a target (shadow_index None) or shadow k; insert one models row per checkpoint. Skips existing ids."""
    exp, conn = ws.exp, ws.conn
    mcfg = exp.config(config_id)
    ids = model_ids(mcfg, seed, shadow_index)
    if all(db.model_exists(conn, m) for m in ids):
        return []
    if shadow_index is None:
        train_idx, out_idx, val_idx = ws.split["member"], ws.split["nonmember"], ws.split["val"]
    else:
        train_idx, out_idx, val_idx = ws.shadow_set(shadow_index)
    base_id = base_model_id(config_id, seed, shadow_index)
    n_train, checkpoints = _fit(ws, mcfg, base_id, seed, shadow_index, train_idx, val_idx)

    done = []
    for epoch, path, secs in checkpoints:
        mid = f"{base_id}_e{epoch}" if mcfg.has_epoch_suffix else base_id
        if db.model_exists(conn, mid):
            continue
        model = load_model(mcfg.family, path, mid)
        if mcfg.family == "yolo":
            P_in, P_out = model.predict_proba(ws.images[train_idx]), model.predict_proba(ws.images[out_idx])
        else:
            P_in, P_out = model.predict_proba(ws.features[train_idx]), model.predict_proba(ws.features[out_idx])
        save_probs(mid, P_in, P_out, train_idx, out_idx)
        y_in, y_out = ws.labels[train_idx], ws.labels[out_idx]
        row = dict(model_id=mid, config_id=config_id, role="target" if shadow_index is None else "shadow",
                   shadow_index=shadow_index, seed=seed, epoch=epoch, artifact_path=Path(path).relative_to(ROOT).as_posix(),
                   n_train=n_train, train_acc=accuracy(P_in, y_in), test_acc=accuracy(P_out, y_out),
                   train_loss=mean_ce_loss(P_in, y_in), test_loss=mean_ce_loss(P_out, y_out), n_params=model.n_params,
                   model_bytes=os.path.getsize(path), train_seconds=secs,
                   infer_ms_per_img=_infer_ms(model, mcfg.family, ws.images[out_idx]))
        with conn:
            if mcfg.base_config_id:
                ensure_config(conn, exp.configs[mcfg.base_config_id])
            ensure_config(conn, mcfg)
            db.insert_model(conn, row)
            if shadow_index is not None:
                db.insert_shadow_membership(conn, mid, train_idx, out_idx)
        done.append(mid)
    if shadow_index is None:
        register_sweep_points(conn, exp)
    return done


def run_jobs(ws: Workspace, jobs: list[tuple[str, int, int | None]]) -> dict:
    """Train every (config_id, seed, shadow_index) job; log failures to results/train_errors.log and continue."""
    summary = {"trained": 0, "skipped": 0, "failed": 0}
    for i, (cid, seed, k) in enumerate(jobs, 1):
        name = base_model_id(cid, seed, k)
        t0 = time.perf_counter()
        try:
            new = train_model(ws, cid, seed, k)
            summary["trained" if new else "skipped"] += 1
            status = f"trained {len(new)} checkpoint(s)" if new else "exists, skipped"
        except Exception as e:  # one failing model must not stop the batch
            summary["failed"] += 1
            ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
            with open(ERROR_LOG, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now().isoformat(timespec='seconds')}] {name}: {e!r}\n{traceback.format_exc()}\n")
            status = f"FAILED ({type(e).__name__}: {e}); see {ERROR_LOG.name}"
        print(f"[{i:>3}/{len(jobs)}] {name:<28} {status}  ({time.perf_counter() - t0:.1f} s)")
    print(f"Summary: {summary}")
    return summary


def register_sweep_points(conn, exp: Experiment) -> None:
    """Idempotently register early-stopping and label-smoothing sweep points for every existing target."""
    with conn:
        for base, epochs in exp.early_stopping.items():
            q = f"SELECT model_id, epoch FROM models WHERE config_id = ? AND role = 'target' AND epoch IN ({','.join('?' * len(epochs))})"
            for mid, epoch in conn.execute(q, (base, *epochs)):
                db.insert_sweep_point(conn, "early_stopping", base, mid, epoch)
        for base in exp.ls_bases:
            final = exp.configs[base].epochs
            for (mid,) in conn.execute("SELECT model_id FROM models WHERE config_id = ? AND role = 'target' "
                                       "AND seed = 0 AND epoch = ?", (base, final)):
                db.insert_sweep_point(conn, "label_smoothing", base, mid, 0.0)
            for eps in exp.ls_epsilons:
                if eps > 0:
                    for (mid,) in conn.execute("SELECT model_id FROM models WHERE config_id = ? AND role = 'target' "
                                               "AND seed = 0", (exp.ls_config(base, eps).config_id,)):
                        db.insert_sweep_point(conn, "label_smoothing", base, mid, eps)
