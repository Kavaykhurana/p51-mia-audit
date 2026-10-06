"""CIFAR-10 download and parsing, the stratified member/non-member split, shadow sampling, YOLO folder export."""
from __future__ import annotations

import pickle
import tarfile
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .config import Experiment

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CLASS_NAMES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
PARTITIONS = ("member", "nonmember", "val", "shadow")


def _archive_ok(path: Path) -> bool:
    try:
        with tarfile.open(path, "r:gz") as tf:
            return any(n.endswith("test_batch") for n in tf.getnames())
    except (tarfile.TarError, OSError, EOFError):
        return False


def download_cifar10(raw_dir: Path) -> Path:
    """Download (with retries) and extract CIFAR-10; returns the cifar-10-batches-py directory."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive, out = raw_dir / "cifar-10-python.tar.gz", raw_dir / "cifar-10-batches-py"
    if archive.exists() and not _archive_ok(archive):
        archive.unlink()  # corrupt archive: delete and re-download once
    if not archive.exists():
        for attempt in range(3):
            try:
                with urllib.request.urlopen(CIFAR_URL, timeout=60) as r, open(archive, "wb") as f:
                    total, done, step = int(r.headers.get("Content-Length") or 0), 0, 1
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        if total and done * 10 >= total * step:
                            print(f"  CIFAR-10 download {done / 2**20:6.1f} / {total / 2**20:.1f} MB")
                            step += 1
                if _archive_ok(archive):
                    break
                archive.unlink()
            except OSError:
                archive.unlink(missing_ok=True)
            if attempt < 2:
                time.sleep(5)
        else:
            raise RuntimeError(f"CIFAR-10 download failed after 3 attempts. Download {CIFAR_URL} manually "
                               f"and place it at {archive}")
    if not (out / "test_batch").exists():
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(raw_dir, filter="data")
    return out


def load_pool(raw_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """60,000-image pool: (uint8 BGR images (60000,32,32,3), int64 labels, source_split strings).

    Index 0-49,999 = data_batch_1..5 in order, 50,000-59,999 = test_batch. This is the one place where
    CIFAR's RGB is converted to OpenCV's BGR.
    """
    batch_dir = Path(raw_dir) / "cifar-10-batches-py"
    imgs, labels = [], []
    for name in [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"]:
        # CIFAR-10 is only distributed as Python pickles; these come from the official archive above.
        with open(batch_dir / name, "rb") as f:
            d = pickle.load(f, encoding="bytes")
        imgs.append(d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
        labels.append(np.asarray(d[b"labels"], dtype=np.int64))
    rgb = np.ascontiguousarray(np.concatenate(imgs))
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    source = np.array(["cifar_train"] * 50000 + ["cifar_test"] * 10000)
    return bgr, np.concatenate(labels), source


def make_split(labels: np.ndarray, exp: Experiment) -> dict[str, np.ndarray]:
    """Per class: member / nonmember / val / shadow, drawn from one permutation within the class."""
    rng = np.random.default_rng(exp.split_seed)
    n = exp.per_class
    out = {p: [] for p in PARTITIONS}
    for c in range(10):
        idx = rng.permutation(np.flatnonzero(labels == c))
        bounds = np.cumsum([0, n["member"], n["nonmember"], n["val"], n["shadow"]])
        for p, lo, hi in zip(PARTITIONS, bounds[:-1], bounds[1:]):
            out[p].append(idx[lo:hi])
    split = {p: np.sort(np.concatenate(v)) for p, v in out.items()}
    for p in PARTITIONS:
        assert len(split[p]) == 10 * n[p], f"partition {p} has {len(split[p])} images"
    all_idx = np.concatenate(list(split.values()))
    assert len(np.unique(all_idx)) == len(all_idx) == 60000, "partitions must be disjoint and cover the pool"
    return split


def make_shadow_sets(shadow_idx: np.ndarray, labels: np.ndarray, k: int, exp: Experiment) -> dict[str, np.ndarray]:
    """Shadow k: per class `per_class_in` IN and `per_class_out` OUT images, plus a validation set from the rest."""
    rng, rng_val = np.random.default_rng(1000 + k), np.random.default_rng(2000 + k)
    n_in, n_out, n_val = exp.per_class_in, exp.per_class_out, exp.per_class_val
    ins, outs, vals = [], [], []
    for c in range(10):
        pool = shadow_idx[labels[shadow_idx] == c]
        drawn = rng.permutation(pool)
        ins.append(drawn[:n_in])
        outs.append(drawn[n_in:n_in + n_out])
        vals.append(rng_val.choice(drawn[n_in + n_out:], size=n_val, replace=False))
    return {"in_idx": np.sort(np.concatenate(ins)), "out_idx": np.sort(np.concatenate(outs)),
            "val_idx": np.sort(np.concatenate(vals))}


def save_split(split: dict, shadow_sets: list[dict], path: Path) -> None:
    arrays = dict(split)
    for k, s in enumerate(shadow_sets):
        arrays[f"shadow_{k}_in"], arrays[f"shadow_{k}_out"], arrays[f"shadow_{k}_val"] = s["in_idx"], s["out_idx"], s["val_idx"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def load_split(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def export_yolo_folder(images: np.ndarray, labels: np.ndarray, train_idx, val_idx, out_dir: Path) -> Path:
    """Write Ultralytics classification folders out_dir/{train,val}/<class_name>/<idx>.png."""
    out_dir = Path(out_dir)
    marker, expected = out_dir / ".done", len(train_idx) + len(val_idx)
    if marker.exists() and marker.read_text().strip() == str(expected):
        return out_dir
    for sub, idxs in (("train", train_idx), ("val", val_idx)):
        for c in CLASS_NAMES:
            (out_dir / sub / c).mkdir(parents=True, exist_ok=True)
        for i in idxs:
            cv2.imwrite(str(out_dir / sub / CLASS_NAMES[labels[i]] / f"{int(i)}.png"), images[i])
    marker.write_text(str(expected))
    return out_dir
