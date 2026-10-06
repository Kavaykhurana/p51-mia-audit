"""Load config/experiment.yaml into frozen dataclasses and validate every value."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "experiment.yaml"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SPLIT_PATH = DATA_DIR / "splits" / "split_seed0.npz"
FEATURES_PATH = DATA_DIR / "features" / "features_all.npy"
AUG_DIR = DATA_DIR / "features" / "features_aug"
YOLO_DS_DIR = DATA_DIR / "yolo_ds"
MODELS_DIR = ROOT / "models"
PROBS_DIR = MODELS_DIR / "probs"
RESULTS_DIR = ROOT / "results"
DB_PATH = RESULTS_DIR / "mia.sqlite"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORTS_DIR = RESULTS_DIR / "reports"
TEMPLATES_DIR = ROOT / "templates"
ERROR_LOG = RESULTS_DIR / "train_errors.log"

FAMILIES = ("nn", "softmax", "lda", "yolo")
YOLO_AUG_KEYS = ("dropout", "patience", "fliplr", "scale", "erasing", "auto_augment", "hsv_h", "hsv_s", "hsv_v")


@dataclass(frozen=True)
class ModelConfig:
    config_id: str
    family: str
    description: str
    params: dict = field(compare=False)  # the raw yaml entry, stored as params_json
    base_config_id: str | None = None
    pca: int | None = None
    l2: float = 0.0
    epochs: int = 0
    label_smoothing: float = 0.0
    aug_copies: int = 0
    checkpoint_epochs: tuple[int, ...] = ()

    @property
    def yolo_train_kwargs(self) -> dict:
        return {k: self.params[k] for k in YOLO_AUG_KEYS}

    @property
    def has_epoch_suffix(self) -> bool:
        """Configs saving several checkpoints (C2, C6) carry `_e{epoch}` in every model id."""
        return len(self.checkpoint_epochs) > 1


@dataclass(frozen=True)
class Experiment:
    split_seed: int
    per_class: dict
    classical_k: int
    yolo_k: int
    per_class_in: int
    per_class_out: int
    per_class_val: int
    seeds_softmax: tuple[int, ...]
    seeds_yolo: tuple[int, ...]
    configs: dict
    early_stopping: dict
    ls_bases: tuple[str, ...]
    ls_epsilons: tuple[float, ...]
    bootstrap: int
    bootstrap_seed: int
    permutations: int
    fpr_points: tuple[float, ...]
    severity_bounds: dict

    def n_shadows(self, mcfg: ModelConfig) -> int:
        return self.yolo_k if mcfg.family == "yolo" else self.classical_k

    def seeds_for(self, mcfg: ModelConfig) -> tuple[int, ...]:
        if mcfg.family == "softmax":
            return self.seeds_softmax
        if mcfg.family == "yolo":
            return self.seeds_yolo
        return (0,)  # C1 and C5 are deterministic: one run

    def ls_config(self, base_id: str, eps: float) -> ModelConfig:
        """Derived label-smoothing config, e.g. C2_ls0.10 (trained to the base's final epoch)."""
        base = self.configs[base_id]
        params = {**base.params, "label_smoothing": eps, "checkpoint_epochs": [base.epochs]}
        return replace(base, config_id=f"{base_id}_ls{eps:.2f}", base_config_id=base_id, params=params,
                       label_smoothing=eps, checkpoint_epochs=(base.epochs,),
                       description=f"{base.description}; label smoothing eps={eps:.2f}")

    def config(self, config_id: str) -> ModelConfig:
        if config_id in self.configs:
            return self.configs[config_id]
        for base in self.ls_bases:
            for eps in self.ls_epsilons:
                if eps > 0 and config_id == f"{base}_ls{eps:.2f}":
                    return self.ls_config(base, eps)
        raise KeyError(f"unknown config_id {config_id!r}")


def _req(d: dict, key: str, path: str):
    if not isinstance(d, dict) or key not in d:
        raise ValueError(f"missing key '{path}{key}' in experiment.yaml")
    return d[key]


def _check(ok: bool, key: str, why: str) -> None:
    if not ok:
        raise ValueError(f"invalid value for '{key}': {why}")


def _model_config(cid: str, entry: dict) -> ModelConfig:
    p = f"configs.{cid}."
    family = _req(entry, "family", p)
    _check(family in FAMILIES, p + "family", f"must be one of {FAMILIES}")
    desc = _req(entry, "description", p)
    if family == "nn":
        return ModelConfig(cid, family, desc, entry)
    if family == "lda":
        pca = _req(entry, "pca", p)
        _check(isinstance(pca, int) and pca > 0, p + "pca", "must be a positive integer")
        return ModelConfig(cid, family, desc, entry, pca=pca)
    epochs = _req(entry, "epochs", p)
    _check(isinstance(epochs, int) and epochs > 0, p + "epochs", "must be a positive integer")
    ckpts = tuple(_req(entry, "checkpoint_epochs", p))
    _check(all(isinstance(e, int) and 1 <= e <= epochs for e in ckpts), p + "checkpoint_epochs",
           f"every epoch must be an integer in [1, {epochs}]")
    if family == "yolo":
        for k in YOLO_AUG_KEYS:
            _req(entry, k, p)
        _check(0 <= entry["dropout"] < 1, p + "dropout", "must be in [0, 1)")
        return ModelConfig(cid, family, desc, entry, epochs=epochs, checkpoint_epochs=ckpts)
    pca, l2, eps = _req(entry, "pca", p), _req(entry, "l2", p), _req(entry, "label_smoothing", p)
    aug = _req(entry, "aug_copies", p)
    _check(pca is None or (isinstance(pca, int) and pca > 0), p + "pca", "must be null or a positive integer")
    _check(l2 >= 0, p + "l2", "must be >= 0")
    _check(0 <= eps < 1, p + "label_smoothing", "must be in [0, 1)")
    _check(isinstance(aug, int) and aug >= 0, p + "aug_copies", "must be a non-negative integer")
    _check(epochs in ckpts, p + "checkpoint_epochs", "must include the final epoch")
    return ModelConfig(cid, family, desc, entry, pca=pca, l2=float(l2), epochs=epochs, label_smoothing=float(eps),
                       aug_copies=aug, checkpoint_epochs=ckpts)


def load_experiment(path: Path = CONFIG_PATH) -> Experiment:
    raw = yaml.safe_load(Path(path).read_text())
    split = _req(raw, "split", "")
    per_class = dict(_req(split, "per_class", "split."))
    for k in ("member", "nonmember", "val", "shadow"):
        _check(int(_req(per_class, k, "split.per_class.")) > 0, f"split.per_class.{k}", "must be positive")
    _check(sum(per_class.values()) == 6000, "split.per_class", "must sum to 6000 (CIFAR-10 images per class)")

    sh = _req(raw, "shadows", "")
    vals = {k: int(_req(sh, k, "shadows.")) for k in ("classical_k", "yolo_k", "per_class_in", "per_class_out", "per_class_val")}
    for k, v in vals.items():
        _check(v > 0, f"shadows.{k}", "must be positive")
    _check(vals["per_class_in"] + vals["per_class_out"] + vals["per_class_val"] <= per_class["shadow"],
           "shadows.per_class_in", "IN + OUT + val per class must fit in the shadow pool")

    seeds = _req(raw, "seeds", "")
    configs = {cid: _model_config(cid, e) for cid, e in _req(raw, "configs", "").items()}

    sweeps = _req(raw, "sweeps", "")
    es = {k: tuple(v) for k, v in _req(sweeps, "early_stopping", "sweeps.").items()}
    for cid, epochs in es.items():
        _check(cid in configs, f"sweeps.early_stopping.{cid}", "unknown config")
        _check(set(epochs) <= set(configs[cid].checkpoint_epochs), f"sweeps.early_stopping.{cid}",
               "every epoch must be one of the config's checkpoint_epochs")
    ls = _req(sweeps, "label_smoothing", "sweeps.")
    bases = tuple(_req(ls, "bases", "sweeps.label_smoothing."))
    for b in bases:
        _check(b in configs and configs[b].family == "softmax", "sweeps.label_smoothing.bases", f"{b} must be a softmax config")
    eps = tuple(float(e) for e in _req(ls, "epsilons", "sweeps.label_smoothing."))
    _check(all(0 <= e < 1 for e in eps) and 0.0 in eps, "sweeps.label_smoothing.epsilons", "must be in [0, 1) and include 0")

    m = _req(raw, "metrics", "")
    sev_raw = _req(raw, "severity_lift_bounds", "")
    sev = {k: float(_req(sev_raw, k, "severity_lift_bounds.")) for k in ("moderate", "high", "critical")}
    _check(1 <= sev["moderate"] < sev["high"] < sev["critical"], "severity_lift_bounds", "must satisfy 1 <= moderate < high < critical")
    fpr = tuple(float(x) for x in _req(m, "fpr_points", "metrics."))
    _check(all(0 < x < 1 for x in fpr), "metrics.fpr_points", "must be in (0, 1)")

    return Experiment(
        split_seed=int(_req(split, "seed", "split.")), per_class=per_class,
        classical_k=vals["classical_k"], yolo_k=vals["yolo_k"], per_class_in=vals["per_class_in"],
        per_class_out=vals["per_class_out"], per_class_val=vals["per_class_val"],
        seeds_softmax=tuple(_req(seeds, "softmax", "seeds.")), seeds_yolo=tuple(_req(seeds, "yolo", "seeds.")),
        configs=configs, early_stopping=es, ls_bases=bases, ls_epsilons=eps,
        bootstrap=int(_req(m, "bootstrap", "metrics.")), bootstrap_seed=int(_req(m, "bootstrap_seed", "metrics.")),
        permutations=int(_req(m, "permutations", "metrics.")), fpr_points=fpr, severity_bounds=sev,
    )
