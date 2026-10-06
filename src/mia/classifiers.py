"""Target / shadow classifiers with one interface: fit, predict_proba -> (n, 10) float64, save, load, n_params."""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NearestNeighbors

from .data import CLASS_NAMES

N_CLASSES = 10


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class Standardizer:
    def __init__(self, mean: np.ndarray | None = None, std: np.ndarray | None = None):
        self.mean, self.std = mean, std

    def fit(self, X: np.ndarray) -> Standardizer:
        X = np.asarray(X, dtype=np.float64)
        self.mean, self.std = X.mean(axis=0), np.maximum(X.std(axis=0), 1e-8)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float64) - self.mean) / self.std


def _pca_fit(Xs: np.ndarray, n_components: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=n_components, random_state=seed, svd_solver="randomized").fit(Xs)
    return pca.components_, pca.mean_


class SoftNN:
    """Config C1: soft 1-nearest-neighbour. p_k ∝ exp(-d_k / tau), d_k = distance to the nearest class-k training point."""

    def __init__(self):
        self.scaler, self.X, self.y, self.tau, self._nn = Standardizer(), None, None, None, None

    def _build_index(self) -> None:
        self._nn = [NearestNeighbors(n_neighbors=1).fit(self.X[self.y == k]) for k in range(N_CLASSES)]

    def _distances(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X).astype(np.float32)
        return np.column_stack([nn.kneighbors(Xs)[0][:, 0] for nn in self._nn]).astype(np.float64)

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray) -> SoftNN:
        self.scaler.fit(X)
        self.X, self.y = self.scaler.transform(X).astype(np.float32), np.asarray(y, dtype=np.int64)
        self._build_index()
        self.calibrate_tau(X_val)
        return self

    def calibrate_tau(self, X_val: np.ndarray) -> None:
        self.tau = float(np.median(self._distances(X_val)))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return softmax(-self._distances(X) / self.tau)

    def save(self, path: Path) -> None:
        np.savez(path, X=self.X, y=self.y, mean=self.scaler.mean, std=self.scaler.std, tau=self.tau)

    @classmethod
    def load(cls, path: Path) -> SoftNN:
        with np.load(path) as z:
            m = cls()
            m.X, m.y, m.tau = z["X"], z["y"], float(z["tau"])
            m.scaler = Standardizer(z["mean"], z["std"])
        m._build_index()
        return m

    @property
    def n_params(self) -> int:
        return int(self.X.size + self.scaler.mean.size + self.scaler.std.size + 1)


class SoftmaxDiscriminant:
    """Configs C2-C4 and the label-smoothing sweep: linear discriminants g_k(x) = w_k.x + b_k with softmax,
    trained by mini-batch SGD with momentum on smoothed cross-entropy plus (l2/2)||W||^2."""

    def __init__(self, pca: int | None = None, l2: float = 0.0, epsilon: float = 0.0, epochs: int = 200, seed: int = 0,
                 batch: int = 256, lr: float = 0.05, momentum: float = 0.9):
        self.pca, self.l2, self.epsilon, self.epochs, self.seed = pca, l2, epsilon, epochs, seed
        self.batch, self.lr, self.momentum = batch, lr, momentum
        self.scaler = Standardizer()
        self.components, self.pca_mean = np.zeros((0, 0)), np.zeros(0)
        self.W, self.b, self.epoch = None, None, 0

    def _project(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        return (Xs - self.pca_mean) @ self.components.T if self.components.size else Xs

    def fit(self, X: np.ndarray, y: np.ndarray, checkpoint_epochs=(), checkpoint_dir: Path | None = None) -> dict:
        """Train; save a checkpoint at each listed epoch and at the final epoch.

        Returns {epoch: (checkpoint_path, seconds_since_start)}.
        """
        t0 = time.perf_counter()
        self.scaler.fit(X)
        if self.pca:
            self.components, self.pca_mean = _pca_fit(self.scaler.transform(X), self.pca, self.seed)
        Z, y = self._project(X), np.asarray(y, dtype=np.int64)
        n, d = Z.shape
        T = np.full((n, N_CLASSES), self.epsilon / N_CLASSES)
        T[np.arange(n), y] += 1 - self.epsilon
        self.W = np.random.default_rng(self.seed).normal(0.0, 0.01, size=(d, N_CLASSES))
        self.b = np.zeros(N_CLASSES)
        vW, vb = np.zeros_like(self.W), np.zeros_like(self.b)
        marks = set(checkpoint_epochs) | {self.epochs}
        saved = {}
        for epoch in range(1, self.epochs + 1):
            order = np.random.default_rng(self.seed + epoch).permutation(n)
            for s in range(0, n, self.batch):
                bi = order[s:s + self.batch]
                G = (softmax(Z[bi] @ self.W + self.b) - T[bi]) / len(bi)
                vW = self.momentum * vW - self.lr * (Z[bi].T @ G + self.l2 * self.W)
                vb = self.momentum * vb - self.lr * G.sum(axis=0)
                self.W += vW
                self.b += vb
            self.epoch = epoch
            if epoch in marks and checkpoint_dir is not None:
                path = Path(checkpoint_dir) / f"epoch{epoch}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                self.save(path)
                saved[epoch] = (path, time.perf_counter() - t0)
        return saved

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return softmax(self._project(X) @ self.W + self.b)

    def save(self, path: Path) -> None:
        np.savez(path, W=self.W, b=self.b, mean=self.scaler.mean, std=self.scaler.std, pca_components=self.components,
                 pca_mean=self.pca_mean, epsilon=self.epsilon, l2=self.l2, epoch=self.epoch, seed=self.seed)

    @classmethod
    def load(cls, path: Path) -> SoftmaxDiscriminant:
        with np.load(path) as z:
            comps = z["pca_components"]
            m = cls(pca=comps.shape[0] or None, l2=float(z["l2"]), epsilon=float(z["epsilon"]), seed=int(z["seed"]))
            m.W, m.b, m.epoch = z["W"], z["b"], int(z["epoch"])
            m.scaler = Standardizer(z["mean"], z["std"])
            m.components, m.pca_mean = comps, z["pca_mean"]
        return m

    @property
    def n_params(self) -> int:
        return int(self.W.size + self.b.size)


class LDAClassifier:
    """Config C5: Standardizer -> PCA(64) -> Fisher LDA (lsqr, Ledoit-Wolf shrinkage). Stored as plain arrays."""

    def __init__(self, pca: int = 64, seed: int = 0):
        self.pca, self.seed = pca, seed
        self.scaler, self.components, self.pca_mean, self.coef, self.intercept = Standardizer(), None, None, None, None

    def fit(self, X: np.ndarray, y: np.ndarray) -> LDAClassifier:
        self.scaler.fit(X)
        self.components, self.pca_mean = _pca_fit(self.scaler.transform(X), self.pca, self.seed)
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(self._project(X), y)
        self.coef, self.intercept = lda.coef_, lda.intercept_
        return self

    def _project(self, X: np.ndarray) -> np.ndarray:
        return (self.scaler.transform(X) - self.pca_mean) @ self.components.T

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # identical to sklearn's multiclass LDA.predict_proba: softmax of the discriminant functions
        return softmax(self._project(X) @ self.coef.T + self.intercept)

    def save(self, path: Path) -> None:
        np.savez(path, mean=self.scaler.mean, std=self.scaler.std, components=self.components, pca_mean=self.pca_mean,
                 coef=self.coef, intercept=self.intercept, seed=self.seed)

    @classmethod
    def load(cls, path: Path) -> LDAClassifier:
        with np.load(path) as z:
            m = cls(pca=z["components"].shape[0], seed=int(z["seed"]))
            m.scaler = Standardizer(z["mean"], z["std"])
            m.components, m.pca_mean, m.coef, m.intercept = z["components"], z["pca_mean"], z["coef"], z["intercept"]
        return m

    @property
    def n_params(self) -> int:
        return int(self.coef.size + self.intercept.size + self.components.size + self.pca_mean.size)


_CPU_WARNED = False


def yolo_device():
    global _CPU_WARNED
    import torch
    if torch.cuda.is_available():
        return 0
    if not _CPU_WARNED:
        print("WARNING: CUDA is not available; YOLO runs on CPU (roughly 10x slower).")
        _CPU_WARNED = True
    return "cpu"


class YoloClassifier:
    """Configs C6, C7: Ultralytics YOLO26n-cls trained from scratch (architecture yaml, no pretrained weights)."""

    def __init__(self, model=None):
        self.model = model

    def fit(self, ds_dir: Path, *, model_id: str, epochs: int, seed: int, train_kwargs: dict, checkpoint_epochs,
            project_dir: Path) -> dict:
        """Train; returns {completed_epoch: (weights_path, seconds_since_start)}.

        Ultralytics' own save_period names files by 0-based epoch index (epoch5.pt is written after 6 epochs),
        so an on_model_save callback copies last.pt to e{N}.pt exactly when N epochs have completed. With no
        checkpoint epochs (C7) the result is best.pt, keyed by the epoch it was saved at.
        Copies are optimizer-stripped like Ultralytics' own final weights.
        """
        import torch
        from ultralytics import YOLO
        from ultralytics.utils.torch_utils import strip_optimizer

        device, marks = yolo_device(), set(checkpoint_epochs)
        state: dict = {}

        def on_model_save(trainer):
            e = trainer.epoch + 1
            if trainer.best_fitness == trainer.fitness:
                state["best"] = (e, time.perf_counter() - state["t0"])
            if e in marks:
                dst = Path(trainer.wdir) / f"e{e}.pt"
                shutil.copyfile(trainer.last, dst)
                strip_optimizer(dst)  # same on-disk format as the final best.pt / last.pt, so model_bytes compare
                state["saved"][e] = (dst, time.perf_counter() - state["t0"])

        for batch in (256, 128, 64):
            model = YOLO("yolo26n-cls.yaml")
            model.add_callback("on_model_save", on_model_save)
            state.update(t0=time.perf_counter(), saved={})
            try:
                model.train(data=str(ds_dir), epochs=epochs, imgsz=64, batch=batch, seed=seed, deterministic=True,
                            pretrained=False, project=str(project_dir), name=model_id, exist_ok=True, workers=2,
                            device=device, verbose=False, **train_kwargs)
                break
            except torch.cuda.OutOfMemoryError:
                if batch == 64:
                    raise
                torch.cuda.empty_cache()
                print(f"{model_id}: CUDA out of memory at batch {batch}; retrying with batch {batch // 2}")
        if marks:
            return state["saved"]
        best_epoch, secs = state["best"]
        return {best_epoch: (Path(project_dir) / model_id / "weights" / "best.pt", secs)}

    def predict_proba(self, images_bgr, batch: int = 256) -> np.ndarray:
        out = []
        for s in range(0, len(images_bgr), batch):
            chunk = [np.ascontiguousarray(im) for im in images_bgr[s:s + batch]]
            results = self.model.predict(chunk, imgsz=64, batch=batch, verbose=False)
            out.append(np.stack([r.probs.data.cpu().numpy() for r in results]).astype(np.float64))
        P = np.concatenate(out)
        return P / P.sum(axis=1, keepdims=True)

    def save(self, path: Path) -> None:
        self.model.save(str(path))

    @classmethod
    def load(cls, path: Path) -> YoloClassifier:
        from ultralytics import YOLO
        if not Path(path).exists():
            raise FileNotFoundError(f"YOLO checkpoint not found: {path}")
        model = YOLO(str(path))
        names = tuple(model.names[i] for i in range(len(model.names)))
        if names != CLASS_NAMES:
            raise ValueError(f"{path}: class order {names} does not match CIFAR-10 {CLASS_NAMES}")
        return cls(model)

    @property
    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.model.model.parameters()))
