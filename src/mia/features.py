"""The 426-dim hand-crafted descriptor: HOG 324 | uniform LBP 40 | Gabor 16 | Haar DWT 14 | HSV histogram 32."""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pywt

from .preprocess import affine_augment, enhance, to_gray

BLOCKS = (("HOG", 324), ("LBP", 40), ("Gabor", 16), ("DWT", 14), ("HSV", 32))
N_DIMS = sum(n for _, n in BLOCKS)  # 426

_HOG = cv2.HOGDescriptor((32, 32), (16, 16), (8, 8), (8, 8), 9)
GABOR_PARAMS = [(theta, lambd) for theta in (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4) for lambd in (4, 8)]
_GABOR = [cv2.getGaborKernel((9, 9), 2.0, theta, lambd, 0.5, 0, ktype=cv2.CV_32F) for theta, lambd in GABOR_PARAMS]

# LBP P=8, R=1: neighbour offsets (dy, dx) at angles 2*pi*p/8, sampled with bilinear interpolation.
_ANGLES = 2 * np.pi * np.arange(8) / 8
_LBP_DY, _LBP_DX = -np.sin(_ANGLES), np.cos(_ANGLES)


def hog(gray: np.ndarray) -> np.ndarray:
    return _HOG.compute(gray).ravel()


def lbp_codes(gray: np.ndarray) -> np.ndarray:
    """Rotation-invariant uniform LBP code per pixel: number of ones if <= 2 transitions, else 9."""
    g = gray.astype(np.float64)
    h, w = g.shape
    pad = np.pad(g, 2, mode="reflect")
    rows, cols = np.mgrid[0:h, 0:w] + 2.0
    bits = np.empty((8, h, w), dtype=np.uint8)
    for p in range(8):
        y, x = rows + _LBP_DY[p], cols + _LBP_DX[p]
        y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int)
        fy, fx = y - y0, x - x0
        val = (pad[y0, x0] * (1 - fy) * (1 - fx) + pad[y0, x0 + 1] * (1 - fy) * fx
               + pad[y0 + 1, x0] * fy * (1 - fx) + pad[y0 + 1, x0 + 1] * fy * fx)
        bits[p] = val >= g - 1e-9  # tolerance absorbs interpolation round-off on flat regions
    transitions = np.abs(np.diff(bits.astype(np.int8), axis=0, append=bits[:1])).sum(axis=0)
    return np.where(transitions <= 2, bits.sum(axis=0), 9).astype(np.uint8)


def lbp_uniform(gray: np.ndarray) -> np.ndarray:
    codes = lbp_codes(gray)
    out = []
    for r in (0, 16):
        for c in (0, 16):
            hist = np.bincount(codes[r:r + 16, c:c + 16].ravel(), minlength=10).astype(np.float64)
            out.append(hist / hist.sum())
    return np.concatenate(out)


def gabor_responses(gray: np.ndarray) -> list[np.ndarray]:
    g = gray.astype(np.float32) / 255.0
    return [cv2.filter2D(g, cv2.CV_32F, k) for k in _GABOR]


def gabor_stats(gray: np.ndarray) -> np.ndarray:
    return np.array([s for r in gabor_responses(gray) for s in (np.abs(r).mean(), r.std())])


def dwt_bands(gray: np.ndarray) -> list[np.ndarray]:
    """[cA2, cH2, cV2, cD2, cH1, cV1, cD1] of a 2-level Haar DWT."""
    ca2, (ch2, cv2_, cd2), (ch1, cv1, cd1) = pywt.wavedec2(gray / 255.0, "haar", level=2)
    return [ca2, ch2, cv2_, cd2, ch1, cv1, cd1]


def dwt_stats(gray: np.ndarray) -> np.ndarray:
    return np.array([s for b in dwt_bands(gray) for s in (np.abs(b).mean(), b.std())])


def hsv_hist(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    out = []
    for ch, bins, hi in ((0, 16, 180), (1, 8, 256), (2, 8, 256)):
        hist = cv2.calcHist([hsv], [ch], None, [bins], [0, hi]).ravel().astype(np.float64)
        out.append(hist / hist.sum())
    return np.concatenate(out)


def describe(img_bgr: np.ndarray) -> np.ndarray:
    """enhance -> [HOG, LBP, Gabor, DWT on grayscale] + [HSV histogram on the enhanced colour image]."""
    enh = enhance(img_bgr)
    gray = to_gray(enh)
    vec = np.concatenate([hog(gray), lbp_uniform(gray), gabor_stats(gray), dwt_stats(gray), hsv_hist(enh)])
    vec = vec.astype(np.float32)
    if not np.all(np.isfinite(vec)):
        raise ValueError("descriptor contains NaN or inf")
    return vec


def describe_batch(images: np.ndarray, offset: int = 0) -> np.ndarray:
    out = np.empty((len(images), N_DIMS), dtype=np.float32)
    for i, img in enumerate(images):
        try:
            out[i] = describe(img)
        except ValueError as e:
            raise ValueError(f"image index {offset + i}: {e}") from e
    return out


def build_feature_cache(images: np.ndarray, path: Path, n_jobs: int = 1, chunk: int = 5000) -> np.ndarray:
    """Descriptors for every pool image, cached as float32 .npy; reused when the shape matches."""
    path = Path(path)
    if path.exists():
        cached = np.load(path, mmap_mode="r")
        if cached.shape == (len(images), N_DIMS):
            return np.asarray(cached)
    path.parent.mkdir(parents=True, exist_ok=True)
    starts = list(range(0, len(images), chunk))
    t0, parts = time.perf_counter(), []
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            for s, part in zip(starts, ex.map(describe_batch, [images[s:s + chunk] for s in starts], starts)):
                parts.append(part)
                print(f"  features {s + len(part):>6}/{len(images)}  {time.perf_counter() - t0:6.1f} s")
    else:
        for s in starts:
            parts.append(describe_batch(images[s:s + chunk], s))
            print(f"  features {s + len(parts[-1]):>6}/{len(images)}  {time.perf_counter() - t0:6.1f} s")
    feats = np.concatenate(parts)
    np.save(path, feats)
    return feats


def augmented_features(images: np.ndarray, idx: np.ndarray, n_copies: int, seed: int, cache_path: Path) -> np.ndarray:
    """n_copies affine-augmented descriptors per index, ordered [idx0 copy0, idx0 copy1, idx1 copy0, ...]."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape == (len(idx) * n_copies, N_DIMS):
            return cached
    rng = np.random.default_rng(seed)
    out = np.empty((len(idx) * n_copies, N_DIMS), dtype=np.float32)
    for j, i in enumerate(idx):
        for c in range(n_copies):
            try:
                out[j * n_copies + c] = describe(affine_augment(images[i], rng))
            except ValueError as e:
                raise ValueError(f"image index {int(i)} (augmented copy {c}): {e}") from e
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, out)
    return out
