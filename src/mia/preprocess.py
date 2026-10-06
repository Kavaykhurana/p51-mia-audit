"""Image enhancement (CLAHE on luminance + Gaussian smoothing) and affine augmentation. Images are uint8 BGR."""
from __future__ import annotations

import cv2
import numpy as np

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))


def enhance(img_bgr: np.ndarray) -> np.ndarray:
    ycc = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    ycc[..., 0] = _CLAHE.apply(np.ascontiguousarray(ycc[..., 0]))
    return cv2.GaussianBlur(cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR), (3, 3), sigmaX=0.5)


def to_gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def affine_augment(img_bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Rotation U(-10, 10) deg about the centre, translation U(-2, 2) px, horizontal flip with p=0.5."""
    angle, tx, ty, flip = rng.uniform(-10, 10), rng.uniform(-2, 2), rng.uniform(-2, 2), rng.random() < 0.5
    h, w = img_bgr.shape[:2]
    m = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle, 1.0)
    m[:, 2] += (tx, ty)
    src = cv2.flip(img_bgr, 1) if flip else img_bgr
    return cv2.warpAffine(src, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
