"""Visualisations of every syllabus descriptor for notebook 01 and the Feature Explorer page."""
from __future__ import annotations

import cv2
import numpy as np

from .features import dwt_bands, gabor_responses, hsv_hist, lbp_codes


def _norm(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    span = a.max() - a.min()
    return np.zeros(a.shape, np.uint8) if span == 0 else ((a - a.min()) / span * 255).astype(np.uint8)


def upscale(img: np.ndarray, factor: int = 8) -> np.ndarray:
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)


def hog_glyphs(gray: np.ndarray, scale: int = 4) -> np.ndarray:
    """One line per orientation bin per 8x8 cell (9 unsigned bins), length proportional to bin weight."""
    g = gray.astype(np.float32)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=1), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=1)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    bins = ((ang % 180) / 20).astype(int) % 9
    canvas = np.zeros((32 * scale, 32 * scale), np.uint8)
    cells = np.zeros((4, 4, 9))
    for r in range(4):
        for c in range(4):
            cells[r, c] = np.bincount(bins[r * 8:(r + 1) * 8, c * 8:(c + 1) * 8].ravel(),
                                      weights=mag[r * 8:(r + 1) * 8, c * 8:(c + 1) * 8].ravel(), minlength=9)
    peak = cells.max() or 1.0
    half = 4 * scale
    for r in range(4):
        for c in range(4):
            cy, cx = r * 8 * scale + half, c * 8 * scale + half
            for b in range(9):
                theta = np.deg2rad(b * 20 + 10 + 90)  # draw the edge direction, perpendicular to the gradient
                length = half * cells[r, c, b] / peak
                dx, dy = length * np.cos(theta), length * np.sin(theta)
                cv2.line(canvas, (int(cx - dx), int(cy - dy)), (int(cx + dx), int(cy + dy)), 255, 1, cv2.LINE_AA)
    return canvas


def lbp_code_map(gray: np.ndarray) -> np.ndarray:
    return (lbp_codes(gray).astype(np.float64) * 255 / 9).astype(np.uint8)


def gabor_grid(gray: np.ndarray) -> np.ndarray:
    """2x4 tile: rows = wavelength (4, 8), columns = orientation (0, 45, 90, 135 deg)."""
    resp = [_norm(np.abs(r)) for r in gabor_responses(gray)]  # ordered theta-major, lambda-minor
    rows = [np.hstack([resp[t * 2 + li] for t in range(4)]) for li in range(2)]
    return np.vstack(rows)


def dwt_subbands(gray: np.ndarray) -> np.ndarray:
    """Standard wavelet mosaic: level-2 bands in the top-left quadrant, level-1 bands around it."""
    ca2, ch2, cv2_, cd2, ch1, cv1, cd1 = [_norm(np.abs(b)) for b in dwt_bands(gray)]
    tl = np.vstack([np.hstack([ca2, ch2]), np.hstack([cv2_, cd2])])
    return np.vstack([np.hstack([tl, ch1]), np.hstack([cv1, cd1])])


def fft_spectrum(gray: np.ndarray) -> np.ndarray:
    return _norm(np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray.astype(np.float64))))))


def canny(gray: np.ndarray, lo: int = 50, hi: int = 150) -> np.ndarray:
    return cv2.Canny(gray, lo, hi)


def log_edges(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray.astype(np.float64), (0, 0), 1.0)
    return _norm(np.abs(cv2.Laplacian(blur, cv2.CV_64F)))


def dog(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float64)
    return _norm(cv2.GaussianBlur(g, (0, 0), 1.0) - cv2.GaussianBlur(g, (0, 0), 2.0))


def hsv_hist_data(img_bgr: np.ndarray) -> dict[str, np.ndarray]:
    """The three L1-normalised HSV histograms used in the descriptor (H 16, S 8, V 8 bins)."""
    h = hsv_hist(img_bgr)
    return {"Hue": h[:16], "Saturation": h[16:24], "Value": h[24:]}
