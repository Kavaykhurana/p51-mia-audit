import numpy as np

from mia.features import describe, dwt_stats, hog, hsv_hist, lbp_uniform
from mia.preprocess import enhance, to_gray


def _img(seed=0):
    return np.random.default_rng(seed).integers(0, 256, (32, 32, 3), dtype=np.uint8)


def test_describe_shape_dtype_finite_deterministic():
    a, b = describe(_img()), describe(_img())
    assert a.shape == (426,) and a.dtype == np.float32
    assert np.all(np.isfinite(a))
    assert np.array_equal(a, b)


def test_block_lengths_and_normalisation():
    gray = to_gray(enhance(_img(1)))
    assert hog(gray).shape == (324,)
    assert dwt_stats(gray).shape == (14,)
    lbp = lbp_uniform(gray)
    assert lbp.shape == (40,)
    np.testing.assert_allclose(lbp.reshape(4, 10).sum(axis=1), 1.0)
    h = hsv_hist(enhance(_img(2)))
    np.testing.assert_allclose([h[:16].sum(), h[16:24].sum(), h[24:].sum()], 1.0)
