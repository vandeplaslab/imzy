import numpy as np
import pytest

from imzy._normalizations._extract import (
    _get_outlier_mask,
    calculate_normalizations,
    calculate_normalizations_optimized,
    get_normalizations,
)


def test_calculate_normalizations_optimized():
    y = np.arange(10, dtype=np.float32)
    res = calculate_normalizations_optimized(y)
    assert res.shape == (12,)


def test_calculate_normalizations():
    y = np.arange(10, dtype=np.float32)
    res = calculate_normalizations(y)
    assert res.shape == (12,)


def test_get_normalizations():
    res = get_normalizations()
    assert len(res) == 12
    assert res[0] == "TIC"


def test__get_outlier_mask():
    array = np.random.random(100)
    mask = _get_outlier_mask(array)
    assert mask.shape == (100,)


def _reference_normalizations(spectrum: np.ndarray) -> np.ndarray:
    """Original normalization formulas, evaluated in float64 with plain numpy."""
    y = spectrum.astype(np.float64)
    out = np.zeros(12)
    out[0] = y.sum()
    if out[0] == 0:
        return out
    positive = y[y > 0]
    out[1] = np.sqrt(np.mean(y**2))
    out[2] = np.median(positive) if positive.size else 0
    q95, q90, q10, q5 = np.quantile(positive, [0.95, 0.9, 0.1, 0.05]) if positive.size > 1 else (0, 0, 0, 0)
    out[3] = y[y < q95].sum()
    out[4] = y[y < q90].sum()
    out[5] = y[y > q5].sum()
    out[6] = y[y > q10].sum()
    out[7] = y[(y > q5) & (y < q95)].sum()
    out[8] = y[(y > q10) & (y < q90)].sum()
    out[9] = np.count_nonzero(y)
    out[10] = np.linalg.norm(y, 2)
    out[11] = np.sum(np.abs(y) ** 3) ** (1 / 3)
    return out


@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64])
@pytest.mark.parametrize("n_nonzero", [0, 1, 2, 500, 10_000])
def test_calculate_normalizations_optimized_matches_reference(dtype, n_nonzero):
    rng = np.random.default_rng(42)
    y = np.zeros(10_000, dtype=dtype)
    y[rng.choice(y.size, n_nonzero, replace=False)] = rng.integers(1, 5_000, n_nonzero)
    np.testing.assert_allclose(calculate_normalizations_optimized(y), _reference_normalizations(y), rtol=1e-5, atol=1e-3)
