import numpy as np

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
