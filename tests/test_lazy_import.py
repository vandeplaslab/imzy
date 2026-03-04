"""Test lazy imports."""

import imzy


def test_lazy_imports():
    """Test that lazy imports work."""
    # Accessing the attributes should trigger the lazy import
    assert "BaseReader" in dir(imzy)
    assert "H5CentroidsStore" in dir(imzy)
    assert "get_reader" in dir(imzy)

    from imzy import BaseReader, H5CentroidsStore, get_reader

    assert BaseReader is not None, "BaseReader should be imported"
    assert H5CentroidsStore is not None, "H5CentroidsStore should be imported"
    assert get_reader is not None, "get_reader should be imported"
