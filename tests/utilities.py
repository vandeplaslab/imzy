from pathlib import Path


def get_imzml_data():
    """Get data from the `_test_data` folder."""
    test_dir = Path(__file__).parent / "_test_data"
    return list(test_dir.glob("*.imzML"))


def get_tsf_data():
    """Get data from the `_test_data` folder."""
    test_dir = Path(__file__).parent / "_test_data"
    return list(test_dir.glob("*_tsf.d"))


def get_tdf_data():
    """Get data from the `_test_data` folder."""
    test_dir = Path(__file__).parent / "_test_data"
    return list(test_dir.glob("*_tdf.d"))
