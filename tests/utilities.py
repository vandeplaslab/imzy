from pathlib import Path


def get_imzml_data(pattern: str = "*.imzML") -> list[Path]:
    """Get data from the `_test_data` folder."""
    test_dir = Path(__file__).parent / "_test_data"
    return list(test_dir.glob(pattern))


def get_tsf_data(pattern: str = "*_tsf.d") -> list[Path]:
    """Get data from the `_test_data` folder."""
    test_dir = Path(__file__).parent / "_test_data"
    return list(test_dir.glob(pattern))


def get_tdf_data(pattern: str = "*_tdf.d"):
    """Get data from the `_test_data` folder."""
    test_dir = Path(__file__).parent / "_test_data"
    return list(test_dir.glob(pattern))
