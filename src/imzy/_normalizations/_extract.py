"""Get normalization utilities."""


def get_normalizations() -> list[str]:
    """Get list of available normalizations."""
    return [
        "TIC",
        "RMS",
        "Median",
        "0-95% TIC",
        "0-90% TIC",
        "5-100% TIC",
        "10-100% TIC",
        "5-95% TIC",
        "10-90% TIC",
        "0-norm",
        "2-norm",
        "3-norm",
    ]
