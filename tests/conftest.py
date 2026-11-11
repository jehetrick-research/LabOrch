import shutil
import warnings

import pytest

from laborch import Bank, config


def _have_audio():
    try:
        import pydub  # noqa: F401
    except ImportError:
        return False
    return shutil.which("ffmpeg") is not None


requires_audio = pytest.mark.skipif(not _have_audio(), reason="needs pydub + ffmpeg")


@pytest.fixture(scope="session")
def bank():
    """Bank v2 (the default): power spectra with levels and metadata."""
    if not config.BANK_FILE.exists():
        pytest.skip(f"Bank v2 not found at {config.BANK_FILE}")
    return Bank.load()


@pytest.fixture(scope="session")
def bank_v1():
    """Bank v1: the FFTBank text files (unit-sum magnitude spectra)."""
    if not config.FFT_DIR.exists():
        pytest.skip(f"Bank v1 not found at {config.FFT_DIR}")
    return Bank.load(version=1)


@pytest.fixture(scope="session")
def old_bank():
    if not config.OLD_FFT_DIR.exists():
        pytest.skip(f"old Bank not found at {config.OLD_FFT_DIR}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Bank.load_old()
