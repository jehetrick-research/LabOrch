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
    if not config.FFT_DIR.exists():
        pytest.skip(f"Bank not found at {config.FFT_DIR}")
    return Bank.load()


@pytest.fixture(scope="session")
def old_bank():
    if not config.OLD_FFT_DIR.exists():
        pytest.skip(f"old Bank not found at {config.OLD_FFT_DIR}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Bank.load_old()
