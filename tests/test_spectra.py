"""The FFT pipeline must rebuild the stored Bank spectra from the .aif files."""
import numpy as np
import pytest

from laborch.spectra import fft_of_file, make_fft
from tests.conftest import requires_audio


@requires_audio
@pytest.mark.parametrize("name", ["flu-e4-ord-mf", "vln-a4x-ord-mf", "pno-c4-ord-ff", "tpc-c5-ord-mf"])
def test_rebuilds_bank_file(bank, name):
    ref = bank.spectrum(name)
    got = fft_of_file(bank.sound_path(name))
    assert np.max(np.abs(got - ref)) / ref.max() < 5e-3


def test_pure_tone_peak():
    sr, f0 = 44100, 440.0
    t = np.arange(2 * sr) / sr
    spec = make_fft(np.sin(2 * np.pi * f0 * t))
    assert abs(spec.argmax() * sr / 4096 - f0) < sr / 4096
    assert abs(spec.sum() - 1) < 1e-12


def test_too_short():
    with pytest.raises(ValueError):
        make_fft(np.zeros(3000))
