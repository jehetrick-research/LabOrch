"""laborch.synth: the notebook 1.2 C3 'marimba' (partials 1, 4, 10)."""
import numpy as np
import pytest

from laborch import nnls_fit, parse_name
from laborch import config
from laborch.spectra import make_fft, spectral_peaks
from laborch.synth import spectrum_to_audio, tone

F0 = parse_name("mrb-c3-ord-mf").freq            # 130.81 Hz
PARTIALS, AMPS = [1, 4, 10], [1.0, 0.5, 0.25]


@pytest.fixture(scope="module")
def spec():
    return make_fft(tone(F0, PARTIALS, AMPS))


def test_tone_basics():
    x = tone(440, seconds=0.5, peak=2**15 - 1)
    assert len(x) == config.SAMPLE_RATE // 2
    assert abs(np.abs(x).max() - (2**15 - 1)) < 1e-6
    with pytest.raises(ValueError):
        tone(440, partials=[1, 2], amps=[1])
    with pytest.raises(ValueError):
        tone(15000, partials=[1, 2])               # 30 kHz > Nyquist


def test_decay_envelope():
    x = tone(220, decay=10.0)
    first, last = np.abs(x[:4410]).max(), np.abs(x[-4410:]).max()
    assert last < 0.001 * first                    # e^-10 over one second


def test_peaks_at_partials(spec):
    peaks = spectral_peaks(spec, max_hz=3000)
    assert len(peaks) == 3
    for (hz, _), p in zip(peaks, PARTIALS):
        assert abs(hz - p * F0) < config.BIN_HZ
    rel = [h for _, h in peaks]
    assert np.allclose(rel, [1.0, 0.45, 0.21], atol=0.04)   # window smears 0.5, 0.25


def test_marimba_decomposition(bank, spec):
    """NNLS explains the synthetic tone with a real marimba C3 plus
    marimba notes on the 4th (C5) and 10th (about E6) partials."""
    fit = nnls_fit(bank.select(instruments=["mrb"]), spec)
    top = [n for n, _ in fit.ranked(3)]
    assert top[0] == "mrb-c3-ord-mf"
    assert set(top) == {"mrb-c3-ord-mf", "mrb-c5-ord-mf", "mrb-e6-ord-mf"}
    assert fit.relative_error < 0.3


def test_spectrum_round_trip(spec):
    """Spectrum -> audio -> spectrum keeps the shape (cosine about 0.97).

    Not 1.0: each already window-smeared bin is resynthesized as its own
    sine and smeared again by the second analysis."""
    y = spectrum_to_audio(spec)
    back = make_fft(y)
    cos = back @ spec / np.linalg.norm(back) / np.linalg.norm(spec)
    assert cos > 0.95


def test_resynthesis_is_repeatable(spec):
    assert np.array_equal(spectrum_to_audio(spec, seed=3), spectrum_to_audio(spec, seed=3))
