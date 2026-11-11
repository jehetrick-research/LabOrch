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


def test_marimba_decomposition(bank_v1, spec):
    """NNLS explains the synthetic tone with a real marimba C3 plus
    marimba notes on the 4th (C5) and 10th (about E6) partials."""
    fit = nnls_fit(bank_v1.select(instruments=["mrb"]), spec)
    top = [n for n, _ in fit.ranked(3)]
    assert top[0] == "mrb-c3-ord-mf"
    assert set(top) == {"mrb-c3-ord-mf", "mrb-c5-ord-mf", "mrb-e6-ord-mf"}
    assert fit.relative_error < 0.3


def test_spectrum_round_trip(spec):
    """Spectrum -> audio -> spectrum keeps the shape.

    method="bins" reaches only about 0.97: each already window-smeared bin
    is resynthesized as its own sine and smeared again by the second
    analysis. method="peaks" rebuilds one sine per partial."""
    def cos(y):
        back = make_fft(y)
        return back @ spec / np.linalg.norm(back) / np.linalg.norm(spec)
    assert cos(spectrum_to_audio(spec, method="bins")) > 0.95
    assert cos(spectrum_to_audio(spec)) > 0.995


def test_resynthesis_is_repeatable(spec):
    assert np.array_equal(spectrum_to_audio(spec, seed=3), spectrum_to_audio(spec, seed=3))


def _wobble(y, sr=config.SAMPLE_RATE):
    """Depth of 5-40 Hz amplitude modulation, from a 5 ms RMS envelope."""
    fr = int(0.005 * sr)
    env = np.sqrt(np.mean(y[:len(y) // fr * fr].reshape(-1, fr) ** 2, axis=1))[20:-20]
    E = np.abs(np.fft.rfft(env - env.mean()))
    f = np.fft.rfftfreq(len(env), 0.005)
    band = (f > 5) & (f < 40)
    return E[band].max() / (len(env) * env.mean() / 2)


def test_peaks_remove_bin_beating():
    """A sawtooth-like F2 (87.31 Hz, between bins): bin-by-bin resynthesis
    wavers at about 10.8 Hz; peak resynthesis stays near the original."""
    n = np.arange(1, 16)
    x = tone(87.31, partials=n, amps=1 / n, seconds=2)
    spec = make_fft(x)
    assert _wobble(spectrum_to_audio(spec, seconds=2, method="bins")) > 0.1
    assert _wobble(spectrum_to_audio(spec, seconds=2)) < 2 * _wobble(x) + 0.02


def test_peak_frequencies_are_interpolated():
    from laborch.synth import _interpolated_peaks
    spec = make_fft(tone(87.31, partials=[1, 3], amps=[1, 0.5]))
    hz, amp = _interpolated_peaks(spec, 1e-3 * spec.max(), config.BIN_HZ)
    main = np.sort(hz[np.argsort(-amp)[:2]])                 # skip window side-lobes
    assert np.allclose(main, [87.31, 261.93], atol=0.5)      # bin grid is off by ~1.2 Hz
