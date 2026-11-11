"""Bank v2 analysis: power spectra, levels and sustain detection."""
import numpy as np
import pytest

from laborch import config
from laborch.spectra import find_sustain, frame_power, power_of_file, power_spectrum
from tests.conftest import requires_audio

SR = config.SAMPLE_RATE


def _t(seconds):
    return np.arange(int(seconds * SR)) / SR


def test_parseval_level():
    """A sine of amplitude A has mean square A^2/2: the spectrum sums to it."""
    x = 0.5 * np.sin(2 * np.pi * 440 * _t(2))
    spec, sus = power_spectrum(x)
    assert abs(10 * np.log10(spec.sum()) - 10 * np.log10(0.125)) < 0.05
    assert abs(sus.level_db - 10 * np.log10(0.125)) < 0.05
    assert abs(spec.argmax() * config.BIN_HZ - 440) < config.BIN_HZ


def test_powers_of_uncorrelated_sounds_add():
    rng = np.random.default_rng(1)
    a = 0.1 * rng.standard_normal(2 * SR)
    b = 0.2 * np.sin(2 * np.pi * 300 * _t(2))
    pa, pb, pab = (power_spectrum(x)[0] for x in (a, b, a + b))
    assert abs(pab.sum() / (pa.sum() + pb.sum()) - 1) < 0.01


def test_sustain_skips_fades_and_silence():
    t = _t(2.5)
    env = np.interp(t, [0, 0.3, 1.3, 1.6, 2.5], [0, 1, 1, 0, 0])
    _, sus = power_spectrum(0.5 * env * np.sin(2 * np.pi * 440 * t))
    assert sus.method == "sustain"
    assert 0.1 < sus.start_s < 0.35 and 1.2 < sus.end_s < 1.45


def test_decay_window_starts_at_attack():
    t = _t(2)
    x = 0.5 * np.exp(-6 * t) * np.sin(2 * np.pi * 440 * t)
    x = np.concatenate([np.zeros(SR // 4), x])
    _, sus = power_spectrum(x, envelope="decaying")
    assert sus.method == "decay"
    assert 0.15 < sus.start_s < 0.3
    assert sus.end_s - sus.start_s <= config.DECAY_MAX_S + config.NFFT / SR + 1e-9


def test_fallback_to_trim():
    """Too few stable frames -> the v1 15% trim; a steady note never falls back."""
    from laborch.spectra import _trim_frames
    first, last, method = _trim_frames(100)
    assert method == "trim" and 10 <= first < last <= 90
    for n in (6, 8):                       # very short notes, steep ramp
        ms = 10 ** (np.arange(n) * 3.0)    # +30 dB per frame
        sus = find_sustain(ms)
        assert sus.method in ("sustain", "trim") and 0 <= sus.first <= sus.last < n
    assert find_sustain(np.full(50, 1e-3)).method == "sustain"


def test_given_region():
    x = 0.3 * np.sin(2 * np.pi * 220 * _t(2))
    _, sus = power_spectrum(x, region=(0.5, 1.0))
    assert sus.method == "given" and 0.49 < sus.start_s < 0.52 and 0.9 < sus.end_s <= 1.0


def test_bad_envelope():
    _, ms = frame_power(np.ones(10000))
    with pytest.raises(ValueError):
        find_sustain(ms, "pulsing")


@requires_audio
@pytest.mark.parametrize("name", ["flu-e4-ord-mf", "vln-a4x-ord-mf", "pno-c4-ord-ff", "mrb-c5-ord-mf"])
def test_rebuilds_bank_v2(bank, name):
    env = bank.meta[name]["envelope"]
    got, sus = power_of_file(bank.sound_path(name), env)
    ref = bank.spectrum(name)
    assert np.max(np.abs(got - ref)) / ref.max() < 1e-3
    assert abs(sus.start_s - bank.meta[name]["sustain_start_s"]) < 1e-3


def test_level_matches_spectrum_sum(bank):
    for n in bank.names[::50]:
        assert abs(bank.level_db(n) - bank.meta[n]["level_db"]) < 0.01


def test_unit_normalize_option(bank):
    from laborch import Bank
    b = Bank.load(normalize="unit")
    assert np.allclose(b.spectra.sum(axis=1), 1)
    assert b.names == bank.names
