"""Notebook 1.12 / RH2009 Fig. 1: Ravel, Daphnis et Chloe, rehearsal 17.

The woodwind chord (2 oboes, English horn, Eb clarinet; transposed to C)
should resemble a cup-muted trumpet more than a straight-muted one.

Note: by plain Euclidean distance between the unit-sum spectra the two
mutes come out nearly tied (straight is marginally closer), while every
shape-based measure picks the cup mute. That gap is exactly the
weighting question for Phase 2.
"""
import numpy as np
import pytest

from laborch import config
from laborch.spectra import make_fft, segment_samples, split_on_silence
from tests.conftest import requires_audio

MUTES = config.MISC_SOUNDS_DIR / "Trumpet-mutes_Basson-multiphics"
WOODWINDS = ["obo-c5-ord-mf", "obo-c6-ord-mf", "enh-c5-ord-mf", "cle-c6-ord-mf"]


def _mute_c5(track):
    chunks = split_on_silence(MUTES / track)   # notebook settings
    assert len(chunks) == 6
    return make_fft(segment_samples(chunks[2]))  # third note = C5


@pytest.fixture(scope="module")
def spectra(bank_v1):
    if not MUTES.exists():
        pytest.skip("MiscSounds not present")
    wood = sum(bank_v1.spectrum(n) for n in WOODWINDS)
    wood = wood / wood.sum()
    return wood, _mute_c5("5 Track 05.aiff"), _mute_c5("11 Track 11.aiff")


def _cos(a, b):
    return a @ b / np.linalg.norm(a) / np.linalg.norm(b)


def _log_dist(a, b, lo=5, hi=1000):
    return np.linalg.norm(np.log10(a[lo:hi] + 1e-12) - np.log10(b[lo:hi] + 1e-12))


@requires_audio
def test_cup_mute_is_closer(spectra):
    wood, straight, cup = spectra
    assert _cos(wood, cup) > _cos(wood, straight)
    assert _log_dist(wood, cup) < _log_dist(wood, straight)


@requires_audio
def test_energy_sits_where_the_paper_says(spectra):
    """Woodwinds and cup mute peak near the 2nd partial (about 1,050 Hz);
    the straight mute peaks much higher."""
    wood, straight, cup = spectra
    hz = lambda s: s.argmax() * config.BIN_HZ
    assert abs(hz(cup) - hz(wood)) < 50
    assert hz(straight) > 2 * hz(wood)


# ---- the same example on Bank v2 (power spectra, detected sustain) --------

@pytest.fixture(scope="module")
def spectra_v2(bank):
    if not MUTES.exists():
        pytest.skip("MiscSounds not present")
    from laborch.spectra import power_spectrum, segment_float

    def mute(track):
        chunks = split_on_silence(MUTES / track)
        return power_spectrum(segment_float(chunks[2]))[0]

    wood = sum(bank.spectrum(n) for n in WOODWINDS)   # powers add
    return wood, mute("5 Track 05.aiff"), mute("11 Track 11.aiff")


@requires_audio
def test_cup_mute_is_closer_v2(spectra_v2):
    wood, straight, cup = spectra_v2
    assert _cos(wood, cup) > _cos(wood, straight)
    assert _cos(np.sqrt(wood), np.sqrt(cup)) > _cos(np.sqrt(wood), np.sqrt(straight))
    unit = lambda s: s / s.sum()
    assert _log_dist(unit(wood), unit(cup)) < _log_dist(unit(wood), unit(straight))


@requires_audio
def test_energy_sits_where_the_paper_says_v2(spectra_v2):
    wood, straight, cup = spectra_v2
    hz = lambda s: s.argmax() * config.BIN_HZ
    assert abs(hz(cup) - hz(wood)) < 50
    assert hz(straight) > 2 * hz(wood)
