"""Synthetic sounds: build tones from partials, and turn spectra back into audio.

Reproduces the experiments of notebooks 1.2/1.3 (a C3 "marimba" made of the
1st, 4th and 10th partials) in a form that plugs into the rest of laborch:

    from laborch.synth import tone, spectrum_to_audio
    from laborch.spectra import make_fft
    x = tone(130.81, partials=[1, 4, 10], amps=[1, 0.5, 0.25])
    spec = make_fft(x)                    # a Target in Bank format
    y = spectrum_to_audio(spec)           # back to sound (random phases)

Bank spectra keep magnitudes only, so ``spectrum_to_audio`` has to invent
phases. The result has the right timbre (which partials, how strong) but
not the original waveform, attack or decay.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from . import config


def tone(f0: float, partials: Sequence[float] = (1,), amps: Optional[Sequence[float]] = None,
         seconds: float = 1.0, sr: int = config.SAMPLE_RATE, decay: float = 0.0,
         phases: Optional[Sequence[float]] = None, peak: float = 1.0) -> np.ndarray:
    """Sum of sine partials at multiples of ``f0``.

    Parameters
    ----------
    f0        fundamental in Hz (``parse_name("mrb-c3-ord-mf").freq`` = 130.81)
    partials  multiples of f0; need not be integers (e.g. 2.76 for a bar)
    amps      amplitude of each partial (default all 1)
    seconds   duration
    sr        sample rate
    decay     exponential decay rate in 1/s; 0 = steady (1.2 used 10)
    phases    starting phase of each partial in radians (default 0)
    peak      the largest |sample| after scaling; 2**15 - 1 gives 16-bit scale

    Returns a float array of ``int(seconds * sr)`` samples.
    """
    partials = np.asarray(partials, float)
    amps = np.ones_like(partials) if amps is None else np.asarray(amps, float)
    phases = np.zeros_like(partials) if phases is None else np.asarray(phases, float)
    if not (len(partials) == len(amps) == len(phases)):
        raise ValueError("partials, amps and phases must have the same length")
    if np.any(partials * f0 >= sr / 2):
        raise ValueError(f"a partial is at or above the Nyquist frequency ({sr / 2:g} Hz)")
    t = np.arange(int(seconds * sr)) / sr
    x = np.zeros_like(t)
    for p, a, ph in zip(partials, amps, phases):
        x += a * np.sin(2 * np.pi * p * f0 * t + ph)
    if decay:
        x *= np.exp(-decay * t)
    m = np.abs(x).max()
    return x * (peak / m) if m > 0 else x


def spectrum_to_audio(spec: np.ndarray, seconds: float = 1.0, sr: int = config.SAMPLE_RATE,
                      seed: Optional[int] = 0, threshold: float = 1e-4,
                      peak: float = 1.0, method: str = "peaks") -> np.ndarray:
    """Additive resynthesis of a magnitude spectrum with random phases.

    Works for Bank spectra, ``Fit.model`` and ``make_fft`` output alike.
    ``seed`` makes the random phases repeatable.

    method="peaks" (default)
        Each local maximum above ``threshold * max(spec)`` becomes one sine.
        Its frequency and amplitude come from a parabola through the log
        magnitudes of the peak bin and its two neighbours, so a partial
        between bins (e.g. F2 = 87.3 Hz) gets its own frequency instead of
        the bin grid's.
    method="bins"
        Every bin k above the threshold becomes a sine at k * BIN_HZ (the
        original behaviour). The Hamming window spreads each partial over
        about four bins, and those neighbouring sines, 10.77 Hz apart, beat:
        the result wavers at about 10.8 Hz.
    """
    spec = np.asarray(spec, float)
    if spec.ndim != 1:
        raise ValueError("spec must be one-dimensional")
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    bin_hz = sr / (2 * len(spec))
    cut = threshold * spec.max()
    if method == "bins":
        k = np.nonzero(spec > cut)[0]
        freqs, amps = k * bin_hz, spec[k]
    elif method == "peaks":
        freqs, amps = _interpolated_peaks(spec, cut, bin_hz)
    else:
        raise ValueError(f"method must be 'peaks' or 'bins', not {method!r}")
    ph = rng.uniform(0, 2 * np.pi, len(freqs))
    x = np.zeros_like(t)
    for f, a, p in zip(freqs, amps, ph):   # one sine at a time keeps memory small
        x += a * np.sin(2 * np.pi * f * t + p)
    m = np.abs(x).max()
    return x * (peak / m) if m > 0 else x


def _interpolated_peaks(spec: np.ndarray, cut: float, bin_hz: float):
    """Frequencies (Hz) and heights of the local maxima of ``spec`` above
    ``cut``, refined by quadratic interpolation of the log magnitude."""
    s = spec
    k = np.nonzero((s[1:-1] > s[:-2]) & (s[1:-1] >= s[2:]) & (s[1:-1] > cut))[0] + 1
    with np.errstate(divide="ignore"):
        a, b, c = (np.log(np.maximum(s[k + j], 1e-300)) for j in (-1, 0, 1))
    den = a - 2 * b + c
    d = np.where(den != 0, 0.5 * (a - c) / np.where(den != 0, den, 1), 0.0)
    d = np.clip(d, -0.5, 0.5)
    return (k + d) * bin_hz, np.exp(b - 0.25 * (a - c) * d)
