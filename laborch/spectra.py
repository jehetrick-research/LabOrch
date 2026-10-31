"""Audio loading and the averaged-spectrum pipeline (notebooks 1.4, 1.11, 1.12).

``make_fft`` reproduces exactly how ``Soundbank_McGill/FFTBank`` was built:
trim 15% from each end, slide a 4,096-sample Hamming window in steps of 512,
combine the frames as sqrt(sum |X_k|^2) / n, keep bins 0..2047
(bin 0 = DC, bin spacing 10.77 Hz), then divide by the sum so every
spectrum has unit area.

Consequences worth knowing (Phase 1 will revisit them):
  * every Bank sound has the same total magnitude, so recorded loudness is
    gone and NNLS weights are shares, not playing levels;
  * the result is a magnitude, not a power, spectrum.
Recomputing a Bank file from its .aif matches the stored values to about
0.1% of the peak (decoder differences).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np

from . import config


def _pydub():
    try:
        from pydub import AudioSegment
    except ImportError as e:  # pragma: no cover
        raise ImportError("audio I/O needs pydub (and ffmpeg): pip install pydub") from e
    return AudioSegment


def load_segment(path: Union[str, Path]):
    """Read any ffmpeg-readable file as a pydub AudioSegment."""
    return _pydub().from_file(str(path))


def segment_samples(segment) -> np.ndarray:
    """Mono samples of an AudioSegment as float, in the file's integer scale.

    The integer scale (not -1..1) matches the values used to build the Bank.
    """
    if segment.frame_rate != config.SAMPLE_RATE:
        warnings.warn(f"sample rate {segment.frame_rate} Hz, Bank uses "
                      f"{config.SAMPLE_RATE} Hz; bins will not line up")
    mono = segment.set_channels(1)
    return np.asarray(mono.get_array_of_samples(), dtype=float)


def load_samples(path: Union[str, Path]) -> np.ndarray:
    return segment_samples(load_segment(path))


def make_fft(samples: np.ndarray, shave: float = config.SHAVE,
             normalize: bool = True) -> np.ndarray:
    """Averaged magnitude spectrum of the sustained part of a note.

    Parameters
    ----------
    samples   mono samples (see ``segment_samples``)
    shave     fraction removed from the start and from the end
    normalize divide by the sum (True reproduces FFTBank; False keeps level)
    """
    x = np.asarray(samples, dtype=float)
    n_samp = len(x)
    begin = int(shave * n_samp)
    end = int((1 - shave) * n_samp)
    win = np.hamming(config.NFFT)
    acc = np.zeros(config.NBINS)
    n = 0
    for i in range(begin, end - config.NFFT, config.HOP):
        frame = x[i:i + config.NFFT] * win
        acc += (np.abs(np.fft.fft(frame)) ** 2)[:config.NBINS]
        n += 1
    if n == 0:
        raise ValueError(f"sound too short: {n_samp} samples after trimming "
                         f"leaves less than one {config.NFFT}-sample window")
    spec = np.sqrt(acc) / n
    if normalize:
        spec = spec / spec.sum()
    return spec


def fft_of_file(path: Union[str, Path], **kw) -> np.ndarray:
    return make_fft(load_samples(path), **kw)


def split_on_silence(path: Union[str, Path], min_silence_len: int = 500,
                     silence_thresh: float = -50, keep_silence: int = 100) -> list:
    """Split a multi-note recording into notes (notebooks 1.8, 1.9, 1.12)."""
    from pydub.silence import split_on_silence as _split
    return _split(load_segment(path), min_silence_len=min_silence_len,
                  silence_thresh=silence_thresh, keep_silence=keep_silence)


def spectral_peaks(spec: np.ndarray, rel_height: float = 0.02,
                   max_hz: Optional[float] = None) -> list:
    """Local maxima of a spectrum, as [(Hz, height relative to the tallest)].

    Only peaks at least ``rel_height`` times the tallest are kept, lowest
    frequency first.
    """
    spec = np.asarray(spec, float)
    top = spec.max()
    hz = np.arange(len(spec)) * config.BIN_HZ
    out = []
    for k in range(1, len(spec) - 1):
        if max_hz is not None and hz[k] > max_hz:
            break
        if spec[k] > spec[k - 1] and spec[k] >= spec[k + 1] and spec[k] >= rel_height * top:
            out.append((float(hz[k]), float(spec[k] / top)))
    return out
