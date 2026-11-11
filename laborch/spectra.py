"""Audio loading and the averaged-spectrum pipelines.

Bank v2 (Phase 1, the default Bank)
-----------------------------------
``power_spectrum`` finds the sustained part of a note automatically
(``find_sustain``) and averages the **power** |X_k|^2 of 4,096-point Hamming
frames (hop 512) over it. Samples are scaled to -1..1 and the spectrum is
scaled so that its sum is the mean-square level of the analysed region
(Parseval): ``10*log10(spec.sum())`` is the level in dBFS. Nothing is
normalized away, so a Bank sound keeps its recorded level, and spectra of
sounds played together add (for uncorrelated sounds).

Bank v1 (Phase 0, ``Soundbank_McGill/FFTBank``)
-----------------------------------------------
``make_fft`` reproduces exactly how ``Soundbank_McGill/FFTBank`` was built:
trim 15% from each end, slide a 4,096-sample Hamming window in steps of 512,
combine the frames as sqrt(sum |X_k|^2) / n, keep bins 0..2047
(bin 0 = DC, bin spacing 10.77 Hz), then divide by the sum so every
spectrum has unit area.

Consequences worth knowing (Phase 1 addressed them in v2):
  * every Bank sound has the same total magnitude, so recorded loudness is
    gone and NNLS weights are shares, not playing levels;
  * the result is a magnitude, not a power, spectrum.
Recomputing a Bank file from its .aif matches the stored values to about
0.1% of the peak (decoder differences).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Union

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


def segment_float(segment) -> np.ndarray:
    """Mono samples of an AudioSegment scaled to -1..1 (v2)."""
    full_scale = float(2 ** (8 * segment.sample_width - 1))
    return segment_samples(segment) / full_scale


def load_float(path: Union[str, Path]) -> np.ndarray:
    """Mono samples of a file scaled to -1..1 (v2)."""
    return segment_float(load_segment(path))


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


# ---------------------------------------------------------------------------
# Bank v2: power spectra over a detected sustain
# ---------------------------------------------------------------------------

@dataclass
class Sustain:
    """The analysed region of a note, in frames and seconds.

    ``method`` is 'sustain' (stable region found), 'decay' (struck sound,
    window from the attack), 'trim' (fallback: fixed 15% trim) or 'given'.
    """
    first: int            # first frame index used
    last: int             # last frame index used (inclusive)
    method: str
    level_db: float       # mean-square level of the region, dBFS
    peak_db: float        # loudest frame, dBFS

    @property
    def n_frames(self) -> int:
        return self.last - self.first + 1

    @property
    def start_s(self) -> float:
        return self.first * config.HOP / config.SAMPLE_RATE

    @property
    def end_s(self) -> float:
        return (self.last * config.HOP + config.NFFT) / config.SAMPLE_RATE


def frames(x: np.ndarray) -> np.ndarray:
    """Overlapping 4,096-sample frames at hop 512, shape (n_frames, NFFT)."""
    x = np.asarray(x, float)
    if len(x) < config.NFFT:
        raise ValueError(f"sound too short: {len(x)} samples, need {config.NFFT}")
    n = 1 + (len(x) - config.NFFT) // config.HOP
    return np.lib.stride_tricks.as_strided(
        x, shape=(n, config.NFFT), strides=(x.strides[0] * config.HOP, x.strides[0]))


def frame_power(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-frame power spectra and per-frame mean-square level.

    Returns ``(P, ms)``: ``P`` has shape (n_frames, NBINS) and is scaled so
    that ``P[i].sum()`` is the windowed mean square of frame ``i`` (one-sided
    spectrum, Parseval); ``ms`` is the plain mean square of each frame.
    """
    F = frames(x)
    win = np.hamming(config.NFFT)
    X = np.fft.rfft(F * win, axis=1)[:, :config.NBINS]
    P = np.abs(X) ** 2 / (config.NFFT * np.sum(win ** 2))
    P[:, 1:] *= 2.0                                  # fold negative frequencies
    ms = np.mean(F ** 2, axis=1)
    return P, ms


def _db(v):
    return 10 * np.log10(np.maximum(v, 1e-20))


def _smooth_db(e: np.ndarray, width: int) -> np.ndarray:
    """Centred moving average of a dB envelope (shorter at the ends)."""
    if width <= 1 or len(e) < 2:
        return e
    k = np.ones(width)
    return np.convolve(e, k, "same") / np.convolve(np.ones_like(e), k, "same")


def _longest_run(mask: np.ndarray) -> Tuple[int, int]:
    """(first, last) of the longest run of True, or (0, -1) if none."""
    best, cur, start = (0, -1), 0, 0
    for i, m in enumerate(mask):
        if m:
            if cur == 0:
                start = i
            cur += 1
            if cur > best[1] - best[0] + 1:
                best = (start, i)
        else:
            cur = 0
    return best


def find_sustain(ms: np.ndarray, envelope: str = "sustained") -> Sustain:
    """Choose the frames to average, from per-frame mean-square levels.

    sustained  the level is smoothed over ``SMOOTH_FRAMES`` (about 0.2 s,
               so tremolo and vibrato do not break it up); the reference is
               the median of the loud part (within ``LOUD_DB`` of the
               smoothed peak); the region is the longest run of frames within
               ``SUSTAIN_DB`` of it, less ``EDGE_FRAMES`` at each end. Falls
               back to the v1 fixed trim when fewer than ``MIN_FRAMES`` remain.
    decaying   from the loudest frame until the level has fallen
               ``DECAY_DB`` below it, at most ``DECAY_MAX_S``.
    """
    e = _db(ms)
    n = len(e)
    peak = float(e.max())
    ip = int(e.argmax())
    if envelope == "decaying":
        max_frames = max(1, int(round(config.DECAY_MAX_S * config.SAMPLE_RATE / config.HOP)))
        below = np.nonzero(e[ip:] < peak - config.DECAY_DB)[0]
        last = ip + (int(below[0]) - 1 if len(below) else n - 1 - ip)
        last = min(last, ip + max_frames - 1, n - 1)
        last = max(last, min(ip + config.MIN_FRAMES - 1, n - 1))
        first, method = ip, "decay"
    elif envelope == "sustained":
        es = _smooth_db(e, config.SMOOTH_FRAMES)
        loud = es > es.max() - config.LOUD_DB
        ref = float(np.median(es[loud]))
        first, last = _longest_run(np.abs(es - ref) <= config.SUSTAIN_DB)
        if last - first + 1 > 2 * config.EDGE_FRAMES + config.MIN_FRAMES:
            first, last = first + config.EDGE_FRAMES, last - config.EDGE_FRAMES
        method = "sustain"
        if last - first + 1 < config.MIN_FRAMES:
            first, last, method = _trim_frames(n)
    else:
        raise ValueError(f"envelope must be 'sustained' or 'decaying', not {envelope!r}")
    level = float(_db(np.mean(ms[first:last + 1])))
    return Sustain(first, last, method, level, peak)


def _trim_frames(n_frames: int) -> Tuple[int, int, str]:
    """The v1 15% trim, expressed in frames."""
    n_samp = (n_frames - 1) * config.HOP + config.NFFT
    first = int(np.ceil(config.SHAVE * n_samp / config.HOP))
    last = int(((1 - config.SHAVE) * n_samp - config.NFFT) // config.HOP)
    if last < first:
        first, last = 0, n_frames - 1
    return first, min(last, n_frames - 1), "trim"


def power_spectrum(x: np.ndarray, envelope: str = "sustained",
                   region: Optional[Tuple[float, float]] = None
                   ) -> Tuple[np.ndarray, Sustain]:
    """Averaged power spectrum of a note (Bank v2), and the region used.

    Parameters
    ----------
    x         mono samples scaled to -1..1 (``load_float``)
    envelope  'sustained' (bowed, blown) or 'decaying' (struck); see
              ``find_sustain`` and ``metadata.envelope_class``
    region    (start_s, end_s) to use instead of detecting the sustain
    """
    P, ms = frame_power(x)
    if region is not None:
        first = int(np.ceil(region[0] * config.SAMPLE_RATE / config.HOP))
        last = int((region[1] * config.SAMPLE_RATE - config.NFFT) // config.HOP)
        last = min(last, len(ms) - 1)
        if last < first:
            raise ValueError(f"region {region} holds no complete frame")
        e = _db(ms)
        sus = Sustain(first, last, "given", float(_db(np.mean(ms[first:last + 1]))), float(e.max()))
    else:
        sus = find_sustain(ms, envelope)
    return P[sus.first:sus.last + 1].mean(axis=0), sus


def power_of_file(path: Union[str, Path], envelope: str = "sustained",
                  region: Optional[Tuple[float, float]] = None
                  ) -> Tuple[np.ndarray, Sustain]:
    """``power_spectrum`` of an audio file (any ffmpeg-readable format)."""
    return power_spectrum(load_float(path), envelope, region)
