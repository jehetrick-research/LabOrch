"""Hearing a Recommendation: mixing Bank recordings (notebook 1.7)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from .bank import Bank
from .solve import Fit
from .spectra import load_segment


def weights_to_db(weights: Sequence[float], floor_db: float = -60.0) -> np.ndarray:
    """Gain in dB for each weight, relative to the largest weight.

    The Bank holds magnitude spectra, so a weight scales amplitude:
    gain = 20 log10(w / w_max). Zero weights get ``floor_db``.

    This is approximate: Bank spectra are normalized to unit sum, so the
    recordings' own loudness differences are not accounted for (Phase 1).
    """
    w = np.asarray(weights, float)
    out = np.full(w.shape, floor_db)
    pos = w > 0
    if pos.any():
        out[pos] = np.maximum(20 * np.log10(w[pos] / w[pos].max()), floor_db)
    return out


def mix(bank: Bank, names: Sequence[str], gains_db: Sequence[float]):
    """Overlay Bank recordings, each with its own gain, into one AudioSegment.

    The result is as long as the longest sound. (Notebook 1.7 indexed the
    gains in Palette order instead of ranked order; here each gain travels
    with its name.)
    """
    if len(names) != len(gains_db):
        raise ValueError("names and gains_db differ in length")
    segs = [load_segment(bank.sound_path(n)) + float(g) for n, g in zip(names, gains_db)]
    segs.sort(key=len, reverse=True)
    out = segs[0]
    for s in segs[1:]:
        out = out.overlay(s)
    return out


def render_fit(bank: Bank, fit: Fit, n: int = 8,
               path: Optional[Union[str, Path]] = None):
    """Mix the top-n sounds of a Fit at their relative gains.

    Writes a file if ``path`` is given (format from its extension).
    """
    ranked = fit.ranked(n)
    names = [r[0] for r in ranked]
    gains = weights_to_db([r[1] for r in ranked])
    seg = mix(bank, names, gains)
    if path is not None:
        path = Path(path)
        seg.export(str(path), format=path.suffix.lstrip(".") or "wav")
    return seg
