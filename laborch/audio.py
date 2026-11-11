"""Hearing a Recommendation: mixing Bank recordings (notebook 1.7)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from .bank import Bank
from .solve import Fit
from .spectra import load_segment


def weights_to_db(weights: Sequence[float], floor_db: float = -60.0,
                  kind: str = "magnitude", relative: bool = True) -> np.ndarray:
    """Gain in dB for each weight.

    kind      'magnitude' (Bank v1): a weight scales amplitude, 20 log10 w.
              'power' (Bank v2): a weight scales power, 10 log10 w.
    relative  True: relative to the largest weight (the loudest sound at
              0 dB). False: absolute, i.e. relative to each recording's own
              level, which is what v2 weights mean (w = 1: as recorded).

    Zero weights get ``floor_db``.
    """
    w = np.asarray(weights, float)
    k = {"magnitude": 20.0, "power": 10.0}[kind]
    out = np.full(w.shape, floor_db)
    pos = w > 0
    if pos.any():
        ref = w[pos].max() if relative else 1.0
        out[pos] = np.maximum(k * np.log10(w[pos] / ref), floor_db)
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
               path: Optional[Union[str, Path]] = None,
               headroom_db: float = 0.0):
    """Mix the top-n sounds of a Fit at their gains.

    With Bank v2 (power spectra at recorded level) each weight is a power
    gain on its recording, so the gains are absolute: 10 log10(w) dB. If the
    loudest gain would be above ``headroom_db``, all gains are lowered
    together, keeping the balance. With Bank v1 the gains are relative to
    the largest weight.

    Writes a file if ``path`` is given (format from its extension).
    """
    ranked = fit.ranked(n)
    names = [r[0] for r in ranked]
    w = [r[1] for r in ranked]
    if bank.kind == "power":
        gains = weights_to_db(w, kind="power", relative=False)
        if gains.max() > headroom_db:
            gains = gains - (gains.max() - headroom_db)
    else:
        gains = weights_to_db(w)
    seg = mix(bank, names, gains)
    if path is not None:
        path = Path(path)
        seg.export(str(path), format=path.suffix.lstrip(".") or "wav")
    return seg
