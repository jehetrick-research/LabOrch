"""LabOrch: computer-assisted orchestration by spectral decomposition.

Phase 0 gathered the 2019-2020 notebooks into this package; Phase 1 made
the Bank a dataset (Bank v2: power spectra over a detected sustain, with
recorded levels and a metadata table).

Modules
-------
config   paths and FFT constants (one place to change paths)
bank     sound names, the Bank of spectra (v2 default, v1), Palette selection
metadata the Bank's metadata table: families, envelopes, ranges
spectra  audio loading; v2 power spectra with sustain detection; v1 make_fft
build    rebuild Bank v2 from the recordings (python -m laborch.build)
solve    DOT, pseudo-inverse and NNLS decompositions (notebook 1.5, RH2009)
audio    mixing Recommendations for playback (notebook 1.7)
synth    synthetic tones and spectrum-to-audio resynthesis (notebook 1.2)
"""

from .bank import Bank, Palette, SoundName, parse_name, INSTRUMENT_NAMES
from .solve import Fit, dot, nnls_fit, pinv_fit, chi2
from .spectra import make_fft, power_spectrum, find_sustain, spectral_peaks
from .metadata import FAMILIES
from .synth import tone, spectrum_to_audio

__all__ = [
    "Bank", "Palette", "SoundName", "parse_name", "INSTRUMENT_NAMES",
    "Fit", "dot", "nnls_fit", "pinv_fit", "chi2", "make_fft", "power_spectrum",
    "find_sustain", "spectral_peaks", "FAMILIES",
    "tone", "spectrum_to_audio",
]
__version__ = "0.2.0"
