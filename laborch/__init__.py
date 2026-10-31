"""LabOrch: computer-assisted orchestration by spectral decomposition.

Phase 0 package: the logic from the 2019-2020 notebooks (1.5, 1.7, 1.11,
1.12 and the tool*.py widget scripts), gathered into importable modules.

Modules
-------
config   paths and FFT constants (one place to change paths)
bank     sound names, the Bank of spectra, Palette selection
spectra  audio loading and the averaged-FFT pipeline (notebook 1.11)
solve    DOT, pseudo-inverse and NNLS decompositions (notebook 1.5, RH2009)
audio    mixing Recommendations for playback (notebook 1.7)
synth    synthetic tones and spectrum-to-audio resynthesis (notebook 1.2)
"""

from .bank import Bank, Palette, SoundName, parse_name, INSTRUMENT_NAMES
from .solve import Fit, dot, nnls_fit, pinv_fit, chi2
from .spectra import make_fft, spectral_peaks
from .synth import tone, spectrum_to_audio

__all__ = [
    "Bank", "Palette", "SoundName", "parse_name", "INSTRUMENT_NAMES",
    "Fit", "dot", "nnls_fit", "pinv_fit", "chi2", "make_fft", "spectral_peaks",
    "tone", "spectrum_to_audio",
]
__version__ = "0.1.0"
