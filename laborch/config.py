"""Paths and analysis constants.

All file locations are derived from ``ROOT``, the LabOrch repository folder.
Set the environment variable ``LABORCH_ROOT`` to point somewhere else
(e.g. when the package is installed outside the repo).
"""

import os
from pathlib import Path

ROOT = Path(os.environ.get("LABORCH_ROOT", Path(__file__).resolve().parent.parent))

BANK_DIR = ROOT / "Soundbank_McGill"
FFT_DIR = BANK_DIR / "FFTBank"            # current Bank: 1,515 x 2,048-bin .txt spectra
SOUND_DIRS = [                             # matching single-note .aif files
    BANK_DIR / "Soundbank_WB",
    BANK_DIR / "Soundbank_Str",
    BANK_DIR / "Soundbank_PianoPercSax",
]
OLD_FFT_DIR = ROOT / "OldFFTBank"          # 2009 .nbb spectra (float32), reference only
MISC_SOUNDS_DIR = ROOT / "MiscSounds"
CACHE_FILE = ROOT / ".cache" / "fftbank.npz"

# FFT pipeline (RH2009; notebooks 1.4 and 1.11)
SAMPLE_RATE = 44100
NFFT = 4096                  # Hamming window length
HOP = 512                    # frame step
NBINS = NFFT // 2            # 2,048 positive-frequency bins kept
BIN_HZ = SAMPLE_RATE / NFFT  # 10.77 Hz per bin
SHAVE = 0.15                 # fraction trimmed from each end (attack / decay)
