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
CACHE_FILE = ROOT / ".cache" / "fftbank.npz"   # Bank v1 cache (text files are slow)

# Bank v2 (Phase 1): power spectra with levels, built by laborch.build
BANK_VERSION = 2
BANK_V2_DIR = BANK_DIR / "Bank_v2"
BANK_FILE = BANK_V2_DIR / "bank_v2.npz"     # names + power spectra (float32)
BANK_META = BANK_V2_DIR / "bank_v2.csv"     # metadata table, one row per sound
BUILD_CACHE = ROOT / ".cache" / "build_v2.npz"  # resumable build state

# FFT pipeline (RH2009; notebooks 1.4 and 1.11)
SAMPLE_RATE = 44100
NFFT = 4096                  # Hamming window length
HOP = 512                    # frame step
NBINS = NFFT // 2            # 2,048 positive-frequency bins kept
BIN_HZ = SAMPLE_RATE / NFFT  # 10.77 Hz per bin
SHAVE = 0.15                 # v1: fraction trimmed from each end (attack / decay)

# Sustain detection (v2; see spectra.find_sustain)
SMOOTH_FRAMES = 17           # level smoothed over 17 frames (about 0.2 s)
LOUD_DB = 10.0               # reference = median level of frames within 10 dB of the peak
SUSTAIN_DB = 4.0             # stable = within +/- this many dB of the reference
EDGE_FRAMES = 2              # frames dropped at each end of the stable run
MIN_FRAMES = 3               # fewer stable frames than this -> fixed-trim fallback
DECAY_DB = 20.0              # decaying sounds: stop when 20 dB below the peak
DECAY_MAX_S = 0.5            # ... or after this long
