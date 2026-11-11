# LabOrch

Computer-assisted orchestration: given a **Target** sound and an **Ensemble**
of instruments, find the notes (the **Recommendation**) whose combined
spectrum best matches the Target. Formerly *Orchestration Engine*; based on
F. Rose and J. Hetrick, *Enhancing Orchestration Technique via Spectrally
Based Linear Algebra Methods*, Computer Music Journal 33(1), 2009 (RH2009; PDF in `Reading/`).

## Layout

| Path | What |
| --- | --- |
| `laborch/` | Python package: Bank (v2 and v1), metadata, sustain detection and spectra, DOT / NNLS solvers, mixing, synthetic tones |
| `tests/` | pytest suite (61 tests), incl. regressions for notebook 1.5 and the Ravel example (1.12) |
| `Soundbank_McGill/Bank_v2/` | **Bank v2** (default): power spectra and the metadata table (`bank_v2.csv`) |
| `Soundbank_McGill/FFTBank/` | Bank v1: the 2019-2020 spectra of the McGill (MUMS) single-note samples |
| `OldFFTBank/` | 2009 `.nbb` spectra, kept for reference and the 1.5 regression test |
| `docs/` | Package manual (`laborch_manual` .md / .docx / .tex / .pdf); Bank v2 report (`bank_v2.md`) |
| `Reading/` | The RH2009 paper, project overview deck (`LabOrch.pptx`) and background reading |

## Quick start

```bash
pip install -e ".[dev]"      # numpy, scipy, pydub, pytest; audio also needs ffmpeg
pytest                       # ~15 s; the first run caches Bank v1 (about 1 min)
```

```python
from laborch import Bank, nnls_fit

bank = Bank.load()                                    # Bank v2: 1,515 power spectra x 2,048 bins
target = bank.spectrum("trb-f2-ord-mf")               # a Target from the Bank
palette = bank.select(instruments=["clb", "bcl"],     # the Ensemble
                      exclude=["trb-f2-ord-mf"])
fit = nnls_fit(palette, target)                       # non-negative least squares
print(fit.summary(8))
fit.top(3, refit=True).ranked()                       # keep 3 sounds and refit
bank.meta["trb-f2-ord-mf"]                            # metadata: family, range, sustain, level...
```

To hear a result: `from laborch.audio import render_fit; render_fit(bank, fit, n=4, path="mix.wav")`.

## Data

The repo includes the spectra (`Bank_v2/`, `FFTBank/`, `OldFFTBank/`) but not the audio.
The McGill University Master Samples (MUMS) recordings and the extra Targets
in `MiscSounds/` are licensed material and are not redistributed. If you have
them, place them at `Soundbank_McGill/Soundbank_{WB,Str,PianoPercSax}/` and
`MiscSounds/` (or point `LABORCH_ROOT` at a folder that has them). Tests that
need them are skipped otherwise.

## Conventions

- Sound names: `inst-pitch-mode-dyn`, e.g. `vln-a4x-ord-mf` (see `laborch/bank.py`).
- Bank v2 spectra (default): 4,096-point Hamming windows, hop 512, bins 0-2,047
  at 10.77 Hz; **power**, averaged over each note's automatically detected sustain
  (struck sounds: the first 0.5 s from the attack), at the recorded level, so
  `10*log10(sum)` is the level in dBFS and powers of sounds played together add.
  Rebuild with `python -m laborch.build`; see `docs/bank_v2.md`.
- Bank v1 (`Bank.load(version=1)`): magnitude over the middle 70% of the note,
  sqrt(sum |X|^2)/n, normalized to unit sum.
- Paths are set in `laborch/config.py`; override the root with `LABORCH_ROOT`.

## Status

Phase 0 (consolidation) and Phase 1 (the Bank as a dataset: metadata table,
sustain detection, power spectra with levels) done. Next: Phase 2, the
imitation engine (choice of frequency weighting, a loudness model, and
one-note-per-player constraints).
