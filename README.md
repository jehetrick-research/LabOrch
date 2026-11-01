# LabOrch

Computer-assisted orchestration: given a **Target** sound and an **Ensemble**
of instruments, find the notes (the **Recommendation**) whose combined
spectrum best matches the Target. Formerly *Orchestration Engine*; based on
F. Rose and J. Hetrick, *Enhancing Orchestration Technique via Spectrally
Based Linear Algebra Methods*, Computer Music Journal 33(1), 2009 (RH2009; PDF in `Reading/`).

## Layout

| Path | What |
| --- | --- |
| `laborch/` | Python package (Phase 0): Bank, FFT pipeline, DOT / NNLS solvers, mixing, synthetic tones |
| `tests/` | pytest suite (28 tests), incl. regressions for notebook 1.5 and the Ravel example (1.12) |
| `Soundbank_McGill/FFTBank/` | Spectra of the McGill (MUMS) single-note samples |
| `OldFFTBank/` | 2009 `.nbb` spectra, kept for reference and the 1.5 regression test |
| `docs/` | Package manual (`laborch_manual` .md / .docx / .tex / .pdf) |
| `Reading/` | The RH2009 paper, project overview deck (`LabOrch.pptx`) and background reading |

## Quick start

```bash
pip install -e ".[dev]"      # numpy, scipy, pydub, pytest; audio also needs ffmpeg
pytest                       # ~10 s; first run builds .cache/fftbank.npz (about 1 min)
```

```python
from laborch import Bank, nnls_fit

bank = Bank.load()                                    # 1,515 spectra x 2,048 bins
target = bank.spectrum("trb-f2-ord-mf")               # a Target from the Bank
palette = bank.select(instruments=["clb", "bcl"],     # the Ensemble
                      exclude=["trb-f2-ord-mf"])
fit = nnls_fit(palette, target)                       # non-negative least squares
print(fit.summary(8))
fit.top(3, refit=True).ranked()                       # keep 3 sounds and refit
```

To hear a result: `from laborch.audio import render_fit; render_fit(bank, fit, n=4, path="mix.wav")`.

## Data

The repo includes the spectra (`FFTBank/`, `OldFFTBank/`) but not the audio.
The McGill University Master Samples (MUMS) recordings and the extra Targets
in `MiscSounds/` are licensed material and are not redistributed. If you have
them, place them at `Soundbank_McGill/Soundbank_{WB,Str,PianoPercSax}/` and
`MiscSounds/` (or point `LABORCH_ROOT` at a folder that has them). Tests that
need them are skipped otherwise.

## Conventions

- Sound names: `inst-pitch-mode-dyn`, e.g. `vln-a4x-ord-mf` (see `laborch/bank.py`).
- Spectra: 4,096-point Hamming windows, hop 512, middle 70% of the note,
  bins 0-2,047 at 10.77 Hz, combined as sqrt(sum |X|^2)/n and **normalized to unit sum**.
- Paths are set in `laborch/config.py`; override the root with `LABORCH_ROOT`.

## Status

Phase 0 (consolidation) done. Next: Phase 1, the Bank as a dataset
(metadata table, power spectra, loudness), then Phase 2, the imitation engine
(weighted NNLS + one-note-per-player constraints).
