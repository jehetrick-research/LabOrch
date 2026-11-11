"""Build Bank v2 from the McGill recordings (Phase 1).

    python -m laborch.build            # resumable; rerun until it says done
    python -m laborch.build --fresh    # start over

For each .aif in ``config.SOUND_DIRS`` it finds the sustain (or, for
struck sounds, the decay window), averages the power spectrum over it, and
records the level. Results are saved to ``config.BUILD_CACHE`` after every
batch, so a run can be stopped and continued. When every sound is done it
writes:

  Soundbank_McGill/Bank_v2/bank_v2.npz   names, spectra (float32, power)
  Soundbank_McGill/Bank_v2/bank_v2.csv   the metadata table

``similarity_v1`` is the cosine similarity between the new spectrum (as a
magnitude, sqrt of power) and the v1 FFTBank spectrum of the same sound;
low values flag sounds whose analysis region changed a lot.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from . import config
from .metadata import describe, write_table, COLUMNS


# Recordings whose file name has a typo, mapped to the Bank name
# (the spreadsheet row for WW-17 reads 'tpt' instead of 'tpc').
NAME_FIXES = {"tpt-fs3-hmo-mf": "tpc-fs3-hmo-mf"}


def _sound_files():
    out = {}
    for d in config.SOUND_DIRS:
        for f in sorted(Path(d).glob("*.aif")):
            out[NAME_FIXES.get(f.stem, f.stem)] = f
    return dict(sorted(out.items()))


def _load_state(path: Path, fresh: bool):
    if fresh or not path.exists():
        return {}
    z = np.load(path, allow_pickle=True)
    return z["state"].item()


def _save_state(path: Path, state: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        np.savez(f, state=np.array(state, dtype=object))


def analyse(name: str, path: Path) -> dict:
    """Spectrum and metadata row for one Bank recording."""
    from .spectra import load_segment, segment_float, power_spectrum
    row = describe(name)
    seg = load_segment(path)
    if seg.frame_rate != config.SAMPLE_RATE:
        raise ValueError(f"{name}: {seg.frame_rate} Hz")
    x = segment_float(seg)
    spec, sus = power_spectrum(x, row["envelope"])
    row.update({
        "sub_bank": Path(path).parent.name,
        "aif": Path(path).relative_to(config.ROOT).as_posix(),
        "duration_s": round(len(x) / config.SAMPLE_RATE, 4),
        "sustain_start_s": round(sus.start_s, 4),
        "sustain_end_s": round(sus.end_s, 4),
        "n_frames": sus.n_frames,
        "method": sus.method,
        "level_db": round(float(10 * np.log10(spec.sum())), 3),
        "peak_db": round(sus.peak_db, 3),
    })
    return {"row": row, "spec": spec.astype(np.float32)}


def build(budget_s: float = 150.0, fresh: bool = False, batch: int = 50,
          verbose: bool = True) -> bool:
    """Analyse sounds until done or ``budget_s`` runs out. True when done."""
    files = _sound_files()
    state = _load_state(config.BUILD_CACHE, fresh)
    state = {n: v for n, v in state.items() if n in files}   # drop stale names
    todo = [n for n in files if n not in state]
    t0 = time.time()
    while todo and time.time() - t0 < budget_s:
        for n in todo[:batch]:
            state[n] = analyse(n, files[n])
        todo = todo[batch:]
        _save_state(config.BUILD_CACHE, state)
        if verbose:
            print(f"{len(state)}/{len(files)} analysed", flush=True)
    if todo:
        if verbose:
            print(f"{len(todo)} left; run again to continue")
        return False
    finish(state)
    return True


def finish(state: dict) -> None:
    """Write the npz and CSV from a complete build state."""
    names = sorted(state)
    spectra = np.vstack([state[n]["spec"] for n in names])
    rows = [state[n]["row"] for n in names]
    v1 = _v1_spectra()
    for r, s in zip(rows, spectra):
        old = v1.get(r["name"])
        if old is not None:
            m = np.sqrt(s.astype(float))
            r["similarity_v1"] = round(float(m @ old / np.linalg.norm(m) / np.linalg.norm(old)), 4)
        else:
            r["similarity_v1"] = ""
    config.BANK_V2_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        config.BANK_FILE, names=np.array(names), spectra=spectra,
        bank_version=config.BANK_VERSION, kind="power",
        sample_rate=config.SAMPLE_RATE, nfft=config.NFFT, hop=config.HOP)
    write_table(rows, config.BANK_META)
    print(f"wrote {config.BANK_FILE} and {config.BANK_META} ({len(names)} sounds)")


def _v1_spectra() -> dict:
    try:
        from .bank import Bank
        b = Bank.load(version=1)
    except Exception:
        return {}
    return {n: b.spectrum(n) for n in b.names}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fresh", action="store_true", help="discard earlier progress")
    ap.add_argument("--budget", type=float, default=150.0, help="seconds per run")
    a = ap.parse_args(argv)
    build(a.budget, a.fresh)


if __name__ == "__main__":
    main()
