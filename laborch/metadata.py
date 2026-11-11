"""The Bank's metadata table: one row per sound (Phase 1).

Most columns come from the sound name; the analysis columns (level,
sustain region, method) are filled in by ``laborch.build``. The table is
stored as CSV next to the spectra (``config.BANK_META``) so it can be read
without laborch, e.g. in a spreadsheet or with pandas.

Columns
-------
name, instrument, instrument_name, family, envelope,
note, octave, pitch, midi, freq_hz, string, variant, variant_name,
mode, mode_name, dynamic, sub_bank, aif,
duration_s, sustain_start_s, sustain_end_s, n_frames, method,
level_db, peak_db, similarity_v1
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from . import config
from .bank import INSTRUMENT_NAMES, MODE_NAMES, VARIANT_NAMES, parse_name

FAMILIES = {
    "strings": ["vln", "vla", "vlc", "ctb"],
    "flutes": ["pic", "flu", "afl", "bfl"],
    "double reeds": ["obo", "enh", "bsn", "cbs"],
    "clarinets": ["cle", "clb", "bcl", "cbc"],
    "saxophones": ["spx", "alx", "tnx", "brx", "bsx"],
    "brass": ["tpc", "frh", "trb", "btb", "tba"],
    "keyboard": ["pno"],
    "percussion": ["mrb", "xyl", "vib", "glk", "cro", "tbb"],
}
FAMILY_OF = {inst: fam for fam, insts in FAMILIES.items() for inst in insts}

# Struck sounds have no sustain; the analysis window starts at the attack.
DECAYING_INSTRUMENTS = {"pno", "mrb", "xyl", "glk", "cro", "tbb"}
DECAYING_MODES = {("vib", "hdm")}

COLUMNS = [
    "name", "instrument", "instrument_name", "family", "envelope",
    "note", "octave", "pitch", "midi", "freq_hz", "string", "variant",
    "variant_name", "mode", "mode_name", "dynamic", "sub_bank", "aif",
    "duration_s", "sustain_start_s", "sustain_end_s", "n_frames", "method",
    "level_db", "peak_db", "similarity_v1",
]
_INT = {"octave", "midi", "n_frames"}
_FLOAT = {"freq_hz", "duration_s", "sustain_start_s", "sustain_end_s",
          "level_db", "peak_db", "similarity_v1"}


def envelope_class(instrument: str, mode: str) -> str:
    """'decaying' for struck sounds, 'sustained' otherwise."""
    if instrument in DECAYING_INSTRUMENTS or (instrument, mode) in DECAYING_MODES:
        return "decaying"
    return "sustained"


def describe(name: str) -> Dict[str, object]:
    """The name-derived columns for one sound."""
    s = parse_name(name)
    return {
        "name": str(s),
        "instrument": s.instrument,
        "instrument_name": INSTRUMENT_NAMES.get(s.instrument, ""),
        "family": FAMILY_OF.get(s.instrument, ""),
        "envelope": envelope_class(s.instrument, s.mode),
        "note": s.note, "octave": s.octave, "pitch": s.pitch,
        "midi": s.midi, "freq_hz": round(s.freq, 3),
        "string": "" if s.string is None else s.string,
        "variant": s.variant, "variant_name": VARIANT_NAMES.get(s.variant, ""),
        "mode": s.mode, "mode_name": MODE_NAMES.get(s.mode, ""),
        "dynamic": s.dynamic,
    }


def write_table(rows: Iterable[dict], path: Path = config.BANK_META) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_table(path: Path = config.BANK_META) -> List[dict]:
    """Rows as dicts, with numeric columns converted."""
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k in _INT:
                if r.get(k, "") != "":
                    r[k] = int(r[k])
            for k in _FLOAT:
                if r.get(k, "") != "":
                    r[k] = float(r[k])
            r["string"] = int(r["string"]) if r.get("string") else None
            out.append(r)
    return out


def ranges(rows: Iterable[dict]) -> Dict[tuple, tuple]:
    """(instrument, mode) -> (lowest MIDI, highest MIDI) in the Bank."""
    out: Dict[tuple, tuple] = {}
    for r in rows:
        key = (r["instrument"], r["mode"])
        lo, hi = out.get(key, (r["midi"], r["midi"]))
        out[key] = (min(lo, r["midi"]), max(hi, r["midi"]))
    return out


def spreadsheet_names(folder: Path = config.BANK_DIR / "ForJim-04-08-20") -> List[str]:
    """Sound names listed in François' per-instrument spreadsheets.

    Needs openpyxl. Columns are: number, instrument, pitch, mode, dynamic.
    """
    import openpyxl
    names = []
    for f in sorted(Path(folder).glob("*.xlsx")):
        if f.name.startswith("Instrument"):
            continue
        ws = openpyxl.load_workbook(f, data_only=True, read_only=True).active
        for r in ws.iter_rows(min_row=2, values_only=True):
            if len(r) >= 5 and r[1] and r[2]:
                names.append("-".join(str(c).strip() for c in r[1:5]))
    return names
