"""The Bank of averaged spectra, sound names, and Palette selection.

Vocabulary (see CLAUDE.md):
  Bank     every recorded note of every instrument, as averaged spectra
  Ensemble the instruments the composer writes for (a list of codes)
  Palette  every Bank sound those instruments can play, as a matrix
  Target   the spectrum to imitate

Sound names follow François' scheme ``inst-pitch-mode-dyn``, e.g.
``vln-a4-ord-mf``. The pitch token may carry a string number prefix
(``ctb-1b4-nhr-mf``) and a one-letter variant suffix (``vln-a4o-ord-mf`` open
string, ``vln-a4x-ord-mf`` stopped).
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from . import config

# Instrument codes in the current Bank (from Notebooks/toolprepcode.py).
INSTRUMENT_NAMES = {
    "afl": "Alto Flute", "alx": "Alto Sax", "bcl": "Bass Clarinet",
    "bfl": "Bass Flute", "brx": "Baritone Sax", "bsn": "Bassoon",
    "bsx": "Bass Saxophone", "btb": "Bass Trombone",
    "cbc": "Contra Bass Clarinet", "cbs": "Contra Bassoon",
    "clb": "Bb Clarinet", "cle": "Eb Clarinet", "cro": "Crotales",
    "ctb": "Contrabass", "enh": "English Horn", "flu": "Flute",
    "frh": "French Horn", "glk": "Glockenspiel", "mrb": "Marimba",
    "obo": "Oboe", "pic": "Piccolo", "pno": "Piano", "spx": "Soprano Sax",
    "tba": "Tuba", "tbb": "Tubular Bells", "tnx": "Tenor Sax",
    "tpc": "C Trumpet", "trb": "Tenor Trombone", "vib": "Vibraphone",
    "vla": "Viola", "vlc": "Cello", "vln": "Violin", "xyl": "Xylophone",
}

# Mode codes, from François' "Instrument Index" spreadsheet
# (Soundbank_McGill/ForJim-04-08-20). Wind "ord" is played with vibrato.
MODE_NAMES = {
    "ord": "ordinary", "mtd": "muted", "nhr": "natural harmonic",
    "ahr": "artificial harmonic", "hmo": "harmon mute (open)",
    "stm": "straight mute", "ped": "pedal tone", "bwd": "bowed",
    "hdm": "hard mallet", "bbt": "brass beater",
}

# Pitch-token variants on string instruments: the same pitch played on an
# open string ("o") or stopped on the next string down ("x"). They occur
# only on open-string pitches (e.g. vln g3, d4, a4, e5).
VARIANT_NAMES = {"": "", "o": "open string", "x": "stopped"}

_NOTE_INDEX = {"c": 0, "cs": 1, "d": 2, "ds": 3, "e": 4, "f": 5,
               "fs": 6, "g": 7, "gs": 8, "a": 9, "as": 10, "b": 11}

_NAME_RE = re.compile(
    r"^(?P<inst>[a-z]{3})-"
    r"(?P<string>[1-4])?(?P<note>[a-g]s?)(?P<octave>\d)(?P<variant>[a-z]?)-"
    r"(?P<mode>[a-z]+)-(?P<dynamic>pp|mf|ff)$"
)


@dataclass(frozen=True)
class SoundName:
    instrument: str
    note: str                 # 'c', 'cs', ... 'b'
    octave: int
    mode: str
    dynamic: str
    string: Optional[int] = None
    variant: str = ""

    @property
    def pitch(self) -> str:
        return f"{self.note}{self.octave}"

    @property
    def midi(self) -> int:
        """MIDI note number (C4 = 60)."""
        return 12 * (self.octave + 1) + _NOTE_INDEX[self.note]

    @property
    def freq(self) -> float:
        """Equal-tempered fundamental in Hz (A4 = 440)."""
        return 440.0 * 2 ** ((self.midi - 69) / 12)

    def __str__(self) -> str:
        s = "" if self.string is None else str(self.string)
        return f"{self.instrument}-{s}{self.pitch}{self.variant}-{self.mode}-{self.dynamic}"


def parse_name(name: str) -> SoundName:
    """Parse ``'vln-a4-ord-mf'`` (with or without a file extension)."""
    stem = Path(name).stem if "." in name else name
    m = _NAME_RE.match(stem)
    if not m:
        raise ValueError(f"not a Bank sound name: {name!r}")
    d = m.groupdict()
    return SoundName(
        instrument=d["inst"], note=d["note"], octave=int(d["octave"]),
        mode=d["mode"], dynamic=d["dynamic"],
        string=int(d["string"]) if d["string"] else None,
        variant=d["variant"],
    )


def _as_set(x) -> Optional[set]:
    """None stays None; a single string becomes a one-element set."""
    if x is None:
        return None
    if isinstance(x, str):
        return {x}
    return set(x)


@dataclass
class Palette:
    """Palette spectra as columns: ``matrix`` has shape (NBINS, n_sounds).

    This is the notebooks' ``palate`` matrix, the basis X of RH2009.
    """
    names: list
    matrix: np.ndarray

    def __len__(self) -> int:
        return len(self.names)


class Bank:
    """All Bank spectra, one row per sound, rows sorted by name.

    Two versions exist (``Bank.load(version=...)``):

    2  (default, Phase 1) power spectra over each note's detected sustain,
       at the recorded level: ``10*log10(spectrum.sum())`` is the sound's
       level in dBFS. Comes with a metadata table (``bank.meta``).
    1  (Phase 0) the FFTBank text files: magnitude spectra normalized to
       unit sum.

    ``kind`` is 'power' or 'magnitude'; ``version`` is 2 or 1.
    """

    def __init__(self, names: Sequence[str], spectra: np.ndarray, source: str = "",
                 kind: str = "magnitude", version: int = 1,
                 meta: Optional[dict] = None):
        spectra = np.asarray(spectra, dtype=float)
        if spectra.shape != (len(names), config.NBINS):
            raise ValueError(f"spectra shape {spectra.shape} does not match "
                             f"({len(names)}, {config.NBINS})")
        if kind not in ("power", "magnitude"):
            raise ValueError(f"kind must be 'power' or 'magnitude', not {kind!r}")
        self.names = list(names)
        self.spectra = spectra
        self.source = source
        self.kind = kind
        self.version = version
        self._index = {n: i for i, n in enumerate(self.names)}
        self.info = {n: parse_name(n) for n in self.names}
        self.meta = meta or {}

    def __repr__(self) -> str:
        return (f"<Bank v{self.version}: {len(self)} sounds, {self.kind} spectra, "
                f"{len(self.instruments)} instruments>")

    # ---- loading -------------------------------------------------------
    @classmethod
    def load(cls, version: int = config.BANK_VERSION,
             normalize: Optional[str] = None, **kw) -> "Bank":
        """Load a Bank: version 2 (default) or 1.

        ``normalize='unit'`` divides every spectrum by its sum, keeping only
        its shape (the v1 convention); the default keeps the levels.
        Other keyword arguments go to ``load_v2`` or ``load_v1``.
        """
        if version == 2:
            b = cls.load_v2(**kw)
        elif version == 1:
            b = cls.load_v1(**kw)
        else:
            raise ValueError(f"unknown Bank version {version}")
        if normalize == "unit":
            b.spectra = b.spectra / b.spectra.sum(axis=1, keepdims=True)
        elif normalize is not None:
            raise ValueError(f"normalize must be None or 'unit', not {normalize!r}")
        return b

    @classmethod
    def load_v2(cls, path: Path = config.BANK_FILE,
                meta_path: Path = config.BANK_META) -> "Bank":
        """Bank v2: ``bank_v2.npz`` and its metadata table ``bank_v2.csv``."""
        from .metadata import read_table
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; build it with "
                                    f"'python -m laborch.build' or use Bank.load(version=1)")
        z = np.load(path, allow_pickle=False)
        names = [str(n) for n in z["names"]]
        meta = {}
        if Path(meta_path).exists():
            meta = {r["name"]: r for r in read_table(meta_path)}
        return cls(names, z["spectra"], source=str(path), kind=str(z["kind"]),
                   version=int(z["bank_version"]), meta=meta)

    @classmethod
    def load_v1(cls, fft_dir: Path = config.FFT_DIR, use_cache: bool = True,
                cache_file: Path = config.CACHE_FILE) -> "Bank":
        """Bank v1 (``FFTBank/*.txt``), unit-sum magnitude spectra.

        Reading 1,515 text files is slow, so the matrix is cached in
        ``.cache/fftbank.npz``; the cache is reused only if it holds exactly
        the same set of sound names. Pass ``use_cache=False`` to rebuild.
        """
        fft_dir = Path(fft_dir)
        files = sorted(fft_dir.glob("*.txt"))
        names = [f.stem for f in files]
        if use_cache and cache_file.exists():
            c = np.load(cache_file, allow_pickle=False)
            if list(c["names"]) == names:
                return cls(names, c["spectra"], source=str(fft_dir))
        spectra = np.vstack([np.loadtxt(f) for f in files])
        if use_cache:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_file, names=np.array(names), spectra=spectra)
        return cls(names, spectra, source=str(fft_dir))

    @classmethod
    def load_old(cls, nbb_dir: Path = config.OLD_FFT_DIR) -> "Bank":
        """Load the 2009 ``.nbb`` spectra (2,048 float32 values each).

        Names that do not follow the naming scheme are skipped with a warning.
        """
        names, rows, skipped = [], [], []
        for f in sorted(Path(nbb_dir).glob("*.nbb")):
            try:
                parse_name(f.stem)
            except ValueError:
                skipped.append(f.name)
                continue
            names.append(f.stem)
            rows.append(np.fromfile(f, dtype="float32"))
        if skipped:
            warnings.warn(f"skipped {len(skipped)} unparseable files: {skipped}")
        return cls(names, np.vstack(rows), source=str(nbb_dir))

    # ---- access --------------------------------------------------------
    def __len__(self) -> int:
        return len(self.names)

    def __contains__(self, name: str) -> bool:
        return name in self._index

    def spectrum(self, name: str) -> np.ndarray:
        return self.spectra[self._index[name]]

    @property
    def instruments(self) -> list:
        return sorted({i.instrument for i in self.info.values()})

    def select(self, instruments: Optional[Iterable[str]] = None,
               modes: Optional[Iterable[str]] = None,
               dynamics: Optional[Iterable[str]] = None,
               midi_range: Optional[tuple] = None,
               exclude: Iterable[str] = (),
               families: Optional[Iterable[str]] = None) -> Palette:
        """Build a Palette from an Ensemble's instrument codes plus filters.

        ``exclude`` drops named sounds (e.g. the Target itself, when the
        Target is taken from the Bank).

        Each of ``instruments``, ``modes``, ``dynamics`` and ``exclude``
        takes a list or a single string: ``exclude="trb-f2-ord-mf"`` is the
        same as ``exclude=["trb-f2-ord-mf"]``.

        ``families`` selects by instrument family (``metadata.FAMILIES``:
        'strings', 'flutes', 'double reeds', 'clarinets', 'saxophones',
        'brass', 'keyboard', 'percussion').
        """
        if families is not None:
            from .metadata import FAMILIES
            fam = _as_set(families)
            unknown = fam - set(FAMILIES)
            if unknown:
                raise ValueError(f"unknown families {sorted(unknown)}; choose from {list(FAMILIES)}")
            from_fam = {i for f in fam for i in FAMILIES[f]}
            instruments = from_fam if instruments is None else _as_set(instruments) & from_fam
        inst = _as_set(instruments)
        modes = _as_set(modes)
        dyns = _as_set(dynamics)
        exclude = _as_set(exclude) or set()
        unknown = exclude - set(self._index)
        if unknown:
            warnings.warn(f"exclude: not in the Bank: {sorted(unknown)}")
        keep = []
        for n in self.names:
            s = self.info[n]
            if n in exclude:
                continue
            if inst is not None and s.instrument not in inst:
                continue
            if modes is not None and s.mode not in modes:
                continue
            if dyns is not None and s.dynamic not in dyns:
                continue
            if midi_range is not None and not (midi_range[0] <= s.midi <= midi_range[1]):
                continue
            keep.append(n)
        if not keep:
            raise ValueError("selection is empty")
        idx = [self._index[n] for n in keep]
        return Palette(keep, self.spectra[idx].T.copy())

    def level_db(self, name: str) -> float:
        """Level of a sound in dB: 10 log10 of its spectrum's sum.

        For Bank v2 this is the mean-square level of the analysed region in
        dBFS; for unit-normalized spectra it is 0.
        """
        s = self.spectrum(name).sum()
        return float(10 * np.log10(s)) if self.kind == "power" else float(20 * np.log10(s))

    def sound_path(self, name: str) -> Path:
        """The .aif recording for a Bank sound."""
        for d in config.SOUND_DIRS:
            p = d / f"{name}.aif"
            if p.exists():
                return p
        raise FileNotFoundError(f"no .aif for {name} in {config.SOUND_DIRS}")


def frequencies() -> np.ndarray:
    """Centre frequency (Hz) of each spectrum bin, bin 0 = DC."""
    return np.arange(config.NBINS) * config.BIN_HZ
