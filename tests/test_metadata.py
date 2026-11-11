"""The Bank v2 metadata table."""
import numpy as np
import pytest

from laborch import FAMILIES, INSTRUMENT_NAMES, config
from laborch.metadata import COLUMNS, describe, envelope_class, ranges


def test_families_cover_every_instrument_once():
    insts = [i for f in FAMILIES.values() for i in f]
    assert sorted(insts) == sorted(INSTRUMENT_NAMES)


def test_describe():
    d = describe("vln-a4o-ord-mf")
    assert d["family"] == "strings" and d["variant_name"] == "open string"
    assert d["midi"] == 69 and d["envelope"] == "sustained"
    assert describe("vib-c4-hdm-mf")["envelope"] == "decaying"
    assert describe("vib-c4-bwd-mf")["envelope"] == "sustained"
    assert envelope_class("pno", "ord") == "decaying"


def test_every_sound_has_a_complete_row(bank):
    assert set(bank.meta) == set(bank.names)
    for n in bank.names:
        r = bank.meta[n]
        assert set(COLUMNS) <= set(r)
        assert r["instrument_name"] and r["family"] and r["mode_name"]
        assert 0 <= r["sustain_start_s"] < r["sustain_end_s"] <= r["duration_s"] + 1e-6
        assert r["n_frames"] >= config.MIN_FRAMES - 1
        assert r["method"] in ("sustain", "decay", "trim")


def test_variants_are_open_string_pitches(bank):
    open_strings = {"vln": {"g3", "d4", "a4", "e5"}, "vla": {"c3", "g3", "d4", "a4"},
                    "vlc": {"c2", "g2", "d3", "a3"}, "ctb": {"e1", "a1", "d2", "g2"}}
    for r in bank.meta.values():
        if r["variant"]:
            assert r["pitch"] in open_strings[r["instrument"]], r["name"]


def test_ranges(bank):
    rg = ranges(bank.meta.values())
    assert rg[("vln", "ord")] == (55, max(r["midi"] for r in bank.meta.values()
                                         if r["instrument"] == "vln" and r["mode"] == "ord"))
    for (inst, mode), (lo, hi) in rg.items():
        assert lo <= hi


def test_sustained_regions_agree_with_v1(bank):
    """For bowed and blown notes the new analysis region changes little."""
    sims = [r["similarity_v1"] for r in bank.meta.values() if r["envelope"] == "sustained"]
    assert np.median(sims) > 0.99 and np.percentile(sims, 1) > 0.9


def test_select_by_family(bank):
    pal = bank.select(families="brass", modes="ord")
    assert {bank.info[n].instrument for n in pal.names} == set(FAMILIES["brass"])
    pal = bank.select(families=["strings"], instruments=["vln", "flu"])
    assert {bank.info[n].instrument for n in pal.names} == {"vln"}
    with pytest.raises(ValueError):
        bank.select(families="kazoos")
