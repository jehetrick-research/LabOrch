"""Bank.select() filters, on a small synthetic Bank (no data files needed)."""
import numpy as np
import pytest

from laborch import Bank, config

NAMES = ["trb-f2-ord-mf", "vla-c4-ord-mf", "vla-c4-mtd-mf", "vlc-f2-ord-mf", "vln-a5-ord-mf"]


@pytest.fixture
def small_bank():
    rng = np.random.default_rng(0)
    return Bank(NAMES, rng.random((len(NAMES), config.NBINS)))


def test_exclude_accepts_a_single_name(small_bank):
    as_str = small_bank.select(exclude="trb-f2-ord-mf")
    as_list = small_bank.select(exclude=["trb-f2-ord-mf"])
    assert "trb-f2-ord-mf" not in as_str.names
    assert as_str.names == as_list.names
    assert len(as_str) == len(NAMES) - 1


def test_single_string_filters(small_bank):
    assert small_bank.select(instruments="vla").names == ["vla-c4-ord-mf", "vla-c4-mtd-mf"]
    assert small_bank.select(instruments=["vla", "vlc"], modes="ord").names == \
        ["vla-c4-ord-mf", "vlc-f2-ord-mf"]
    assert len(small_bank.select(dynamics="mf")) == len(NAMES)


def test_exclude_unknown_name_warns(small_bank):
    with pytest.warns(UserWarning, match="not in the Bank"):
        small_bank.select(exclude="trb-f2-ord-mx")


def test_palette_matches_bank_spectra(small_bank):
    pal = small_bank.select(instruments="vln")
    np.testing.assert_array_equal(pal.matrix[:, 0], small_bank.spectrum("vln-a5-ord-mf"))
