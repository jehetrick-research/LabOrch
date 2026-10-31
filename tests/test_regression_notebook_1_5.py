"""Notebook 1.5: reproduce a violin A4 with flutes.

The notebook used the 2009 .nbb spectra and printed these NNLS weights;
the package must give the same numbers.
"""
import numpy as np

from laborch import nnls_fit, pinv_fit

NOTEBOOK_TOP5 = [
    ("flu-a4-ord-mf", 0.297),
    ("flu-a6-ord-mf", 0.091),
    ("flu-e6-ord-mf", 0.088),
    ("flu-c4-ord-mf", 0.033),
    ("flu-gs4-ord-mf", 0.025),
]


def test_old_bank_matches_notebook(old_bank):
    pal = old_bank.select(instruments=["flu"])
    assert len(pal) == 37
    f = nnls_fit(pal, old_bank.spectrum("vln-a4-ord-mf"))
    top = f.ranked(5)
    assert [n for n, _ in top] == [n for n, _ in NOTEBOOK_TOP5]
    assert np.allclose([w for _, w in top], [w for _, w in NOTEBOOK_TOP5], atol=6e-4)


def test_pinv_gives_negative_weights_nnls_does_not(old_bank):
    pal = old_bank.select(instruments=["flu"])
    t = old_bank.spectrum("vln-a4-ord-mf")
    assert (pinv_fit(pal, t).weights < 0).any()
    assert (nnls_fit(pal, t).weights >= 0).all()


def test_current_bank_same_answer(bank):
    """With the rebuilt Bank the flute A4 still carries the violin A4."""
    pal = bank.select(instruments=["flu"])
    for target in ["vln-a4o-ord-mf", "vln-a4x-ord-mf"]:
        f = nnls_fit(pal, bank.spectrum(target))
        assert f.ranked(1)[0][0] == "flu-a4-ord-mf"
        assert {"flu-a6-ord-mf", "flu-gs4-ord-mf"} <= {n for n, _ in f.ranked(4)}
