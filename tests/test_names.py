import pytest

from laborch import parse_name, INSTRUMENT_NAMES


def test_parse_plain():
    s = parse_name("vln-a4x-ord-mf")
    assert (s.instrument, s.pitch, s.variant, s.mode, s.dynamic) == ("vln", "a4", "x", "ord", "mf")
    assert s.midi == 69 and abs(s.freq - 440.0) < 1e-9
    assert str(s) == "vln-a4x-ord-mf"


def test_parse_string_number_and_sharp():
    s = parse_name("ctb-2fs4-nhr-mf.txt")
    assert s.string == 2 and s.note == "fs" and s.octave == 4
    assert str(s) == "ctb-2fs4-nhr-mf"


def test_middle_c():
    assert parse_name("pno-c4-ord-pp").midi == 60


def test_bad_name():
    with pytest.raises(ValueError):
        parse_name("obo-f5-c6")


def test_every_bank_name_parses(bank):
    assert len(bank) == 1515
    assert set(bank.instruments) == set(INSTRUMENT_NAMES)
    for n in bank.names:
        assert str(bank.info[n]) == n


def test_bank_spectra_are_unit_normalized(bank):
    sums = bank.spectra.sum(axis=1)
    assert abs(sums - 1).max() < 1e-6
