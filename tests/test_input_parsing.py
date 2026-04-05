import pytest

from reserving_app.services.input_parsing import parse_exclusion_cells


def test_parse_exclusion_cells_valid():
    assert parse_exclusion_cells("0,1; 2,3") == {(0, 1), (2, 3)}


def test_parse_exclusion_cells_empty():
    assert parse_exclusion_cells("") == set()


def test_parse_exclusion_cells_invalid():
    with pytest.raises(ValueError):
        parse_exclusion_cells("bad-token")
