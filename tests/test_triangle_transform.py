import pandas as pd
import pytest

from reserving_app.services.triangle_builder import (
    build_triangle,
    build_triangle_from_development_matrix,
    convert_origin_calendar_to_development_triangle,
    parse_period_label,
)


def test_build_triangle_from_incremental_development_matrix():
    df = pd.DataFrame(
        {
            "origin": ["2018Q1", "2018Q2"],
            "Dev 1": [100, 120],
            "Dev 2": [80, 60],
        }
    )
    tri = build_triangle_from_development_matrix(df, "origin", ["Dev 1", "Dev 2"], "Incremental")
    assert tri.incremental.loc["2018Q1", "Dev 2"] == 80
    assert tri.cumulative.loc["2018Q1", "Dev 2"] == 180


def test_convert_origin_calendar_to_development_triangle():
    df = pd.DataFrame(
        {
            "origin": ["2018Q1", "2018Q2"],
            "2018Q1": [100, 0],
            "2018Q2": [80, 120],
            "2018Q3": [60, 90],
        }
    )
    tri = convert_origin_calendar_to_development_triangle(df, "origin", ["2018Q1", "2018Q2", "2018Q3"])
    assert tri.incremental.loc["2018Q1", "Dev 1"] == 100
    assert tri.incremental.loc["2018Q2", "Dev 1"] == 120
    assert tri.incremental.loc["2018Q2", "Dev 2"] == 90


def test_build_triangle_from_development_matrix_empty_input_raises():
    df = pd.DataFrame({"origin": [None], "Dev 1": [100]})
    with pytest.raises(ValueError, match="No valid origin periods found"):
        build_triangle_from_development_matrix(df, "origin", ["Dev 1"], "Incremental")


def test_build_triangle_from_development_matrix_missing_required_column_raises():
    df = pd.DataFrame({"origin": ["2018Q1"], "Dev 1": [100]})
    with pytest.raises(ValueError, match="Development columns not found"):
        build_triangle_from_development_matrix(df, "origin", ["Dev 1", "Dev 2"], "Incremental")


def test_build_triangle_from_development_matrix_invalid_origin_mapping_raises():
    df = pd.DataFrame({"origin": ["invalid-period"], "Dev 1": [100]})
    with pytest.raises(ValueError, match="Could not parse period label"):
        build_triangle_from_development_matrix(df, "origin", ["Dev 1"], "Incremental")


def test_convert_origin_calendar_handles_duplicates_and_unsorted_records():
    # Regression: duplicate origins previously caused _latest_diagonal to fail due to duplicate index labels.
    df = pd.DataFrame(
        {
            "origin": ["2018Q2", "2018Q1", "2018Q1"],
            "2018Q1": [0, 100, 25],
            "2018Q2": [120, 80, 20],
            "2018Q3": [90, 60, 10],
        }
    )
    tri = convert_origin_calendar_to_development_triangle(df, "origin", ["2018Q1", "2018Q2", "2018Q3"])
    assert tri.incremental.index.tolist() == ["2018Q1", "2018Q2"]
    assert tri.incremental.loc["2018Q1", "Dev 1"] == 125
    assert tri.incremental.loc["2018Q1", "Dev 2"] == 100
    assert tri.incremental.loc["2018Q1", "Dev 3"] == 70
    assert tri.incremental.loc["2018Q2", "Dev 1"] == 120
    assert tri.incremental.loc["2018Q2", "Dev 2"] == 90


def test_convert_origin_calendar_sparse_triangle_fills_missing_cells_with_zero():
    df = pd.DataFrame(
        {
            "origin": ["2018Q1", "2018Q3"],
            "2018Q1": [50, 0],
            "2018Q2": [0, 0],
            "2018Q3": [25, 35],
        }
    )
    tri = convert_origin_calendar_to_development_triangle(df, "origin", ["2018Q1", "2018Q2", "2018Q3"])
    assert tri.incremental.loc["2018Q3", "Dev 2"] == 0
    assert tri.incremental.loc["2018Q1", "Dev 2"] == 0


def test_convert_origin_calendar_empty_input_raises():
    df = pd.DataFrame({"origin": [None], "2018Q1": [100]})
    with pytest.raises(ValueError, match="No valid origin periods found"):
        convert_origin_calendar_to_development_triangle(df, "origin", ["2018Q1"])


def test_convert_origin_calendar_missing_required_column_raises():
    df = pd.DataFrame({"origin": ["2018Q1"], "2018Q1": [100]})
    with pytest.raises(ValueError, match="Calendar columns not found"):
        convert_origin_calendar_to_development_triangle(df, "origin", ["2018Q1", "2018Q2"])


def test_convert_origin_calendar_invalid_period_mapping_raises():
    df = pd.DataFrame({"origin": ["2018Q1"], "not_a_period": [100]})
    with pytest.raises(ValueError, match="Could not parse period label"):
        convert_origin_calendar_to_development_triangle(df, "origin", ["not_a_period"])


def test_build_triangle_from_mapped_transactional_data_supports_segment_filter():
    df = pd.DataFrame(
        {
            "accident_date": ["2020-01-01", "2020-01-01", "2021-01-01"],
            "valuation_date": ["2020-12-31", "2021-12-31", "2021-12-31"],
            "paid": [100, 50, 70],
            "segment": ["A", "A", "B"],
        }
    )
    mapping = {
        "origin_period": "accident_date",
        "development_period": "valuation_date",
        "paid_amount": "paid",
        "incurred_amount": None,
        "reported_count": None,
        "paid_count": None,
        "claim_id": None,
        "segment": "segment",
    }
    tri = build_triangle(df, mapping, "paid_amount", "Yearly", segment_filter="A")
    assert tri.incremental.shape == (1, 2)
    assert tri.incremental.iloc[0, 0] == 100
    assert tri.incremental.iloc[0, 1] == 50
