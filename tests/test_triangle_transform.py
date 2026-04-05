import pandas as pd

from reserving_app.services.triangle_builder import (
    build_triangle_from_development_matrix,
    convert_origin_calendar_to_development_triangle,
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
