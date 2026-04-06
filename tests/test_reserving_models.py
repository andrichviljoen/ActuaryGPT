import pandas as pd
import pytest

from reserving_app.services.reserving_models import (
    cumulative_to_chainladder_triangle,
    run_chain_ladder,
    run_chainladder_model,
)


def test_run_chain_ladder_uses_latest_observed_development_age_per_origin():
    cumulative = pd.DataFrame(
        {
            "Dev 1": [100.0, 120.0, 90.0],
            "Dev 2": [160.0, 170.0, 0.0],
            "Dev 3": [200.0, 0.0, 0.0],
        },
        index=["2019", "2020", "2021"],
    )

    result = run_chain_ladder(cumulative, apply_tail_factor=False)

    # Old bug: rows with trailing zeroes were treated as fully developed, understating ultimates.
    assert result.ultimates["2020"] > 170.0
    assert result.ultimates["2021"] > 90.0


def test_cumulative_to_chainladder_triangle():
    cl = pytest.importorskip("chainladder")
    cumulative = pd.DataFrame({"Dev 0": [100.0], "Dev 1": [150.0]}, index=["2018-01-01"])
    tri = cumulative_to_chainladder_triangle(cumulative, grain="Yearly")
    assert isinstance(tri, cl.Triangle)


def test_chainladder_and_mack_model_execution():
    pytest.importorskip("chainladder")
    cumulative = pd.DataFrame(
        {"Dev 0": [100.0, 120.0, 150.0], "Dev 1": [140.0, 170.0, 0.0], "Dev 2": [180.0, 0.0, 0.0]},
        index=["2018-01-01", "2019-01-01", "2020-01-01"],
    )
    det_cl, _ = run_chainladder_model(cumulative, model_name="Chainladder", grain="Yearly")
    det_mack, _ = run_chainladder_model(cumulative, model_name="MackChainladder", grain="Yearly")
    assert det_cl.ibnr.sum() >= 0
    assert det_mack.ibnr.sum() >= 0
