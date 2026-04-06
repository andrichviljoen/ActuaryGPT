import pandas as pd

from reserving_app.services.reserving_models import run_chain_ladder


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
