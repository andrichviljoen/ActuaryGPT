import pandas as pd

import reserving_app.services.chainladder_demo as chainladder_demo
from reserving_app.services.chainladder_demo import _load_local_genins_snapshot, _triangle_to_dataframe


class FakeTriangle:
    def to_frame(self):
        return pd.DataFrame(
            {
                "origin": ["2018", "2018", "2019", "2019"],
                "development": [12, 24, 12, 24],
                "value": [100, 150, 120, 180],
            }
        )


def test_triangle_to_dataframe_creates_dev_columns():
    df = _triangle_to_dataframe(FakeTriangle())
    assert df.columns.tolist() == ["Dev 0", "Dev 1"]
    assert df.loc["2018", "Dev 0"] == 100


def test_local_genins_snapshot_loads():
    df = _load_local_genins_snapshot()
    assert df.shape[0] >= 10
    assert df.columns[0] == "Dev 0"


def test_load_genins_demo_falls_back_when_chainladder_missing(monkeypatch):
    monkeypatch.setattr(chainladder_demo, "cl", None)
    tri, sim_ldf, source = chainladder_demo.load_genins_demo()
    assert source == "local_snapshot"
    assert tri.cumulative.shape[0] >= 10
    assert len(sim_ldf) == tri.cumulative.shape[1] - 1
