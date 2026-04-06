import pandas as pd

from reserving_app.services.chainladder_demo import _triangle_to_dataframe


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
