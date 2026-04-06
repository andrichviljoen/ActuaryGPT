import numpy as np
import pandas as pd

from reserving_app.services.json_utils import to_jsonable


def test_to_jsonable_handles_pandas_numpy_time_types():
    payload = {
        "period": pd.Period("2018Q1", freq="Q"),
        "timestamp": pd.Timestamp("2018-01-01"),
        "series": pd.Series([np.int64(1), np.float64(2.5)], index=["a", "b"]),
    }
    converted = to_jsonable(payload)
    assert converted["period"] == "2018Q1"
    assert converted["timestamp"].startswith("2018-01-01")
    assert converted["series"]["a"] == 1
    assert converted["series"]["b"] == 2.5
