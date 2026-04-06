from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from decimal import Decimal

import numpy as np
import pandas as pd


def to_jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, pd.Period):
        return str(value)
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, pd.Series):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, pd.DataFrame):
        return [{str(k): to_jsonable(v) for k, v in row.items()} for row in value.to_dict(orient="records")]
    if isinstance(value, np.ndarray):
        return [to_jsonable(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    return str(value)
