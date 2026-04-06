from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TriangleArtifacts:
    incremental: pd.DataFrame
    cumulative: pd.DataFrame
    latest_diagonal: pd.DataFrame


PERIOD_PATTERNS = {
    "quarterly": re.compile(r"^(?P<y>\d{4})Q(?P<p>[1-4])$"),
    "monthly": re.compile(r"^(?P<y>\d{4})[-/]?(?P<p>0[1-9]|1[0-2])$"),
    "half_yearly": re.compile(r"^(?P<y>\d{4})[- ]?H(?P<p>[1-2])$"),
    "yearly": re.compile(r"^(?P<y>\d{4})$"),
}

GRAIN_TO_FREQ = {
    "Yearly": "Y",
    "Half-Yearly": "S",
    "Quarterly": "Q",
    "Monthly": "M",
}


def _parse_raw_period(value: object, grain: str) -> pd.Period:
    if pd.isna(value):
        raise ValueError("Empty period value.")

    text = str(value).strip()
    if not text:
        raise ValueError("Empty period value.")

    normalized = text.upper().replace(" ", "")
    if grain == "Half-Yearly":
        match = PERIOD_PATTERNS["half_yearly"].match(normalized)
        if match:
            year = int(match.group("y"))
            half = int(match.group("p"))
            month = 1 if half == 1 else 7
            return pd.Period(pd.Timestamp(year=year, month=month, day=1), freq="2Q")

    if grain == "Quarterly":
        match = PERIOD_PATTERNS["quarterly"].match(normalized)
        if match:
            return pd.Period(f"{match.group('y')}Q{match.group('p')}", freq="Q")

    if grain == "Yearly":
        match = PERIOD_PATTERNS["yearly"].match(normalized)
        if match:
            return pd.Period(f"{match.group('y')}", freq="Y")

    if grain == "Monthly":
        match = PERIOD_PATTERNS["monthly"].match(normalized)
        if match:
            return pd.Period(f"{match.group('y')}-{match.group('p')}", freq="M")

    dt = pd.to_datetime([value], errors="coerce")[0]
    if pd.isna(dt):
        raise ValueError(f"Could not parse period value '{value}' for grain '{grain}'.")

    freq = GRAIN_TO_FREQ[grain]
    if freq == "S":
        month = 1 if dt.month <= 6 else 7
        return pd.Period(pd.Timestamp(year=dt.year, month=month, day=1), freq="2Q")
    return pd.Period(dt, freq=freq)


def _lag_between(origin: pd.Period, development: pd.Period, grain: str) -> int:
    if grain == "Yearly":
        lag = development.year - origin.year
    elif grain == "Quarterly":
        lag = (development.year - origin.year) * 4 + (development.quarter - origin.quarter)
    elif grain == "Half-Yearly":
        def _half(p: pd.Period) -> int:
            month = p.start_time.month
            return 1 if month <= 6 else 2
        lag = (development.year - origin.year) * 2 + (_half(development) - _half(origin))
    else:
        lag = (development.year - origin.year) * 12 + (development.month - origin.month)
    return int(lag)


def _format_period(series: pd.Series, grain: str) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    if dt.notna().sum() == 0:
        return series.astype(str)

    if grain == "Yearly":
        return dt.dt.to_period("Y").astype(str)
    if grain == "Half-Yearly":
        return (dt.dt.year.astype(str) + "-H" + (((dt.dt.month - 1) // 6) + 1).astype(str))
    if grain == "Quarterly":
        return dt.dt.to_period("Q").astype(str)
    return dt.dt.to_period("M").astype(str)


def build_triangle(
    df: pd.DataFrame,
    mapping: dict[str, str | None],
    value_field: str,
    period_grain: str,
    segment_filter: str | None = None,
) -> TriangleArtifacts:
    work = df.copy()

    if segment_filter and mapping.get("segment"):
        work = work[work[mapping["segment"]] == segment_filter]

    origin_col = mapping["origin_period"]
    dev_col = mapping["development_period"]
    value_col = mapping[value_field]

    try:
        work["_origin_period"] = work[origin_col].map(lambda x: _parse_raw_period(x, period_grain))
        work["_dev_period"] = work[dev_col].map(lambda x: _parse_raw_period(x, period_grain))
    except ValueError as exc:
        raise ValueError(f"Invalid period field in mapped data: {exc}") from exc

    work["_origin"] = work["_origin_period"].astype(str)
    work["_lag"] = [
        _lag_between(origin, development, period_grain)
        for origin, development in zip(work["_origin_period"], work["_dev_period"])
    ]
    if (work["_lag"] < 0).any():
        bad_rows = work.loc[work["_lag"] < 0, [origin_col, dev_col]].head(5)
        raise ValueError(
            "Development period is earlier than origin period for some rows. "
            f"Sample rows: {bad_rows.to_dict(orient='records')}"
        )

    work["_value"] = pd.to_numeric(work[value_col], errors="coerce").fillna(0.0)

    incr = (
        work.groupby(["_origin", "_lag"], dropna=False)["_value"]
        .sum()
        .reset_index()
        .pivot(index="_origin", columns="_lag", values="_value")
        .fillna(0.0)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    incr.columns = [f"Dev {int(c)}" for c in incr.columns]

    cumulative = incr.cumsum(axis=1)
    latest_diag = _latest_diagonal(cumulative)

    logger.info("Built triangle with shape=%s", incr.shape)
    return TriangleArtifacts(incremental=incr, cumulative=cumulative, latest_diagonal=latest_diag)


def _latest_diagonal(cumulative_triangle: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx in range(len(cumulative_triangle.index)):
        origin = cumulative_triangle.index[idx]
        row = cumulative_triangle.iloc[idx]
        observed = row[row > 0]
        if observed.empty:
            continue
        latest_dev = observed.index[-1]
        rows.append(
            {
                "origin": origin,
                "latest_development": latest_dev,
                "latest_cumulative": float(observed.iloc[-1]),
            }
        )
    return pd.DataFrame(rows)


def link_ratio_matrix(cumulative_triangle: pd.DataFrame) -> pd.DataFrame:
    lr = cumulative_triangle.iloc[:, 1:].values / np.where(
        cumulative_triangle.iloc[:, :-1].values == 0,
        np.nan,
        cumulative_triangle.iloc[:, :-1].values,
    )
    return pd.DataFrame(lr, index=cumulative_triangle.index, columns=cumulative_triangle.columns[1:])


def parse_period_label(label: str) -> tuple[int, int]:
    """
    Parse a period label into a sortable tuple.
    Supports yearly, half-yearly, quarterly, and monthly formats.
    """
    text = str(label).strip()
    normalized = text.upper().replace(" ", "")
    if PERIOD_PATTERNS["quarterly"].match(normalized):
        p = _parse_raw_period(text, "Quarterly")
        return p.year, p.quarter
    if PERIOD_PATTERNS["half_yearly"].match(normalized):
        p = _parse_raw_period(text, "Half-Yearly")
        return p.year, 1 if p.start_time.month <= 6 else 2
    if PERIOD_PATTERNS["monthly"].match(normalized):
        p = _parse_raw_period(text, "Monthly")
        return p.year, p.month
    if PERIOD_PATTERNS["yearly"].match(normalized):
        p = _parse_raw_period(text, "Yearly")
        return p.year, 1
    try:
        p = _parse_raw_period(text, "Monthly")
        return p.year, p.month
    except ValueError:
        pass
    try:
        p = _parse_raw_period(text, "Yearly")
        return p.year, 1
    except ValueError:
        pass
    raise ValueError(f"Could not parse period label '{label}'.")


def order_period_labels(labels: list[str]) -> list[str]:
    parsed = []
    for label in labels:
        parsed.append((label, parse_period_label(label)))
    parsed_sorted = sorted(parsed, key=lambda x: x[1])
    return [x[0] for x in parsed_sorted]


def parse_development_period_label(label: str) -> int:
    text = str(label).strip().lower().replace("dev", "").replace("development", "").strip()
    match = re.search(r"\d+", text)
    if not match:
        raise ValueError(f"Development column '{label}' is not a valid development period label.")
    value = int(match.group(0))
    if value <= 0:
        raise ValueError(f"Development column '{label}' must be a positive integer.")
    return value


def build_triangle_from_development_matrix(
    df: pd.DataFrame,
    origin_col: str,
    development_columns: list[str],
    data_type: str,
) -> TriangleArtifacts:
    if origin_col not in df.columns:
        raise ValueError(f"Origin column '{origin_col}' not found.")
    if not development_columns:
        raise ValueError("No development period columns selected.")
    missing_development_columns = [col for col in development_columns if col not in df.columns]
    if missing_development_columns:
        raise ValueError(f"Development columns not found: {missing_development_columns}")

    dev_pairs = sorted([(col, parse_development_period_label(col)) for col in development_columns], key=lambda x: x[1])
    ordered_cols = [pair[0] for pair in dev_pairs]
    dev_labels = [f"Dev {pair[1]}" for pair in dev_pairs]

    work = df[[origin_col] + ordered_cols].copy()
    work = work.dropna(subset=[origin_col])
    if work.empty:
        raise ValueError("No valid origin periods found.")
    work[origin_col] = work[origin_col].astype(str).str.strip()
    origin_sort_keys = {origin: parse_period_label(origin) for origin in work[origin_col].unique()}
    numeric = work[ordered_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    numeric.columns = dev_labels
    numeric["_origin"] = work[origin_col].values
    numeric = numeric.groupby("_origin", as_index=True).sum()
    numeric = numeric.loc[sorted(numeric.index, key=lambda label: origin_sort_keys[label])]

    if data_type == "Incremental":
        incremental = numeric
        cumulative = numeric.cumsum(axis=1)
    else:
        incremental = numeric.diff(axis=1)
        incremental.iloc[:, 0] = numeric.iloc[:, 0]
        incremental = incremental.fillna(0.0)
        cumulative = numeric
    return TriangleArtifacts(incremental=incremental, cumulative=cumulative, latest_diagonal=_latest_diagonal(cumulative))


def convert_origin_calendar_to_development_triangle(
    df: pd.DataFrame,
    origin_col: str,
    calendar_columns: list[str],
) -> TriangleArtifacts:
    if origin_col not in df.columns:
        raise ValueError(f"Origin column '{origin_col}' not found.")
    if not calendar_columns:
        raise ValueError("No calendar period columns selected.")
    missing_calendar_columns = [col for col in calendar_columns if col not in df.columns]
    if missing_calendar_columns:
        raise ValueError(f"Calendar columns not found: {missing_calendar_columns}")

    origins_raw = df[origin_col].dropna().astype(str).str.strip().tolist()
    if not origins_raw:
        raise ValueError("No valid origin periods found.")

    all_periods = list(set(origins_raw + [str(c) for c in calendar_columns]))
    ordered_periods = order_period_labels(all_periods)
    period_to_idx = {p: i for i, p in enumerate(ordered_periods)}

    values_by_origin_and_lag: dict[tuple[str, int], float] = {}
    origins_seen = set()
    dev_max = 0
    for _, row in df.iterrows():
        origin = str(row.get(origin_col)).strip()
        if origin not in period_to_idx:
            raise ValueError(f"Origin period '{origin}' cannot be aligned to ordered periods.")
        origins_seen.add(origin)
        origin_idx = period_to_idx[origin]

        for cal_col in calendar_columns:
            cal_label = str(cal_col)
            if cal_label not in period_to_idx:
                raise ValueError(f"Calendar period '{cal_label}' cannot be aligned to ordered periods.")
            cal_idx = period_to_idx[cal_label]
            if cal_idx < origin_idx:
                continue
            lag = cal_idx - origin_idx + 1
            value = pd.to_numeric(pd.Series([row.get(cal_col)]), errors="coerce").iloc[0]
            value = 0.0 if pd.isna(value) else float(value)
            key = (origin, lag)
            values_by_origin_and_lag[key] = values_by_origin_and_lag.get(key, 0.0) + value
            dev_max = max(dev_max, lag)

    sorted_origins = sorted(origins_seen, key=parse_period_label)
    dev_columns = [f"Dev {i}" for i in range(1, dev_max + 1)]
    inc_matrix = [
        [values_by_origin_and_lag.get((origin, i), 0.0) for i in range(1, dev_max + 1)]
        for origin in sorted_origins
    ]
    incremental = pd.DataFrame(inc_matrix, index=sorted_origins, columns=dev_columns)
    cumulative = incremental.cumsum(axis=1)
    return TriangleArtifacts(incremental=incremental, cumulative=cumulative, latest_diagonal=_latest_diagonal(cumulative))
