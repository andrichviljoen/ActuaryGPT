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

    work["_origin"] = _format_period(work[origin_col], period_grain)
    work["_dev"] = _format_period(work[dev_col], period_grain)
    work["_value"] = pd.to_numeric(work[value_col], errors="coerce").fillna(0.0)

    incr = (
        work.groupby(["_origin", "_dev"], dropna=False)["_value"]
        .sum()
        .reset_index()
        .pivot(index="_origin", columns="_dev", values="_value")
        .fillna(0.0)
        .sort_index()
    )

    cumulative = incr.cumsum(axis=1)
    latest_diag = _latest_diagonal(cumulative)

    logger.info("Built triangle with shape=%s", incr.shape)
    return TriangleArtifacts(incremental=incr, cumulative=cumulative, latest_diagonal=latest_diag)


def _latest_diagonal(cumulative_triangle: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cols = list(cumulative_triangle.columns)
    for idx, origin in enumerate(cumulative_triangle.index):
        row = cumulative_triangle.loc[origin]
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
    text = str(label).strip().upper().replace(" ", "")
    for _, pattern in PERIOD_PATTERNS.items():
        match = pattern.match(text)
        if match:
            year = int(match.group("y"))
            period = int(match.group("p")) if "p" in match.groupdict() else 1
            return year, period
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

    dev_pairs = sorted([(col, parse_development_period_label(col)) for col in development_columns], key=lambda x: x[1])
    ordered_cols = [pair[0] for pair in dev_pairs]
    dev_labels = [f"Dev {pair[1]}" for pair in dev_pairs]

    work = df[[origin_col] + ordered_cols].copy()
    work = work.dropna(subset=[origin_col])
    work[origin_col] = work[origin_col].astype(str)
    numeric = work[ordered_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    numeric.columns = dev_labels
    numeric.index = work[origin_col]

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

    origins_raw = df[origin_col].dropna().astype(str).tolist()
    if not origins_raw:
        raise ValueError("No valid origin periods found.")

    all_periods = list(set(origins_raw + [str(c) for c in calendar_columns]))
    ordered_periods = order_period_labels(all_periods)
    period_to_idx = {p: i for i, p in enumerate(ordered_periods)}

    rows = []
    dev_max = 0
    for _, row in df.iterrows():
        origin = str(row.get(origin_col))
        if origin not in period_to_idx:
            raise ValueError(f"Origin period '{origin}' cannot be aligned to ordered periods.")
        origin_idx = period_to_idx[origin]

        dev_values: dict[int, float] = {}
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
            dev_values[lag] = dev_values.get(lag, 0.0) + value
            dev_max = max(dev_max, lag)

        rows.append((origin, dev_values))

    dev_columns = [f"Dev {i}" for i in range(1, dev_max + 1)]
    inc_matrix = []
    origin_labels = []
    for origin, dev_values in rows:
        origin_labels.append(origin)
        inc_matrix.append([dev_values.get(i, 0.0) for i in range(1, dev_max + 1)])

    incremental = pd.DataFrame(inc_matrix, index=origin_labels, columns=dev_columns)
    cumulative = incremental.cumsum(axis=1)
    return TriangleArtifacts(incremental=incremental, cumulative=cumulative, latest_diagonal=_latest_diagonal(cumulative))
