from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TriangleArtifacts:
    incremental: pd.DataFrame
    cumulative: pd.DataFrame
    latest_diagonal: pd.DataFrame


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
