from __future__ import annotations

import pandas as pd


def detect_outlier_link_ratios(link_ratios: pd.DataFrame, z_threshold: float = 2.0) -> pd.DataFrame:
    stacked = link_ratios.stack(dropna=True).reset_index()
    if stacked.empty:
        return pd.DataFrame(columns=["origin", "development", "link_ratio", "z_score"])

    stacked.columns = ["origin", "development", "link_ratio"]
    mean = stacked["link_ratio"].mean()
    std = stacked["link_ratio"].std() or 1.0
    stacked["z_score"] = (stacked["link_ratio"] - mean) / std
    return stacked[stacked["z_score"].abs() >= z_threshold].sort_values("z_score", ascending=False)


def sparse_data_warnings(triangle: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    non_zero_ratio = (triangle > 0).sum().sum() / (triangle.shape[0] * triangle.shape[1])

    if triangle.shape[0] < 5:
        warnings.append("Triangle has fewer than 5 origin periods; reserve estimates may be unstable.")
    if triangle.shape[1] < 5:
        warnings.append("Triangle has fewer than 5 development periods; diagnostics may be limited.")
    if non_zero_ratio < 0.4:
        warnings.append("Triangle is sparse (<40% populated). Consider aggregating to coarser periods.")

    return warnings
