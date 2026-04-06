from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import pandas as pd

from reserving_app.services.triangle_builder import link_ratio_matrix

try:
    import chainladder as cl
except Exception:  # pragma: no cover
    cl = None


@dataclass
class DeterministicResult:
    selected_ldf: pd.Series
    cdf: pd.Series
    ultimates: pd.Series
    ibnr: pd.Series
    diagnostics: dict[str, pd.DataFrame | pd.Series]


@dataclass
class BootstrapResult:
    reserve_distribution: pd.Series
    summary: pd.Series


def _origin_label_to_timestamp(origin_label: object, grain: str) -> pd.Timestamp:
    text = str(origin_label).strip()
    if grain == "Quarterly":
        q_match = re.match(r"^(\d{4})Q([1-4])$", text.upper())
        if q_match:
            year, quarter = int(q_match.group(1)), int(q_match.group(2))
            month = 3 * (quarter - 1) + 1
            return pd.Timestamp(year=year, month=month, day=1)
    if grain == "Half-Yearly":
        h_match = re.match(r"^(\d{4})[- ]?H([1-2])$", text.upper())
        if h_match:
            year, half = int(h_match.group(1)), int(h_match.group(2))
            month = 1 if half == 1 else 7
            return pd.Timestamp(year=year, month=month, day=1)
    ts = pd.to_datetime([origin_label], errors="coerce")[0]
    if pd.isna(ts):
        y_match = re.match(r"^(\d{4})$", text)
        if y_match:
            return pd.Timestamp(year=int(y_match.group(1)), month=1, day=1)
        raise ValueError(f"Could not parse origin label '{origin_label}' to a date.")
    return pd.Timestamp(ts)


def _grain_offset(grain: str, lag: int) -> pd.DateOffset:
    if grain == "Yearly":
        return pd.DateOffset(years=lag)
    if grain == "Quarterly":
        return pd.DateOffset(months=3 * lag)
    if grain == "Half-Yearly":
        return pd.DateOffset(months=6 * lag)
    return pd.DateOffset(months=lag)


def cumulative_to_chainladder_triangle(cumulative: pd.DataFrame, grain: str = "Yearly"):
    if cl is None:
        raise RuntimeError("chainladder package is not available.")

    records = []
    for origin_label in cumulative.index:
        origin_dt = _origin_label_to_timestamp(origin_label, grain)
        for dev_col in cumulative.columns:
            lag_match = re.search(r"\d+", str(dev_col))
            lag = int(lag_match.group(0)) if lag_match else 0
            development_dt = origin_dt + _grain_offset(grain, lag)
            value = cumulative.loc[origin_label, dev_col]
            if pd.isna(value):
                continue
            records.append(
                {
                    "origin": origin_dt,
                    "development": development_dt,
                    "value": float(value),
                }
            )
    long_df = pd.DataFrame(records)
    return cl.Triangle(long_df, origin="origin", development="development", columns=["value"], cumulative=True)


def _to_chainladder_triangle(cumulative: pd.DataFrame):
    records = []
    for i, origin_label in enumerate(cumulative.index):
        origin_dt = pd.Timestamp(year=2000 + i, month=1, day=1)
        row_values = cumulative.loc[origin_label].astype(float).values
        non_zero_positions = np.where(row_values != 0)[0]
        last_observed = int(non_zero_positions.max()) if len(non_zero_positions) > 0 else -1
        for j, dev_col in enumerate(cumulative.columns):
            valuation_dt = origin_dt + pd.DateOffset(months=12 * j)
            value = float(cumulative.loc[origin_label, dev_col]) if j <= last_observed else np.nan
            records.append(
                {
                    "origin": origin_dt.strftime("%Y-%m-%d"),
                    "development": valuation_dt.strftime("%Y-%m-%d"),
                    "value": value,
                }
            )
    long_df = pd.DataFrame(records)
    return cl.Triangle(
        long_df,
        origin="origin",
        development="development",
        columns=["value"],
        cumulative=True,
    )


def _selected_ldf(cumulative: pd.DataFrame, exclusions: set[tuple[int, int]] | None = None) -> pd.Series:
    lr = link_ratio_matrix(cumulative)
    weights = cumulative.iloc[:, :-1].copy()

    if exclusions:
        for r, c in exclusions:
            if r < len(lr.index) and c < len(lr.columns):
                lr.iat[r, c] = np.nan
                weights.iat[r, c] = np.nan

    vw = (lr * weights).sum(axis=0) / weights.where(~lr.isna()).sum(axis=0)
    vw = vw.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return vw


def run_chain_ladder(cumulative: pd.DataFrame, apply_tail_factor: bool, exclusions: set[tuple[int, int]] | None = None) -> DeterministicResult:
    selected = _selected_ldf(cumulative, exclusions=exclusions)

    if apply_tail_factor:
        selected = pd.concat([selected, pd.Series([1.02], index=["Tail"])])

    cdf_values = []
    running = 1.0
    for val in reversed(selected.values):
        running *= float(val)
        cdf_values.append(running)
    cdf = pd.Series(list(reversed(cdf_values)), index=selected.index)

    observed_triangle = cumulative.replace(0, np.nan)
    latest = observed_triangle.ffill(axis=1).iloc[:, -1].fillna(0.0)

    observed_counts = observed_triangle.notna().sum(axis=1)
    last_dev_position = (observed_counts - 1).clip(lower=0)
    row_cdf = []
    for pos, latest_value in zip(last_dev_position, latest.values):
        if float(latest_value) <= 0:
            row_cdf.append(1.0)
            continue
        pos = max(min(int(pos), len(cdf) - 1), 0)
        row_cdf.append(float(cdf.iloc[pos]))

    row_cdf_series = pd.Series(row_cdf, index=cumulative.index)
    ultimates = latest * row_cdf_series
    ibnr = (ultimates - latest).clip(lower=0)

    diagnostics = {
        "link_ratios": link_ratio_matrix(cumulative),
        "volume_weighted_ldf": selected,
        "cdf": cdf,
    }

    return DeterministicResult(selected_ldf=selected, cdf=cdf, ultimates=ultimates, ibnr=ibnr, diagnostics=diagnostics)


def run_chainladder_model(
    cumulative: pd.DataFrame,
    model_name: str,
    grain: str = "Yearly",
    n_sims: int = 1000,
    apriori: float = 1.0,
    benktander_iters: int = 2,
) -> tuple[DeterministicResult, BootstrapResult | None]:
    if cl is None:
        if model_name != "Chainladder":
            raise RuntimeError("chainladder package is not available for selected model.")
        return run_chain_ladder(cumulative, apply_tail_factor=False), run_bootstrap_chain_ladder(cumulative, n_sims=n_sims)

    tri = cumulative_to_chainladder_triangle(cumulative, grain=grain)
    model_lookup = {
        "Chainladder": cl.Chainladder(),
        "MackChainladder": cl.MackChainladder(),
        "BornhuetterFerguson": cl.BornhuetterFerguson(apriori=apriori),
        "Benktander": cl.Benktander(apriori=apriori, n_iters=benktander_iters),
        "CapeCod": cl.CapeCod(),
        "Development": cl.Development(),
    }
    if model_name not in model_lookup and model_name != "BootstrapODPSample":
        raise ValueError(f"Unsupported model '{model_name}'.")

    if model_name == "Development":
        dev = model_lookup[model_name].fit(tri)
        selected_ldf = pd.Series(np.asarray(dev.ldf_.values).flatten(), index=[f"Dev {i}" for i in range(np.asarray(dev.ldf_.values).size)])
        det = run_chain_ladder(cumulative, apply_tail_factor=False)
        det.selected_ldf = selected_ldf
        return det, None

    if model_name == "BootstrapODPSample":
        det_model = cl.Chainladder().fit(tri)
        sims = cl.BootstrapODPSample(n_sims=n_sims).fit_transform(tri)
        boot_model = cl.Chainladder().fit(sims)
        boot_values = np.asarray(boot_model.ibnr_.values).sum(axis=(1, 2, 3)).flatten()
        boot_series = pd.Series(boot_values)
        det = _deterministic_from_chainladder(det_model, cumulative)
        return det, BootstrapResult(reserve_distribution=boot_series, summary=boot_series.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))

    model = model_lookup[model_name]
    if model_name in {"BornhuetterFerguson", "Benktander", "CapeCod"}:
        latest = cumulative.replace(0, np.nan).ffill(axis=1).iloc[:, -1].fillna(0.0)
        exposure_records = []
        for origin_label, exposure_value in latest.items():
            origin_dt = _origin_label_to_timestamp(origin_label, grain)
            development_dt = origin_dt
            exposure_records.append({"origin": origin_dt, "development": development_dt, "value": float(max(exposure_value, 1.0))})
        exposure_df = pd.DataFrame(exposure_records)
        exposure_triangle = cl.Triangle(exposure_df, origin="origin", development="development", columns=["value"], cumulative=True)
        model = model.fit(tri, sample_weight=exposure_triangle)
    else:
        model = model.fit(tri)
    det = _deterministic_from_chainladder(model, cumulative)
    return det, None


def _deterministic_from_chainladder(model, cumulative: pd.DataFrame) -> DeterministicResult:
    try:
        ibnr_vals = np.asarray(model.ibnr_.values).reshape(-1)
    except Exception:
        ibnr_vals = np.zeros(len(cumulative.index))
    try:
        ult_vals = np.asarray(model.ultimate_.values).reshape(-1)
    except Exception:
        ult_vals = cumulative.replace(0, np.nan).ffill(axis=1).iloc[:, -1].fillna(0.0).values
    try:
        ldf_vals = np.asarray(model.ldf_.values).reshape(-1)
    except Exception:
        ldf_vals = np.asarray(_selected_ldf(cumulative).values)

    selected = pd.Series(ldf_vals[: max(len(cumulative.columns) - 1, 1)], index=[f"Dev {i}" for i in range(max(len(ldf_vals[: max(len(cumulative.columns) - 1, 1)]), 1))])
    cdf_values = []
    running = 1.0
    for val in reversed(selected.values):
        running *= float(val)
        cdf_values.append(running)
    cdf = pd.Series(list(reversed(cdf_values)), index=selected.index)

    ibnr = pd.Series(ibnr_vals[: len(cumulative.index)], index=cumulative.index).fillna(0.0)
    ultimates = pd.Series(ult_vals[: len(cumulative.index)], index=cumulative.index).fillna(0.0)
    diagnostics = {"link_ratios": link_ratio_matrix(cumulative), "volume_weighted_ldf": selected, "cdf": cdf}
    return DeterministicResult(selected_ldf=selected, cdf=cdf, ultimates=ultimates, ibnr=ibnr, diagnostics=diagnostics)


def run_bootstrap_chain_ladder(cumulative: pd.DataFrame, n_sims: int = 1000, grain: str = "Yearly") -> BootstrapResult:
    if cl is not None:
        try:
            tri = cumulative_to_chainladder_triangle(cumulative, grain=grain)
            sims = cl.BootstrapODPSample(n_sims=n_sims).fit_transform(tri)
            model = cl.Chainladder().fit(sims)
            reserve_dist = pd.Series(np.asarray(model.ibnr_.values).sum(axis=(1, 2, 3)).flatten())
        except Exception:
            reserve_dist = _fallback_bootstrap(cumulative, n_sims)
    else:
        reserve_dist = _fallback_bootstrap(cumulative, n_sims)

    summary = reserve_dist.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99])
    return BootstrapResult(reserve_distribution=reserve_dist, summary=summary)


def _fallback_bootstrap(cumulative: pd.DataFrame, n_sims: int) -> pd.Series:
    base = run_chain_ladder(cumulative, apply_tail_factor=False).ibnr.sum()
    noise = np.random.normal(loc=1.0, scale=0.15, size=n_sims)
    noise = np.clip(noise, 0.5, 1.8)
    return pd.Series(base * noise)


def run_bootstrap_odp_distribution(
    cumulative: pd.DataFrame,
    n_sims: int,
    random_state: int | None = None,
    drop_high=None,
    drop_low=None,
) -> pd.Series:
    if cl is None:
        raise RuntimeError("chainladder package is not available.")
    try:
        tri = _to_chainladder_triangle(cumulative)
        sims = cl.BootstrapODPSample(
            n_sims=n_sims,
            random_state=random_state,
            drop_high=drop_high,
            drop_low=drop_low,
        ).fit_transform(tri)
        model = cl.Chainladder().fit(sims)
        ibnr_values = np.asarray(model.ibnr_.values)
        return pd.Series(ibnr_values.sum(axis=(1, 2, 3)).flatten(), name="ibnr")
    except Exception as exc:
        raise ValueError(f"Triangle is not suitable for BootstrapODPSample: {exc}") from exc


def run_bootstrap_odp_variability_comparison(
    cumulative: pd.DataFrame,
    n_sims: int,
    random_state: int | None,
    drop_high_count: int,
    drop_low_count: int,
) -> pd.DataFrame:
    try:
        tri = _to_chainladder_triangle(cumulative)
        dev_count = max(tri.shape[-1] - 1, 1)
        drop_high = [True] * min(drop_high_count, dev_count) + [False] * max(dev_count - drop_high_count, 0)
        drop_low = [True] * min(drop_low_count, dev_count) + [False] * max(dev_count - drop_low_count, 0)

        s1 = cl.BootstrapODPSample(n_sims=n_sims, random_state=random_state).fit(tri).resampled_triangles_
        s2 = cl.BootstrapODPSample(
            drop_high=drop_high,
            drop_low=drop_low,
            n_sims=n_sims,
            random_state=random_state,
        ).fit_transform(tri)

        original_values = np.asarray(cl.Chainladder().fit(s1).ibnr_.values)
        dropped_values = np.asarray(cl.Chainladder().fit(s2).ibnr_.values)
        original = pd.Series(original_values.sum(axis=(1, 2, 3)).flatten(), name="Original")
        dropped = pd.Series(dropped_values.sum(axis=(1, 2, 3)).flatten(), name="Dropped")
        return pd.concat([original, dropped], axis=1)
    except Exception as exc:
        raise ValueError(f"Triangle is not suitable for bootstrap variability comparison: {exc}") from exc
