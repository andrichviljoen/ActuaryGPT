from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.services.triangle_builder import link_ratio_matrix

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

    latest = cumulative.replace(0, np.nan).ffill(axis=1).iloc[:, -1].fillna(0.0)

    development_cols = [c for c in cumulative.columns if c in selected.index]
    last_dev_position = cumulative.notna().sum(axis=1) - 2
    row_cdf = []
    for pos in last_dev_position:
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


def run_bootstrap_chain_ladder(cumulative: pd.DataFrame, n_sims: int = 1000) -> BootstrapResult:
    if cl is not None:
        try:
            long_df = cumulative.reset_index().melt(id_vars=cumulative.index.name or "index")
            long_df.columns = ["origin", "development", "value"]
            tri = cl.Triangle(
                long_df,
                origin="origin",
                development="development",
                columns=["value"],
                cumulative=True,
            )
            sims = cl.BootstrapODPSample(n_sims=n_sims).fit_transform(tri)
            model = cl.Chainladder().fit(sims)
            reserve_dist = pd.Series(model.ibnr_.sum(axis=(1, 2)).values.flatten())
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
