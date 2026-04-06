from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from reserving_app.services.triangle_builder import TriangleArtifacts

try:
    import chainladder as cl
except Exception:  # pragma: no cover
    cl = None


def load_genins_demo() -> tuple[TriangleArtifacts, pd.Series, str]:
    if cl is None:
        cumulative = _load_local_genins_snapshot()
        sim_ldf = _fallback_sim_ldf(cumulative)
        source = "local_snapshot"
    else:
        tri = cl.load_sample("genins")
        sims = cl.BootstrapODPSample(random_state=42).fit_transform(tri)
        sim_ldf = cl.Development().fit(sims).ldf_
        cumulative = _triangle_to_dataframe(tri)
        source = "chainladder_package"
    incremental = cumulative.diff(axis=1).fillna(cumulative.iloc[:, 0], axis=0)
    latest = cumulative.replace(0, np.nan).ffill(axis=1).iloc[:, -1].fillna(0.0)
    latest_df = pd.DataFrame(
        {"origin": cumulative.index.astype(str), "latest_development": cumulative.columns[-1], "latest_cumulative": latest.values}
    )
    return (
        TriangleArtifacts(incremental=incremental, cumulative=cumulative, latest_diagonal=latest_df),
        pd.Series(np.asarray(sim_ldf.values).flatten()),
        source,
    )


def _triangle_to_dataframe(tri) -> pd.DataFrame:
    if hasattr(tri, "to_frame"):
        df = tri.to_frame()
    elif hasattr(tri, "to_pandas"):
        df = tri.to_pandas()
    else:
        raise ValueError("Unsupported chainladder Triangle object; no to_frame/to_pandas method found.")

    if isinstance(df.index, pd.MultiIndex):
        if "origin" in df.index.names:
            df = df.reset_index().pivot(index="origin", columns="development", values=df.columns[-1])
        else:
            df = df.reset_index().pivot(index=df.columns[0], columns=df.columns[1], values=df.columns[-1])
    elif {"origin", "development"}.issubset(df.columns):
        value_col = "value" if "value" in df.columns else df.columns[-1]
        df = df.pivot(index="origin", columns="development", values=value_col)

    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df = df.sort_index(axis=0).sort_index(axis=1)
    df.index = pd.Index([str(i) for i in df.index])
    df.columns = [f"Dev {i}" for i in range(len(df.columns))]
    return df


def _load_local_genins_snapshot() -> pd.DataFrame:
    path = Path("data/genins_sample.csv")
    df = pd.read_csv(path, index_col=0).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df.columns = [f"Dev {i}" for i in range(len(df.columns))]
    df.index = df.index.astype(str)
    return df


def _fallback_sim_ldf(cumulative: pd.DataFrame) -> pd.Series:
    base_ldf = []
    for i in range(cumulative.shape[1] - 1):
        prev = cumulative.iloc[:, i].replace(0, np.nan)
        nxt = cumulative.iloc[:, i + 1].replace(0, np.nan)
        ratio = (nxt / prev).replace([np.inf, -np.inf], np.nan).dropna()
        base_ldf.append(float(ratio.mean()) if not ratio.empty else 1.0)
    return pd.Series(base_ldf, index=[f"Dev {i}" for i in range(len(base_ldf))])
