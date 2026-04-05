from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def heatmap_from_triangle(triangle: pd.DataFrame, title: str):
    fig = px.imshow(
        triangle.values,
        labels={"x": "Development", "y": "Origin", "color": "Value"},
        x=[str(c) for c in triangle.columns],
        y=[str(i) for i in triangle.index],
        title=title,
        aspect="auto",
        color_continuous_scale="Blues",
    )
    return fig


def development_factor_chart(ldf: pd.Series):
    fig = go.Figure(data=[go.Bar(x=[str(i) for i in ldf.index], y=ldf.values)])
    fig.update_layout(title="Selected Development Factors", xaxis_title="Development Age", yaxis_title="Factor")
    return fig


def reserve_by_origin_chart(ibnr: pd.Series):
    fig = px.bar(x=[str(i) for i in ibnr.index], y=ibnr.values, labels={"x": "Origin Period", "y": "IBNR"}, title="Reserve by Origin")
    return fig


def cumulative_vs_ultimate(latest: pd.Series, ultimate: pd.Series):
    df = pd.DataFrame({"latest": latest, "ultimate": ultimate})
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Latest", x=df.index.astype(str), y=df["latest"]))
    fig.add_trace(go.Bar(name="Ultimate", x=df.index.astype(str), y=df["ultimate"]))
    fig.update_layout(title="Latest Cumulative vs Ultimate", barmode="group")
    return fig


def bootstrap_histogram(dist: pd.Series):
    fig = px.histogram(dist, nbins=40, title="Bootstrap Reserve Distribution")
    return fig


def percentile_chart(summary: pd.Series):
    points = ["50%", "75%", "90%", "95%", "99%"]
    values = [summary.get(p, None) for p in points]
    fig = px.line(x=points, y=values, markers=True, title="Reserve Uncertainty Percentiles")
    fig.update_yaxes(title="Reserve")
    return fig


def bootstrap_comparison_histogram(comparison_df: pd.DataFrame):
    fig = go.Figure()
    for col in comparison_df.columns:
        fig.add_trace(
            go.Histogram(
                x=comparison_df[col].dropna(),
                name=str(col),
                opacity=0.6,
                nbinsx=40,
            )
        )
    fig.update_layout(
        title="Bootstrap ODP Variability Comparison",
        barmode="overlay",
        xaxis_title="IBNR / Reserve",
        yaxis_title="Count",
    )
    return fig
