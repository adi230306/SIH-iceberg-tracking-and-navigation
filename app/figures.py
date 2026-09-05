"""Pure Plotly figure-building functions — themed to match the app's
cream/black/green palette. No Dash-specific code, easy to test standalone."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

INK = "#0B0F0D"
GREEN = "#1B7A3D"
RED = "#B23A3A"
PANEL = "#FFFFFF"


def build_map_figure(track_df: pd.DataFrame, forecast_df: pd.DataFrame, expanded: bool = False) -> go.Figure:
    """Build the map figure: observed track + forecast (with uncertainty).

    Args:
        expanded: when True (the "map only" popout view), render taller
            and with a bit more legend/margin room, since the map is the
            only thing on screen rather than sharing space with the
            controls/list below it.
    """
    fig = go.Figure()
    scatter_map = go.Scattermap if hasattr(go, "Scattermap") else go.Scattermapbox

    fig.add_trace(scatter_map(
        lat=track_df.lat, lon=track_df.lon, mode="lines+markers",
        name="Observed", marker=dict(size=6, color=INK),
        line=dict(width=2, color=INK),
    ))

    if len(forecast_df):
        fig.add_trace(scatter_map(
            lat=forecast_df.lat, lon=forecast_df.lon, mode="lines+markers",
            name="Forecast", marker=dict(size=6, color=GREEN),
            line=dict(width=2, color=GREEN),
        ))
        if "uncertainty_km" in forecast_df.columns:
            fig.add_trace(scatter_map(
                lat=forecast_df.lat, lon=forecast_df.lon, mode="markers",
                name="Uncertainty", showlegend=True,
                marker=dict(size=(forecast_df["uncertainty_km"] * 1.5).clip(upper=40),
                            color=GREEN, opacity=0.12),
            ))

    center_lat = track_df.lat.iloc[-1]
    center_lon = track_df.lon.iloc[-1]
    map_layout_key = "map" if hasattr(go, "Scattermap") else "mapbox"
    fig.update_layout(
        **{
            f"{map_layout_key}_style": "light",
            f"{map_layout_key}_zoom": 5 if not expanded else 6,
            f"{map_layout_key}_center": {"lat": center_lat, "lon": center_lon},
        },
        paper_bgcolor=PANEL,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="left", x=0.01,
                    font=dict(family="Inter, sans-serif", size=11, color=INK),
                    bgcolor="rgba(255,255,255,0.85)"),
        height=340 if not expanded else None,
        autosize=expanded,
        font=dict(family="Inter, sans-serif"),
    )
    return fig


def build_diagnostics_figure(metrics: dict) -> go.Figure:
    """Build the physics-only vs hybrid FDE comparison bar chart."""
    colors = [INK if k != "hybrid" else GREEN for k in metrics.keys()]
    fig = go.Figure(go.Bar(
        x=list(metrics.keys()), y=list(metrics.values()),
        marker_color=colors,
        text=[f"{v:.1f} km" for v in metrics.values()], textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace", size=13, color=INK),
    ))
    fig.update_layout(
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        yaxis_title="Final displacement error (km)",
        font=dict(family="Inter, sans-serif", color=INK),
        margin=dict(l=50, r=20, t=20, b=40),
        height=320,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#E4E1D8"),
    )
    return fig


if __name__ == "__main__":
    dummy_track = pd.DataFrame({"lat": [-65.0, -64.9], "lon": [-55.0, -54.8]})
    dummy_forecast = pd.DataFrame({"lat": [-64.8, -64.7], "lon": [-54.6, -54.4], "uncertainty_km": [3, 6]})
    f1 = build_map_figure(dummy_track, dummy_forecast)
    f2 = build_diagnostics_figure({"physics_only": 18.4, "hybrid": 6.7})
    assert f1.data and f2.data
    print("figures OK")