"""All @app.callback definitions — the only place Dash 'logic' lives."""
from __future__ import annotations

import numpy as np
import pandas as pd
from dash import Dash, Input, Output, html

import src.decision_support as decision_support
from src.physics import free_drift_velocity, geodesic_distance_km, step_position
from app.figures import build_map_figure, build_diagnostics_figure

TRACK_CSV_PATH = "data/synthetic_track.csv"
DT_HOURS = 6
WINDOW_SIZE = 5


def _load_track(_iceberg_id: str) -> pd.DataFrame:
    return pd.read_csv(TRACK_CSV_PATH, parse_dates=["timestamp"])


def _build_future_environmental_forecast(track: pd.DataFrame, n_steps: int) -> pd.DataFrame:
    """Stand-in forecast: hold the last observed wind/current constant."""
    last_row = track.iloc[-1]
    timestamps = last_row["timestamp"] + pd.to_timedelta(
        np.arange(1, n_steps + 1) * DT_HOURS, unit="h"
    )
    return pd.DataFrame({
        "timestamp": timestamps,
        "u_wind": last_row["u_wind"],
        "v_wind": last_row["v_wind"],
        "u_current": last_row["u_current"],
        "v_current": last_row["v_current"],
    })


def _physics_only_rollout(last_known_row: pd.Series, future_env: pd.DataFrame, dt_seconds: float) -> pd.DataFrame:
    current_lat, current_lon = last_known_row["lat"], last_known_row["lon"]
    records = []
    for _, row in future_env.iterrows():
        u, v = free_drift_velocity(row["u_wind"], row["v_wind"], row["u_current"], row["v_current"], current_lat)
        current_lat, current_lon = step_position(current_lat, current_lon, u, v, dt_seconds)
        records.append({"timestamp": row["timestamp"], "lat": current_lat, "lon": current_lon})
    return pd.DataFrame.from_records(records)


def _pill(text: str, level: str) -> html.Span:
    return html.Span(text, className=f"iw-pill {level}")


def _build_list_rows(forecast: pd.DataFrame, last_known_row: pd.Series, active_step: int,
                      vessel_lat, vessel_lon) -> list:
    """Build the themed row list: one row per forecast step, distance from
    the previous point, and a status pill (green/amber/red) driven by
    proximity to the vessel when one is set, else a plain 'forecast' tag."""
    rows = []
    prev_lat, prev_lon = last_known_row["lat"], last_known_row["lon"]

    for i, rec in enumerate(forecast.itertuples()):
        step_km = geodesic_distance_km(prev_lat, prev_lon, rec.lat, rec.lon)
        prev_lat, prev_lon = rec.lat, rec.lon

        if vessel_lat is not None and vessel_lon is not None:
            dist_to_vessel = geodesic_distance_km(vessel_lat, vessel_lon, rec.lat, rec.lon)
            if dist_to_vessel < 10:
                pill = _pill("DANGER", "red")
            elif dist_to_vessel < 30:
                pill = _pill("WATCH", "amber")
            else:
                pill = _pill("CLEAR", "green")
            meta = f"{rec.timestamp} · {dist_to_vessel:.1f} km from vessel"
        else:
            pill = _pill("FORECAST", "grey")
            meta = f"{rec.timestamp} · step {step_km:.1f} km"

        rows.append(html.Div(className=f"iw-row{' active' if i == active_step else ''}", children=[
            html.Div(className="iw-row-left", children=[
                html.Span(f"{i + 1:02d}", className="iw-row-idx"),
                html.Div([
                    html.Div(f"{rec.lat:.3f}, {rec.lon:.3f}", className="iw-row-title"),
                    html.Div(meta, className="iw-row-meta"),
                ]),
            ]),
            html.Div(className="iw-row-right", children=[pill]),
        ]))

    return rows


def register_callbacks(app: Dash, model: dict, metrics: dict) -> None:

    @app.callback(
        Output("forecast-store", "data"),
        Output("timestep-slider", "max"),
        Output("timestep-slider", "value"),
        Input("iceberg-select", "value"),
        Input("horizon-slider", "value"),
        Input("mode-select", "value"),
    )
    def compute_forecast(iceberg_id: str, horizon_hours: int, mode: str):
        track = _load_track(iceberg_id)
        n_steps = max(int(round(horizon_hours / DT_HOURS)), 1)
        future_env = _build_future_environmental_forecast(track, n_steps)
        dt_seconds = DT_HOURS * 3600.0
        last_known_row = track.iloc[-1]

        if mode == "hybrid":
            history_window = track.iloc[-WINDOW_SIZE:][decision_support.LAG_BASE_COLUMNS].reset_index(drop=True)
            forecast = decision_support.rollout_forecast(
                model, last_known_row, history_window, future_env, model["feature_cols"], dt_seconds
            )
        else:
            forecast = _physics_only_rollout(last_known_row, future_env, dt_seconds)

        forecast_json = forecast.assign(timestamp=forecast["timestamp"].astype(str)).to_dict("records")
        return forecast_json, max(len(forecast) - 1, 0), max(len(forecast) - 1, 0)

    @app.callback(
        Output("map-graph", "figure"),
        Output("forecast-list", "children"),
        Input("forecast-store", "data"),
        Input("iceberg-select", "value"),
        Input("timestep-slider", "value"),
        Input("vessel-lat", "value"),
        Input("vessel-lon", "value"),
        Input("map-only-toggle", "value"),
    )
    def render_map_and_list(forecast_data, iceberg_id: str, step: int, vessel_lat, vessel_lon, map_only_value):
        track = _load_track(iceberg_id)
        full_forecast = pd.DataFrame(forecast_data) if forecast_data else pd.DataFrame(columns=["timestamp", "lat", "lon"])

        rows = []
        if len(full_forecast):
            step = min(step, len(full_forecast) - 1)
            visible_forecast = full_forecast.iloc[: step + 1]
            rows = _build_list_rows(full_forecast, track.iloc[-1], step, vessel_lat, vessel_lon)
        else:
            visible_forecast = full_forecast

        is_map_only = bool(map_only_value and "map_only" in map_only_value)
        fig = build_map_figure(track, visible_forecast, expanded=is_map_only)
        return fig, rows

    @app.callback(
        Output("cpa-text", "children"),
        Output("risk-pill", "children"),
        Input("forecast-store", "data"),
        Input("vessel-lat", "value"),
        Input("vessel-lon", "value"),
    )
    def update_risk(forecast_data, vessel_lat, vessel_lon):
        if not forecast_data or vessel_lat is None or vessel_lon is None:
            return "Enter a vessel position to compute CPA.", ""

        forecast = pd.DataFrame(forecast_data)
        forecast["timestamp"] = pd.to_datetime(forecast["timestamp"])

        cpa = decision_support.compute_cpa(forecast, vessel_lat, vessel_lon)
        risk = decision_support.risk_score(cpa)

        text = f"CPA {cpa['cpa_distance_km']:.1f} km in {cpa['time_to_cpa_hours']:.0f}h"
        pill = _pill(risk["level"].upper(), risk["level"])
        return text, pill

    @app.callback(
        Output("diagnostics-graph", "figure"),
        Input("mode-select", "value"),
    )
    def render_diagnostics(_mode):
        return build_diagnostics_figure(metrics)

    @app.callback(
        Output("iw-details", "style"),
        Output("live-panel", "className"),
        Input("map-only-toggle", "value"),
    )
    def toggle_map_only(toggle_value):
        is_map_only = bool(toggle_value and "map_only" in toggle_value)
        details_style = {"display": "none"} if is_map_only else {"display": "block"}
        panel_class = "iw-panel iw-panel-fullscreen" if is_map_only else "iw-panel"
        return details_style, panel_class