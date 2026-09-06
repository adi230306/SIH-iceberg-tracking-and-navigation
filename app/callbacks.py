"""All @app.callback definitions — the only place Dash 'logic' lives."""
from __future__ import annotations

import numpy as np
import pandas as pd
from dash import Dash, Input, Output, State, html

import decision_support
import weather_api
from physics import free_drift_velocity, geodesic_distance_km, step_position
from app.figures import build_map_figure, build_heatmap_figure, build_diagnostics_figure

TRACK_CSV_PATH = "data/synthetic_track.csv"
DT_HOURS = 6
WINDOW_SIZE = 5
N_BOOTSTRAP_SAMPLES = 25
NOISE_STD_WIND = 1.0
NOISE_STD_CURRENT = 0.05


def _load_track(_iceberg_id: str) -> pd.DataFrame:
    return pd.read_csv(TRACK_CSV_PATH, parse_dates=["timestamp"])


def _build_future_environmental_forecast(track: pd.DataFrame, n_steps: int) -> pd.DataFrame:
    """Get the wind/current forecast for the next n_steps timesteps.

    Tries a real forecast from Open-Meteo first (see weather_api.py);
    if that fails for any reason -- no internet, the free tier's rate
    limit, the API being down -- falls back to holding the last
    observed wind/current constant, so the app degrades gracefully
    rather than crashing mid-demo.
    """
    last_row = track.iloc[-1]
    try:
        return weather_api.fetch_future_environmental_forecast(
            lat=last_row["lat"], lon=last_row["lon"], n_steps=n_steps, dt_hours=DT_HOURS
        )
    except Exception as exc:
        print(f"[weather_api] live forecast fetch failed, using stand-in: {exc}")
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


def _physics_only_bootstrap(
    last_known_row: pd.Series, future_env: pd.DataFrame, dt_seconds: float,
    n_samples: int = N_BOOTSTRAP_SAMPLES, noise_std_wind: float = NOISE_STD_WIND,
    noise_std_current: float = NOISE_STD_CURRENT,
) -> list:
    """Physics-only counterpart to decision_support.bootstrap_uncertainty_cone,
    for when 'Physics only' mode is selected -- that function requires a
    trained model dict to run rollout_forecast internally, which doesn't
    apply here. Same idea: perturb the environmental forecast with
    Gaussian noise n_samples times and re-roll each one, so the spread
    across samples approximates forecast uncertainty."""
    rng = np.random.default_rng()
    samples = []
    for _ in range(n_samples):
        noisy = future_env.copy()
        n = len(noisy)
        noisy["u_wind"] = noisy["u_wind"] + rng.normal(0, noise_std_wind, n)
        noisy["v_wind"] = noisy["v_wind"] + rng.normal(0, noise_std_wind, n)
        noisy["u_current"] = noisy["u_current"] + rng.normal(0, noise_std_current, n)
        noisy["v_current"] = noisy["v_current"] + rng.normal(0, noise_std_current, n)
        samples.append(_physics_only_rollout(last_known_row, noisy, dt_seconds))
    return samples


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


def _meta_row(label: str, value: str) -> html.Div:
    return html.Div(className="iw-heatmap-meta-row", children=[
        html.Span(label), html.Span(value),
    ])


def _build_heatmap_meta(samples: list, mode: str, horizon_hours: int, n_steps: int,
                         track: pd.DataFrame) -> list:
    """Build the small metadata card shown over the heatmap: what actually
    generated it, so 'a glowing blob' reads as a specific, reproducible
    computation rather than decoration."""
    final_lats = [s["lat"].iloc[-1] for s in samples if len(s)]
    final_lons = [s["lon"].iloc[-1] for s in samples if len(s)]

    if final_lats:
        center_lat, center_lon = float(np.mean(final_lats)), float(np.mean(final_lons))
        spread_km = max(
            geodesic_distance_km(center_lat, center_lon, lat, lon)
            for lat, lon in zip(final_lats, final_lons)
        )
        spread_text = f"{spread_km:.1f} km"
    else:
        spread_text = "n/a"

    mode_label = "Hybrid (Physics + ML)" if mode == "hybrid" else "Physics only"
    as_of = track["timestamp"].iloc[-1]

    return [
        html.Div("FORECAST SPREAD", className="iw-heatmap-meta-title"),
        _meta_row("Samples", str(len(samples))),
        _meta_row("Horizon", f"{horizon_hours}h ({n_steps} steps)"),
        _meta_row("Model", mode_label),
        _meta_row("Wind noise", f"\u00b1{NOISE_STD_WIND:.1f} m/s"),
        _meta_row("Current noise", f"\u00b1{NOISE_STD_CURRENT:.2f} m/s"),
        _meta_row("Max spread", spread_text),
        _meta_row("As of", str(as_of)),
    ]


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
        Output("heatmap-meta", "children"),
        Output("heatmap-meta", "style"),
        Input("forecast-store", "data"),
        Input("iceberg-select", "value"),
        Input("timestep-slider", "value"),
        Input("vessel-lat", "value"),
        Input("vessel-lon", "value"),
        Input("fullscreen-state", "data"),
        Input("heatmap-state", "data"),
        State("horizon-slider", "value"),
        State("mode-select", "value"),
    )
    def render_map_and_list(forecast_data, iceberg_id: str, step: int, vessel_lat, vessel_lon,
                             is_fullscreen: bool, is_heatmap: bool, horizon_hours: int, mode: str):
        track = _load_track(iceberg_id)
        full_forecast = pd.DataFrame(forecast_data) if forecast_data else pd.DataFrame(columns=["timestamp", "lat", "lon"])

        rows = []
        if len(full_forecast):
            step = min(step, len(full_forecast) - 1)
            visible_forecast = full_forecast.iloc[: step + 1]
            rows = _build_list_rows(full_forecast, track.iloc[-1], step, vessel_lat, vessel_lon)
        else:
            visible_forecast = full_forecast

        meta_children, meta_style = [], {"display": "none"}

        if is_heatmap:
            n_steps = max(int(round(horizon_hours / DT_HOURS)), 1)
            future_env = _build_future_environmental_forecast(track, n_steps)
            dt_seconds = DT_HOURS * 3600.0
            last_known_row = track.iloc[-1]

            if mode == "hybrid":
                history_window = track.iloc[-WINDOW_SIZE:][decision_support.LAG_BASE_COLUMNS].reset_index(drop=True)
                samples = decision_support.bootstrap_uncertainty_cone(
                    model, last_known_row, history_window, future_env, model["feature_cols"], dt_seconds,
                    n_samples=N_BOOTSTRAP_SAMPLES, noise_std_wind=NOISE_STD_WIND, noise_std_current=NOISE_STD_CURRENT,
                )
            else:
                samples = _physics_only_bootstrap(last_known_row, future_env, dt_seconds)

            fig = build_heatmap_figure(track, samples, expanded=is_fullscreen)
            meta_children = _build_heatmap_meta(samples, mode, horizon_hours, n_steps, track)
            meta_style = {"display": "block"}
        else:
            fig = build_map_figure(track, visible_forecast, expanded=is_fullscreen)

        return fig, rows, meta_children, meta_style

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
        Output("fullscreen-state", "data"),
        Input("fullscreen-btn", "n_clicks"),
        State("fullscreen-state", "data"),
        prevent_initial_call=True,
    )
    def toggle_fullscreen(_n_clicks, is_fullscreen):
        return not is_fullscreen

    @app.callback(
        Output("heatmap-state", "data"),
        Input("heatmap-btn", "n_clicks"),
        State("heatmap-state", "data"),
        prevent_initial_call=True,
    )
    def toggle_heatmap(_n_clicks, is_heatmap):
        return not is_heatmap

    @app.callback(
        Output("iw-details", "style"),
        Output("live-panel", "className"),
        Output("fullscreen-btn", "className"),
        Output("panel-title", "children"),
        Input("fullscreen-state", "data"),
    )
    def apply_fullscreen(is_fullscreen: bool):
        details_style = {"display": "none"} if is_fullscreen else {"display": "block"}
        panel_class = "iw-panel iw-panel-fullscreen" if is_fullscreen else "iw-panel"
        btn_class = "iw-map-tool-btn active" if is_fullscreen else "iw-map-tool-btn"
        title = "LIVE FORECAST — FULLSCREEN" if is_fullscreen else "LIVE FORECAST"
        return details_style, panel_class, btn_class, title

    @app.callback(
        Output("heatmap-btn", "className"),
        Input("heatmap-state", "data"),
    )
    def apply_heatmap_button_state(is_heatmap: bool):
        return "iw-map-tool-btn active" if is_heatmap else "iw-map-tool-btn"