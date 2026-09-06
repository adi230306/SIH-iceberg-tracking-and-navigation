"""weather_api.py

Fetches real short-range wind and ocean current forecasts from
Open-Meteo (free, no API key) for a single point, and reshapes them
into the {timestamp, u_wind, v_wind, u_current, v_current} schema that
decision_support.rollout_forecast() expects as future_environmental_forecast.

This is deliberately separate from data_ingest.py's fetch_era5_wind()/
fetch_copernicus_currents(): those pull HISTORICAL reanalysis data (for
building training tracks), whereas this pulls an actual forward-looking
forecast (for decision_support.py's rollout). Different problem, so a
different module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import requests

WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
MARINE_FORECAST_URL = "https://marine-api.open-meteo.com/v1/marine"


def _speed_dir_to_uv(speed: np.ndarray, direction_deg: np.ndarray, convention: str) -> tuple[np.ndarray, np.ndarray]:
    """Convert speed + compass direction into eastward/northward components.

    Args:
        speed: Speed values, m/s.
        direction_deg: Compass direction, degrees (0 = north, clockwise).
        convention: "from" (meteorological wind convention -- direction
            the wind is BLOWING FROM) or "to" (oceanographic current
            convention -- direction the current is FLOWING TOWARD).
            Getting this backwards flips the sign of both components,
            so it matters a lot for physics.free_drift_velocity().

    Returns:
        (u, v) eastward/northward component arrays, m/s.
    """
    direction_rad = np.radians(direction_deg)
    if convention == "from":
        # Wind vector points opposite the "from" direction.
        u = -speed * np.sin(direction_rad)
        v = -speed * np.cos(direction_rad)
    elif convention == "to":
        u = speed * np.sin(direction_rad)
        v = speed * np.cos(direction_rad)
    else:
        raise ValueError(f"convention must be 'from' or 'to', got {convention!r}")
    return u, v


def fetch_wind_forecast(lat: float, lon: float, n_hours: int) -> pd.DataFrame:
    """Fetch an hourly 10m wind forecast for a point from Open-Meteo.

    Args:
        lat, lon: Point to forecast at, degrees.
        n_hours: Number of hourly forecast steps to request.

    Returns:
        DataFrame with columns timestamp, u_wind, v_wind (m/s).

    Raises:
        RuntimeError: If the request fails or the response is malformed.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "wind_speed_10m,wind_direction_10m",
        "wind_speed_unit": "ms",
        "forecast_hours": n_hours,
        "timezone": "UTC",
    }
    try:
        resp = requests.get(WEATHER_FORECAST_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()["hourly"]
    except Exception as exc:
        raise RuntimeError(f"fetch_wind_forecast: request to Open-Meteo failed: {exc}") from exc

    speed = np.array(data["wind_speed_10m"], dtype=float)
    direction = np.array(data["wind_direction_10m"], dtype=float)
    u_wind, v_wind = _speed_dir_to_uv(speed, direction, convention="from")

    return pd.DataFrame({
        "timestamp": pd.to_datetime(data["time"]),
        "u_wind": u_wind,
        "v_wind": v_wind,
    })


def fetch_current_forecast(lat: float, lon: float, n_hours: int) -> pd.DataFrame:
    """Fetch an hourly ocean surface current forecast for a point from
    Open-Meteo's Marine API.

    Args:
        lat, lon: Point to forecast at, degrees.
        n_hours: Number of hourly forecast steps to request.

    Returns:
        DataFrame with columns timestamp, u_current, v_current (m/s).

    Raises:
        RuntimeError: If the request fails, the response is malformed,
            or the point has no marine data (e.g. far inland -- not a
            concern for iceberg tracks, but worth knowing).
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "ocean_current_velocity,ocean_current_direction",
        "length_unit": "metric",
        "forecast_hours": n_hours,
        "timezone": "UTC",
    }
    try:
        resp = requests.get(MARINE_FORECAST_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()["hourly"]
    except Exception as exc:
        raise RuntimeError(f"fetch_current_forecast: request to Open-Meteo Marine failed: {exc}") from exc

    # ocean_current_velocity comes back in km/h even with length_unit=metric;
    # Open-Meteo doesn't expose an m/s option for this field, so convert.
    speed_kmh = np.array(data["ocean_current_velocity"], dtype=float)
    speed_ms = speed_kmh / 3.6
    direction = np.array(data["ocean_current_direction"], dtype=float)
    u_current, v_current = _speed_dir_to_uv(speed_ms, direction, convention="to")

    return pd.DataFrame({
        "timestamp": pd.to_datetime(data["time"]),
        "u_current": u_current,
        "v_current": v_current,
    })


def fetch_future_environmental_forecast(lat: float, lon: float, n_steps: int, dt_hours: int) -> pd.DataFrame:
    """Fetch a combined wind+current forecast, resampled to the project's
    dt_hours step size, matching the schema decision_support.rollout_forecast()
    expects for future_environmental_forecast.

    Args:
        lat, lon: The iceberg's current (last-known) position -- used as
            a fixed point for the whole forecast horizon. Good enough
            for a hackathon; a more accurate version would re-fetch at
            the iceberg's PREDICTED position at each step, but that
            means fetching one step at a time instead of one batched
            call, which is much slower and hits rate limits fast.
        n_steps: Number of dt_hours-sized forecast steps needed.
        dt_hours: The project's fixed timestep size (e.g. 6).

    Returns:
        DataFrame with columns timestamp, u_wind, v_wind, u_current,
        v_current, one row per step, ascending by time.

    Raises:
        RuntimeError: If either underlying fetch fails.
    """
    n_hours_needed = n_steps * dt_hours
    wind_df = fetch_wind_forecast(lat, lon, n_hours_needed)
    current_df = fetch_current_forecast(lat, lon, n_hours_needed)

    merged = pd.merge(wind_df, current_df, on="timestamp", how="inner")

    # Open-Meteo returns hourly rows; downsample to the project's
    # dt_hours step size by taking every dt_hours-th row.
    merged = merged.iloc[::dt_hours].reset_index(drop=True)
    merged = merged.iloc[:n_steps]

    if len(merged) < n_steps:
        raise RuntimeError(
            f"fetch_future_environmental_forecast: Open-Meteo only returned "
            f"{len(merged)} usable step(s) but {n_steps} were requested. "
            f"Try a smaller horizon or check the API response."
        )

    return merged[["timestamp", "u_wind", "v_wind", "u_current", "v_current"]]