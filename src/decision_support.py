"""
decision_support.py

Turns a trained hybrid model's predictions into an actual forward-
looking forecast with uncertainty, plus a risk score relative to a
vessel/platform position -- the "decision support" half of the
project, distinct from pure trajectory prediction (train_model.py) or
feature engineering (features.py).

Everything here operates on FUTURE, not-yet-observed timesteps, which
is a fundamentally different problem from evaluate_trajectory()'s
historical rollout in train_model.py: we no longer have real future
wind/current values to fall back on, only a forecast of them (however
crude), so uncertainty in that forecast has to be propagated forward
explicitly rather than ignored.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.physics import free_drift_velocity, geodesic_distance_km, step_position
from src.train_model import predict_residual

# The environmental variables that get lagged into sliding-window
# features (must match features.LAG_BASE_COLUMNS exactly, since the
# feature rows built here have to line up with what the models were
# trained on).
LAG_BASE_COLUMNS: list[str] = ["u_wind", "v_wind", "u_current", "v_current", "area_km2"]


def _infer_window_size(feature_cols: list[str]) -> int:
    """Infer the sliding-window size used to build feature_cols from their names.

    Lag feature columns are named "{var}_t-{lag}" (see
    features.build_sliding_window_features); the window size is the
    maximum lag number present across all such columns. Reimplemented
    here (rather than imported from train_model, where an equivalent
    private helper lives) to keep this module's dependencies limited to
    the public contract listed in its docstring.

    Args:
        feature_cols: The feature column names to inspect.

    Returns:
        The inferred window size, or 0 if no lag columns are found.
    """
    lag_pattern = re.compile(r"_t-(\d+)$")
    lags = [int(m.group(1)) for col in feature_cols if (m := lag_pattern.search(col))]
    return max(lags) if lags else 0


def build_feature_row(
    history_window: pd.DataFrame,
    current_lat: float,
    current_lon: float,
    phys_u: float,
    phys_v: float,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Assemble a single-row feature DataFrame matching features.py's exact column layout.

    During forecast rollout we don't have a pre-built feature table --
    the future is being generated one step at a time -- so each
    feature row has to be built on the fly from a rolling window of
    recent conditions plus the current step's state.

    history_window is expected to hold the `window_size` (inferred from
    feature_cols) most recent OBSERVED rows of u_wind, v_wind,
    u_current, v_current, area_km2, sorted ascending by time, with the
    LAST row being the most recent one (i.e. "t-1" relative to the step
    being forecast) -- this exactly matches the lag convention used by
    features.build_sliding_window_features (df[col].shift(1) == the
    immediately preceding row).

    Args:
        history_window: DataFrame with at least the LAG_BASE_COLUMNS
            columns, at least `window_size` rows, most-recent-last.
        current_lat: The iceberg's current (starting-point-for-this-
            step) latitude, degrees.
        current_lon: The iceberg's current longitude, degrees.
        phys_u: The physics-baseline eastward velocity prediction for
            this step, m/s.
        phys_v: The physics-baseline northward velocity prediction for
            this step, m/s.
        feature_cols: The exact feature column names (and order) the
            trained models expect.

    Returns:
        A single-row DataFrame with columns exactly matching
        feature_cols, in that order.

    Raises:
        ValueError: If history_window has fewer rows than the window
            size implied by feature_cols, or is missing any required
            lag base column.
    """
    window_size = _infer_window_size(feature_cols)

    missing_cols = [c for c in LAG_BASE_COLUMNS if c not in history_window.columns]
    if missing_cols:
        raise ValueError(
            f"build_feature_row: history_window is missing required column(s) "
            f"{missing_cols}. It must contain: {LAG_BASE_COLUMNS}."
        )

    if len(history_window) < window_size:
        raise ValueError(
            f"build_feature_row: history_window has only {len(history_window)} row(s), "
            f"but feature_cols implies a window size of {window_size}. Provide at least "
            f"{window_size} rows of recent history (most-recent-last)."
        )

    feature_values: dict[str, float] = {
        "lat": current_lat,
        "lon": current_lon,
        "phys_u": phys_u,
        "phys_v": phys_v,
    }
    # history_window.iloc[-lag] is the row `lag` steps before "now",
    # matching features.py's df[col].shift(lag) convention exactly.
    for col in LAG_BASE_COLUMNS:
        for lag in range(1, window_size + 1):
            feature_values[f"{col}_t-{lag}"] = history_window[col].iloc[-lag]

    feature_row_df = pd.DataFrame([feature_values])
    return feature_row_df[feature_cols]


def rollout_forecast(
    models: dict,
    last_known_row: pd.Series,
    history_window: pd.DataFrame,
    future_environmental_forecast: pd.DataFrame,
    feature_cols: list[str],
    dt_seconds: float,
) -> pd.DataFrame:
    """Roll the hybrid model forward one future timestep at a time.

    future_environmental_forecast supplies the wind/current inputs for
    each future step -- in a real system this would come from a
    weather/ocean forecast API (e.g. a short-range ERA5/GFS or
    Copernicus Marine forecast product), but for this hackathon we
    ASSUME it is already provided as a DataFrame with columns
    timestamp, u_wind, v_wind, u_current, v_current (one row per future
    step, ascending order). A reasonable stand-in when no real forecast
    is wired up yet is to reuse the most recent historical values or
    hold them constant -- see this file's __main__ block for exactly
    that stand-in in action. Getting a *real* environmental forecast
    feed is a separate, later integration problem this function
    deliberately does not solve.

    A single dt_seconds is used for every step, which assumes uniform
    time spacing across future_environmental_forecast's rows (matching
    the project's fixed dt_hours convention, e.g. 6h steps); if you
    need irregular future spacing, compute per-row dt from
    future_environmental_forecast's own timestamp column instead.

    Since future area_km2 isn't part of future_environmental_forecast
    (we don't have a melt forecast), the iceberg's last known area is
    held constant throughout the rollout -- a reasonable simplification
    since melting is slow relative to typical short-term forecast
    horizons, but worth revisiting for very long horizons.

    Args:
        models: The dict from train_model.train_residual_model()
            ("u", "v", "feature_cols").
        last_known_row: A pd.Series with at least timestamp, lat, lon,
            area_km2 -- the iceberg's most recent real observation,
            i.e. the starting point of the forecast.
        history_window: The `window_size` most recent OBSERVED rows of
            u_wind/v_wind/u_current/v_current/area_km2, most-recent-
            last, where the last row corresponds to last_known_row's
            own timestep (see build_feature_row's docstring for the
            exact lag convention this assumes).
        future_environmental_forecast: DataFrame with columns
            timestamp, u_wind, v_wind, u_current, v_current, one row
            per future step to forecast, ascending by time.
        feature_cols: The feature column names/order the models
            expect.
        dt_seconds: Elapsed time between consecutive forecast steps, in
            seconds (assumed uniform -- see note above).

    Returns:
        A DataFrame with columns {timestamp, lat, lon}, one row per
        row of future_environmental_forecast, giving the forecasted
        iceberg position at each future step.

    Raises:
        ValueError: If future_environmental_forecast is missing any
            required column.
    """
    required_cols = ["timestamp", "u_wind", "v_wind", "u_current", "v_current"]
    missing_cols = [c for c in required_cols if c not in future_environmental_forecast.columns]
    if missing_cols:
        raise ValueError(
            f"rollout_forecast: future_environmental_forecast is missing required "
            f"column(s) {missing_cols}."
        )

    window = history_window[LAG_BASE_COLUMNS].reset_index(drop=True).copy()
    current_lat = last_known_row["lat"]
    current_lon = last_known_row["lon"]
    last_known_area_km2 = last_known_row["area_km2"]

    forecast_records: list[dict] = []

    # This loop is another instance of the "sequential position
    # integration" exception to vectorization: each step's forecast
    # depends on the PREVIOUS step's forecasted (not real) position and
    # updated history window, so it cannot be vectorized across steps.
    for _, future_row in future_environmental_forecast.reset_index(drop=True).iterrows():
        u_wind_f = future_row["u_wind"]
        v_wind_f = future_row["v_wind"]
        u_current_f = future_row["u_current"]
        v_current_f = future_row["v_current"]

        phys_u, phys_v = free_drift_velocity(
            u_wind=u_wind_f, v_wind=v_wind_f, u_current=u_current_f, v_current=v_current_f, lat_deg=current_lat
        )

        feature_row_df = build_feature_row(window, current_lat, current_lon, phys_u, phys_v, feature_cols)
        residual_u, residual_v = predict_residual(models, feature_row_df)

        final_u = phys_u + residual_u
        final_v = phys_v + residual_v

        new_lat, new_lon = step_position(
            lat=current_lat, lon=current_lon, u_ms=final_u, v_ms=final_v, dt_seconds=dt_seconds
        )

        forecast_records.append({"timestamp": future_row["timestamp"], "lat": new_lat, "lon": new_lon})

        # Slide the history window forward: drop the oldest row, append
        # this step's (forecast-input) environmental conditions as the
        # new most-recent row, so the NEXT iteration's lag features are
        # built correctly.
        new_hist_row = pd.DataFrame(
            [
                {
                    "u_wind": u_wind_f,
                    "v_wind": v_wind_f,
                    "u_current": u_current_f,
                    "v_current": v_current_f,
                    "area_km2": last_known_area_km2,
                }
            ]
        )
        window = pd.concat([window, new_hist_row], ignore_index=True).iloc[1:].reset_index(drop=True)

        current_lat, current_lon = new_lat, new_lon

    return pd.DataFrame.from_records(forecast_records)


def bootstrap_uncertainty_cone(
    models: dict,
    last_known_row: pd.Series,
    history_window: pd.DataFrame,
    future_environmental_forecast: pd.DataFrame,
    feature_cols: list[str],
    dt_seconds: float,
    n_samples: int = 30,
    noise_std_wind: float = 1.0,
    noise_std_current: float = 0.05,
) -> list[pd.DataFrame]:
    """Approximate a forecast uncertainty cone via bootstrapped environmental noise.

    Calls rollout_forecast() n_samples times, each time perturbing
    future_environmental_forecast's wind and current columns with
    independent Gaussian noise (noise_std_wind, noise_std_current) --
    this approximates how uncertainty in the environmental FORECAST
    itself (which is never perfect) propagates into position
    uncertainty, without requiring a full probabilistic weather/ocean
    model, which is out of scope for a hackathon.

    The caller (e.g. app.py) can compute a spatial envelope from the
    returned list to draw a cone on a map -- for example, taking the
    min/max lat and min/max lon across all samples at each future
    timestep, or computing a convex hull of all sampled positions at
    each timestep for a tighter, non-axis-aligned envelope.

    Args:
        models, last_known_row, history_window,
            future_environmental_forecast, feature_cols, dt_seconds:
            Same as rollout_forecast().
        n_samples: Number of bootstrap rollouts to run.
        noise_std_wind: Standard deviation (m/s) of the Gaussian noise
            added independently to each future u_wind/v_wind value in
            each sample.
        noise_std_current: Standard deviation (m/s) of the Gaussian
            noise added independently to each future u_current/
            v_current value in each sample.

    Returns:
        A list of n_samples DataFrames, each in the same {timestamp,
        lat, lon} format returned by rollout_forecast().
    """
    rng = np.random.default_rng()
    samples: list[pd.DataFrame] = []

    for _ in range(n_samples):
        noisy_forecast = future_environmental_forecast.copy()
        n_steps = len(noisy_forecast)
        noisy_forecast["u_wind"] = noisy_forecast["u_wind"] + rng.normal(0, noise_std_wind, n_steps)
        noisy_forecast["v_wind"] = noisy_forecast["v_wind"] + rng.normal(0, noise_std_wind, n_steps)
        noisy_forecast["u_current"] = noisy_forecast["u_current"] + rng.normal(0, noise_std_current, n_steps)
        noisy_forecast["v_current"] = noisy_forecast["v_current"] + rng.normal(0, noise_std_current, n_steps)

        sample_forecast = rollout_forecast(
            models, last_known_row, history_window, noisy_forecast, feature_cols, dt_seconds
        )
        samples.append(sample_forecast)

    return samples


def compute_cpa(iceberg_forecast_df: pd.DataFrame, vessel_lat: float, vessel_lon: float) -> dict:
    """Compute the Closest Point of Approach between a forecasted iceberg track and a fixed vessel.

    Args:
        iceberg_forecast_df: A forecast DataFrame with columns
            timestamp, lat, lon (e.g. from rollout_forecast()).
        vessel_lat: Fixed vessel/platform latitude, degrees.
        vessel_lon: Fixed vessel/platform longitude, degrees.

    Returns:
        A dict: {"cpa_distance_km": float, "cpa_timestamp": pd.Timestamp,
        "time_to_cpa_hours": float}. time_to_cpa_hours is measured
        relative to iceberg_forecast_df's FIRST timestamp (i.e. "hours
        from now").
    """
    # geodesic_distance_km() takes scalar points; each forecasted step
    # is independent of the others (no sequential dependency here), so
    # this list comprehension is a simple, readable per-row evaluation
    # rather than a sequential-integration-style loop.
    distances_km = [
        geodesic_distance_km(vessel_lat, vessel_lon, row.lat, row.lon)
        for row in iceberg_forecast_df.itertuples()
    ]

    min_idx = int(np.argmin(distances_km))
    cpa_distance_km = float(distances_km[min_idx])
    cpa_timestamp = iceberg_forecast_df["timestamp"].iloc[min_idx]
    first_timestamp = iceberg_forecast_df["timestamp"].iloc[0]
    time_to_cpa_hours = (cpa_timestamp - first_timestamp).total_seconds() / 3600.0

    return {
        "cpa_distance_km": cpa_distance_km,
        "cpa_timestamp": cpa_timestamp,
        "time_to_cpa_hours": time_to_cpa_hours,
    }


def risk_score(
    cpa_result: dict,
    uncertainty_cone: list[pd.DataFrame] | None = None,
    vessel_lat: float | None = None,
    vessel_lon: float | None = None,
    danger_threshold_km: float = 10.0,
    watch_threshold_km: float = 30.0,
    uncertainty_escalation_fraction: float = 0.3,
) -> dict:
    """Produce a tiered (red/yellow/green) risk assessment from a CPA result.

    Base tier is determined purely by cpa_distance_km against the two
    thresholds. If uncertainty_cone is also provided (along with
    vessel_lat/vessel_lon), this additionally computes CPA distance for
    every bootstrap sample and checks whether the samples disagree
    widely (std/mean of sampled CPA distances exceeding
    uncertainty_escalation_fraction) -- if so, the risk level is
    escalated by one tier (green->yellow->red, red stays red), since
    high forecast uncertainty means the "true" risk could plausibly be
    worse than the single best-estimate forecast suggests.

    This function always returns one of the three levels -- it never
    silently returns an ungraded/unlabeled result.

    Args:
        cpa_result: The dict from compute_cpa() for the best-estimate
            (non-bootstrapped) forecast.
        uncertainty_cone: Optional list of bootstrap forecast
            DataFrames (from bootstrap_uncertainty_cone()), used to
            assess forecast confidence. If provided, vessel_lat and
            vessel_lon must also be provided.
        vessel_lat: Fixed vessel latitude, degrees. Required if
            uncertainty_cone is provided.
        vessel_lon: Fixed vessel longitude, degrees. Required if
            uncertainty_cone is provided.
        danger_threshold_km: CPA distance below which risk is "red".
        watch_threshold_km: CPA distance below which (and at/above
            danger_threshold_km) risk is "yellow"; at or above this,
            risk is "green".
        uncertainty_escalation_fraction: If the bootstrap CPA distances'
            (std / mean) exceeds this fraction, escalate the risk level
            by one tier.

    Returns:
        A dict: {"level": "red"|"yellow"|"green", "cpa_distance_km":
        float, "time_to_cpa_hours": float, "confidence_note": str}.

    Raises:
        ValueError: If uncertainty_cone is provided but vessel_lat or
            vessel_lon is missing.
    """
    cpa_distance_km = cpa_result["cpa_distance_km"]
    time_to_cpa_hours = cpa_result["time_to_cpa_hours"]

    tiers = ["green", "yellow", "red"]
    if cpa_distance_km < danger_threshold_km:
        level = "red"
    elif cpa_distance_km < watch_threshold_km:
        level = "yellow"
    else:
        level = "green"

    confidence_note = "Uncertainty cone not provided; risk level reflects a single best-estimate forecast only."

    if uncertainty_cone is not None:
        if vessel_lat is None or vessel_lon is None:
            raise ValueError(
                "risk_score: uncertainty_cone was provided but vessel_lat/vessel_lon were not -- "
                "both are required to evaluate CPA against the bootstrap samples."
            )

        sample_cpa_distances_km = [
            compute_cpa(sample_df, vessel_lat, vessel_lon)["cpa_distance_km"] for sample_df in uncertainty_cone
        ]
        mean_cpa_km = float(np.mean(sample_cpa_distances_km))
        std_cpa_km = float(np.std(sample_cpa_distances_km))
        relative_spread = std_cpa_km / mean_cpa_km if mean_cpa_km > 0 else float("inf")

        if relative_spread > uncertainty_escalation_fraction:
            escalated_idx = min(tiers.index(level) + 1, len(tiers) - 1)
            level = tiers[escalated_idx]
            confidence_note = (
                f"Elevated due to high forecast uncertainty: bootstrap CPA distances vary by "
                f"{relative_spread:.0%} (std={std_cpa_km:.1f} km) around a mean of {mean_cpa_km:.1f} km."
            )
        else:
            confidence_note = (
                f"Forecast uncertainty is low for this horizon: bootstrap CPA distances vary by only "
                f"{relative_spread:.0%} (std={std_cpa_km:.1f} km) around a mean of {mean_cpa_km:.1f} km."
            )

    return {
        "level": level,
        "cpa_distance_km": cpa_distance_km,
        "time_to_cpa_hours": time_to_cpa_hours,
        "confidence_note": confidence_note,
    }

if __name__ == "__main__":
    from data_ingest import generate_synthetic_track
    from features import build_sliding_window_features
    from train_model import train_residual_model, train_test_split_by_time

    WINDOW_SIZE = 5
    HORIZON_STEPS = 10
    DT_HOURS = 6

    track = generate_synthetic_track(n_steps=150, dt_hours=DT_HOURS, seed=7)
    feature_df, feature_cols, target_cols = build_sliding_window_features(track, window_size=WINDOW_SIZE)
    train_df, _test_df = train_test_split_by_time(feature_df, test_fraction=0.2)
    models = train_residual_model(train_df, feature_cols, target_cols)

    # Build a forecast setup by carving the LAST 10 rows of the real
    # synthetic track's environmental data off as a stand-in "future
    # forecast" -- NOT a real weather forecast, just a convenient
    # historical block to demonstrate rollout_forecast() end to end
    # without needing a live forecast API wired up yet.
    n = len(track)
    last_known_idx = n - HORIZON_STEPS - 1
    last_known_row = track.iloc[last_known_idx]
    history_window = track.iloc[last_known_idx - WINDOW_SIZE : last_known_idx].reset_index(drop=True)
    future_environmental_forecast = track.iloc[last_known_idx + 1 : last_known_idx + 1 + HORIZON_STEPS][
        ["timestamp", "u_wind", "v_wind", "u_current", "v_current"]
    ].reset_index(drop=True)
    dt_seconds = DT_HOURS * 3600.0

    forecast_df = rollout_forecast(
        models, last_known_row, history_window, future_environmental_forecast, feature_cols, dt_seconds
    )
    print("Forecasted track:")
    print(forecast_df.to_string(index=False))

    uncertainty_cone = bootstrap_uncertainty_cone(
        models, last_known_row, history_window, future_environmental_forecast, feature_cols, dt_seconds,
        n_samples=30,
    )

    # Pick a vessel position a plausible distance from the forecasted
    # track's end point (roughly tens of km away at these latitudes) so
    # compute_cpa/risk_score have something meaningful to evaluate.
    end_lat, end_lon = forecast_df["lat"].iloc[-1], forecast_df["lon"].iloc[-1]
    vessel_lat, vessel_lon = end_lat + 0.15, end_lon + 0.25
    vessel_distance_km = geodesic_distance_km(vessel_lat, vessel_lon, end_lat, end_lon)
    print(f"\nVessel position: ({vessel_lat:.4f}, {vessel_lon:.4f}), "
          f"~{vessel_distance_km:.1f} km from the forecast's final point.")

    cpa_result = compute_cpa(forecast_df, vessel_lat, vessel_lon)
    risk_result = risk_score(cpa_result, uncertainty_cone=uncertainty_cone, vessel_lat=vessel_lat, vessel_lon=vessel_lon)

    print("\nCPA result:")
    print(cpa_result)
    print("\nRisk assessment:")
    print(risk_result)

    final_lats = [sample_df["lat"].iloc[-1] for sample_df in uncertainty_cone]
    final_lons = [sample_df["lon"].iloc[-1] for sample_df in uncertainty_cone]
    print(f"\nUncertainty cone spread at final forecasted timestep:")
    print(f"  lat: {min(final_lats):.5f} to {max(final_lats):.5f}")
    print(f"  lon: {min(final_lons):.5f} to {max(final_lons):.5f}")
    assert max(final_lats) > min(final_lats), "uncertainty cone collapsed to a single point (lat)"
    assert max(final_lons) > min(final_lons), "uncertainty cone collapsed to a single point (lon)"

    print("\nAll decision_support.py sanity checks passed.")
