"""
decision_support.py

Turns a drift model into an operational answer: where will this iceberg
be, how sure are we, and does it threaten a given vessel or platform.

This is the half of the project that is NOT trajectory prediction. A
forecast track is not a decision; a closest-point-of-approach distance
with an uncertainty envelope and a graded risk level is.

THE HONEST UNCERTAINTY STORY
============================
Two things separate a real forecast from the hindcast evaluation in
train_model.py:

1. We do not know the future wind and current. The hindcast is allowed
   to read them from the record; a real forecast must be handed an
   environmental forecast, and that forecast is itself wrong.

2. Consequently the single most useful output is not the track, it is
   the SPREAD. bootstrap_uncertainty_cone() propagates plausible
   environmental error into position error by re-running the rollout
   many times with perturbed forcing. The perturbation scale is a
   parameter with a real meaning -- roughly the RMS error of the
   environmental forecast at the relevant lead time -- not a decoration.

The cone is a lower bound on true uncertainty: it captures forcing
error, but not error in the drift model itself. On the real NIC record
the calibrated physics has a ~53 km 3-week ADE, so risk_score() adds a
lead-time-scaled model-error floor (see model_error_km) and the cone is
never allowed to claim more confidence than the model has earned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from features import LAG_BASE_COLUMNS, build_single_feature_row
from physics import (
    DriftParams,
    free_drift_velocity,
    geodesic_distance_km,
    get_default_drift_params,
    step_position,
)
from train_model import predict_residual

# Displacement error of the calibrated physics baseline as a function of
# forecast lead time, fitted to the leave-one-iceberg-out rollout on the
# real NIC record (34.7 km at 7 days, 53.9 at 14, 69.3 at 21):
#
#     error_km ~= MODEL_ERROR_COEFF_KM * days ** MODEL_ERROR_EXPONENT
#
# The sub-linear exponent is real and worth knowing: error does NOT grow
# proportionally with lead time, because drift direction partly
# decorrelates and excursions cancel rather than accumulate.
#
# Expressing this as a function of DAYS rather than of forecast STEPS
# matters for the dashboard, which steps at 6 hours rather than the
# ~weekly cadence of the training record -- a per-step constant would
# overstate the error by a factor of ~28 there and paint every forecast
# red.
MODEL_ERROR_COEFF_KM: float = 10.4
MODEL_ERROR_EXPONENT: float = 0.63


def model_error_km(lead_time_hours: float) -> float:
    """Estimate the drift model's own displacement error at a given lead time.

    Args:
        lead_time_hours: Forecast lead time, hours.

    Returns:
        Expected displacement error in km, from the power law fitted to
        the leave-one-iceberg-out rollout on the real record.
    """
    days = max(float(lead_time_hours), 0.0) / 24.0
    return MODEL_ERROR_COEFF_KM * (days ** MODEL_ERROR_EXPONENT)


def _normalise_history(
    history_window: pd.DataFrame | list[dict[str, float]] | None,
    last_known: pd.Series,
    n_lags: int,
) -> list[dict[str, float]]:
    """Coerce the several shapes of lag history callers pass into one form.

    The frontend hands over a DataFrame slice whose LAST row is the most
    recent observation; internal callers use a list ordered most recent
    FIRST. Both are accepted, and a missing history falls back to
    repeating the last known row, so a short track still forecasts
    instead of raising.

    Args:
        history_window: A DataFrame (oldest first), a list of dicts
            (newest first), or None.
        last_known: The most recent row, used as the fallback.
        n_lags: Number of lag entries required.

    Returns:
        A list of exactly n_lags dicts, most recent first, each holding
        obs_u/obs_v/residual_u/residual_v.
    """
    keys = ("obs_u", "obs_v", "residual_u", "residual_v")

    if isinstance(history_window, pd.DataFrame) and not history_window.empty:
        # Reverse: the frontend's slice runs oldest -> newest. Rows whose
        # lag values are not finite are skipped rather than coerced to
        # zero: when several icebergs are displayed at once their tracks
        # are concatenated with NaN separator rows to break the map
        # polylines, and a separator landing inside the history window
        # would otherwise be read as "the iceberg was stationary", which
        # is a silently wrong forecast rather than a visible failure.
        entries = []
        for _, row in history_window.iloc[::-1].iterrows():
            values = {k: float(row.get(k, np.nan)) for k in keys}
            if all(np.isfinite(v) for v in values.values()):
                entries.append(values)
    elif isinstance(history_window, list) and history_window:
        entries = [{k: float(entry.get(k, 0.0) or 0.0) for k in keys} for entry in history_window]
    else:
        entries = []

    if not entries:
        fallback = {k: float(last_known.get(k, 0.0) or 0.0) for k in keys}
        entries = [dict(fallback)]

    # Pad by repeating the oldest available entry so a history shorter
    # than n_lags degrades gracefully rather than raising.
    while len(entries) < n_lags:
        entries.append(dict(entries[-1]))
    return entries[:n_lags]


def rollout_forecast(
    models: dict | None,
    last_known: pd.Series,
    history_window: pd.DataFrame | list[dict[str, float]] | None = None,
    future_environment: pd.DataFrame | None = None,
    feature_cols: list[str] | None = None,
    dt_seconds: float | None = None,
    drift_params: DriftParams | None = None,
    n_lags: int = config.DEFAULT_N_LAGS,
    mode: str | None = None,
) -> pd.DataFrame:
    """Forecast an iceberg's position forward over a sequence of future segments.

    Steps through `future_environment` one row at a time: compute the
    free-drift velocity from that row's forcing, optionally add the
    learned residual, advance the position geodesically, and feed the
    step's own output back in as the next step's lag features.

    The environmental forecast is supplied by the caller. In deployment
    it would come from a numerical weather/ocean forecast; for a demo,
    holding recent conditions constant or replaying the last few observed
    segments is a reasonable stand-in, and the demo block below does the
    latter. Either way, its error is what
    bootstrap_uncertainty_cone() propagates.

    ARGUMENT ORDER: the first six parameters are positional-compatible
    with the call the Dash frontend makes,

        rollout_forecast(model, last_known_row, history_window,
                         future_env, feature_cols, dt_seconds)

    so app/callbacks.py works against this module unmodified.

    Args:
        models: A trained bundle, required when mode resolves to
            "hybrid"; None forecasts with physics alone.
        last_known: The iceberg's most recent enriched row, with
            timestamp, lat, lon, area_km2 and (for the lag features)
            obs_u/obs_v/residual_u/residual_v.
        history_window: Lag history. Either a DataFrame whose LAST row is
            the most recent observation (the shape the frontend passes,
            from `track.iloc[-window:]`) or a list of dicts ordered most
            recent FIRST. Defaults to repeating `last_known`'s values.
        future_environment: One row per future segment, with u_wind,
            v_wind, u_current, v_current and either a `timestamp` column
            or a `dt_hours` column giving each segment's length.
        feature_cols: Accepted for call compatibility with the frontend
            and ignored -- the bundle already carries the authoritative
            column order, and honouring a second, possibly stale list
            would be a way to silently mispredict.
        dt_seconds: A single segment length applied to every step. When
            given it overrides whatever the timestamps imply, which is
            what the frontend relies on for its fixed 6-hour cadence.
        drift_params: Calibrated drift coefficients. Defaults to the
            bundle's, else the module defaults.
        n_lags: Number of lags the model expects.
        mode: "physics" or "hybrid". Defaults to "hybrid" when a bundle
            is supplied (the frontend routes here only for hybrid; its
            physics path is computed inline) and "physics" otherwise.

    Returns:
        A DataFrame with timestamp, lat, lon, u_forecast, v_forecast for
        each future segment.

    Raises:
        ValueError: If mode resolves to "hybrid" without models, or if
            future_environment provides neither timestamps nor dt_hours.
    """
    if mode is None:
        mode = "hybrid" if models is not None else "physics"
    if mode == "hybrid" and models is None:
        raise ValueError("rollout_forecast: mode='hybrid' requires a trained `models` bundle.")
    if future_environment is None:
        raise ValueError("rollout_forecast: future_environment is required.")
    if drift_params is None:
        drift_params = (
            models["drift_params"] if models is not None else get_default_drift_params()
        )

    env = future_environment.reset_index(drop=True)
    if "dt_hours" in env.columns:
        dt_hours = env["dt_hours"].to_numpy(dtype=float)
        timestamps = (
            env["timestamp"].to_list()
            if "timestamp" in env.columns
            else [
                pd.Timestamp(last_known["timestamp"]) + pd.Timedelta(hours=float(h))
                for h in np.cumsum(dt_hours)
            ]
        )
    elif "timestamp" in env.columns:
        stamps = pd.to_datetime(env["timestamp"])
        edges = pd.concat([pd.Series([pd.Timestamp(last_known["timestamp"])]), stamps])
        dt_hours = edges.diff().dt.total_seconds().to_numpy()[1:] / 3600.0
        timestamps = stamps.to_list()
    else:
        raise ValueError(
            "rollout_forecast: future_environment must have a 'timestamp' or 'dt_hours' "
            f"column to know how long each segment is; it has {list(env.columns)}."
        )

    if dt_seconds is not None:
        dt_hours = np.full(len(env), float(dt_seconds) / 3600.0)

    history = _normalise_history(history_window, last_known, n_lags)

    lat = float(last_known["lat"])
    lon = float(last_known["lon"])
    area = float(last_known["area_km2"])

    records: list[dict[str, object]] = []
    for step, row in env.iterrows():
        phys_u, phys_v = free_drift_velocity(
            u_wind=float(row["u_wind"]),
            v_wind=float(row["v_wind"]),
            u_current=float(row["u_current"]),
            v_current=float(row["v_current"]),
            lat_deg=lat,
            **drift_params.as_kwargs(),
        )

        residual_u = residual_v = 0.0
        if mode == "hybrid":
            state: dict[str, float] = {
                "lat": lat,
                "lon": lon,
                "area_km2": area,
                "u_wind": float(row["u_wind"]),
                "v_wind": float(row["v_wind"]),
                "u_current": float(row["u_current"]),
                "v_current": float(row["v_current"]),
                "phys_u": phys_u,
                "phys_v": phys_v,
                "dt_hours": float(dt_hours[step]),
            }
            for lag, past in enumerate(history, start=1):
                for col in ("obs_u", "obs_v", "residual_u", "residual_v"):
                    state[f"{col}_t-{lag}"] = past[col]
            residual_u, residual_v = predict_residual(
                models, build_single_feature_row(state, n_lags=n_lags)
            )

        total_u, total_v = phys_u + residual_u, phys_v + residual_v
        lat, lon = step_position(lat, lon, total_u, total_v, float(dt_hours[step]) * 3600.0)

        records.append(
            {
                "timestamp": timestamps[step],
                "lat": lat,
                "lon": lon,
                "u_forecast": total_u,
                "v_forecast": total_v,
            }
        )

        # The next step's lag features are this step's own output.
        history.insert(
            0,
            {"obs_u": total_u, "obs_v": total_v, "residual_u": residual_u, "residual_v": residual_v},
        )
        history = history[:n_lags]

    return pd.DataFrame.from_records(records)


def bootstrap_uncertainty_cone(
    models: dict | None,
    last_known: pd.Series,
    history_window: pd.DataFrame | list[dict[str, float]] | None = None,
    future_environment: pd.DataFrame | None = None,
    feature_cols: list[str] | None = None,
    dt_seconds: float | None = None,
    drift_params: DriftParams | None = None,
    n_samples: int = 40,
    noise_std_wind: float = 2.5,
    noise_std_current: float = 0.04,
    n_lags: int = config.DEFAULT_N_LAGS,
    mode: str | None = None,
    seed: int = 42,
) -> list[pd.DataFrame]:
    """Propagate environmental forecast error into position uncertainty.

    Re-runs rollout_forecast() `n_samples` times, each with independent
    Gaussian noise added to the future wind and current, producing a
    family of plausible tracks whose spread at each future timestep is
    the uncertainty cone.

    The default noise scales are not arbitrary: 2.5 m/s is a
    representative RMS error for a multi-day 10 m wind forecast, and
    0.04 m/s is a similar figure for surface currents -- comparable to
    the 0.05-0.10 m/s the icebergs in this record actually drift at,
    which is precisely why the cone is wide.

    The noise is applied INDEPENDENTLY per future step, which treats
    successive forecast errors as uncorrelated. Real forecast errors are
    strongly correlated in time (a forecast that is too westerly today
    is likely too westerly tomorrow), so independent noise partially
    cancels over a rollout and this cone is, if anything, narrower than
    the truth. risk_score()'s model-error floor is what keeps that from
    being read as confidence.

    ARGUMENT ORDER matches rollout_forecast(), so the dashboard can call
    both the same way:

        bootstrap_uncertainty_cone(model, last_known_row, history_window,
                                   future_env, feature_cols, dt_seconds, ...)

    Args:
        models: A trained bundle, or None to perturb the physics baseline
            alone.
        last_known: The iceberg's most recent enriched row.
        history_window: Lag history; see rollout_forecast().
        future_environment: Forecast forcing per future segment.
        feature_cols: Accepted for call compatibility and ignored -- the
            bundle carries the authoritative column order.
        dt_seconds: A single segment length applied to every step.
        drift_params: Calibrated drift coefficients.
        n_samples: Number of perturbed rollouts.
        noise_std_wind: Std dev of the wind perturbation, m/s.
        noise_std_current: Std dev of the current perturbation, m/s.
        n_lags: Number of lags the model expects.
        mode: "physics" or "hybrid"; defaults to the bundle's validated
            mode.
        seed: Seed for the perturbation generator, for reproducibility.

    Returns:
        A list of n_samples forecast DataFrames, each shaped like
        rollout_forecast()'s output. Call cone_envelope() to reduce them
        to a per-timestep spatial envelope for plotting.
    """
    if future_environment is None:
        raise ValueError("bootstrap_uncertainty_cone: future_environment is required.")

    rng = np.random.default_rng(seed)
    shape = (len(future_environment),)

    samples: list[pd.DataFrame] = []
    for _ in range(n_samples):
        perturbed = future_environment.reset_index(drop=True).copy()
        perturbed["u_wind"] = perturbed["u_wind"] + rng.normal(0.0, noise_std_wind, shape)
        perturbed["v_wind"] = perturbed["v_wind"] + rng.normal(0.0, noise_std_wind, shape)
        perturbed["u_current"] = perturbed["u_current"] + rng.normal(0.0, noise_std_current, shape)
        perturbed["v_current"] = perturbed["v_current"] + rng.normal(0.0, noise_std_current, shape)
        samples.append(
            rollout_forecast(
                models,
                last_known,
                history_window,
                perturbed,
                feature_cols,
                dt_seconds,
                drift_params=drift_params,
                n_lags=n_lags,
                mode=mode,
            )
        )
    return samples


def cone_envelope(cone: list[pd.DataFrame]) -> pd.DataFrame:
    """Reduce a bootstrap cone to a per-timestep envelope for plotting.

    Args:
        cone: The list of forecast DataFrames from
            bootstrap_uncertainty_cone().

    Returns:
        A DataFrame with one row per future timestep: timestamp, the
        mean lat/lon, the min/max of each, and spread_km -- the mean
        geodesic distance of the samples from their own centroid, which
        is the number to quote as "the forecast is good to +/- X km".
    """
    stacked = pd.concat(cone, ignore_index=True)
    rows: list[dict[str, object]] = []
    for timestamp, group in stacked.groupby("timestamp", sort=True):
        mean_lat = float(group["lat"].mean())
        mean_lon = float(group["lon"].mean())
        spread = float(
            np.mean(
                [
                    geodesic_distance_km(mean_lat, mean_lon, lat, lon)
                    for lat, lon in zip(group["lat"], group["lon"])
                ]
            )
        )
        rows.append(
            {
                "timestamp": timestamp,
                "lat": mean_lat,
                "lon": mean_lon,
                "lat_min": float(group["lat"].min()),
                "lat_max": float(group["lat"].max()),
                "lon_min": float(group["lon"].min()),
                "lon_max": float(group["lon"].max()),
                "spread_km": spread,
            }
        )
    return pd.DataFrame(rows)


def compute_cpa(
    forecast: pd.DataFrame, vessel_lat: float, vessel_lon: float
) -> dict:
    """Find the closest point of approach between a forecast track and a fixed asset.

    Args:
        forecast: A forecast track with timestamp, lat, lon.
        vessel_lat: Vessel or platform latitude, degrees.
        vessel_lon: Vessel or platform longitude, degrees.

    Returns:
        A dict with cpa_distance_km, cpa_timestamp, time_to_cpa_hours
        (measured from the first forecast timestamp, i.e. "hours from
        now") and cpa_step_index.

    Raises:
        ValueError: If the forecast is empty.
    """
    if forecast.empty:
        raise ValueError("compute_cpa: the forecast track is empty.")

    # A multi-iceberg forecast is several tracks concatenated with NaN
    # separator rows (they make the map draw one polyline per iceberg).
    # Drop those before measuring distance, and the CPA then correctly
    # becomes the closest approach of ANY forecast iceberg -- which is
    # the question a bridge officer is actually asking.
    points = forecast.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    if points.empty:
        raise ValueError("compute_cpa: the forecast track has no valid positions.")

    distances = np.array(
        [
            geodesic_distance_km(lat, lon, vessel_lat, vessel_lon)
            for lat, lon in zip(points["lat"], points["lon"])
        ]
    )
    idx = int(np.argmin(distances))
    cpa_time = pd.Timestamp(points["timestamp"].iloc[idx])
    origin = pd.Timestamp(points["timestamp"].iloc[0])

    return {
        "cpa_distance_km": float(distances[idx]),
        "cpa_timestamp": cpa_time,
        "time_to_cpa_hours": float((cpa_time - origin).total_seconds() / 3600.0),
        "cpa_step_index": idx,
        # Which iceberg makes the closest approach, when several are
        # forecast together.
        "cpa_iceberg_id": (
            str(points["iceberg_id"].iloc[idx])
            if "iceberg_id" in points.columns else None
        ),
    }


def risk_score(
    cpa_result: dict,
    uncertainty_cone: list[pd.DataFrame] | None = None,
    vessel_lat: float | None = None,
    vessel_lon: float | None = None,
    danger_threshold_km: float = 10.0,
    watch_threshold_km: float = 30.0,
) -> dict:
    """Grade the threat a forecast iceberg poses to a fixed asset.

    Returns one of three levels, always -- an ungraded result is not a
    useful output for a bridge officer.

      red    -- forecast CPA inside danger_threshold_km
      amber  -- inside watch_threshold_km
      green  -- outside both

    The middle tier is named "amber" rather than "yellow" because the
    dashboard renders the level directly as a CSS class (`iw-pill
    {level}`) and the stylesheet defines green/amber/red/grey. Returning
    "yellow" would silently render an unstyled pill.

    Two things escalate the level beyond what the nominal CPA implies:

    1. Cone disagreement. If the bootstrap samples disagree widely about
       the CPA, the nominal number is not trustworthy and the level is
       raised one tier.

    2. The model's own error. The calibrated drift model has a known
       per-step displacement error on real icebergs, independent of any
       forcing uncertainty. A forecast CPA of 35 km three steps out, with
       a ~18 km/step model error, does NOT mean the asset is clear. The
       level is raised whenever the CPA lies inside the model's own error
       budget at that lead time.

    This deliberately errs toward escalation: a false yellow costs a
    course check, a false green costs a hull.

    Args:
        cpa_result: The dict from compute_cpa().
        uncertainty_cone: Optional bootstrap samples for the spread check.
        vessel_lat: Vessel latitude; required to use the cone.
        vessel_lon: Vessel longitude; required to use the cone.
        danger_threshold_km: Red threshold.
        watch_threshold_km: Amber threshold.

    Returns:
        A dict with level ("green"/"amber"/"red"), cpa_distance_km,
        time_to_cpa_hours,
        effective_threat_km (CPA minus the total error budget),
        cpa_spread_km and confidence_note.
    """
    order = ["green", "amber", "red"]
    cpa_km = float(cpa_result["cpa_distance_km"])

    level = "red" if cpa_km < danger_threshold_km else (
        "amber" if cpa_km < watch_threshold_km else "green"
    )
    notes: list[str] = []

    # --- Model error budget at this lead time ------------------------
    lead_hours = float(cpa_result.get("time_to_cpa_hours", 0.0))
    model_error = model_error_km(lead_hours)

    # --- Cone disagreement -------------------------------------------
    cpa_spread_km = float("nan")
    if uncertainty_cone and vessel_lat is not None and vessel_lon is not None:
        sample_cpas = np.array(
            [compute_cpa(sample, vessel_lat, vessel_lon)["cpa_distance_km"]
             for sample in uncertainty_cone]
        )
        cpa_spread_km = float(sample_cpas.std())
        if cpa_spread_km > 0.5 * max(sample_cpas.mean(), 1e-6):
            level = order[min(order.index(level) + 1, 2)]
            notes.append(
                f"forecast spread is large relative to the approach distance "
                f"(CPA std {cpa_spread_km:.1f} km across {len(sample_cpas)} perturbed runs)"
            )
    total_error_km = model_error + (0.0 if np.isnan(cpa_spread_km) else cpa_spread_km)

    if cpa_km < watch_threshold_km + total_error_km and level != "red":
        level = order[min(order.index(level) + 1, 2)]
        notes.append(
            f"the {cpa_km:.0f} km closest approach is inside the model's own "
            f"+/-{total_error_km:.0f} km error budget at {lead_hours:.0f} h lead"
        )

    if notes:
        confidence_note = "Escalated: " + "; ".join(notes) + "."
    else:
        confidence_note = (
            f"Closest approach {cpa_km:.0f} km clears the {watch_threshold_km:.0f} km watch "
            f"threshold by more than the +/-{total_error_km:.0f} km forecast error budget."
        )

    return {
        "level": level,
        "cpa_distance_km": cpa_km,
        "time_to_cpa_hours": float(cpa_result["time_to_cpa_hours"]),
        "effective_threat_km": cpa_km - total_error_km,
        "cpa_spread_km": cpa_spread_km,
        "model_error_km": model_error,
        "confidence_note": confidence_note,
    }


if __name__ == "__main__":
    from data_ingest import build_real_dataset
    from features import calibrate_drift_params, compute_physics_residual, compute_observed_velocity

    pooled, _motion = build_real_dataset(verbose=False)
    params = calibrate_drift_params(compute_observed_velocity(pooled))
    enriched = compute_physics_residual(pooled, params=params)

    # Take the fastest-drifting berg in the record as the demo subject.
    speeds = enriched.groupby("iceberg_id").apply(
        lambda g: float(np.hypot(g["obs_u"], g["obs_v"]).mean()), include_groups=False
    )
    berg_id = str(speeds.idxmax())
    berg = enriched[enriched["iceberg_id"] == berg_id].sort_values("timestamp").reset_index(drop=True)
    last_known = berg.iloc[-1]

    print(f"Iceberg {berg_id}: {len(berg)} forced segments, last fix "
          f"{last_known['timestamp']:%Y-%m-%d} at ({last_known['lat']:.2f}, {last_known['lon']:.2f}), "
          f"mean speed {speeds.max():.3f} m/s")

    # STAND-IN FOR A REAL FORECAST: replay this berg's own last four
    # observed segments as if they were the coming four weeks' forecast.
    # A deployed system would call a weather/ocean forecast API here;
    # the point of the cone below is precisely that this is uncertain.
    horizon = 4
    future_env = berg.tail(horizon)[
        ["u_wind", "v_wind", "u_current", "v_current", "dt_hours"]
    ].reset_index(drop=True)

    forecast = rollout_forecast(None, last_known, future_environment=future_env,
                                drift_params=params, mode="physics")
    print(f"\n{horizon}-segment forecast from {last_known['timestamp']:%Y-%m-%d}:")
    print(forecast.to_string(index=False))

    cone = bootstrap_uncertainty_cone(
        None, last_known, future_environment=future_env,
        drift_params=params, mode="physics", n_samples=40
    )
    envelope = cone_envelope(cone)
    print(f"\nUncertainty cone ({len(cone)} perturbed rollouts):")
    print(envelope[["timestamp", "lat", "lon", "lat_min", "lat_max", "spread_km"]].to_string(index=False))
    assert envelope["spread_km"].iloc[-1] > 0, "the cone collapsed to a point"
    assert envelope["spread_km"].is_monotonic_increasing, "uncertainty should grow with lead time"

    # Place a vessel a plausible distance off the forecast end point.
    end = forecast.iloc[-1]
    vessel_lat = float(end["lat"]) + 0.25
    vessel_lon = float(end["lon"]) + 0.25

    cpa = compute_cpa(forecast, vessel_lat, vessel_lon)
    risk = risk_score(cpa, uncertainty_cone=cone, vessel_lat=vessel_lat, vessel_lon=vessel_lon)

    print(f"\nVessel at ({vessel_lat:.2f}, {vessel_lon:.2f})")
    print(f"CPA: {cpa['cpa_distance_km']:.1f} km at {cpa['cpa_timestamp']:%Y-%m-%d} "
          f"({cpa['time_to_cpa_hours']:.0f} h out)")
    print(f"\nRISK: {risk['level'].upper()}")
    for key, value in risk.items():
        if key != "level":
            print(f"  {key}: {value}")
    assert risk["level"] in {"red", "amber", "green"}, "risk must always be graded"
    print("\ndecision_support.py checks passed.")
