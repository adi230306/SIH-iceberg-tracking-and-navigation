"""
features.py

Turns a pooled multi-iceberg track table into the ML feature table for
the hybrid physics + gradient-boosting residual model.

The idea is unchanged from the synthetic prototype: for each observed
segment we compute what free-drift physics SAYS the mean velocity should
have been, compare it to what the two position fixes say it ACTUALLY
was, and train the model on the difference. What changes for real NIC
data is the bookkeeping around it:

  * Everything is grouped by `iceberg_id`. A velocity, a lag, or a
    residual must never be differenced across an iceberg boundary, and
    the pooled table interleaves fifteen separate tracks.

  * The physics coefficients are FITTED, not assumed. calibrate_drift_
    params() refits the wind factor, deflection angle and current factor
    on the training rows only. This moves error out of the ML residual
    and into the physical term, which generalises to unseen icebergs in
    a way a tree ensemble trained on fifteen bergs does not.

  * Longitude enters as sin/cos, not as a raw number. Two of the fifteen
    icebergs (B22A, B22F) straddle the antimeridian, where raw longitude
    jumps from +179 to -179 and would put physically adjacent fixes in
    completely different regions of every tree.

  * The most informative feature is the PREVIOUS segment's residual. Its
    physical meaning is precisely the thing free drift cannot see: this
    particular berg's draft, keel shape and sea-ice contact, which
    persist from week to week. Using it is legitimate -- it is computed
    from past position fixes only -- but it means a multi-step forecast
    must feed its own predicted residual back in for steps beyond the
    first, which is exactly what the rollout in train_model.py does.

TRAIN/SERVE SKEW: the batch builder and the single-row builder used
during forecast rollout both go through _derive_current_features(), so
there is exactly one definition of every derived column. Adding a
feature in one place cannot silently diverge from the other.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pyproj import Geod

import config
from physics import (
    DriftParams,
    calibrate_free_drift_params,
    free_drift_velocity_array,
    geodesic_distance_km,
    step_position,
)

_GEOD = Geod(ellps="WGS84")

# Features describing the segment currently being predicted.
CURRENT_FEATURE_COLUMNS: list[str] = [
    "phys_u",
    "phys_v",
    "u_wind",
    "v_wind",
    "u_current",
    "v_current",
    "wind_speed",
    "current_speed",
    "lat",
    "lon_sin",
    "lon_cos",
    "log_area",
    "dt_hours",
]

# Columns lagged into the feature row from previous segments of the SAME
# iceberg. obs_* is the previous observed velocity (drift is strongly
# autocorrelated week to week); residual_* is the previous physics error,
# the per-iceberg signature the model is really keyed on.
LAG_BASE_COLUMNS: list[str] = ["obs_u", "obs_v", "residual_u", "residual_v"]

TARGET_COLUMNS: list[str] = ["residual_u", "residual_v"]

# Carried through the feature table for grouping, plotting and
# leakage-safe splitting -- never fed to the model.
METADATA_COLUMNS: list[str] = ["iceberg_id", "timestamp", "lat", "lon", "obs_u", "obs_v"]


def compute_observed_velocity(
    track_df: pd.DataFrame, group_col: str | None = "iceberg_id"
) -> pd.DataFrame:
    """Compute the observed mean velocity over each inter-fix segment.

    Uses the WGS84 geodesic inverse to get the true distance and forward
    azimuth between consecutive fixes, then decomposes the azimuth into
    eastward/northward components (obs_u = speed*sin(az), obs_v =
    speed*cos(az)) -- the exact inverse of physics.step_position()'s
    bearing convention. Naive lat/lon differencing would be badly wrong
    at 65 deg S and catastrophically wrong for the two icebergs that
    cross the antimeridian.

    Elapsed time comes from the timestamps themselves, never from an
    assumed fixed step: NIC fixes are 6 to 15 days apart.

    The velocity for the segment (t[k-1], t[k]] is written to row k, so
    it pairs with the segment-mean forcing that data_ingest wrote to that
    same row. The first row of each iceberg has no preceding fix and is
    dropped -- so every row returned has a valid obs_u/obs_v, and no
    downstream consumer has to think about NaNs.

    When the table carries the `segment_hours` column written by
    data_ingest.sample_environment_along_segments(), this checks it
    against the interval actually recomputed here and drops any row where
    they disagree. That catches the one silent corruption this pipeline
    is prone to: if a middle row was filtered out after its forcing was
    computed, the surviving row's forcing would describe a different
    interval than its velocity, and every residual for it would be
    quietly wrong.

    Args:
        track_df: A pooled (or single) track DataFrame.
        group_col: Column identifying each iceberg, or None if the table
            holds exactly one track.

    Returns:
        A copy of track_df with obs_u and obs_v (m/s) added and the first
        row of each iceberg removed.

    Raises:
        ValueError: If any two consecutive timestamps within an iceberg
            are equal or out of order, making velocity undefined.
    """
    sort_cols = ([group_col] if group_col else []) + ["timestamp"]
    df = track_df.sort_values(sort_cols).reset_index(drop=True).copy()

    if group_col is None:
        df = df.assign(_group=0)
        group_key = "_group"
    else:
        group_key = group_col

    lats = df["lat"].to_numpy()
    lons = df["lon"].to_numpy()

    # One vectorized geodesic inverse over every consecutive row pair;
    # pairs that straddle an iceberg boundary are masked out below.
    fwd_az, _back_az, distance_m = _GEOD.inv(lons[:-1], lats[:-1], lons[1:], lats[1:])
    dt_seconds = df["timestamp"].diff().dt.total_seconds().to_numpy()[1:]

    same_iceberg = (df[group_key].to_numpy()[1:] == df[group_key].to_numpy()[:-1])

    bad = same_iceberg & (dt_seconds <= 0)
    if bad.any():
        rows = (np.where(bad)[0] + 1).tolist()
        raise ValueError(
            f"compute_observed_velocity: non-positive elapsed time before row(s) {rows}. "
            f"Timestamps must be strictly increasing within each iceberg -- check for "
            f"duplicate snapshot dates."
        )

    # np.where guards the division so masked (cross-iceberg) pairs never
    # divide by a negative or zero dt.
    safe_dt = np.where(same_iceberg, dt_seconds, np.nan)
    speed_ms = distance_m / safe_dt
    azimuth_rad = np.radians(fwd_az)

    df["obs_u"] = np.concatenate(([np.nan], speed_ms * np.sin(azimuth_rad)))
    df["obs_v"] = np.concatenate(([np.nan], speed_ms * np.cos(azimuth_rad)))
    df["dt_hours"] = np.concatenate(([np.nan], safe_dt / 3600.0))

    df = df.dropna(subset=["obs_u", "obs_v"]).reset_index(drop=True)

    # Reject segments implying a physically impossible speed. These are
    # bad position fixes, not fast icebergs, and a single one of them
    # dominates a least-squares calibration: the raw BYU daily record has
    # a median speed of 0.037 m/s but an RMS of 0.28 m/s entirely because
    # of this tail.
    speed = np.hypot(df["obs_u"], df["obs_v"])
    df = df.loc[speed < config.MAX_PLAUSIBLE_SPEED_MS].reset_index(drop=True)

    # Verify the forcing on each row really covers the interval its
    # velocity was measured over (see the docstring).
    if "segment_hours" in df.columns:
        mismatch = (df["segment_hours"] - df["dt_hours"]).abs() > 1.0
        if mismatch.any():
            df = df.loc[~mismatch].reset_index(drop=True)

    if group_col is None:
        df = df.drop(columns=["_group"])
    return df


def calibrate_drift_params(
    track_df: pd.DataFrame, deflection_grid_deg: np.ndarray | None = None
) -> DriftParams:
    """Fit the free-drift coefficients on the rows of a track table.

    Thin wrapper over physics.calibrate_free_drift_params() that pulls
    the columns out of a DataFrame. Call it on the TRAINING rows only --
    fitting on the full table before splitting would leak information
    about the test period into the physics baseline, which is the subtler
    cousin of the random-shuffle-split mistake.

    Args:
        track_df: A table with u_wind/v_wind/u_current/v_current/lat and
            obs_u/obs_v (i.e. after compute_observed_velocity()).
        deflection_grid_deg: Candidate deflection angles; see
            physics.calibrate_free_drift_params().

    Returns:
        The fitted DriftParams.

    Raises:
        KeyError: If obs_u/obs_v are absent -- run
            compute_observed_velocity() first.
    """
    missing = [c for c in ("obs_u", "obs_v") if c not in track_df.columns]
    if missing:
        raise KeyError(
            f"calibrate_drift_params: {missing} not found. Run compute_observed_velocity() "
            f"on the track table before calibrating."
        )
    return calibrate_free_drift_params(
        u_wind=track_df["u_wind"].to_numpy(),
        v_wind=track_df["v_wind"].to_numpy(),
        u_current=track_df["u_current"].to_numpy(),
        v_current=track_df["v_current"].to_numpy(),
        lat_deg=track_df["lat"].to_numpy(),
        obs_u=track_df["obs_u"].to_numpy(),
        obs_v=track_df["obs_v"].to_numpy(),
        deflection_grid_deg=deflection_grid_deg,
    )


def compute_physics_residual(
    track_df: pd.DataFrame,
    params: DriftParams | None = None,
    group_col: str | None = "iceberg_id",
) -> pd.DataFrame:
    """Add the free-drift baseline and its residual against observed velocity.

    If obs_u/obs_v are absent, compute_observed_velocity() is called
    first, so a raw track table can be passed straight in.

    phys_u/phys_v are the free-drift prediction for the segment, driven
    by that row's segment-mean forcing; residual_* = obs_* - phys_* is
    the ML target. The residual isolates exactly what free drift cannot
    represent -- the iceberg's draft and keel, which we never observe.

    Args:
        track_df: A track table, with or without obs_u/obs_v.
        params: Drift coefficients. Defaults to the literature values in
            config; pass the output of calibrate_drift_params() (fitted
            on training rows only) for the real pipeline.
        group_col: Iceberg grouping column, or None for a single track.

    Returns:
        A copy of the table with phys_u, phys_v, residual_u, residual_v
        added.
    """
    if params is None:
        params = DriftParams(
            wind_factor=config.DEFAULT_WIND_FACTOR,
            deflection_deg=config.DEFAULT_DEFLECTION_DEG,
            current_factor=config.DEFAULT_CURRENT_FACTOR,
        )

    if "obs_u" not in track_df.columns or "obs_v" not in track_df.columns:
        df = compute_observed_velocity(track_df, group_col=group_col)
    else:
        df = track_df.reset_index(drop=True).copy()

    phys_u, phys_v = free_drift_velocity_array(
        u_wind=df["u_wind"].to_numpy(),
        v_wind=df["v_wind"].to_numpy(),
        u_current=df["u_current"].to_numpy(),
        v_current=df["v_current"].to_numpy(),
        lat_deg=df["lat"].to_numpy(),
        **params.as_kwargs(),
    )
    df["phys_u"] = phys_u
    df["phys_v"] = phys_v
    df["residual_u"] = df["obs_u"] - df["phys_u"]
    df["residual_v"] = df["obs_v"] - df["phys_v"]
    return df


def _derive_current_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived (non-lag) feature columns to a table, in place-safe fashion.

    This is the single definition of every derived feature, shared by the
    batch builder and by the single-row builder used during forecast
    rollout, so the two can never drift apart.

    Longitude becomes sin/cos because it is a circular coordinate: two
    of the tracked icebergs cross the antimeridian, where raw longitude
    leaps from +179.7 to -179.7 for a step of a few kilometres. Area
    becomes log-area because iceberg areas span 50 to 3000 km^2 and the
    physically relevant quantity (keel depth, hence the depth of current
    the berg feels) scales with a root of the linear dimension, not with
    area itself.

    Args:
        df: A table containing lon, area_km2 and the wind/current columns.

    Returns:
        A copy of df with wind_speed, current_speed, lon_sin, lon_cos and
        log_area added.
    """
    out = df.copy()
    out["wind_speed"] = np.hypot(out["u_wind"], out["v_wind"])
    out["current_speed"] = np.hypot(out["u_current"], out["v_current"])
    lon_rad = np.radians(out["lon"])
    out["lon_sin"] = np.sin(lon_rad)
    out["lon_cos"] = np.cos(lon_rad)
    # +1 keeps this finite for a hypothetical zero-area record.
    out["log_area"] = np.log1p(out["area_km2"].clip(lower=0.0))
    return out


def feature_column_names(n_lags: int = config.DEFAULT_N_LAGS) -> list[str]:
    """Return the canonical feature column names, in the canonical order.

    Both the training table and every single row built during forecast
    rollout are indexed by this list, so the model always sees its
    columns in the order it was fitted on.

    Args:
        n_lags: Number of previous segments folded into each row.

    Returns:
        The ordered list of model input column names.
    """
    lag_names = [f"{col}_t-{lag}" for col in LAG_BASE_COLUMNS for lag in range(1, n_lags + 1)]
    return CURRENT_FEATURE_COLUMNS + lag_names


def build_feature_table(
    track_df: pd.DataFrame,
    n_lags: int = config.DEFAULT_N_LAGS,
    params: DriftParams | None = None,
    group_col: str | None = "iceberg_id",
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build the flat, model-ready feature table from a pooled track table.

    Runs compute_observed_velocity() and compute_physics_residual()
    internally, so a raw pooled table from data_ingest.build_real_dataset()
    can be passed straight in.

    Lags are taken with groupby().shift(), never a bare shift(), so no
    row ever inherits history from a different iceberg. Rows without a
    full lag history are dropped, which costs `n_lags` rows per iceberg
    -- with only ~10 usable segments per berg that is a real price, and
    it is why config.DEFAULT_N_LAGS is 1 rather than the 5 that suited
    the 120-step synthetic track.

    Args:
        track_df: A pooled track table (POOLED_SCHEMA_COLUMNS) or a
            single-iceberg track.
        n_lags: Number of previous segments to fold into each row.
        params: Drift coefficients for the physics baseline; see
            compute_physics_residual().
        group_col: Iceberg grouping column, or None for a single track.

    Returns:
        A (feature_df, feature_cols, target_cols) tuple. feature_df
        carries METADATA_COLUMNS (for grouping/splitting/plotting; not
        model inputs), then every feature column, then the targets, and
        contains no NaNs.

    Raises:
        ValueError: If no rows survive (too few fixes per iceberg for the
            requested n_lags).
    """
    df = compute_physics_residual(track_df, params=params, group_col=group_col)
    df = _derive_current_features(df)

    if group_col is None:
        df = df.assign(_group=0)
        group_key = "_group"
    else:
        group_key = group_col

    for col in LAG_BASE_COLUMNS:
        for lag in range(1, n_lags + 1):
            df[f"{col}_t-{lag}"] = df.groupby(group_key)[col].shift(lag)

    feature_cols = feature_column_names(n_lags)
    metadata = [c for c in METADATA_COLUMNS if c in df.columns]
    # `lat` is both a model feature and useful metadata, so de-duplicate
    # while preserving order -- selecting a duplicated name would hand
    # back a two-column DataFrame for it and break X[feature_cols].
    output_cols = list(dict.fromkeys(metadata + feature_cols + TARGET_COLUMNS))

    feature_df = df[output_cols].dropna().reset_index(drop=True)
    if feature_df.empty:
        raise ValueError(
            f"build_feature_table: no rows survived with n_lags={n_lags}. Each iceberg "
            f"needs at least {n_lags + 2} fixes (one lost to the velocity difference, "
            f"{n_lags} to the lags). Reduce n_lags or supply longer tracks."
        )
    return feature_df, feature_cols, TARGET_COLUMNS


def build_single_feature_row(
    state: dict[str, float], n_lags: int = config.DEFAULT_N_LAGS
) -> pd.DataFrame:
    """Assemble one feature row during forecast rollout, matching the training layout.

    During a rollout there is no pre-built feature table: each future
    step's features depend on the previous step's forecast, so they must
    be built one at a time. This routes the raw quantities through the
    same _derive_current_features() the batch builder uses and then
    re-indexes to feature_column_names(), guaranteeing identical columns
    in identical order.

    Args:
        state: A dict holding every raw quantity needed -- lat, lon,
            area_km2, u_wind, v_wind, u_current, v_current, phys_u,
            phys_v, dt_hours, and one entry per lag column (e.g.
            "residual_u_t-1").
        n_lags: Number of lags the model was trained with.

    Returns:
        A single-row DataFrame containing exactly the model's feature
        columns, in training order.

    Raises:
        KeyError: If any required feature is missing from `state`, naming
            all of them at once rather than failing on the first.
    """
    row = _derive_current_features(pd.DataFrame([state]))
    required = feature_column_names(n_lags)
    missing = [c for c in required if c not in row.columns]
    if missing:
        raise KeyError(
            f"build_single_feature_row: missing required feature(s) {missing}. The state "
            f"dict must supply lat, lon, area_km2, the four environmental components, "
            f"phys_u/phys_v, dt_hours, and every lag column for n_lags={n_lags}."
        )
    return row[required]


if __name__ == "__main__":
    from data_ingest import build_real_dataset, generate_synthetic_track

    try:
        pooled, _summary = build_real_dataset(verbose=False)
        source = "real NIC + ERA5 + CMEMS"
        group_col: str | None = "iceberg_id"
    except Exception as exc:  # noqa: BLE001 - demo falls back when offline
        print(f"Real data unavailable ({type(exc).__name__}: {exc}); using synthetic.\n")
        pooled = generate_synthetic_track()
        source = "synthetic"
        group_col = None

    print(f"Source: {source} -- {len(pooled)} rows")

    with_velocity = compute_observed_velocity(pooled, group_col=group_col)
    print(f"Rows with an observed velocity: {len(with_velocity)}")

    default_params = DriftParams()
    fitted_params = calibrate_drift_params(with_velocity)
    print(f"\nDefault physics params: {default_params}")
    print(f"Fitted physics params:  {fitted_params}")

    for label, params in (("default", default_params), ("fitted", fitted_params)):
        resid = compute_physics_residual(with_velocity, params=params, group_col=group_col)
        rms = float(np.sqrt((resid["residual_u"] ** 2 + resid["residual_v"] ** 2).mean()))
        obs_rms = float(np.sqrt((resid["obs_u"] ** 2 + resid["obs_v"] ** 2).mean()))
        print(
            f"  {label:>7} physics: residual RMS {rms:.4f} m/s "
            f"vs observed-speed RMS {obs_rms:.4f} m/s "
            f"(explains {100 * (1 - rms / obs_rms):.1f}% of the drift magnitude)"
        )

    feature_df, feature_cols, target_cols = build_feature_table(
        pooled, params=fitted_params, group_col=group_col
    )
    print(f"\nFeature table: {feature_df.shape[0]} rows x {len(feature_cols)} features")
    print(f"Features: {feature_cols}")
    print(f"Targets:  {target_cols}")
    print("\nFirst 3 rows (features only):")
    print(feature_df[feature_cols].head(3).to_string(index=False))

    print("\nResidual summary (m/s):")
    print(feature_df[target_cols].agg(["mean", "std", "min", "max"]))

    # Round-trip check: stepping from a fix with the observed velocity
    # for that segment must land on the next fix. This validates that
    # compute_observed_velocity's azimuth decomposition really is the
    # inverse of physics.step_position's bearing projection.
    first = with_velocity.iloc[0]
    if group_col is None:
        prev = pooled.iloc[0]
    else:
        prev_rows = pooled[
            (pooled[group_col] == first[group_col]) & (pooled["timestamp"] < first["timestamp"])
        ]
        prev = prev_rows.iloc[-1]
    predicted_lat, predicted_lon = step_position(
        prev["lat"], prev["lon"], first["obs_u"], first["obs_v"], first["dt_hours"] * 3600.0
    )
    error_km = geodesic_distance_km(predicted_lat, predicted_lon, first["lat"], first["lon"])
    print(f"\nObserved-velocity round-trip error: {error_km:.6f} km (should be ~0)")
    assert error_km < 0.01, "observed velocity must reproduce the next fix exactly"
    print("features.py checks passed.")
