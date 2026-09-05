"""
features.py

Turns a track DataFrame into an ML-ready feature table for training the
hybrid physics+GBM residual model.

Pipeline: for each timestep, compute the OBSERVED velocity from actual
consecutive positions (geodesic distance/bearing between real fixes),
compute what the PHYSICS free-drift baseline would have predicted for
that same step, and take the difference -- the residual -- as the ML
training target. A sliding window of recent environmental conditions is
then flattened into a single feature row per timestep so the residual
model can learn from short-term history, not just the instantaneous
conditions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pyproj import Geod

from physics import free_drift_velocity, geodesic_distance_km, step_position

# Same WGS84 geodesic calculator used in physics.py. Recreated here
# (rather than imported) because we need direct access to `.inv()` with
# vectorized array inputs to get both distance AND forward azimuth in a
# single call; geodesic_distance_km() (imported above and used in the
# __main__ sanity check) only exposes the distance.
_GEOD = Geod(ellps="WGS84")

# Base environmental variables that get lagged into sliding-window
# features. Order here also fixes the column ordering used everywhere
# below, so the lag columns for a given variable always appear as a
# contiguous, most-recent-first block: {var}_t-1, {var}_t-2, ...
LAG_BASE_COLUMNS: list[str] = ["u_wind", "v_wind", "u_current", "v_current", "area_km2"]


def compute_observed_velocity(track_df: pd.DataFrame) -> pd.DataFrame:
    """Compute observed eastward/northward velocity between consecutive track fixes.

    Uses pyproj's WGS84 geodesic inverse (`Geod.inv`) to get the true
    distance and forward azimuth between each pair of consecutive
    points -- not naive lat/lon differencing, which would be badly
    wrong near the poles and across the antimeridian. The azimuth
    (bearing, clockwise from north) is then decomposed into
    eastward/northward components: obs_u = speed*sin(azimuth),
    obs_v = speed*cos(azimuth), matching the same eastward/northward
    convention used everywhere else in this project (and the inverse of
    the bearing calculation in physics.step_position).

    Elapsed time between rows is computed from the timestamp column
    itself (not assumed to be a fixed dt), since real-world tracks
    (e.g. from load_usnic_track) may have gaps or irregular sampling.

    NOTE ON THE FIRST ROW: the very first row of track_df has no prior
    fix to difference against, so it cannot have an observed velocity.
    We compute obs_u/obs_v as NaN for that row and then DROP it, so
    the returned DataFrame has one fewer row than the input, with its
    index reset to start at 0. (The alternative -- keeping the row with
    NaN velocities -- would push NaN-handling responsibility onto every
    downstream consumer; dropping it here keeps the contract simple:
    every row returned by this function has valid obs_u/obs_v.)

    Args:
        track_df: A track DataFrame sorted by timestamp, ascending.

    Returns:
        A copy of track_df (minus its first row) with two new columns,
        obs_u and obs_v (m/s), for the eastward/northward velocity
        observed between each row and the one before it.

    Raises:
        ValueError: If any two consecutive timestamps are identical or
            out of order (non-positive elapsed time), since that makes
            velocity undefined/infinite.
    """
    df = track_df.sort_values("timestamp").reset_index(drop=True)

    lats = df["lat"].to_numpy()
    lons = df["lon"].to_numpy()

    # Vectorized geodesic inverse over all consecutive pairs at once:
    # fwd_azimuth_deg[k] / distance_m[k] describe the step from row k
    # to row k+1.
    fwd_azimuth_deg, _back_azimuth_deg, distance_m = _GEOD.inv(
        lons[:-1], lats[:-1], lons[1:], lats[1:]
    )

    dt_seconds = df["timestamp"].diff().dt.total_seconds().to_numpy()[1:]
    if np.any(dt_seconds <= 0):
        bad_rows = np.where(dt_seconds <= 0)[0] + 1  # +1: index of the *later* row in each pair
        raise ValueError(
            f"compute_observed_velocity: found non-positive elapsed time before row(s) "
            f"{bad_rows.tolist()} -- timestamps must be strictly increasing. Check for "
            f"duplicate or out-of-order timestamps in track_df."
        )

    speed_ms = distance_m / dt_seconds
    azimuth_rad = np.radians(fwd_azimuth_deg)
    obs_u = speed_ms * np.sin(azimuth_rad)
    obs_v = speed_ms * np.cos(azimuth_rad)

    df["obs_u"] = np.concatenate(([np.nan], obs_u))
    df["obs_v"] = np.concatenate(([np.nan], obs_v))

    # Drop the first row (no prior fix to diff against) -- see NOTE ON
    # THE FIRST ROW above.
    df = df.iloc[1:].reset_index(drop=True)
    return df


def compute_physics_residual(track_df: pd.DataFrame) -> pd.DataFrame:
    """Compute the free-drift physics baseline and its residual against observed velocity.

    If track_df does not already have obs_u/obs_v columns, this calls
    compute_observed_velocity() internally first, so callers may pass
    either a raw track DataFrame or one that has already been through
    compute_observed_velocity() -- either way the result is the same.

    For each row, physics.free_drift_velocity() is evaluated using that
    row's own u_wind/v_wind/u_current/v_current/lat to get the
    physics-predicted velocity (phys_u, phys_v). The residual --
    residual_u = obs_u - phys_u, residual_v = obs_v - phys_v -- is what
    the ML model is trained to predict; it isolates exactly the part of
    real iceberg motion that free-drift physics (which knows nothing
    about the iceberg's draft or shape) fails to capture.

    free_drift_velocity() is evaluated per row via DataFrame.apply()
    rather than a fully vectorized numpy expression, since it operates
    on scalar Python floats (math.radians etc.) and rows are
    independent of each other (no sequential dependency, unlike
    position integration) -- for very large tables this could be
    reimplemented with pure-numpy trig without changing this function's
    interface.

    Args:
        track_df: A track DataFrame, with or without obs_u/obs_v
            already computed.

    Returns:
        A copy of track_df with four new columns added: phys_u, phys_v
        (the physics baseline prediction, m/s) and residual_u,
        residual_v (obs minus phys, m/s).
    """
    if "obs_u" not in track_df.columns or "obs_v" not in track_df.columns:
        df = compute_observed_velocity(track_df)
    else:
        df = track_df.reset_index(drop=True).copy()

    phys_velocities = df.apply(
        lambda row: free_drift_velocity(
            u_wind=row["u_wind"],
            v_wind=row["v_wind"],
            u_current=row["u_current"],
            v_current=row["v_current"],
            lat_deg=row["lat"],
        ),
        axis=1,
        result_type="expand",
    )
    phys_velocities.columns = ["phys_u", "phys_v"]

    df = pd.concat([df, phys_velocities], axis=1)
    df["residual_u"] = df["obs_u"] - df["phys_u"]
    df["residual_v"] = df["obs_v"] - df["phys_v"]
    return df


def build_sliding_window_features(
    track_df: pd.DataFrame, window_size: int = 5
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build a flat, sliding-window feature table ready for sklearn/xgboost.

    Internally calls compute_physics_residual() (which itself calls
    compute_observed_velocity() if needed) on track_df first, so the
    caller can pass a raw track DataFrame straight in.

    For each timestep t (once at least `window_size` prior rows exist),
    the feature row contains:
      - The most recent `window_size` values of u_wind, v_wind,
        u_current, v_current, and area_km2, each flattened into
        separate columns named "{var}_t-1" (most recent) through
        "{var}_t-{window_size}" (oldest), grouped by variable in the
        order given by LAG_BASE_COLUMNS.
      - The CURRENT row's lat, lon (drift is latitude-dependent via
        Coriolis, so location is informative even without explicit
        physical modeling of regional effects).
      - The CURRENT row's physics prediction phys_u, phys_v -- so the
        model can learn when to trust vs. override the physics
        baseline.
    TARGET columns: residual_u, residual_v (current row).

    Rows at the start of the series that don't yet have a full window
    of history (their lag columns would be NaN) are dropped, so the
    returned DataFrame has no NaNs anywhere.

    Args:
        track_df: A raw (or partially-processed) track DataFrame.
        window_size: Number of prior timesteps of history to include
            per feature row.

    Returns:
        A tuple (feature_df, feature_column_names, target_column_names):
          - feature_df: a flat DataFrame with a "timestamp" column (for
            bookkeeping/joins only -- not a model feature), all feature
            columns, and all target columns, with no NaNs.
          - feature_column_names: the list of column names in
            feature_df that are model INPUTS.
          - target_column_names: the list of column names in
            feature_df that are model OUTPUTS -- ["residual_u",
            "residual_v"].
    """
    df = compute_physics_residual(track_df)

    # Vectorized lag generation: pandas .shift() on a whole column at
    # once, looped only over (variable, lag) combinations -- not over
    # rows -- so this stays vectorized per the project's style rules.
    lag_column_names: list[str] = []
    for col in LAG_BASE_COLUMNS:
        for lag in range(1, window_size + 1):
            lag_col = f"{col}_t-{lag}"
            df[lag_col] = df[col].shift(lag)
            lag_column_names.append(lag_col)

    feature_column_names = lag_column_names + ["lat", "lon", "phys_u", "phys_v"]
    target_column_names = ["residual_u", "residual_v"]

    output_columns = ["timestamp"] + feature_column_names + target_column_names
    feature_df = df[output_columns].dropna().reset_index(drop=True)

    return feature_df, feature_column_names, target_column_names


if __name__ == "__main__":
    import os

    from data_ingest import generate_synthetic_track

    csv_path = "data/synthetic_track.csv"
    if os.path.exists(csv_path):
        track = pd.read_csv(csv_path, parse_dates=["timestamp"])
    else:
        track = generate_synthetic_track()
        os.makedirs("data", exist_ok=True)
        track.to_csv(csv_path, index=False)

    obs_df = compute_observed_velocity(track)
    resid_df = compute_physics_residual(obs_df)
    feature_df, feature_cols, target_cols = build_sliding_window_features(track, window_size=5)

    print(f"Feature table shape: {feature_df.shape}")
    print(f"\nFeature columns ({len(feature_cols)}): {feature_cols}")
    print(f"Target columns: {target_cols}")
    print("\nFirst 3 rows:")
    print(feature_df.head(3).to_string(index=False))

    print("\nResidual summary stats (should be small / near zero -- our synthetic")
    print("generator's only 'unmodeled' component was small random noise, so a")
    print("near-zero mean/std here validates the physics baseline is doing most")
    print("of the work, which is the whole point of the hybrid approach):")
    print(resid_df[["residual_u", "residual_v"]].agg(["mean", "std"]))

    # Bonus round-trip sanity check: stepping forward from row i's
    # position using the OBSERVED velocity computed for the transition
    # into row i+1 should land close to row i+1's actual recorded
    # position. This validates that compute_observed_velocity's
    # azimuth/decomposition logic is the correct inverse of
    # physics.step_position's bearing/projection logic.
    i = 0
    dt_s = (track["timestamp"].iloc[i + 1] - track["timestamp"].iloc[i]).total_seconds()
    predicted_lat, predicted_lon = step_position(
        lat=track["lat"].iloc[i],
        lon=track["lon"].iloc[i],
        u_ms=obs_df["obs_u"].iloc[i],
        v_ms=obs_df["obs_v"].iloc[i],
        dt_seconds=dt_s,
    )
    actual_lat, actual_lon = track["lat"].iloc[i + 1], track["lon"].iloc[i + 1]
    round_trip_error_km = geodesic_distance_km(predicted_lat, predicted_lon, actual_lat, actual_lon)
    print(f"\nRound-trip sanity check error: {round_trip_error_km:.6f} km (should be ~0)")
    assert round_trip_error_km < 0.01, "observed-velocity round trip should return to the actual next fix"

    print("\nAll features.py sanity checks passed.")