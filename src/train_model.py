"""
train_model.py

Trains the hybrid physics + gradient-boosting residual model on the real
NIC/ERA5/CMEMS dataset and scores it the way a navigation user would
care about: displacement error in kilometres over a multi-step forecast,
not regression error on a velocity residual.

WHAT IS BEING MEASURED
======================
evaluate_trajectory() performs an autoregressive rollout. From a real
position fix it forecasts forward `horizon_steps` weekly segments, each
step starting from its own previous FORECAST rather than from ground
truth, so errors compound the way they do in deployment. It reports:

  ADE -- average displacement error over every forecast step
  FDE -- displacement error at the final step only, the harder number

Three modes share that machinery so the comparison is exactly
apples-to-apples:

  persistence -- the iceberg keeps its last observed velocity. The
                 trivial baseline any forecasting system must beat, and
                 a strong one for a body with as much inertia as a
                 gigatonne of ice.
  physics     -- calibrated free drift, no ML.
  hybrid      -- calibrated free drift plus the learned residual.

HOW IT IS VALIDATED
===================
Two splits, because with fifteen icebergs they answer different
questions and only reporting the flattering one would be misleading:

  by time     -- train on the earlier weeks, test on the later ones.
                 This is the deployment-realistic question ("can it
                 forecast next week from what we know now?") but the
                 test set shares icebergs with the training set, so the
                 model has seen each berg's personal residual signature.

  leave-one-iceberg-out -- train on fourteen icebergs, forecast the
                 fifteenth, repeat for all of them. This is the honest
                 generalisation question ("does it work on a berg it has
                 never seen?") and it is the headline number.

In BOTH cases the physics coefficients are recalibrated on the training
fold only. Calibrating once on the full dataset before splitting is a
subtle leak -- the test period's observations would be helping to set
the baseline it is then scored against.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from features import (
    LAG_BASE_COLUMNS,
    TARGET_COLUMNS,
    build_feature_table,
    build_single_feature_row,
    calibrate_drift_params,
    compute_observed_velocity,
    compute_physics_residual,
    feature_column_names,
)
from physics import DriftParams, free_drift_velocity, geodesic_distance_km, step_position

# Deliberately small-capacity settings. The real pooled dataset is ~130
# labelled segments with 17 features; the defaults that suited a
# 120-step synthetic track (200 trees, depth 4) memorise it outright.
# Depth 2 with heavy subsampling and a large min_child_weight keeps the
# model to roughly "a few interaction terms on top of the physics".
XGB_PARAMS: dict[str, object] = {
    "n_estimators": 300,
    "max_depth": 2,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 2.0,
    "random_state": 42,
    "n_jobs": 2,
}


# =====================================================================
# Splitting
# =====================================================================


def train_test_split_by_time(
    feature_df: pd.DataFrame, test_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a feature table chronologically into train and test sets.

    The split is on a TIMESTAMP CUTOFF, not on row position: the pooled
    table interleaves fifteen icebergs, so slicing by row index would put
    the same calendar week on both sides of the split for different
    bergs. Every row at or after the cutoff date is test, everything
    before is train.

    A random shuffle split would be badly wrong here. Adjacent segments
    of one iceberg share environmental forcing and a lag-1 residual, so a
    shuffled test row would sit between two training rows that all but
    give away its answer.

    Args:
        feature_df: A feature table from features.build_feature_table().
        test_fraction: Approximate fraction of rows to hold out, taken
            from the end of the record.

    Returns:
        A (train_df, test_df) tuple.

    Raises:
        ValueError: If either side of the split comes out empty.
    """
    df = feature_df.sort_values("timestamp").reset_index(drop=True)
    cutoff = df["timestamp"].quantile(1.0 - test_fraction)
    train_df = df[df["timestamp"] < cutoff].reset_index(drop=True)
    test_df = df[df["timestamp"] >= cutoff].reset_index(drop=True)

    if train_df.empty or test_df.empty:
        raise ValueError(
            f"train_test_split_by_time: test_fraction={test_fraction} produced an empty "
            f"split ({len(train_df)} train / {len(test_df)} test rows) at cutoff {cutoff}. "
            f"The record spans {df['timestamp'].min()} to {df['timestamp'].max()}."
        )
    return train_df, test_df


# =====================================================================
# Training
# =====================================================================


def train_residual_model(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str] = TARGET_COLUMNS,
    model_type: str = "xgb",
    drift_params: DriftParams | None = None,
    save_dir: str | Path | None = None,
    forecast_mode: str = "hybrid",
) -> dict:
    """Fit one regressor per residual component and return them as a bundle.

    XGBoost's sklearn API is single-output, so residual_u and residual_v
    get a model each. `model_type="ridge"` fits a standardised linear
    model instead -- worth having, not as a toy: with ~130 training rows
    a linear correction is a serious competitor to a tree ensemble, and
    if it wins, the honest thing is to ship it.

    Args:
        train_df: Training rows, containing feature_cols and target_cols.
        feature_cols: Model input column names, in the order the model
            will be fitted (and must later be predicted) on.
        target_cols: Must contain "residual_u" and "residual_v".
        model_type: "xgb" or "ridge".
        drift_params: The calibrated physics coefficients this model was
            trained to correct. Stored in the bundle (and on disk) so a
            forecast can never be run with a different baseline than the
            residual was fitted against -- which would silently produce a
            correction for physics that is not the physics being used.
        save_dir: Directory to persist the bundle to; None skips saving.
        forecast_mode: "hybrid" or "physics" -- recorded in the bundle so
            decision_support.py knows whether the learned residual was
            actually validated as an improvement (see
            select_forecast_mode()). The model is saved either way.

    Returns:
        A dict with keys "u", "v" (fitted estimators), "feature_cols",
        "drift_params" and "model_type".

    Raises:
        ValueError: If target_cols lacks either residual component, or
            model_type is unrecognised.
    """
    missing = [c for c in ("residual_u", "residual_v") if c not in target_cols]
    if missing:
        raise ValueError(f"train_residual_model: target_cols must include {missing}.")

    def _new_model():
        if model_type == "xgb":
            return xgb.XGBRegressor(**XGB_PARAMS)
        if model_type == "ridge":
            # Standardise first: the features mix m/s (~0.05), degrees
            # (~-65) and hours (~170), so an unscaled ridge penalty would
            # fall almost entirely on the small-magnitude columns.
            return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        raise ValueError(
            f"train_residual_model: unknown model_type={model_type!r}; expected 'xgb' or 'ridge'."
        )

    X = train_df[feature_cols]
    model_u = _new_model().fit(X, train_df["residual_u"])
    model_v = _new_model().fit(X, train_df["residual_v"])

    bundle = {
        "u": model_u,
        "v": model_v,
        "feature_cols": list(feature_cols),
        "drift_params": drift_params or DriftParams(),
        "model_type": model_type,
        "forecast_mode": forecast_mode,
    }

    if save_dir is not None:
        save_dir = Path(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(model_u, save_dir / "residual_u.joblib")
        joblib.dump(model_v, save_dir / "residual_v.joblib")
        # The feature order and the drift coefficients are not
        # recoverable from the .joblib files, and a forecast is wrong
        # without both, so they are written alongside.
        with open(save_dir / "model_meta.json", "w") as handle:
            json.dump(
                {
                    "feature_cols": list(feature_cols),
                    "drift_params": asdict(bundle["drift_params"]),
                    "model_type": model_type,
                    "forecast_mode": forecast_mode,
                    "n_train_rows": int(len(train_df)),
                },
                handle,
                indent=2,
            )

    return bundle


def load_residual_model(save_dir: str | Path = config.MODELS_DIR) -> dict:
    """Load a saved model bundle from disk.

    Args:
        save_dir: Directory previously passed to train_residual_model().

    Returns:
        A bundle dict in the same shape train_residual_model() returns.

    Raises:
        FileNotFoundError: If the directory lacks the expected files.
    """
    save_dir = Path(save_dir)
    meta_path = save_dir / "model_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"load_residual_model: {meta_path} not found. Train and save a model first "
            f"(python src/train_on_real_data.py)."
        )
    with open(meta_path) as handle:
        meta = json.load(handle)
    return {
        "u": joblib.load(save_dir / "residual_u.joblib"),
        "v": joblib.load(save_dir / "residual_v.joblib"),
        "feature_cols": meta["feature_cols"],
        "drift_params": DriftParams(**meta["drift_params"]),
        "model_type": meta.get("model_type", "xgb"),
        "forecast_mode": meta.get("forecast_mode", "hybrid"),
    }


def predict_residual(models: dict, feature_row: pd.Series | pd.DataFrame) -> tuple[float, float]:
    """Predict (residual_u, residual_v) for one feature row.

    Accepts a Series or a single-row DataFrame -- both shapes come up
    naturally during rollout. Columns are reordered to the bundle's
    feature_cols before prediction, so callers need not match training
    order themselves.

    Args:
        models: A bundle from train_residual_model() or
            load_residual_model().
        feature_row: One row of features.

    Returns:
        A (residual_u, residual_v) tuple of floats. For a multi-row
        DataFrame, a pair of arrays is returned instead.

    Raises:
        KeyError: If the row is missing any of the bundle's feature_cols.
    """
    frame = feature_row.to_frame().T if isinstance(feature_row, pd.Series) else feature_row
    missing = [c for c in models["feature_cols"] if c not in frame.columns]
    if missing:
        raise KeyError(f"predict_residual: feature row is missing {missing}.")
    frame = frame[models["feature_cols"]].astype(float)

    pred_u = models["u"].predict(frame)
    pred_v = models["v"].predict(frame)
    if len(frame) == 1:
        return float(pred_u[0]), float(pred_v[0])
    return pred_u, pred_v


# =====================================================================
# Trajectory evaluation
# =====================================================================


def evaluate_trajectory(
    track_df: pd.DataFrame,
    mode: str = "physics",
    models: dict | None = None,
    drift_params: DriftParams | None = None,
    n_lags: int = config.DEFAULT_N_LAGS,
    horizon_steps: int = config.DEFAULT_HORIZON_STEPS,
    group_col: str | None = "iceberg_id",
    restrict_to: set[str] | None = None,
) -> dict:
    """Roll forecasts forward step by step and score them in kilometres.

    For every valid starting fix of every iceberg, this forecasts
    `horizon_steps` segments ahead and measures the geodesic distance
    from each forecast position to the real fix at that time. Crucially
    the rollout continues from its OWN previous forecast, so error
    compounds as it does in deployment -- scoring one-step residual
    accuracy instead would flatter every model.

    Future wind and current are taken from the real record. That is
    legitimate for a hindcast: it isolates the drift model's error from
    the weather forecast's error, which is a separate system. True
    forward forecasting has to live with an imperfect weather forecast,
    and decision_support.py handles that case with its uncertainty cone.

    Iceberg AREA is held at its last known value through the rollout
    rather than read from the future record, since a real forecast would
    not know it.

    For the hybrid mode, lag features beyond the first step come from the
    model's OWN predictions -- its previous predicted residual and
    velocity feed the next step. Reading the real observed values at
    every step would be leakage: those are exactly the future positions
    being forecast.

    Args:
        track_df: A pooled track table (raw, from
            data_ingest.build_real_dataset()).
        mode: "persistence", "physics" or "hybrid".
        models: The bundle to use; required for mode="hybrid".
        drift_params: Calibrated drift coefficients. Defaults to the
            bundle's if models is given, else the config defaults.
        n_lags: Number of lags the model expects.
        horizon_steps: Segments to forecast ahead per rollout.
        group_col: Iceberg grouping column, or None for a single track.
        restrict_to: If given, only evaluate these iceberg ids (used by
            the leave-one-out loop to score the held-out berg).

    Returns:
        A dict with ade_km, fde_km, n_rollouts, horizon_steps, mode,
        per_step_km (mean error at each step) and n_icebergs.

    Raises:
        ValueError: If mode is unrecognised, or mode="hybrid" without
            models.
    """
    if mode not in {"persistence", "physics", "hybrid"}:
        raise ValueError(
            f"evaluate_trajectory: unknown mode={mode!r}; expected 'persistence', "
            f"'physics' or 'hybrid'."
        )
    if mode == "hybrid" and models is None:
        raise ValueError("evaluate_trajectory: mode='hybrid' requires a trained `models` bundle.")

    if drift_params is None:
        drift_params = models["drift_params"] if models is not None else DriftParams()

    enriched = compute_physics_residual(track_df, params=drift_params, group_col=group_col)
    if group_col is None:
        enriched = enriched.assign(iceberg_id="track")
        group_key = "iceberg_id"
    else:
        group_key = group_col

    max_segment_hours = config.MAX_SEGMENT_DAYS * 24.0
    step_errors: list[list[float]] = [[] for _ in range(horizon_steps)]
    n_rollouts = 0
    icebergs_used: set[str] = set()

    for iceberg_id, group in enriched.groupby(group_key, sort=True):
        if restrict_to is not None and iceberg_id not in restrict_to:
            continue
        berg = group.sort_values("timestamp").reset_index(drop=True)
        n = len(berg)

        # A start needs n_lags rows of history behind it (for the lag
        # features) and horizon_steps real fixes ahead to score against.
        # The same bound is applied in every mode so all three are
        # compared over an identical set of rollouts.
        for start in range(n_lags, n - horizon_steps):
            steps = berg.iloc[start + 1 : start + 1 + horizon_steps]
            if (steps["dt_hours"] > max_segment_hours).any():
                # A multi-week gap in the record: forecasting "one step"
                # across it is not the same task as the other rollouts.
                continue

            current_lat = float(berg["lat"].iloc[start])
            current_lon = float(berg["lon"].iloc[start])
            last_known_area = float(berg["area_km2"].iloc[start])

            # Seed the lag history from real observations, most recent
            # first; predictions are pushed onto the front as we roll.
            history = [
                {col: float(berg[col].iloc[start - offset]) for col in LAG_BASE_COLUMNS}
                for offset in range(n_lags)
            ]

            for step in range(1, horizon_steps + 1):
                target = berg.iloc[start + step]
                dt_seconds = float(target["dt_hours"]) * 3600.0

                if mode == "persistence":
                    total_u = history[0]["obs_u"]
                    total_v = history[0]["obs_v"]
                    residual_u = residual_v = 0.0
                else:
                    phys_u, phys_v = free_drift_velocity(
                        u_wind=float(target["u_wind"]),
                        v_wind=float(target["v_wind"]),
                        u_current=float(target["u_current"]),
                        v_current=float(target["v_current"]),
                        lat_deg=current_lat,
                        **drift_params.as_kwargs(),
                    )
                    residual_u = residual_v = 0.0
                    if mode == "hybrid":
                        state: dict[str, float] = {
                            "lat": current_lat,
                            "lon": current_lon,
                            "area_km2": last_known_area,
                            "u_wind": float(target["u_wind"]),
                            "v_wind": float(target["v_wind"]),
                            "u_current": float(target["u_current"]),
                            "v_current": float(target["v_current"]),
                            "phys_u": phys_u,
                            "phys_v": phys_v,
                            "dt_hours": float(target["dt_hours"]),
                        }
                        for lag, past in enumerate(history, start=1):
                            for col in LAG_BASE_COLUMNS:
                                state[f"{col}_t-{lag}"] = past[col]
                        row = build_single_feature_row(state, n_lags=n_lags)
                        residual_u, residual_v = predict_residual(models, row)
                    total_u = phys_u + residual_u
                    total_v = phys_v + residual_v

                forecast_lat, forecast_lon = step_position(
                    current_lat, current_lon, total_u, total_v, dt_seconds
                )
                step_errors[step - 1].append(
                    geodesic_distance_km(
                        forecast_lat, forecast_lon, float(target["lat"]), float(target["lon"])
                    )
                )

                # Autoregressive feedback: the next step's lag features
                # are this step's PREDICTIONS, not the truth we are
                # trying to forecast.
                history.insert(
                    0,
                    {
                        "obs_u": total_u,
                        "obs_v": total_v,
                        "residual_u": residual_u,
                        "residual_v": residual_v,
                    },
                )
                history = history[:n_lags]
                current_lat, current_lon = forecast_lat, forecast_lon

            n_rollouts += 1
            icebergs_used.add(str(iceberg_id))

    flat = [e for step in step_errors for e in step]
    return {
        "mode": mode,
        "ade_km": float(np.mean(flat)) if flat else float("nan"),
        "fde_km": float(np.mean(step_errors[-1])) if step_errors[-1] else float("nan"),
        "per_step_km": [float(np.mean(s)) if s else float("nan") for s in step_errors],
        "n_rollouts": n_rollouts,
        "n_icebergs": len(icebergs_used),
        "horizon_steps": horizon_steps,
    }


def leave_one_iceberg_out(
    pooled_df: pd.DataFrame,
    n_lags: int = config.DEFAULT_N_LAGS,
    horizon_steps: int = config.DEFAULT_HORIZON_STEPS,
    model_type: str = "xgb",
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Score generalisation to an unseen iceberg, one held-out berg at a time.

    For each iceberg: recalibrate the physics on the other fourteen,
    train the residual model on the other fourteen, then roll forecasts
    on the held-out one. Nothing about the held-out berg touches either
    the calibration or the fit, so this answers the question that
    actually matters operationally -- a newly calved berg has no history
    for the model to have memorised.

    Args:
        pooled_df: The pooled real track table.
        n_lags: Lags per feature row.
        horizon_steps: Rollout horizon.
        model_type: "xgb" or "ridge".
        verbose: Print a line per held-out iceberg.

    Returns:
        A (per_iceberg_df, aggregate) tuple. per_iceberg_df has one row
        per held-out berg with its ADE/FDE in all three modes; aggregate
        maps each mode to rollout-weighted overall ADE/FDE.
    """
    iceberg_ids = sorted(pooled_df["iceberg_id"].unique())
    rows: list[dict[str, object]] = []

    for held_out in iceberg_ids:
        train_track = pooled_df[pooled_df["iceberg_id"] != held_out]
        test_track = pooled_df[pooled_df["iceberg_id"] == held_out]

        # Calibrate and fit on the training bergs ONLY.
        train_velocity = compute_observed_velocity(train_track)
        params = calibrate_drift_params(train_velocity)
        train_features, feature_cols, target_cols = build_feature_table(
            train_track, n_lags=n_lags, params=params
        )
        if len(train_features) < 20:
            continue
        models = train_residual_model(
            train_features, feature_cols, target_cols, model_type=model_type, drift_params=params
        )

        result: dict[str, dict] = {}
        for mode in ("persistence", "physics", "hybrid"):
            result[mode] = evaluate_trajectory(
                test_track,
                mode=mode,
                models=models if mode == "hybrid" else None,
                drift_params=params,
                n_lags=n_lags,
                horizon_steps=horizon_steps,
            )

        if result["physics"]["n_rollouts"] == 0:
            continue

        rows.append(
            {
                "iceberg_id": held_out,
                "n_rollouts": result["physics"]["n_rollouts"],
                **{f"{mode}_ade_km": result[mode]["ade_km"] for mode in result},
                **{f"{mode}_fde_km": result[mode]["fde_km"] for mode in result},
            }
        )
        if verbose:
            print(
                f"  {held_out:<6} n={result['physics']['n_rollouts']:>2}  "
                f"ADE km  persist {result['persistence']['ade_km']:7.2f} | "
                f"physics {result['physics']['ade_km']:7.2f} | "
                f"hybrid {result['hybrid']['ade_km']:7.2f}"
            )

    per_iceberg = pd.DataFrame(rows)
    if per_iceberg.empty:
        return per_iceberg, {}

    # Weight by rollout count so a berg contributing eight forecasts
    # counts eight times as much as one contributing a single forecast.
    weights = per_iceberg["n_rollouts"].to_numpy(dtype=float)
    aggregate = {
        mode: {
            "ade_km": float(np.average(per_iceberg[f"{mode}_ade_km"], weights=weights)),
            "fde_km": float(np.average(per_iceberg[f"{mode}_fde_km"], weights=weights)),
            "n_rollouts": int(weights.sum()),
        }
        for mode in ("persistence", "physics", "hybrid")
    }
    return per_iceberg, aggregate


def select_forecast_mode(
    aggregate: dict[str, dict], min_improvement_pct: float = 5.0
) -> tuple[str, str]:
    """Decide whether the learned residual has earned its place in the forecast.

    The hybrid model is not shipped just because it was trained. It is
    shipped only if it beats the calibrated physics baseline on the
    leave-one-iceberg-out evaluation by a margin larger than the noise
    of a ~15-fold, ~130-row experiment. Anything smaller than a few
    percent on this much data is not a real improvement, and a forecast
    system that adds an unjustified learned correction is strictly worse
    than one that does not: same accuracy, more ways to fail, and no
    physical interpretation when it goes wrong.

    Args:
        aggregate: The aggregate dict from leave_one_iceberg_out().
        min_improvement_pct: How much better than physics the hybrid must
            be, in percent of physics ADE, to be selected.

    Returns:
        A (mode, rationale) tuple where mode is "hybrid" or "physics".
    """
    if not aggregate or "hybrid" not in aggregate or "physics" not in aggregate:
        return "physics", "No leave-one-out result available; defaulting to physics only."

    physics_ade = aggregate["physics"]["ade_km"]
    hybrid_ade = aggregate["hybrid"]["ade_km"]
    improvement = 100.0 * (physics_ade - hybrid_ade) / physics_ade

    if improvement >= min_improvement_pct:
        return "hybrid", (
            f"The learned residual improves held-out ADE by {improvement:.1f}% "
            f"({physics_ade:.1f} -> {hybrid_ade:.1f} km), above the {min_improvement_pct:.0f}% "
            f"bar; the hybrid is used for forecasting."
        )
    return "physics", (
        f"The learned residual changes held-out ADE by only {-improvement:+.1f}% "
        f"({physics_ade:.1f} -> {hybrid_ade:.1f} km), below the {min_improvement_pct:.0f}% "
        f"bar for this dataset size. Forecasts use calibrated physics alone; the trained "
        f"model is still saved so the comparison can be re-run as the record grows."
    )


def feature_importance(models: dict, top_n: int = 12) -> pd.DataFrame:
    """Summarise which features each residual model actually leaned on.

    Args:
        models: A bundle from train_residual_model().
        top_n: Number of features to return.

    Returns:
        A DataFrame with feature, importance_u, importance_v, sorted by
        their sum descending. Empty if the estimator exposes no
        importances (e.g. a ridge pipeline, whose coefficients are on
        standardised inputs and are reported instead).
    """
    def _scores(model) -> np.ndarray | None:
        if hasattr(model, "feature_importances_"):
            return np.asarray(model.feature_importances_, dtype=float)
        if hasattr(model, "named_steps") and "ridge" in getattr(model, "named_steps", {}):
            return np.abs(np.asarray(model.named_steps["ridge"].coef_, dtype=float))
        return None

    scores_u, scores_v = _scores(models["u"]), _scores(models["v"])
    if scores_u is None or scores_v is None:
        return pd.DataFrame(columns=["feature", "importance_u", "importance_v"])

    frame = pd.DataFrame(
        {"feature": models["feature_cols"], "importance_u": scores_u, "importance_v": scores_v}
    )
    return frame.assign(total=frame["importance_u"] + frame["importance_v"]) \
                .sort_values("total", ascending=False) \
                .drop(columns="total") \
                .head(top_n) \
                .reset_index(drop=True)


if __name__ == "__main__":
    from train_on_real_data import main

    main()
