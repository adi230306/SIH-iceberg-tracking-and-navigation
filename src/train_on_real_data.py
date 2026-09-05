"""
train_on_real_data.py

End-to-end entry point: build the real dataset from the NIC snapshots,
ERA5 wind and Copernicus Marine currents, calibrate the physics, train
the residual model, evaluate it against two baselines under two
different splits, and persist the winning bundle to models/.

    python src/train_on_real_data.py

The first run downloads and caches ERA5 (~85 MB, one CDS request per
month) and one small Copernicus current box per iceberg. Later runs read
the cache and take seconds. Credentials required:

    ~/.cdsapirc              https://cds.climate.copernicus.eu/how-to-api
    ~/.copernicusmarine      `copernicusmarine login`
"""

from __future__ import annotations

import argparse
import textwrap

import numpy as np
import pandas as pd

import config
from data_ingest import build_real_dataset
from features import build_feature_table, calibrate_drift_params, compute_observed_velocity
from train_model import (
    evaluate_trajectory,
    select_forecast_mode,
    feature_importance,
    leave_one_iceberg_out,
    train_residual_model,
    train_test_split_by_time,
)


def _print_comparison(title: str, results: dict[str, dict], horizon_steps: int) -> None:
    """Print one ADE/FDE comparison table across the three forecast modes.

    Args:
        title: Heading for the block.
        results: Mapping of mode name -> evaluate_trajectory() result.
        horizon_steps: Rollout horizon, quoted in the heading.

    Returns:
        None.
    """
    any_result = next(iter(results.values()))
    print()
    print("=" * 72)
    print(f"{title}  --  {horizon_steps}-step (~{horizon_steps} week) rollout, "
          f"{any_result['n_rollouts']} forecasts")
    print("=" * 72)
    print(f"{'Model':<28}{'ADE (km)':>12}{'FDE (km)':>12}{'vs physics':>16}")
    print("-" * 72)

    physics_ade = results.get("physics", {}).get("ade_km", float("nan"))
    labels = {
        "persistence": "Persistence (last velocity)",
        "physics": "Physics only (calibrated)",
        "hybrid": "Hybrid (physics + ML)",
    }
    for mode in ("persistence", "physics", "hybrid"):
        if mode not in results:
            continue
        res = results[mode]
        if mode == "physics" or not np.isfinite(physics_ade):
            delta = "--"
        else:
            change = 100.0 * (res["ade_km"] - physics_ade) / physics_ade
            delta = f"{change:+.1f}%"
        print(f"{labels[mode]:<28}{res['ade_km']:>12.2f}{res['fde_km']:>12.2f}{delta:>16}")
    print("-" * 72)

    steps = results.get("hybrid", any_result)["per_step_km"]
    # The aggregated leave-one-out block has no meaningful per-step
    # breakdown (it averages across folds), so skip the line there.
    if any(np.isfinite(v) for v in steps):
        per_step = "  ".join(f"step {i + 1}: {v:.1f}" for i, v in enumerate(steps))
        print(f"Hybrid error growth (km):  {per_step}")


def main(
    n_lags: int = config.DEFAULT_N_LAGS,
    horizon_steps: int = config.DEFAULT_HORIZON_STEPS,
    model_type: str = "xgb",
    force_refresh: bool = False,
) -> dict:
    """Run the full real-data pipeline and report the results.

    Args:
        n_lags: Number of previous segments folded into each feature row.
        horizon_steps: Rollout horizon, in ~weekly segments.
        model_type: "xgb" or "ridge" for the residual model.
        force_refresh: Bypass the download/track caches.

    Returns:
        A dict with the fitted drift params, both evaluation blocks, the
        per-iceberg leave-one-out table and the saved model bundle.
    """
    # --- 1. Data -----------------------------------------------------
    pooled, motion = build_real_dataset(force_refresh=force_refresh)

    n_grounded = int(motion["is_grounded"].sum())
    print()
    print("=" * 72)
    print("DATASET")
    print("=" * 72)
    print(f"Icebergs tracked by NIC over the record : {len(motion)}")
    print(f"  excluded as grounded / not re-observed: {n_grounded}")
    print(f"  used for drift training               : {pooled['iceberg_id'].nunique()}")
    print(f"Forced observations                     : {len(pooled)}")
    print(f"Record spans                            : "
          f"{pooled['timestamp'].min():%Y-%m-%d} to {pooled['timestamp'].max():%Y-%m-%d}")
    print(f"Segment length (days)                   : "
          f"{pooled['segment_hours'].min() / 24:.0f} to {pooled['segment_hours'].max() / 24:.0f}")

    # --- 2. Physics calibration --------------------------------------
    with_velocity = compute_observed_velocity(pooled)
    params = calibrate_drift_params(with_velocity)
    observed_rms = float(np.sqrt((with_velocity["obs_u"] ** 2 + with_velocity["obs_v"] ** 2).mean()))

    print()
    print("=" * 72)
    print("PHYSICS CALIBRATION  (fitted on the observed drift, not assumed)")
    print("=" * 72)
    print(f"  wind_factor    {params.wind_factor:9.5f}   (literature default 0.018)")
    print(f"  deflection     {params.deflection_deg:9.1f} deg (literature default 20)")
    print(f"  current_factor {params.current_factor:9.5f}   (surface current taken at face value = 1.0)")
    print()
    print(f"  Observed drift speed RMS: {observed_rms:.4f} m/s")
    if params.wind_factor < 1e-3:
        print(
            "  NOTE: the wind term fits to ~zero. This is not a bug -- the Copernicus\n"
            "  analysis current at 0.5 m ALREADY contains the wind-driven Ekman and\n"
            "  Stokes response, so a separate free-drift wind term double-counts it.\n"
            "  The textbook 1.8% factor assumes a geostrophic current that does not."
        )
    print(
        f"  The fitted current factor of {params.current_factor:.2f} is consistent with a tabular\n"
        f"  berg's 150-300 m keel feeling the depth-averaged current rather than the\n"
        f"  surface value the product reports."
    )

    # --- 3. Chronological split --------------------------------------
    feature_df, feature_cols, target_cols = build_feature_table(
        pooled, n_lags=n_lags, params=params
    )
    train_df, test_df = train_test_split_by_time(feature_df, test_fraction=0.25)

    # Recalibrate on the training rows only: calibrating on everything
    # first would let the test period help set the baseline it is scored
    # against.
    train_params = calibrate_drift_params(
        with_velocity[with_velocity["timestamp"] < test_df["timestamp"].min()]
    )
    train_features, feature_cols, target_cols = build_feature_table(
        pooled[pooled["timestamp"] < test_df["timestamp"].min()], n_lags=n_lags, params=train_params
    )
    models_time = train_residual_model(
        train_features, feature_cols, target_cols, model_type=model_type, drift_params=train_params
    )

    cutoff = test_df["timestamp"].min()
    # The rollout needs n_lags rows of history before the first scored
    # step, so the evaluation track reaches back before the cutoff; the
    # MODEL, however, saw nothing at or after it.
    test_track = pooled[pooled["timestamp"] >= cutoff - pd.Timedelta(days=30)]

    time_results = {
        mode: evaluate_trajectory(
            test_track,
            mode=mode,
            models=models_time if mode == "hybrid" else None,
            drift_params=train_params,
            n_lags=n_lags,
            horizon_steps=horizon_steps,
        )
        for mode in ("persistence", "physics", "hybrid")
    }
    _print_comparison(
        f"SPLIT BY TIME  (train < {cutoff:%Y-%m-%d}, {len(train_features)} rows)",
        time_results,
        horizon_steps,
    )

    # --- 4. Leave-one-iceberg-out ------------------------------------
    print()
    print("=" * 72)
    print("LEAVE-ONE-ICEBERG-OUT  (each berg forecast by a model that never saw it)")
    print("=" * 72)
    per_iceberg, loio = leave_one_iceberg_out(
        pooled, n_lags=n_lags, horizon_steps=horizon_steps, model_type=model_type
    )
    if loio:
        _print_comparison("LEAVE-ONE-ICEBERG-OUT AGGREGATE",
                          {m: {**loio[m], "per_step_km": [float("nan")] * horizon_steps}
                           for m in loio},
                          horizon_steps)

    # --- 5. Decide whether the ML stage has earned its place ---------
    forecast_mode, rationale = select_forecast_mode(loio)
    print()
    print("=" * 72)
    print(f"MODEL SELECTION -> {forecast_mode.upper()}")
    print("=" * 72)
    print(textwrap.fill(rationale, width=70, initial_indent="  ", subsequent_indent="  "))

    # --- 6. Final model on everything --------------------------------
    final_models = train_residual_model(
        feature_df,
        feature_cols,
        target_cols,
        model_type=model_type,
        drift_params=params,
        save_dir=config.MODELS_DIR,
        forecast_mode=forecast_mode,
    )
    print()
    print(f"Saved final model ({model_type}, trained on all {len(feature_df)} rows) "
          f"to {config.MODELS_DIR}")

    importance = feature_importance(final_models)
    if not importance.empty:
        print("\nTop features by importance:")
        print(importance.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    return {
        "forecast_mode": forecast_mode,
        "selection_rationale": rationale,
        "drift_params": params,
        "time_split": time_results,
        "leave_one_out": loio,
        "per_iceberg": per_iceberg,
        "models": final_models,
        "pooled": pooled,
        "motion": motion,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-lags", type=int, default=config.DEFAULT_N_LAGS)
    parser.add_argument("--horizon", type=int, default=config.DEFAULT_HORIZON_STEPS)
    parser.add_argument("--model", choices=["xgb", "ridge"], default="xgb")
    parser.add_argument(
        "--force-refresh", action="store_true", help="Bypass the download and track caches."
    )
    args = parser.parse_args()

    results = main(
        n_lags=args.n_lags,
        horizon_steps=args.horizon,
        model_type=args.model,
        force_refresh=args.force_refresh,
    )

    loio = results["leave_one_out"]
    if loio:
        # Sanity floor rather than a success criterion: if the learned
        # correction is dramatically worse than the physics it corrects,
        # something is broken (leakage, a units mismatch, a feature built
        # differently at train and rollout time) and the run should fail
        # loudly rather than print a plausible-looking table.
        assert loio["hybrid"]["ade_km"] < 2.0 * loio["physics"]["ade_km"], (
            f"Hybrid ADE {loio['hybrid']['ade_km']:.2f} km is more than 2x the physics "
            f"baseline {loio['physics']['ade_km']:.2f} km on held-out icebergs -- that is "
            f"a pipeline bug, not a modelling result."
        )
