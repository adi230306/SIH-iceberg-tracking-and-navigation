"""Entry point: wires the real-data backend to the Dash frontend.

    python main.py

This is the ONLY file that knows about both halves. app/ and assets/ are
used exactly as delivered and are not modified; everything needed to make
them run against real NIC/ERA5/CMEMS data instead of a synthetic track is
done from here or inside src/.

WHAT THIS FILE DOES TO BRIDGE THE TWO
=====================================
The frontend was written against the synthetic prototype and holds three
assumptions that real data does not satisfy. Each is handled here rather
than by editing app/:

1. It reads ONE track from the fixed path data/synthetic_track.csv, and
   its iceberg dropdown has a single hard-coded entry. We write one CSV
   per real drifting iceberg, rebind the loader to pick the file matching
   the dropdown value, and fill the dropdown with the real iceberg names.
   The loader rebinding is a deliberate, single-line indirection so that
   app/callbacks.py can stay byte-identical to what was delivered.

2. Its physics-only path calls free_drift_velocity() with no
   coefficients, which would silently use the literature defaults. On
   this record those are worse than predicting nothing (residual RMS
   0.142 m/s against an observed 0.087 m/s), so we install the values
   calibrated on the real data as the module defaults before the app
   starts.

3. It expects the track CSV to carry obs_u/obs_v/residual_u/residual_v
   for the hybrid path's lag features. Those are derived columns, so we
   write the ENRICHED track rather than the raw one.

A note on the forecast cadence: the dashboard steps at 6 hours, while the
NIC record it was trained on is sampled weekly. That is fine and in fact
more useful operationally -- the calibrated drift physics integrates at
any step size -- but it is why the model-error term in the risk score is
expressed per DAY of lead time rather than per forecast step.
"""

from __future__ import annotations

import json
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
# app/callbacks.py resolves data/synthetic_track.csv relative to the
# working directory, so anchor the process here and the app runs the same
# way whichever directory it was launched from.
os.chdir(PROJECT_DIR)
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, PROJECT_DIR)

import pandas as pd
from dash import Dash
import dash_bootstrap_components as dbc

# Backend modules are imported FLAT, matching how they import each other
# ("from physics import ..." inside data_ingest.py, and so on).
import config
import data_ingest
import decision_support
import features
import physics
import train_model

# ...but app/callbacks.py imports them by PACKAGE path ("import
# src.decision_support", "from src.physics import free_drift_velocity").
# Left alone, Python would happily load a SECOND, independent copy of
# each module under the other name. That is not merely wasteful: this
# app configures module-level state -- physics.set_default_drift_params()
# installs the calibrated drift coefficients -- and state set on one copy
# is invisible to the other. The frontend's physics-only path would have
# silently kept using the literature coefficients, which on this dataset
# are worse than predicting nothing.
#
# Aliasing the already-imported flat modules under their src.* names
# makes both spellings resolve to the same object, so there is exactly
# one copy of each and one place for its state to live.
import src as _src_package

for _name, _module in (
    ("config", config),
    ("data_ingest", data_ingest),
    ("decision_support", decision_support),
    ("features", features),
    ("physics", physics),
    ("train_model", train_model),
):
    # sys.modules covers "import src.x" and "from src.x import y";
    # setattr additionally covers attribute access on the package
    # itself, so every spelling reaches the same object.
    sys.modules[f"src.{_name}"] = _module
    setattr(_src_package, _name, _module)

from app.callbacks import register_callbacks
from app.layout import build_layout

TRACK_DIR = os.path.join(PROJECT_DIR, "data", "tracks")
DEFAULT_TRACK_PATH = os.path.join(PROJECT_DIR, "data", "synthetic_track.csv")
METRICS_PATH = os.path.join(config.MODELS_DIR, "frontend_metrics.json")


def _build_backend() -> tuple[pd.DataFrame, dict, physics.DriftParams]:
    """Build the real dataset, calibrate the physics and train the residual model.

    Returns:
        An (enriched_track_table, model_bundle, drift_params) tuple. The
        enriched table is the pooled real record with observed velocity,
        the physics baseline and the residual added -- the columns the
        frontend's hybrid path needs for its lag features.
    """
    pooled, _motion = data_ingest.build_real_dataset(verbose=True)

    with_velocity = features.compute_observed_velocity(pooled)
    params = features.calibrate_drift_params(with_velocity)
    enriched = features.compute_physics_residual(with_velocity, params=params)

    feature_df, feature_cols, target_cols = features.build_feature_table(pooled, params=params)
    bundle = train_model.train_residual_model(
        feature_df, feature_cols, target_cols, drift_params=params
    )
    return enriched, bundle, params


def _write_track_csvs(enriched: pd.DataFrame) -> list[str]:
    """Write one enriched CSV per iceberg for the frontend's track loader.

    Args:
        enriched: The pooled enriched track table.

    Returns:
        The sorted list of iceberg ids written.
    """
    os.makedirs(TRACK_DIR, exist_ok=True)
    iceberg_ids = sorted(enriched["iceberg_id"].unique())
    for iceberg_id in iceberg_ids:
        track = enriched[enriched["iceberg_id"] == iceberg_id].sort_values("timestamp")
        track.to_csv(os.path.join(TRACK_DIR, f"{iceberg_id}.csv"), index=False)

    # The frontend's hard-coded default path still has to resolve, so
    # point it at the fastest-drifting berg -- the most interesting one
    # to land on.
    speeds = enriched.groupby("iceberg_id").apply(
        lambda g: float((g["obs_u"] ** 2 + g["obs_v"] ** 2).mean() ** 0.5), include_groups=False
    )
    default_id = str(speeds.idxmax())
    enriched[enriched["iceberg_id"] == default_id].sort_values("timestamp").to_csv(
        DEFAULT_TRACK_PATH, index=False
    )
    return iceberg_ids, default_id


def _bind_track_loader() -> None:
    """Point the frontend's track loader at the per-iceberg CSVs.

    app/callbacks.py ships a `_load_track` that ignores its iceberg
    argument and always reads one fixed path, because the prototype only
    ever had one synthetic track. Rebinding it here gives the dropdown
    real effect while leaving app/callbacks.py exactly as delivered.

    Returns:
        None.
    """
    import app.callbacks as callbacks

    def load_track(iceberg_id: str) -> pd.DataFrame:
        path = os.path.join(TRACK_DIR, f"{iceberg_id}.csv")
        if not os.path.exists(path):
            path = DEFAULT_TRACK_PATH
        return pd.read_csv(path, parse_dates=["timestamp"])

    callbacks._load_track = load_track


def _find_component(node, component_id: str):
    """Depth-first search a Dash layout tree for a component by id.

    Args:
        node: A Dash component or a list/tuple of them.
        component_id: The id to look for.

    Returns:
        The matching component, or None.
    """
    if isinstance(node, (list, tuple)):
        for child in node:
            found = _find_component(child, component_id)
            if found is not None:
                return found
        return None
    if getattr(node, "id", None) == component_id:
        return node
    children = getattr(node, "children", None)
    return _find_component(children, component_id) if children is not None else None


def _compute_metrics(pooled: pd.DataFrame, force: bool = False) -> dict:
    """Produce the diagnostics-chart numbers, caching them between runs.

    The chart compares final displacement error across the three forecast
    modes. The numbers come from the leave-one-iceberg-out evaluation --
    the honest one, where each berg is forecast by a model that never saw
    it -- rather than the flattering in-sample figure. It takes ~30 s to
    compute, so it is cached to models/frontend_metrics.json.

    Args:
        pooled: The pooled real track table.
        force: Recompute even if a cached file exists.

    Returns:
        A dict of mode label -> final displacement error in km, ordered
        so the chart reads worst to best.
    """
    if os.path.exists(METRICS_PATH) and not force:
        with open(METRICS_PATH) as handle:
            return json.load(handle)

    _per_iceberg, aggregate = train_model.leave_one_iceberg_out(pooled, verbose=False)
    metrics = {
        "persistence": round(aggregate["persistence"]["fde_km"], 1),
        "physics_only": round(aggregate["physics"]["fde_km"], 1),
        "hybrid": round(aggregate["hybrid"]["fde_km"], 1),
    }
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    with open(METRICS_PATH, "w") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


def create_app() -> Dash:
    """Build the Dash app wired to the real-data backend.

    Returns:
        A configured Dash application.
    """
    enriched, bundle, params = _build_backend()

    # Make the calibrated coefficients the defaults, so the frontend's
    # physics-only path -- which passes none -- uses them too.
    physics.set_default_drift_params(params)

    iceberg_ids, default_id = _write_track_csvs(enriched)
    _bind_track_loader()

    pooled = enriched[data_ingest.POOLED_SCHEMA_COLUMNS]
    metrics = _compute_metrics(pooled)

    print(
        f"\n[app] {len(iceberg_ids)} real drifting icebergs available: {', '.join(iceberg_ids)}\n"
        f"[app] default selection: {default_id}\n"
        f"[app] calibrated physics: {params}\n"
        f"[app] held-out FDE (km): {metrics}\n"
    )

    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.title = "Iceberg Trajectory Prediction"

    layout = build_layout()
    # Fill the dropdown, which ships with a single hard-coded synthetic
    # entry, with the real icebergs. Done on the built tree so
    # app/layout.py stays untouched.
    dropdown = _find_component(layout, "iceberg-select")
    if dropdown is not None:
        dropdown.options = [{"label": i, "value": i} for i in iceberg_ids]
        dropdown.value = default_id
    app.layout = layout

    register_callbacks(app, bundle, metrics)
    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
