"""Entry point: wires together data generation/training and the Dash app."""
from __future__ import annotations

import os
import sys

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, SRC_DIR)

import pandas as pd
from dash import Dash
import dash_bootstrap_components as dbc

from app.layout import build_layout
from app.callbacks import register_callbacks

# Flat imports, matching the style used *inside* src/*.py itself
# (physics.py, features.py, data_ingest.py, train_model.py, and
# decision_support.py all import each other flatly, e.g.
# "from physics import ..."). Importing them as "src.train_model"
# instead loads a SECOND copy of the same file under a different
# module name -- and if that happens while decision_support.py's own
# "from train_model import predict_residual" is still running, you get
# exactly the circular "cannot import name X" error you just hit.
# Staying flat everywhere means each file loads exactly once.
import src.data_ingest
import src.features
import src.train_model

DATA_PATH = "data/synthetic_track.csv"
WINDOW_SIZE = 5  # must match app/callbacks.py's WINDOW_SIZE


def _get_or_create_track() -> pd.DataFrame:
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    track = src.data_ingest.generate_synthetic_track(n_steps=400, dt_hours=6, seed=42)
    os.makedirs("data", exist_ok=True)
    track.to_csv(DATA_PATH, index=False)
    return track


def _train_model_and_metrics() -> tuple[dict, dict]:
    track = _get_or_create_track()
    feature_df, feature_cols, target_cols = src.features.build_sliding_window_features(
        track, window_size=WINDOW_SIZE
    )
    train_df, test_df = src.train_model.train_test_split_by_time(feature_df, test_fraction=0.2)

    cutoff_time = test_df["timestamp"].iloc[0]
    test_track_df = track[track["timestamp"] >= cutoff_time].reset_index(drop=True)

    model = src.train_model.train_residual_model(train_df, feature_cols, target_cols)

    horizon_steps = 4
    baseline = src.train_model.evaluate_trajectory(None, test_track_df, feature_cols, horizon_steps)
    hybrid = src.train_model.evaluate_trajectory(model, test_track_df, feature_cols, horizon_steps)

    metrics = {"physics_only": baseline["fde_km"], "hybrid": hybrid["fde_km"]}
    return model, metrics


def create_app() -> Dash:
    model, metrics = _train_model_and_metrics()

    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.title = "Iceberg Trajectory Prediction"
    app.layout = build_layout()
    register_callbacks(app, model, metrics)
    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
