"""Entry point: wires the real-data backend to the Dash frontend.

    python main.py

This is the ONLY file that knows about both halves. app/ and assets/ are
used exactly as delivered and are not modified; everything needed to run
them against the real BYU/ERA5/CMEMS pipeline -- including the
multi-iceberg selection, the metadata panel and the SHAP explanation
panel -- is done from here or inside src/.

HOW THE FRONTEND IS EXTENDED WITHOUT EDITING IT
===============================================
Dash layouts are ordinary Python object trees and callbacks are
registered against ids, so both can be extended after app/layout.py has
built its tree:

  * The iceberg dropdown is switched to multi-select and its options
    filled with the real icebergs (it ships with one hard-coded entry).
  * Two panels are appended to the layout -- iceberg metadata and the
    model explanation -- reusing the stylesheet's existing classes so
    they look native rather than bolted on.
  * New callbacks for those panels are registered alongside the ones
    app/callbacks.py registers. No existing callback is replaced, which
    matters because Dash forbids two callbacks writing the same output.

Three assumptions in app/callbacks.py do not survive contact with real
data, and are handled here rather than by editing it:

1. It reads ONE track from a fixed path and its dropdown has a single
   hard-coded value. `_load_track` is rebound to serve any selection.
   With several icebergs selected it returns their tracks concatenated
   with NaN separator rows, which is what makes Plotly draw separate
   polylines from a single trace. The PRIMARY iceberg (first selected)
   is placed last, so the frontend's `track.iloc[-1]` -- which it uses
   as the forecast origin -- still lands on a real fix.

2. Its physics-only path calls free_drift_velocity() with no
   coefficients, so it would silently use the literature defaults. On
   this record those are worse than predicting nothing, so the fitted
   values are installed as module defaults before the app starts.

3. It expects the track CSV to carry obs_u/obs_v/residual_u/residual_v
   for the hybrid path's lag features, so the ENRICHED track is written
   rather than the raw one.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
# app/callbacks.py resolves data/synthetic_track.csv relative to the
# working directory, so anchor the process here and the app runs the same
# way whichever directory it was launched from.
os.chdir(PROJECT_DIR)
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, PROJECT_DIR)

import numpy as np
import pandas as pd
import dash
from dash import Dash, Input, Output, State, ctx, dcc, html
import dash_bootstrap_components as dbc

# Backend modules are imported FLAT, matching how they import each other
# ("from physics import ..." inside data_ingest.py, and so on).
import config
import data_ingest
import decision_support
import explain
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
# silently kept using the literature coefficients.
#
# Aliasing the already-imported flat modules under their src.* names
# makes both spellings resolve to the same object.
import src as _src_package

for _name, _module in (
    ("config", config),
    ("data_ingest", data_ingest),
    ("decision_support", decision_support),
    ("explain", explain),
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
from app.landing import build_shell
from app.layout import build_layout

TRACK_DIR = os.path.join(PROJECT_DIR, "data", "tracks")
DEFAULT_TRACK_PATH = os.path.join(PROJECT_DIR, "data", "synthetic_track.csv")
METRICS_PATH = os.path.join(config.MODELS_DIR, "frontend_metrics.json")

# ---------------------------------------------------------------------
# WHICH ICEBERGS ARE SELECTED WHEN THE APP OPENS
# ---------------------------------------------------------------------
# Name them explicitly here, e.g. ["D33A", "D33C", "A81"], and exactly
# those are ticked on load. Ids are the iceberg names as they appear in
# the picker (case-insensitive here); run the app once and read the
# "[app] icebergs available:" line for the full list.
#
# Leave it empty to let the app choose: it then picks the tightest
# geographic cluster of DRIFTING icebergs (see _default_cluster), which
# keeps the opening map framed on something sensible instead of on
# bergs scattered around the continent.
DEFAULT_ICEBERGS: list[str] = ["D33C", "A81"]

# How many to pick automatically when DEFAULT_ICEBERGS is empty.
DEFAULT_SELECTION_SIZE = 40


def _build_backend() -> tuple[pd.DataFrame, pd.DataFrame, dict, physics.DriftParams, pd.DataFrame]:
    """Build the real dataset, calibrate the physics and train the residual model.

    DISPLAY AND TRAINING USE DIFFERENT SUBSETS, deliberately. The dataset
    is built with include_grounded=True so that EVERY tracked iceberg is
    available in the dashboard -- a grounded berg is still a hazard to a
    vessel and still belongs on the map. The physics calibration and the
    residual model are then fitted on the DRIFTING subset only, because
    a grounded berg's apparent motion is position noise and fitting to it
    corrupts the drift model (see data_ingest.summarize_iceberg_motion).

    Returns:
        An (enriched_all, motion, model_bundle, drift_params,
        feature_table) tuple. enriched_all covers every iceberg, with
        observed velocity, the physics baseline and the residual added.
    """
    pooled, motion = data_ingest.build_real_dataset(include_grounded=True, verbose=True)

    drifting = set(motion.loc[~motion["is_grounded"], "iceberg_id"])
    training_pooled = pooled[pooled["iceberg_id"].isin(drifting)]

    # Calibrate and fit on drifting icebergs only...
    params = features.calibrate_drift_params(
        features.compute_observed_velocity(training_pooled)
    )
    feature_df, feature_cols, target_cols = features.build_feature_table(
        training_pooled, params=params
    )
    bundle = train_model.train_residual_model(
        feature_df, feature_cols, target_cols, drift_params=params
    )

    # ...but enrich every iceberg, so all of them can be displayed and
    # forecast with the fitted physics.
    enriched_all = features.compute_physics_residual(
        features.compute_observed_velocity(pooled), params=params
    )
    print(
        f"[app] {pooled['iceberg_id'].nunique()} icebergs available for display; "
        f"{len(drifting)} drifting ones used for calibration and training"
    )
    return enriched_all, motion, bundle, params, feature_df


def _write_track_csvs(enriched: pd.DataFrame) -> tuple[list[str], str]:
    """Write one enriched CSV per iceberg for the frontend's track loader.

    Args:
        enriched: The pooled enriched track table.

    Returns:
        An (iceberg_ids, default_id) tuple, the default being the
        fastest-drifting iceberg -- the most interesting one to land on.
    """
    os.makedirs(TRACK_DIR, exist_ok=True)
    iceberg_ids = sorted(enriched["iceberg_id"].unique())
    for iceberg_id in iceberg_ids:
        track = enriched[enriched["iceberg_id"] == iceberg_id].sort_values("timestamp")
        track.to_csv(os.path.join(TRACK_DIR, f"{iceberg_id}.csv"), index=False)

    speeds = enriched.groupby("iceberg_id").apply(
        lambda g: float(np.hypot(g["obs_u"], g["obs_v"]).mean()), include_groups=False
    )
    default_id = str(speeds.idxmax())
    enriched[enriched["iceberg_id"] == default_id].sort_values("timestamp").to_csv(
        DEFAULT_TRACK_PATH, index=False
    )
    return iceberg_ids, default_id


def _read_track(iceberg_id: str) -> pd.DataFrame:
    """Read one iceberg's enriched track from disk.

    Args:
        iceberg_id: The iceberg to load.

    Returns:
        Its track DataFrame, or an empty one if the file is absent.
    """
    path = os.path.join(TRACK_DIR, f"{iceberg_id}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["timestamp"])


def _bind_track_loader(default_id: str) -> None:
    """Point the frontend's track loader at the per-iceberg CSVs.

    app/callbacks.py ships a `_load_track` that ignores its argument and
    always reads one fixed path, because the prototype only ever had one
    synthetic track. Rebinding it here gives the dropdown real effect --
    including multi-select -- while leaving app/callbacks.py exactly as
    delivered.

    For a multi-iceberg selection the tracks are concatenated with a
    single all-NaN row between them. Plotly breaks a line at NaN, so one
    scatter trace renders as one polyline per iceberg, and the frozen
    `build_map_figure` draws a fleet without knowing it. The PRIMARY
    (first-selected) iceberg goes last so that the frontend's
    `track.iloc[-1]`, which it uses as the forecast origin and for the
    stand-in environmental forecast, still lands on a real fix.

    Args:
        default_id: Iceberg to fall back to when the selection is empty.

    Returns:
        None.
    """
    import app.callbacks as callbacks

    def load_track(selection) -> pd.DataFrame:
        if selection is None or (isinstance(selection, list) and not selection):
            selection = [default_id]
        if not isinstance(selection, list):
            selection = [selection]

        primary, others = selection[0], selection[1:]
        frames: list[pd.DataFrame] = []
        # Secondary icebergs first, primary last -- see the docstring.
        for iceberg_id in others:
            track = _read_track(iceberg_id)
            if track.empty:
                continue
            frames.append(track)
            separator = pd.DataFrame([{c: np.nan for c in track.columns}])
            separator["iceberg_id"] = None
            frames.append(separator)

        primary_track = _read_track(primary)
        if primary_track.empty:
            primary_track = _read_track(default_id)
        frames.append(primary_track)

        combined = pd.concat(frames, ignore_index=True)
        # Carry the per-iceberg frames on the result so the forecast
        # stage can roll each one out separately. Using .attrs keeps this
        # request-local -- no module-level "current selection" that two
        # concurrent browser sessions could race on.
        tracks: dict[str, pd.DataFrame] = {}
        for iceberg_id in [primary] + others:
            track = _read_track(iceberg_id)
            if not track.empty:
                tracks[iceberg_id] = track
        combined.attrs["tracks"] = tracks
        return combined

    callbacks._load_track = load_track



# Label style for the per-iceberg tag in the forecast list. The
# delivered .iw-row-idx class fixes the width at 20px for a two-digit
# step number; an iceberg name needs to size to its content instead, and
# must not shrink, or it overlaps the coordinates in the same flex row.
_ROW_LABEL_STYLE: dict[str, str] = {
    "fontFamily": "JetBrains Mono, ui-monospace, monospace",
    "fontSize": "11.5px",
    "fontWeight": "600",
    "color": "var(--ink-soft)",
    "flex": "0 0 auto",
    "width": "auto",
    "minWidth": "46px",
    "whiteSpace": "nowrap",
}


def _separator_row(columns) -> pd.DataFrame:
    """Build the all-NaN row that separates two icebergs' polylines.

    Args:
        columns: Column names the row must carry.

    Returns:
        A one-row DataFrame of NaN.
    """
    return pd.DataFrame([{c: np.nan for c in columns}])


def _bind_multi_forecast(default_id: str) -> None:
    """Make the frontend forecast EVERY selected iceberg, not just the first.

    app/callbacks.py computes one forecast, from `track.iloc[-1]`,
    because the prototype only ever had one track. Rather than edit it,
    the three helpers it calls are rebound so that each runs once per
    selected iceberg and returns the results concatenated -- with the
    same NaN separator rows that make the map draw one polyline per
    iceberg.

    The per-iceberg data travels on the environmental-forecast frame's
    `.attrs`, which app/callbacks.py passes straight through from
    `_build_future_environmental_forecast` to the rollout. That keeps the
    whole mechanism request-local instead of stashing the selection in a
    module-level variable, which two browser sessions would race on.

    Args:
        default_id: Fallback iceberg id.

    Returns:
        None.
    """
    import app.callbacks as callbacks

    original_env = callbacks._build_future_environmental_forecast
    original_physics = callbacks._physics_only_rollout
    original_bootstrap = callbacks._physics_only_bootstrap

    # The new environmental forecast hits a live weather API per call.
    # With several icebergs selected that is one request each, and the
    # map re-renders on every control change, so identical requests are
    # memoised briefly. The TTL keeps "live forecast" honest while
    # stopping a slider drag from firing dozens of requests.
    forecast_cache: dict[tuple, tuple[float, pd.DataFrame]] = {}
    CACHE_TTL_SECONDS = 600.0

    def cached_env(frame: pd.DataFrame, n_steps: int) -> pd.DataFrame:
        last = frame.iloc[-1]
        key = (round(float(last["lat"]), 2), round(float(last["lon"]), 2), int(n_steps))
        hit = forecast_cache.get(key)
        now = time.monotonic()
        if hit is not None and (now - hit[0]) < CACHE_TTL_SECONDS:
            return hit[1].copy()
        env = original_env(frame, n_steps)
        forecast_cache[key] = (now, env.copy())
        return env

    def build_env(track: pd.DataFrame, n_steps: int) -> pd.DataFrame:
        tracks = track.attrs.get("tracks") or {}
        if not tracks:
            return cached_env(track, n_steps)
        envs = {berg: cached_env(frame, n_steps) for berg, frame in tracks.items()}
        primary = next(iter(tracks))
        env = envs[primary].copy()
        env.attrs["tracks"] = tracks
        env.attrs["envs"] = envs
        return env

    def _combine(forecasts: dict[str, pd.DataFrame]) -> pd.DataFrame:
        pieces: list[pd.DataFrame] = []
        for berg, forecast in forecasts.items():
            piece = forecast.copy()
            piece["iceberg_id"] = berg
            pieces.append(piece)
            pieces.append(_separator_row(piece.columns))
        return pd.concat(pieces[:-1], ignore_index=True)

    def physics_rollout(last_known_row, future_env, dt_seconds):
        tracks = future_env.attrs.get("tracks") or {}
        envs = future_env.attrs.get("envs") or {}
        if len(tracks) <= 1:
            return original_physics(last_known_row, future_env, dt_seconds)
        return _combine(
            {
                berg: original_physics(frame.iloc[-1], envs[berg], dt_seconds)
                for berg, frame in tracks.items()
            }
        )

    class _MultiIcebergDecisionSupport:
        """Proxies decision_support, forecasting every selected iceberg.

        Installed only into app/callbacks.py's own namespace, so the
        backend module itself keeps its single-track behaviour for every
        other caller.
        """

        def __getattr__(self, name):
            return getattr(decision_support, name)

        def rollout_forecast(
            self, models, last_known, history_window=None, future_environment=None,
            feature_cols=None, dt_seconds=None, **kwargs
        ):
            tracks = (future_environment.attrs.get("tracks") or {}
                      if future_environment is not None else {})
            envs = (future_environment.attrs.get("envs") or {}
                    if future_environment is not None else {})
            if len(tracks) <= 1:
                return decision_support.rollout_forecast(
                    models, last_known, history_window, future_environment,
                    feature_cols, dt_seconds, **kwargs
                )

            window = len(history_window) if history_window is not None else callbacks.WINDOW_SIZE
            forecasts = {}
            for berg, frame in tracks.items():
                history = frame.iloc[-window:][decision_support.LAG_BASE_COLUMNS]
                forecasts[berg] = decision_support.rollout_forecast(
                    models, frame.iloc[-1], history.reset_index(drop=True),
                    envs[berg], feature_cols, dt_seconds, **kwargs
                )
            return _combine(forecasts)

        def bootstrap_uncertainty_cone(
            self, models, last_known, history_window=None, future_environment=None,
            feature_cols=None, dt_seconds=None, **kwargs
        ):
            tracks = (future_environment.attrs.get("tracks") or {}
                      if future_environment is not None else {})
            envs = (future_environment.attrs.get("envs") or {}
                    if future_environment is not None else {})
            if len(tracks) <= 1:
                return decision_support.bootstrap_uncertainty_cone(
                    models, last_known, history_window, future_environment,
                    feature_cols, dt_seconds, **kwargs
                )
            window = len(history_window) if history_window is not None else callbacks.WINDOW_SIZE
            samples: list[pd.DataFrame] = []
            for berg, frame in tracks.items():
                history = frame.iloc[-window:][decision_support.LAG_BASE_COLUMNS]
                samples.extend(
                    decision_support.bootstrap_uncertainty_cone(
                        models, frame.iloc[-1], history.reset_index(drop=True),
                        envs[berg], feature_cols, dt_seconds, **kwargs
                    )
                )
            return samples

    def build_list_rows(forecast, last_known_row, active_step, vessel_lat, vessel_lon):
        """Row list that labels each entry with its iceberg and skips separators."""
        from physics import geodesic_distance_km

        rows = []
        previous: dict[str, tuple[float, float]] = {}
        for index, record in enumerate(forecast.itertuples()):
            berg = getattr(record, "iceberg_id", None)
            if record.lat is None or (isinstance(record.lat, float) and np.isnan(record.lat)):
                continue  # separator row
            has_berg = bool(berg) and not pd.isna(berg)
            label = str(berg) if has_berg else f"{len(rows) + 1:02d}"
            prev = previous.get(label, (last_known_row["lat"], last_known_row["lon"]))
            step_km = geodesic_distance_km(prev[0], prev[1], record.lat, record.lon)
            previous[label] = (record.lat, record.lon)

            if vessel_lat is not None and vessel_lon is not None:
                distance = geodesic_distance_km(vessel_lat, vessel_lon, record.lat, record.lon)
                level = "red" if distance < 10 else ("amber" if distance < 30 else "green")
                pill = html.Span(
                    {"red": "DANGER", "amber": "WATCH", "green": "CLEAR"}[level],
                    className=f"iw-pill {level}",
                )
                meta = f"{record.timestamp} · {distance:.1f} km from vessel"
            else:
                pill = html.Span("FORECAST", className="iw-pill grey")
                meta = f"{record.timestamp} · step {step_km:.1f} km"

            rows.append(
                html.Div(
                    className=f"iw-row{' active' if index == active_step else ''}",
                    children=[
                        html.Div(
                            className="iw-row-left",
                            children=[
                                # The stylesheet's .iw-row-idx is a hard
                                # 20px wide, sized for a two-digit index.
                                # An iceberg name is longer and would spill
                                # over the coordinates beside it, so names
                                # get their own auto-width style and only
                                # the numeric fallback keeps the class.
                                html.Span(label, className="iw-row-idx")
                                if not has_berg
                                else html.Span(label, style=_ROW_LABEL_STYLE),
                                html.Div(
                                    [
                                        html.Div(f"{record.lat:.3f}, {record.lon:.3f}",
                                                 className="iw-row-title"),
                                        html.Div(meta, className="iw-row-meta"),
                                    ]
                                ),
                            ],
                        ),
                        html.Div(className="iw-row-right", children=[pill]),
                    ],
                )
            )
        return rows

    def physics_bootstrap(last_known_row, future_env, dt_seconds, **kwargs):
        """Physics-only uncertainty cone across every selected iceberg."""
        tracks = future_env.attrs.get("tracks") or {}
        envs = future_env.attrs.get("envs") or {}
        if len(tracks) <= 1:
            return original_bootstrap(last_known_row, future_env, dt_seconds, **kwargs)
        # One cone per iceberg, flattened: the heatmap bins all sample
        # points together, so a longer list of per-iceberg samples gives
        # exactly the right combined spread.
        samples: list[pd.DataFrame] = []
        for berg, frame in tracks.items():
            samples.extend(
                original_bootstrap(frame.iloc[-1], envs[berg], dt_seconds, **kwargs)
            )
        return samples

    callbacks._build_future_environmental_forecast = build_env
    callbacks._physics_only_rollout = physics_rollout
    callbacks._physics_only_bootstrap = physics_bootstrap
    callbacks._build_list_rows = build_list_rows
    callbacks.decision_support = _MultiIcebergDecisionSupport()


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


def _metadata_table(enriched: pd.DataFrame, motion: pd.DataFrame) -> pd.DataFrame:
    """Assemble the per-iceberg metadata shown in the dashboard.

    Combines what the BYU database reports directly (length, width,
    area, tracking sensor) with what only the forced track can say
    (mean and peak drift speed, distance travelled, observation span).

    Args:
        enriched: The pooled enriched track table.
        motion: The motion-classification summary.

    Returns:
        A DataFrame indexed by iceberg_id with one row per iceberg.
    """
    from physics import geodesic_distance_km

    rows: list[dict[str, object]] = []
    for iceberg_id, group in enriched.groupby("iceberg_id"):
        track = group.sort_values("timestamp")
        speeds = np.hypot(track["obs_u"], track["obs_v"])
        lats, lons = track["lat"].to_numpy(), track["lon"].to_numpy()
        distance_km = sum(
            geodesic_distance_km(lats[i], lons[i], lats[i + 1], lons[i + 1])
            for i in range(len(lats) - 1)
        )
        row: dict[str, object] = {
            "iceberg_id": iceberg_id,
            "n_fixes": len(track),
            "first_seen": track["timestamp"].min(),
            "last_seen": track["timestamp"].max(),
            "lat": float(track["lat"].iloc[-1]),
            "lon": float(track["lon"].iloc[-1]),
            "area_km2": float(track["area_km2"].iloc[-1]),
            "mean_speed_ms": float(speeds.mean()),
            "max_speed_ms": float(speeds.max()),
            "distance_km": float(distance_km),
        }
        for column in ("length_km", "width_km", "sensor"):
            if column in track.columns:
                values = track[column].dropna()
                row[column] = values.iloc[-1] if len(values) else None
        rows.append(row)

    table = pd.DataFrame(rows).set_index("iceberg_id")
    if "is_grounded" in motion.columns:
        table = table.join(motion.set_index("iceberg_id")[["is_grounded"]], how="left")
    return table



# Styling for the horizontal iceberg picker. Kept inline rather than in
# assets/style.css so the delivered stylesheet stays untouched; the
# colours are read from the stylesheet's own CSS variables so the chips
# match the theme rather than approximating it.
_CHIP_STYLE: dict[str, str] = {
    "display": "inline-flex",
    "alignItems": "center",
    "gap": "6px",
    "padding": "5px 11px",
    "marginRight": "0px",
    "border": "1px solid var(--line, #E4E1D8)",
    "borderRadius": "999px",
    "background": "var(--panel, #FFFFFF)",
    "fontFamily": "JetBrains Mono, ui-monospace, monospace",
    "fontSize": "12.5px",
    "fontWeight": "600",
    "cursor": "pointer",
    "userSelect": "none",
    "whiteSpace": "nowrap",
}

_PICKER_STYLE: dict[str, str] = {
    "display": "flex",
    "flexWrap": "wrap",
    "gap": "7px",
    "alignItems": "center",
    "maxHeight": "132px",
    "overflowY": "auto",
    "padding": "4px 2px",
}


def _build_selector(iceberg_ids: list[str], selection: list[str], metadata: pd.DataFrame):
    """Build the horizontal iceberg picker that replaces the dropdown.

    app/layout.py ships a `dcc.Dropdown` whose menu opens as a tall
    vertical list -- workable for one iceberg, poor for picking several
    out of twenty-odd. This swaps in a `dcc.Checklist` with `inline=True`,
    rendered as a wrapping row of chips.

    It keeps the SAME component id and the same value semantics (a list
    of iceberg ids), so every callback in app/callbacks.py continues to
    work against it untouched.

    Grounded icebergs are marked with a bullet in their label. They stay
    selectable -- a grounded berg is still a hazard worth plotting -- but
    the marker warns that its track is not drift and its forecast is
    therefore close to stationary.

    Args:
        iceberg_ids: Every selectable iceberg.
        selection: Icebergs selected initially.
        metadata: Metadata table, used for the grounded markers.

    Returns:
        A dcc.Checklist component.
    """
    grounded = set()
    if "is_grounded" in metadata.columns:
        grounded = set(metadata.index[metadata["is_grounded"].fillna(False)])

    options = [
        {"label": f"{berg}\u2009\u2022" if berg in grounded else berg, "value": berg}
        for berg in iceberg_ids
    ]
    return dcc.Checklist(
        id="iceberg-select",
        options=options,
        value=selection,
        inline=True,
        style=_PICKER_STYLE,
        labelStyle=_CHIP_STYLE,
        inputStyle={"marginRight": "2px", "accentColor": "var(--green-deep, #1B7A3D)"},
    )


def _replace_component(node, component_id: str, replacement) -> bool:
    """Swap a component in a Dash layout tree for another, in place.

    Args:
        node: A component or list of components to search.
        component_id: Id of the component to replace.
        replacement: The component to put in its place.

    Returns:
        True if a replacement was made.
    """
    children = getattr(node, "children", None)
    if isinstance(children, list):
        for index, child in enumerate(children):
            if getattr(child, "id", None) == component_id:
                children[index] = replacement
                return True
            if _replace_component(child, component_id, replacement):
                return True
    elif children is not None:
        if getattr(children, "id", None) == component_id:
            node.children = replacement
            return True
        return _replace_component(children, component_id, replacement)
    return False


def _resolve_default_selection(
    requested: list[str], available: list[str], metadata: pd.DataFrame
) -> list[str]:
    """Work out which icebergs are ticked when the app opens.

    Args:
        requested: Names from DEFAULT_ICEBERGS; empty means "choose one".
        available: Every selectable iceberg id.
        metadata: Per-iceberg metadata, used for the automatic choice.

    Returns:
        The iceberg ids to preselect, in the requested order.
    """
    if not requested:
        return _default_cluster(metadata, DEFAULT_SELECTION_SIZE)

    lookup = {berg.upper(): berg for berg in available}
    selection = [lookup[name.upper()] for name in requested if name.upper() in lookup]

    unknown = [name for name in requested if name.upper() not in lookup]
    if unknown:
        # Name it rather than silently dropping it: a typo in
        # DEFAULT_ICEBERGS would otherwise look like the app quietly
        # ignoring the setting.
        warnings.warn(
            f"DEFAULT_ICEBERGS lists {unknown}, which are not in this dataset. "
            f"Available ids: {', '.join(available)}",
            stacklevel=2,
        )
    if not selection:
        warnings.warn(
            "None of DEFAULT_ICEBERGS matched; falling back to the automatic choice.",
            stacklevel=2,
        )
        return _default_cluster(metadata, DEFAULT_SELECTION_SIZE)
    return selection


def _default_cluster(table: pd.DataFrame, count: int) -> list[str]:
    """Choose a default selection of icebergs that are actually near each other.

    The frozen build_map_figure() centres on the primary iceberg at a
    FIXED zoom, so the default selection has to be geographically tight
    or the map opens on empty ocean with the other bergs off-screen.
    Rather than starting from the fastest berg and taking its nearest
    neighbours -- which in this record are still several hundred km away
    -- this picks the tightest cluster in the fleet: the berg whose
    (count - 1) nearest neighbours are closest overall.

    A tight cluster is also the operationally interesting case, since
    what threatens a vessel is the several bergs near it rather than the
    fastest one somewhere else.

    Args:
        table: The metadata table, carrying each iceberg's last position.
        count: Number of icebergs to select.

    Returns:
        A list of iceberg ids, the cluster centre first.
    """
    from physics import geodesic_distance_km

    # Consider only drifting icebergs as cluster candidates. Grounded
    # bergs sit still, so they form the TIGHTEST clusters in the fleet by
    # construction -- and opening the dashboard on three stationary bergs
    # with a flat forecast is exactly the wrong first impression. They
    # remain selectable, just not the default.
    candidates = table
    if "is_grounded" in table.columns:
        drifting = table[~table["is_grounded"].fillna(False)]
        if len(drifting) >= count:
            candidates = drifting

    positions = {
        iceberg_id: (float(row["lat"]), float(row["lon"]))
        for iceberg_id, row in candidates.iterrows()
    }
    if len(positions) <= count:
        return list(positions)

    best: tuple[float, list[str]] | None = None
    for centre, (lat, lon) in positions.items():
        others = sorted(
            ((geodesic_distance_km(lat, lon, *pos), berg)
             for berg, pos in positions.items() if berg != centre)
        )
        neighbours = others[: count - 1]
        spread = sum(distance for distance, _ in neighbours)
        if best is None or spread < best[0]:
            best = (spread, [centre] + [berg for _, berg in neighbours])
    return best[1]


def _metadata_rows(table: pd.DataFrame, selection: list[str], primary: str) -> list:
    """Render metadata for the selected icebergs as themed list rows.

    Args:
        table: The metadata table from _metadata_table().
        selection: Selected iceberg ids.
        primary: The iceberg the forecast is being computed for.

    Returns:
        A list of Dash components using the stylesheet's row classes.
    """
    rows = []
    for index, iceberg_id in enumerate(selection):
        if iceberg_id not in table.index:
            continue
        meta = table.loc[iceberg_id]

        size = "size unknown"
        if meta.get("length_km") is not None and pd.notna(meta.get("length_km")):
            size = f"{meta['length_km']:.0f} x {meta['width_km']:.0f} km"
        detail = (
            f"{size} · {meta['area_km2']:.0f} km² · "
            f"drift {meta['mean_speed_ms']:.3f} m/s (peak {meta['max_speed_ms']:.3f}) · "
            f"{meta['distance_km']:.0f} km travelled · {meta['n_fixes']} fixes "
            f"{meta['first_seen']:%d %b} – {meta['last_seen']:%d %b %Y}"
        )
        if meta.get("sensor"):
            detail += f" · {meta['sensor']}"

        # Every selected iceberg is forecast; the primary is additionally
        # the anchor for the forecast list ordering and the map centre.
        if meta.get("is_grounded"):
            pill = html.Span("GROUNDED", className="iw-pill amber")
        elif iceberg_id == primary:
            pill = html.Span("PRIMARY", className="iw-pill green")
        else:
            pill = html.Span("FORECAST", className="iw-pill green")

        rows.append(
            html.Div(
                className=f"iw-row{' active' if iceberg_id == primary else ''}",
                children=[
                    html.Div(
                        className="iw-row-left",
                        children=[
                            html.Span(f"{index + 1:02d}", className="iw-row-idx"),
                            html.Div(
                                [
                                    html.Div(
                                        f"{iceberg_id}  ({meta['lat']:.2f}, {meta['lon']:.2f})",
                                        className="iw-row-title",
                                    ),
                                    html.Div(detail, className="iw-row-meta"),
                                ]
                            ),
                        ],
                    ),
                    html.Div(className="iw-row-right", children=[pill]),
                ],
            )
        )
    return rows


def _append_panels(layout) -> None:
    """Append the metadata and explanation panels to the built layout.

    Args:
        layout: The tree returned by app.layout.build_layout().

    Returns:
        None. The tree is modified in place.
    """
    def section(title: str, body) -> html.Div:
        return html.Div(
            className="iw-section",
            children=[
                html.Div(title, className="iw-section-title"),
                html.Div(className="iw-panel", children=[body]),
            ],
        )

    panels = [
        # Invisible: drives the map-repaint nudge below.
        dcc.Interval(id="iw-map-nudge", interval=1600, max_intervals=3),
        html.Div(id="iw-map-nudge-sink", style={"display": "none"}),
        section("Iceberg metadata", html.Div(id="iw-metadata", className="iw-list")),
        section(
            "Why the model corrected the physics",
            html.Div(
                [
                    html.Div(id="iw-explain-summary", className="iw-row-title",
                             style={"padding": "14px 16px 4px"}),
                    html.Div(id="iw-explain-rows", className="iw-list"),
                ]
            ),
        ),
    ]
    # Insert before the trailing dcc.Store components so the stores stay
    # at the end of the tree, where Dash's own docs put them.
    stores = [c for c in layout.children if getattr(c, "id", None) in
              {"forecast-store", "fullscreen-state"}]
    body = [c for c in layout.children if c not in stores]
    layout.children = body + panels + stores


def _register_extra_callbacks(
    app: Dash,
    metadata: pd.DataFrame,
    bundle: dict,
    feature_df: pd.DataFrame,
    default_id: str,
) -> None:
    """Register the callbacks driving the two added panels.

    These write only to ids introduced by _append_panels(), so they
    coexist with app/callbacks.py rather than competing with it -- Dash
    rejects two callbacks writing the same output.

    Args:
        app: The Dash app.
        metadata: Per-iceberg metadata table.
        bundle: The trained model bundle.
        feature_df: The feature table, used to look up a representative
            row per iceberg for the explanation.
        default_id: Fallback iceberg id.

    Returns:
        None.
    """
    try:
        explainer = explain.ResidualExplainer(bundle)
    except (RuntimeError, TypeError) as exc:
        explainer = None
        explain_error = str(exc)

    def _selection(value) -> list[str]:
        if value is None or (isinstance(value, list) and not value):
            return [default_id]
        return value if isinstance(value, list) else [value]

    # MAP REPAINT NUDGE.
    #
    # The Plotly map is a maplibre canvas, and it only draws its traces
    # once the basemap STYLE has finished loading. When a new figure
    # arrives before that happens, Plotly throws "Style is not done
    # loading" and the map is left blank -- basemap and traces both --
    # until something forces a relayout.
    #
    # The horizontal iceberg picker made this far more likely, because it
    # is much taller than the dropdown it replaced, so the page height
    # changes as it renders and the map is sized before the layout has
    # settled. A resize call after the figure lands is enough to make the
    # map draw; this fires on load and whenever a control that changes
    # the figure is touched.
    app.clientside_callback(
        """
        function(forecastData, nIntervals, fullscreen, heatmap, viewOpen) {
            // Keep the map canvas matched to its container.
            //
            // Two separate problems are solved here, and the history of
            // getting them wrong is worth recording:
            //
            // A. BLANK MAP. Plotly only draws map traces once maplibre has
            //    loaded its basemap style. A figure arriving first throws
            //    "Style is not done loading" and leaves the map blank with
            //    nothing to retrigger the draw. A one-shot 'idle' listener
            //    on the maplibre instance fixes that at exactly the right
            //    moment, rather than guessing a delay.
            //
            // B. WRONG SIZE. Fullscreen and the landing/open switch change
            //    the container via CSS, and a maplibre canvas does not
            //    re-fit itself. A ResizeObserver on the container is the
            //    correct trigger: it fires on real box changes only.
            //
            // WHAT NOT TO DO, both learned from breakage:
            //   * Never key a callback on `map-graph.figure` and then call
            //     Plotly.Plots.resize -- resize mutates layout.width and
            //     height, Dash sees the figure prop change, and the
            //     callback re-fires forever. That loop is what made the
            //     map strobe under the cursor.
            //   * Never poll with a repeating resize. Each call is a full
            //     canvas redraw, so any that lands while the user is
            //     hovering or panning shows up as a flicker.
            var install = function () {
                var el = document.querySelector('#map-graph .js-plotly-plot');
                if (!el || !window.Plotly) { return; }
                var box = el.closest('.iw-map-wrap') || el.parentElement;
                if (!box) { return; }

                var fit = function () {
                    var w = Math.round(box.clientWidth);
                    var h = Math.round(box.clientHeight);
                    if (!w || !h) { return; }
                    // Only act on a genuine change, so a redraw can never
                    // be triggered by its own side effects.
                    if (el.__ibW === w && el.__ibH === h) { return; }
                    el.__ibW = w; el.__ibH = h;
                    try { window.Plotly.Plots.resize(el); } catch (e) {}
                    // Plotly resizes its own div, but the maplibre canvas
                    // inside keeps whatever size it was created at, so the
                    // basemap ends up letterboxed inside a correctly-sized
                    // plot. maplibre has to be told directly.
                    try {
                        var sp = el._fullLayout &&
                                 (el._fullLayout.map || el._fullLayout.mapbox);
                        var mm = sp && sp._subplot && sp._subplot.map;
                        if (mm && mm.resize) { mm.resize(); }
                    } catch (e) {}
                };

                if (!box.__ibObserved && window.ResizeObserver) {
                    box.__ibObserved = true;
                    var timer = null;
                    new ResizeObserver(function () {
                        clearTimeout(timer);
                        timer = setTimeout(fit, 120);
                    }).observe(box);
                }

                try {
                    var sub = el._fullLayout &&
                              (el._fullLayout.map || el._fullLayout.mapbox);
                    var m = sub && sub._subplot && sub._subplot.map;
                    if (m && !m.__ibReadyHook) {
                        m.__ibReadyHook = true;
                        m.once('idle', function () {
                            el.__ibW = null;  // force the next fit through
                            fit();
                        });
                    }
                } catch (e) {}

                fit();
            };

            setTimeout(install, 250);
            return '';
        }
        """,
        Output("iw-map-nudge-sink", "children"),
        # Deliberately NOT map-graph.figure -- see the loop warning above.
        Input("forecast-store", "data"),
        Input("iw-map-nudge", "n_intervals"),
        Input("fullscreen-state", "data"),
        Input("heatmap-state", "data"),
        Input("ib-view", "data"),
    )

    # LANDING <-> DASHBOARD.
    #
    # One store drives the view; the shell only swaps a class, because
    # the dashboard is mounted once and merely resized rather than being
    # torn down and rebuilt (see app/landing.py).
    @app.callback(
        Output("ib-view", "data"),
        Input("ib-open-btn", "n_clicks"),
        Input("ib-hero-open-btn", "n_clicks"),
        Input("ib-stage-overlay", "n_clicks"),
        Input("ib-back-btn", "n_clicks"),
        State("ib-view", "data"),
        prevent_initial_call=True,
    )
    def set_view(_open_nav, _open_hero, _open_stage, _back, is_open):
        return ctx.triggered_id != "ib-back-btn"

    # Returning to the landing page must also drop fullscreen. The
    # dashboard's fullscreen style is `position: fixed`, so a leftover
    # fullscreen state would pin the panel over the landing page. Its own
    # toggle owns this store, hence allow_duplicate.
    @app.callback(
        Output("fullscreen-state", "data", allow_duplicate=True),
        Input("ib-view", "data"),
        prevent_initial_call=True,
    )
    def clear_fullscreen_on_return(is_open):
        return False if not is_open else dash.no_update

    @app.callback(
        Output("ib-shell", "className"),
        Output("ib-back-btn", "style"),
        Output("ib-open-btn", "style"),
        Input("ib-view", "data"),
    )
    def apply_view(is_open):
        if is_open:
            return "ib-shell ib-view-open", {"display": "inline-flex"}, {"display": "none"}
        return "ib-shell ib-view-landing", {"display": "none"}, {"display": "inline-flex"}

    @app.callback(
        Output("iw-metadata", "children"),
        Input("iceberg-select", "value"),
    )
    def render_metadata(value):
        selection = _selection(value)
        return _metadata_rows(metadata, selection, selection[0])

    @app.callback(
        Output("iw-explain-summary", "children"),
        Output("iw-explain-rows", "children"),
        Input("iceberg-select", "value"),
        Input("mode-select", "value"),
    )
    def render_explanation(value, mode):
        primary = _selection(value)[0]

        if mode != "hybrid":
            return (
                "Physics-only mode: the forecast is calibrated free drift with no ML "
                "correction, so there is nothing to attribute.",
                [],
            )
        if explainer is None:
            return f"Explanations unavailable: {explain_error}", []

        rows = feature_df[feature_df["iceberg_id"] == primary]
        if rows.empty:
            return f"No feature rows for {primary}.", []

        # Explain the most recent segment -- the one a forecast from
        # "now" would actually be conditioned on.
        latest = rows.sort_values("timestamp").iloc[[-1]]
        summary = f"{primary}: {explainer.narrate(latest)}"

        combined = (
            pd.concat([explainer.explain_row(latest, c) for c in ("u", "v")])
            .groupby(["feature", "label"], as_index=False)
            .agg(shap_value=("shap_value", "sum"), abs_shap=("abs_shap", "sum"),
                 value=("value", "first"))
            .sort_values("abs_shap", ascending=False)
            .head(6)
        )

        listed = []
        for index, item in enumerate(combined.itertuples()):
            # km/day is the unit an operator can sanity-check on a map;
            # m/s attributions are too small to reason about.
            contribution_km_day = item.shap_value * 86.4
            direction = "green" if contribution_km_day >= 0 else "amber"
            listed.append(
                html.Div(
                    className="iw-row",
                    children=[
                        html.Div(
                            className="iw-row-left",
                            children=[
                                html.Span(f"{index + 1:02d}", className="iw-row-idx"),
                                html.Div(
                                    [
                                        html.Div(item.label, className="iw-row-title"),
                                        html.Div(
                                            f"feature value {item.value:+.3f}",
                                            className="iw-row-meta",
                                        ),
                                    ]
                                ),
                            ],
                        ),
                        html.Div(
                            className="iw-row-right",
                            children=[
                                html.Span(
                                    f"{contribution_km_day:+.2f} km/day",
                                    className=f"iw-pill {direction}",
                                )
                            ],
                        ),
                    ],
                )
            )
        return summary, listed


def _hero_stats(metadata: pd.DataFrame, metrics: dict) -> list[tuple[str, str]]:
    """Build the hero's statistics strip from measured values.

    Every figure here is something the pipeline actually computed --
    no rounded-up marketing numbers. The forecast error is the
    leave-one-iceberg-out result, i.e. the honest one, where each berg
    is forecast by a model that never saw it.

    Args:
        metadata: Per-iceberg metadata table.
        metrics: Held-out final displacement error per forecast mode.

    Returns:
        A list of (value, label) pairs.
    """
    drifting = int((~metadata["is_grounded"].fillna(False)).sum()) if "is_grounded" in metadata else len(metadata)
    physics = metrics.get("physics_only")
    days = metrics.get("_horizon_days")
    accuracy = metrics.get("_accuracy_pct")
    skill = metrics.get("_skill_pct")

    # Four figures, not five: the strip is a single row in the hero and a
    # fifth forces the values to wrap. "Actively drifting" is the one that
    # goes -- it is already visible per-iceberg in the metadata panel.
    stats = [(f"{len(metadata)}", "icebergs tracked")]
    if physics:
        # Quote the horizon in days, computed from the record's own step
        # length -- hard-coding "7-day" was wrong once the record was
        # re-binned, and a forecast error means nothing without its horizon.
        label = f"{days:.0f}-day forecast error" if days else "forecast error"
        stats.append((f"{physics:.0f} km", label))
    if skill is not None:
        stats.append((f"{skill:.0f}%", "better than persistence"))
    if accuracy is not None:
        stats.append((f"{accuracy:.0f}%", "of movement predicted"))
    return stats


def _compute_metrics(pooled: pd.DataFrame, force: bool = False, horizon_days: float = 14.0) -> dict:
    """Produce the diagnostics-chart numbers, caching them between runs.

    The chart compares final displacement error across the three forecast
    modes. The numbers come from the leave-one-iceberg-out evaluation --
    the honest one, where each berg is forecast by a model that never saw
    it -- rather than the flattering in-sample figure.

    Args:
        pooled: The pooled real track table.
        force: Recompute even if a cached file exists.

    Returns:
        A dict of mode label -> final displacement error in km.
    """
    if os.path.exists(METRICS_PATH) and not force:
        with open(METRICS_PATH) as handle:
            return json.load(handle)

    _per_iceberg, aggregate = train_model.leave_one_iceberg_out(pooled, verbose=False)
    best = "hybrid" if aggregate["hybrid"]["fde_km"] < aggregate["physics"]["fde_km"] else "physics"
    metrics = {
        "persistence": round(aggregate["persistence"]["fde_km"], 1),
        "physics_only": round(aggregate["physics"]["fde_km"], 1),
        "hybrid": round(aggregate["hybrid"]["fde_km"], 1),
        # Shown in the hero. Kept out of the chart dict consumed by
        # build_diagnostics_figure, which plots every key it is given as a
        # bar -- these are percentages and kilometres, not comparable.
        "_accuracy_pct": round(aggregate[best]["accuracy_pct"], 0),
        "_skill_pct": round(aggregate[best]["skill_vs_persistence_pct"], 0),
        "_moved_km": round(aggregate[best]["actual_displacement_km"], 1),
        "_horizon_days": round(horizon_days, 0),
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
    enriched, motion, bundle, params, feature_df = _build_backend()

    # Make the calibrated coefficients the defaults, so the frontend's
    # physics-only path -- which passes none -- uses them too.
    physics.set_default_drift_params(params)

    iceberg_ids, default_id = _write_track_csvs(enriched)
    _bind_track_loader(default_id)
    _bind_multi_forecast(default_id)

    metadata = _metadata_table(enriched, motion)
    # Evaluate on DRIFTING icebergs only, matching what the model was
    # fitted on. Scoring against grounded bergs would be meaningless --
    # they barely move, so they drag the "distance actually travelled"
    # denominator toward zero and make the accuracy figure a statement
    # about stationary ice rather than about forecast quality.
    drifting_ids = set(motion.loc[~motion["is_grounded"].fillna(False), "iceberg_id"])
    pooled_core = enriched.loc[
        enriched["iceberg_id"].isin(drifting_ids),
        [c for c in data_ingest.POOLED_SCHEMA_COLUMNS if c in enriched.columns],
    ]
    step_days = float(pooled_core["segment_hours"].median()) / 24.0
    metrics = _compute_metrics(pooled_core, horizon_days=step_days * config.DEFAULT_HORIZON_STEPS)

    # An explicit DEFAULT_ICEBERGS list wins; otherwise pick the
    # tightest cluster of drifting bergs, so the opening map is framed on
    # something coherent rather than on bergs spread around the
    # continent. The first entry becomes the forecast subject.
    selection = _resolve_default_selection(DEFAULT_ICEBERGS, iceberg_ids, metadata)

    print(
        f"\n[app] icebergs available: {', '.join(iceberg_ids)}\n"
        f"[app] preselected: {', '.join(selection)}  (forecast subject: {selection[0]})\n"
        f"[app] calibrated physics: {params}\n"
        f"[app] held-out FDE (km): {metrics}\n"
    )

    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.title = "Iceberg Trajectory Prediction"

    layout = build_layout()
    _replace_component(
        layout, "iceberg-select", _build_selector(iceberg_ids, selection, metadata)
    )
    _append_panels(layout)
    app.layout = build_shell(
        layout,
        stats=_hero_stats(metadata, metrics),
        live_note=f"{len(iceberg_ids)} icebergs tracked \u00b7 "
                  f"{pooled_core['timestamp'].max():%d %b %Y}",
    )

    # Underscore-prefixed entries are hero copy, not chart series.
    chart_metrics = {k: v for k, v in metrics.items() if not k.startswith('_')}
    register_callbacks(app, bundle, chart_metrics)
    _register_extra_callbacks(app, metadata, bundle, feature_df, default_id)
    return app


app = create_app()

# WSGI entry point for a production server (gunicorn main:server). Dash
# builds on Flask, and `app.run` above is the development server only --
# it is single-threaded and explicitly not for hosting.
server = app.server

if __name__ == "__main__":
    # Defaults are the local development ones. A host reaching in through
    # the environment gets 0.0.0.0 and its own port, and debug off, so the
    # same file serves both without a separate production entry point.
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8050")),
        debug=os.environ.get("DASH_DEBUG", "1") == "1",
    )
