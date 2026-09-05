"""
data_ingest.py

Produces "track DataFrames" (the shared schema used across this project:
timestamp, lat, lon, area_km2, u_wind, v_wind, u_current, v_current)
from either synthetic data (for offline development) or real public
sources (USNIC iceberg positions, ERA5 wind, Copernicus Marine currents).

Downstream modules (features.py, train_model.py, decision_support.py)
consume the DataFrame produced here without modification, so every
function that returns a track DataFrame is careful to emit exactly the
8 required columns, in order, with the documented dtypes.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import pandas as pd

from physics import free_drift_velocity, step_position

# The canonical column order for every track DataFrame in this project.
# Downstream code is allowed to assume this exact order/naming, so any
# function that builds or returns a track DataFrame re-indexes to this
# list as a final step.
TRACK_SCHEMA_COLUMNS: list[str] = [
    "timestamp",
    "lat",
    "lon",
    "area_km2",
    "u_wind",
    "v_wind",
    "u_current",
    "v_current",
]


def generate_synthetic_track(
    start_lat: float = -65.0,
    start_lon: float = -60.0,
    n_steps: int = 120,
    dt_hours: int = 6,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate a physically-plausible synthetic iceberg track for offline development.

    Wind and ocean current fields are generated as smooth (sinusoidal +
    small noise) functions of the timestep index, rather than white
    noise, because real atmospheric/ocean fields are spatiotemporally
    coherent over the 6-hourly / tens-of-km scales relevant here. The
    "true" drift velocity is the free-drift physics estimate PLUS a
    small random residual, so a downstream ML model has a genuine,
    learnable residual signal to fit -- mirroring the real situation
    where free-drift physics alone under/over-shoots true drift because
    it ignores iceberg draft, shape, and sea-ice interaction.

    Args:
        start_lat: Starting latitude in degrees (-90 to 90). Defaults to
            an Antarctic-like latitude since most large tracked icebergs
            calve from Antarctica.
        start_lon: Starting longitude in degrees (-180 to 180).
        n_steps: Number of timesteps to simulate.
        dt_hours: Timestep size in hours.
        seed: Seed for the numpy random Generator, for reproducibility.

    Returns:
        A track DataFrame with exactly the 8 shared-schema columns.
    """
    rng = np.random.default_rng(seed)
    steps = np.arange(n_steps)

    # --- Slowly-varying synthetic wind field (m/s) ---
    # Sine/cosine of the step index gives a smooth "weather system" style
    # oscillation; small Gaussian noise layered on top avoids the field
    # being perfectly periodic (real wind fields are not).
    wind_period = 40.0  # steps per synoptic-scale wind oscillation
    u_wind = 8.0 * np.sin(2 * np.pi * steps / wind_period) + rng.normal(0, 0.8, n_steps)
    v_wind = 4.0 * np.cos(2 * np.pi * steps / wind_period + 0.5) + rng.normal(0, 0.8, n_steps)

    # --- Slowly-varying synthetic surface current field (m/s) ---
    # Ocean currents vary on a slower timescale and with smaller
    # magnitude than wind, consistent with typical circumpolar current
    # behavior near Antarctica.
    current_period = 90.0
    u_current = 0.3 * np.sin(2 * np.pi * steps / current_period + 1.0) + rng.normal(0, 0.05, n_steps)
    v_current = 0.15 * np.cos(2 * np.pi * steps / current_period) + rng.normal(0, 0.05, n_steps)

    # --- Melting: area decreases slowly over time, never negative ---
    initial_area_km2 = 400.0
    melt_rate_per_step = 0.6  # km^2 lost per 6h step, roughly
    area_noise = rng.normal(0, 0.3, n_steps)
    area_km2 = initial_area_km2 - melt_rate_per_step * steps + np.cumsum(area_noise)
    area_km2 = np.maximum(area_km2, 1.0)  # icebergs don't vanish to 0 in this toy model

    # --- Sequential position integration ---
    # This loop is one of the explicitly-allowed exceptions to
    # vectorization: each position depends on the previous one, so it
    # cannot be computed with a single vectorized pandas/numpy op.
    lats = np.empty(n_steps)
    lons = np.empty(n_steps)
    lats[0] = start_lat
    lons[0] = start_lon

    # Small random residual velocity (m/s) representing the real-world
    # deviation from pure free-drift physics -- this is exactly the
    # signal the XGBoost residual model is meant to learn later.
    residual_u = rng.normal(0, 0.05, n_steps)
    residual_v = rng.normal(0, 0.05, n_steps)

    for i in range(n_steps - 1):
        drift_u, drift_v = free_drift_velocity(
            u_wind=u_wind[i],
            v_wind=v_wind[i],
            u_current=u_current[i],
            v_current=v_current[i],
            lat_deg=lats[i],
        )
        total_u = drift_u + residual_u[i]
        total_v = drift_v + residual_v[i]
        new_lat, new_lon = step_position(
            lat=lats[i],
            lon=lons[i],
            u_ms=total_u,
            v_ms=total_v,
            dt_seconds=dt_hours * 3600.0,
        )
        lats[i + 1] = new_lat
        lons[i + 1] = new_lon

    # Clip defensively in case of pathological inputs; geodesic stepping
    # should already keep us in range, but downstream code assumes this
    # invariant strictly.
    lats = np.clip(lats, -90.0, 90.0)
    lons = ((lons + 180.0) % 360.0) - 180.0  # wrap into [-180, 180)

    start_time = pd.Timestamp("2024-01-01T00:00:00")
    timestamps = start_time + pd.to_timedelta(steps * dt_hours, unit="h")

    track_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "lat": lats,
            "lon": lons,
            "area_km2": area_km2,
            "u_wind": u_wind,
            "v_wind": v_wind,
            "u_current": u_current,
            "v_current": v_current,
        }
    )
    return track_df[TRACK_SCHEMA_COLUMNS]


def load_usnic_track(csv_path: str) -> pd.DataFrame:
    """Load a raw USNIC iceberg position CSV and reshape it into the shared track schema.

    TODO: USNIC iceberg position data source:
        https://usicecenter.gov/Products/AntarcIcebergs
    The raw CSV's column names are not guaranteed to match our schema
    (USNIC exports have varied over time, e.g. "DATE", "LAT", "LONG",
    "AREA (SQ KM)"), so this function checks for several plausible
    aliases per required field and fails loudly with a clear
    ValueError -- naming exactly which required fields could not be
    found -- rather than raising a cryptic KeyError deep in pandas.

    USNIC exports do not include wind/current, so u_wind, v_wind,
    u_current, v_current are populated as NaN here; call
    merge_environmental_data() afterwards to fill them in from
    ERA5/Copernicus Marine data.

    Args:
        csv_path: Path to the raw USNIC CSV file on disk.

    Returns:
        A track DataFrame with exactly the 8 shared-schema columns;
        u_wind/v_wind/u_current/v_current are NaN until merged with
        environmental data.

    Raises:
        ValueError: If any required field (timestamp, lat, lon,
            area_km2) cannot be found under any of its known aliases.
    """
    raw = pd.read_csv(csv_path)

    # Plausible column-name aliases seen in various USNIC export
    # vintages. Extend this list as new export formats are encountered.
    alias_map: dict[str, list[str]] = {
        "timestamp": ["timestamp", "DATE", "Date", "OBS_DATE", "date_time"],
        "lat": ["lat", "LAT", "Lat", "Latitude", "LATITUDE"],
        "lon": ["lon", "LON", "LONG", "Lon", "Longitude", "LONGITUDE"],
        "area_km2": ["area_km2", "AREA_KM2", "AREA (SQ KM)", "Area_Sq_Km", "AREA"],
    }

    resolved: dict[str, str] = {}
    missing: list[str] = []
    for schema_col, aliases in alias_map.items():
        found = next((a for a in aliases if a in raw.columns), None)
        if found is None:
            missing.append(schema_col)
        else:
            resolved[schema_col] = found

    if missing:
        raise ValueError(
            f"load_usnic_track: could not find required column(s) {missing} in "
            f"'{csv_path}'. Available columns are: {list(raw.columns)}. "
            f"Update the alias_map in load_usnic_track() to include this file's "
            f"actual column names."
        )

    track_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(raw[resolved["timestamp"]]),
            "lat": raw[resolved["lat"]].astype(float),
            "lon": raw[resolved["lon"]].astype(float),
            "area_km2": raw[resolved["area_km2"]].astype(float),
            "u_wind": np.nan,
            "v_wind": np.nan,
            "u_current": np.nan,
            "v_current": np.nan,
        }
    )
    return track_df[TRACK_SCHEMA_COLUMNS]


def fetch_era5_wind(
    bbox: Sequence[float],
    start_date: str,
    end_date: str,
    out_path: str = "data/era5_wind.nc",
) -> str:
    """Fetch ERA5 10m wind components over a bounding box/date range via the CDS API.

    Requires a free Copernicus Climate Data Store (CDS) API key
    configured in ~/.cdsapirc. See:
        https://cds.climate.copernicus.eu/api-how-to

    Args:
        bbox: Bounding box as [North, West, South, East] (CDS API
            convention -- note this is NOT [min_lon, min_lat, max_lon,
            max_lat]).
        start_date: Start date as an ISO string, e.g. "2024-01-01".
        end_date: End date as an ISO string, e.g. "2024-03-01".
        out_path: Where to save the downloaded NetCDF file.

    Returns:
        The out_path string, for convenience chaining into
        merge_environmental_data().

    Raises:
        RuntimeError: If the cdsapi package is missing or the CDS API
            key is not configured / the request fails, with an
            actionable message instead of a raw stack trace.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    try:
        import cdsapi
    except ImportError as exc:
        raise RuntimeError(
            "fetch_era5_wind: the 'cdsapi' package is not installed. "
            "Install it with `pip install cdsapi`."
        ) from exc

    north, west, south, east = bbox
    date_range = f"{start_date}/{end_date}"

    try:
        client = cdsapi.Client()
        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
                "date": date_range,
                "time": [f"{h:02d}:00" for h in range(0, 24, 6)],
                "area": [north, west, south, east],
                "format": "netcdf",
            },
            out_path,
        )
    except Exception as exc:
        raise RuntimeError(
            "fetch_era5_wind: request to the CDS API failed. This usually means "
            "~/.cdsapirc is missing or misconfigured. Set up a free CDS API key "
            "at https://cds.climate.copernicus.eu/api-how-to and try again. "
            f"Original error: {exc}"
        ) from exc

    return out_path


def fetch_copernicus_currents(
    bbox: Sequence[float],
    start_date: str,
    end_date: str,
    out_path: str = "data/currents.nc",
) -> str:
    """Fetch ocean surface current components (uo, vo) via the Copernicus Marine Toolbox.

    Requires Copernicus Marine credentials configured (via
    `copernicusmarine login` or environment variables). See:
        https://help.marine.copernicus.eu/en/articles/7970514

    Args:
        bbox: Bounding box as [North, West, South, East], matching the
            same convention used elsewhere in this module; converted
            internally to the min/max lon/lat kwargs the
            copernicusmarine package expects.
        start_date: Start date as an ISO string, e.g. "2024-01-01".
        end_date: End date as an ISO string, e.g. "2024-03-01".
        out_path: Where to save the downloaded NetCDF file.

    Returns:
        The out_path string, for convenience chaining into
        merge_environmental_data().

    Raises:
        RuntimeError: If the copernicusmarine package is missing or
            credentials are not configured / the request fails, with an
            actionable message instead of a raw stack trace.
    """
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    out_filename = os.path.basename(out_path)

    try:
        import copernicusmarine
    except ImportError as exc:
        raise RuntimeError(
            "fetch_copernicus_currents: the 'copernicusmarine' package is not "
            "installed. Install it with `pip install copernicusmarine`."
        ) from exc

    north, west, south, east = bbox

    try:
        copernicusmarine.subset(
            dataset_id="cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
            variables=["uo", "vo"],
            minimum_longitude=west,
            maximum_longitude=east,
            minimum_latitude=south,
            maximum_latitude=north,
            start_datetime=start_date,
            end_datetime=end_date,
            minimum_depth=0,
            maximum_depth=1,
            output_directory=out_dir,
            output_filename=out_filename,
        )
    except Exception as exc:
        raise RuntimeError(
            "fetch_copernicus_currents: request to Copernicus Marine failed. "
            "This usually means you are not logged in / credentials are not "
            "configured. Run `copernicusmarine login` or see "
            "https://help.marine.copernicus.eu/en/articles/7970514 and try "
            f"again. Original error: {exc}"
        ) from exc

    return out_path


def _find_variable(ds, candidates: Sequence[str], role: str, source_path: str) -> str:
    """Find the first matching variable/coordinate name in an xarray Dataset.

    Args:
        ds: The xarray Dataset to search.
        candidates: Candidate variable names to look for, in priority order.
        role: Human-readable description of what this variable represents
            (used only in the error message).
        source_path: Path of the source file (used only in the error message).

    Returns:
        The first candidate name found among ds's variables/coordinates.

    Raises:
        ValueError: If none of the candidates are present in ds.
    """
    available = set(ds.variables.keys())
    found = next((c for c in candidates if c in available), None)
    if found is None:
        raise ValueError(
            f"Could not find a '{role}' field in '{source_path}'. Looked for any "
            f"of {list(candidates)} but the file only has: {sorted(available)}. "
            f"Add the actual variable name to the candidates list in "
            f"merge_environmental_data()."
        )
    return found


def merge_environmental_data(
    track_df: pd.DataFrame,
    wind_nc_path: str,
    current_nc_path: str,
) -> pd.DataFrame:
    """Populate wind/current columns of a track DataFrame from ERA5/Copernicus NetCDF files.

    For each row of track_df (which is expected to already have
    timestamp/lat/lon/area_km2, e.g. from load_usnic_track()), this
    extracts the nearest-neighbor (in time AND space) wind and current
    values from the two NetCDF files using xarray's vectorized
    `.sel(..., method="nearest")` indexing (a single vectorized call
    over all rows at once, not a per-row Python loop).

    Args:
        track_df: A track DataFrame with timestamp/lat/lon/area_km2
            already populated (u_wind/v_wind/u_current/v_current may be
            NaN placeholders).
        wind_nc_path: Path to a NetCDF file containing 10m u/v wind
            components (e.g. produced by fetch_era5_wind()).
        current_nc_path: Path to a NetCDF file containing ocean surface
            current u/v components (e.g. produced by
            fetch_copernicus_currents()).

    Returns:
        A track DataFrame with all 8 shared-schema columns populated.

    Raises:
        ValueError: If a required coordinate/variable cannot be found in
            either NetCDF file, or if any row's timestamp/lat/lon falls
            outside the coverage of either file (names the offending row
            index and its date/location rather than silently returning
            NaN or crashing inside xarray).
    """
    try:
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError(
            "merge_environmental_data: the 'xarray' package is not installed. "
            "Install it with `pip install xarray netCDF4`."
        ) from exc

    ds_wind = xr.open_dataset(wind_nc_path)
    ds_current = xr.open_dataset(current_nc_path)

    time_name_w = _find_variable(ds_wind, ["time", "valid_time"], "time coordinate", wind_nc_path)
    lat_name_w = _find_variable(ds_wind, ["latitude", "lat"], "latitude coordinate", wind_nc_path)
    lon_name_w = _find_variable(ds_wind, ["longitude", "lon"], "longitude coordinate", wind_nc_path)
    u_wind_var = _find_variable(ds_wind, ["u10", "u_wind", "u10n"], "10m u-wind", wind_nc_path)
    v_wind_var = _find_variable(ds_wind, ["v10", "v_wind", "v10n"], "10m v-wind", wind_nc_path)

    time_name_c = _find_variable(ds_current, ["time", "valid_time"], "time coordinate", current_nc_path)
    lat_name_c = _find_variable(ds_current, ["latitude", "lat"], "latitude coordinate", current_nc_path)
    lon_name_c = _find_variable(ds_current, ["longitude", "lon"], "longitude coordinate", current_nc_path)
    u_current_var = _find_variable(ds_current, ["uo", "u_current"], "eastward current", current_nc_path)
    v_current_var = _find_variable(ds_current, ["vo", "v_current"], "northward current", current_nc_path)

    # --- Bounds checking, vectorized, before touching xarray's indexing ---
    def _check_bounds(ds, time_name: str, lat_name: str, lon_name: str, source_path: str) -> None:
        t_min, t_max = ds[time_name].values.min(), ds[time_name].values.max()
        lat_min, lat_max = float(ds[lat_name].values.min()), float(ds[lat_name].values.max())
        lon_min, lon_max = float(ds[lon_name].values.min()), float(ds[lon_name].values.max())

        times = track_df["timestamp"].values
        lats = track_df["lat"].values
        lons = track_df["lon"].values

        out_of_range = (
            (times < t_min) | (times > t_max)
            | (lats < lat_min) | (lats > lat_max)
            | (lons < lon_min) | (lons > lon_max)
        )
        if out_of_range.any():
            bad_idx = int(np.argmax(out_of_range))  # first offending row
            raise ValueError(
                f"merge_environmental_data: row {bad_idx} "
                f"(timestamp={track_df['timestamp'].iloc[bad_idx]}, "
                f"lat={track_df['lat'].iloc[bad_idx]}, lon={track_df['lon'].iloc[bad_idx]}) "
                f"falls outside the coverage of '{source_path}' "
                f"(time: [{t_min}, {t_max}], lat: [{lat_min}, {lat_max}], "
                f"lon: [{lon_min}, {lon_max}]). Re-fetch that source with a wider "
                f"bbox/date range, or trim track_df to the covered period."
            )

    _check_bounds(ds_wind, time_name_w, lat_name_w, lon_name_w, wind_nc_path)
    _check_bounds(ds_current, time_name_c, lat_name_c, lon_name_c, current_nc_path)

    # --- Vectorized nearest-neighbor extraction ---
    # Wrapping the row coordinates in xr.DataArrays sharing a common
    # "points" dimension makes xarray perform "pointwise" advanced
    # indexing: one nearest-neighbor lookup per row, all in a single
    # vectorized call rather than a Python loop over rows.
    points_time = xr.DataArray(track_df["timestamp"].values, dims="points")
    points_lat = xr.DataArray(track_df["lat"].values, dims="points")
    points_lon = xr.DataArray(track_df["lon"].values, dims="points")

    wind_sel = ds_wind.sel(
        {time_name_w: points_time, lat_name_w: points_lat, lon_name_w: points_lon},
        method="nearest",
    )
    current_sel = ds_current.sel(
        {time_name_c: points_time, lat_name_c: points_lat, lon_name_c: points_lon},
        method="nearest",
    )

    merged = track_df.copy()
    merged["u_wind"] = wind_sel[u_wind_var].values
    merged["v_wind"] = wind_sel[v_wind_var].values
    merged["u_current"] = current_sel[u_current_var].values
    merged["v_current"] = current_sel[v_current_var].values

    ds_wind.close()
    ds_current.close()

    return merged[TRACK_SCHEMA_COLUMNS]


if __name__ == "__main__":
    # Demonstrate generate_synthetic_track() end to end. The real-data
    # fetchers (fetch_era5_wind, fetch_copernicus_currents,
    # load_usnic_track) require credentials/files we don't have in this
    # demo, so they are intentionally not exercised here.
    track = generate_synthetic_track(n_steps=120, dt_hours=6, seed=42)

    assert list(track.columns) == TRACK_SCHEMA_COLUMNS, "schema column mismatch"
    assert track["lat"].between(-90, 90).all(), "lat out of range"
    assert track["lon"].between(-180, 180).all(), "lon out of range"
    assert (track["area_km2"] >= 0).all(), "area_km2 went negative"

    print("Synthetic track generated successfully.")
    print(f"Shape: {track.shape}")
    print("\nFirst 3 rows:")
    print(track.head(3).to_string(index=False))
    print("\nLast 3 rows:")
    print(track.tail(3).to_string(index=False))

    os.makedirs("data", exist_ok=True)
    out_csv = "data/synthetic_track.csv"
    track.to_csv(out_csv, index=False)
    print(f"\nSaved synthetic track to '{out_csv}' ({len(track)} rows).")