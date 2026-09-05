"""
data_ingest.py

Builds "track DataFrames" (the shared project schema: timestamp, lat,
lon, area_km2, u_wind, v_wind, u_current, v_current) from REAL public
data:

  * positions  -- USNIC/BYU Antarctic iceberg snapshot CSVs on disk
                  (https://usicecenter.gov/Products/AntarcIcebergs)
  * wind       -- ERA5 10 m u/v via the Copernicus CDS API (cdsapi)
  * currents   -- CMEMS surface currents via the Copernicus Marine
                  Toolbox (copernicusmarine)

generate_synthetic_track() is retained for offline development and for
the standalone demo blocks in the other modules, but it is no longer the
pipeline's primary source.


THREE THINGS THAT MAKE REAL NIC DATA DIFFERENT FROM THE SYNTHETIC TRACK
======================================================================

1. A SNAPSHOT IS NOT A TRACK. Each NIC CSV is one row per *iceberg* on
   one date -- all ~33 currently-tracked bergs, one fix each. A single
   file therefore contains no time series at all. A track is recovered
   by stacking many dated snapshots and grouping by iceberg name, which
   is what build_iceberg_tracks() does. Because the pipeline then has
   many short tracks rather than one long one, every downstream stage
   works on a POOLED table keyed by an `iceberg_id` column and must
   never compute a difference across an iceberg boundary.

2. THE SAMPLING IS WEEKLY AND IRREGULAR. Fixes are 6-15 days apart (with
   one 36-day gap), not 6 hours. The velocity implied by two fixes a week
   apart is a WEEK-MEAN velocity, so the physics baseline it is compared
   against must also be driven by week-mean forcing. Sampling ERA5 at the
   instant of the fix -- which is what you would do for a 6-hourly track,
   and what the synthetic pipeline did -- compares a mean against an
   instantaneous storm-scale value and produces a physics baseline that
   is mostly noise. sample_environment_along_segments() instead walks the
   geodesic between each pair of fixes at SEGMENT_SAMPLE_HOURS intervals
   and averages the forcing over the segment. This is the single largest
   accuracy lever in the real-data pipeline.

   Consequence for the schema: in a REAL pooled track, the wind/current
   columns of row k hold the mean forcing over the segment ENDING at row
   k, i.e. over (t[k-1], t[k]] -- pairing exactly with the observed
   velocity features.py computes for that same row. Row 0 of each
   iceberg has no preceding segment and carries the instantaneous value
   at its own fix; it is only ever used as lag history, never as a label.
   The `segment_hours` column records the interval each row's forcing
   covers so downstream code can verify that pairing (see
   features.compute_observed_velocity).

3. MOST NIC ICEBERGS ARE NOT DRIFTING. Roughly two thirds of the tracked
   bergs are grounded on the shelf or locked in fast ice, and several
   have byte-identical positions across every snapshot -- the position is
   being carried forward, not re-observed. Training a free-drift residual
   model on those teaches it that the correct residual is "minus the
   whole physics prediction", which then destroys forecasts for the
   bergs that actually move. summarize_iceberg_motion() separates the
   two populations; see GROUNDED_SPEED_THRESHOLD_MS in config.py.
"""

from __future__ import annotations

import glob
import hashlib
import os
import re
import warnings
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from pyproj import Geod

import config
from physics import free_drift_velocity, geodesic_distance_km, step_position

# The canonical column order for a single-iceberg track DataFrame.
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

# A pooled multi-iceberg table adds the grouping key plus the bookkeeping
# column that records which interval each row's forcing was averaged
# over. Downstream code groups by `iceberg_id` and never differences
# across a group boundary.
POOLED_SCHEMA_COLUMNS: list[str] = ["iceberg_id"] + TRACK_SCHEMA_COLUMNS + ["segment_hours"]

_GEOD = Geod(ellps="WGS84")


# =====================================================================
# Synthetic data (offline development / module self-tests)
# =====================================================================


def generate_synthetic_track(
    start_lat: float = -65.0,
    start_lon: float = -60.0,
    n_steps: int = 120,
    dt_hours: int = 6,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate a physically plausible fake iceberg track for offline development.

    Wind and current vary smoothly (sinusoids plus small noise) rather
    than as white noise, since real environmental fields are
    spatiotemporally smooth. Positions are advanced with the same
    free-drift physics the model later uses, plus a small random
    residual velocity standing in for the real-world deviation from pure
    free drift that the ML stage is trained to recover.

    Args:
        start_lat: Starting latitude, degrees (Antarctic by default).
        start_lon: Starting longitude, degrees.
        n_steps: Number of timesteps to simulate.
        dt_hours: Hours between consecutive timesteps.
        seed: Seed for numpy's random Generator, for reproducibility.

    Returns:
        A track DataFrame with exactly TRACK_SCHEMA_COLUMNS.
    """
    rng = np.random.default_rng(seed)
    dt_seconds = dt_hours * 3600.0

    lat, lon = start_lat, start_lon
    area = 400.0
    start_time = pd.Timestamp("2024-01-01")

    records: list[dict[str, object]] = []
    for step in range(n_steps):
        # Smooth, slowly-varying synoptic-scale forcing plus small noise.
        u_wind = 5.0 * np.sin(step / 12.0) + rng.normal(0.0, 0.8)
        v_wind = 4.0 * np.cos(step / 17.0) + rng.normal(0.0, 0.8)
        u_current = 0.25 * np.sin(step / 30.0) + rng.normal(0.0, 0.02)
        v_current = 0.15 * np.cos(step / 25.0) + rng.normal(0.0, 0.02)

        records.append(
            {
                "timestamp": start_time + pd.Timedelta(hours=dt_hours * step),
                "lat": lat,
                "lon": lon,
                "area_km2": area,
                "u_wind": u_wind,
                "v_wind": v_wind,
                "u_current": u_current,
                "v_current": v_current,
            }
        )

        u_drift, v_drift = free_drift_velocity(u_wind, v_wind, u_current, v_current, lat)
        # The residual the ML model is meant to learn: everything free
        # drift cannot explain (draft, keel shape, sea-ice contact).
        u_drift += rng.normal(0.0, 0.02)
        v_drift += rng.normal(0.0, 0.02)

        lat, lon = step_position(lat, lon, u_drift, v_drift, dt_seconds)
        area = max(0.0, area - abs(rng.normal(0.15, 0.05)))  # icebergs melt

    return pd.DataFrame.from_records(records)[TRACK_SCHEMA_COLUMNS]


# =====================================================================
# USNIC / BYU iceberg position snapshots
# =====================================================================

# Column aliases seen across NIC export vintages. Extend as needed --
# a miss raises a named ValueError rather than a cryptic KeyError.
_NIC_ALIASES: dict[str, list[str]] = {
    "iceberg_id": ["Iceberg", "ICEBERG", "iceberg", "Name", "NAME", "ID", "Iceberg_ID"],
    "timestamp": ["Last Update", "DATE", "Date", "date", "OBS_DATE", "timestamp"],
    "lat": ["Latitude", "LAT", "Lat", "lat", "LATITUDE"],
    "lon": ["Longitude", "LON", "Lon", "lon", "LONG", "LONGITUDE"],
    "area_km2": ["Area (sqKM)", "AREA (SQ KM)", "Area_Sq_Km", "AREA_KM2", "area_km2"],
}


def _resolve_nic_columns(columns: Iterable[str], csv_path: str) -> dict[str, str]:
    """Map this NIC export's actual column names onto our schema names.

    Args:
        columns: The raw column names read from the CSV.
        csv_path: Source path, used only in the error message.

    Returns:
        A dict mapping schema name -> actual column name in the file.

    Raises:
        ValueError: If any required field is missing under every known alias.
    """
    # NIC exports are UTF-8-BOM encoded, which leaves a stray ﻿ on
    # the first header cell if it is not stripped.
    available = {str(c).strip().lstrip("﻿"): str(c) for c in columns}

    resolved: dict[str, str] = {}
    missing: list[str] = []
    for schema_name, aliases in _NIC_ALIASES.items():
        found = next((a for a in aliases if a in available), None)
        if found is None:
            missing.append(schema_name)
        else:
            resolved[schema_name] = available[found]

    if missing:
        raise ValueError(
            f"load_nic_snapshot: could not find required field(s) {missing} in "
            f"'{csv_path}'. The file's columns are {sorted(available)}. Add this "
            f"export's actual column names to _NIC_ALIASES in data_ingest.py."
        )
    return resolved


def load_nic_snapshot(csv_path: str) -> pd.DataFrame:
    """Load one dated NIC snapshot CSV into long format.

    A NIC snapshot holds every currently-tracked iceberg on one date, so
    this returns one row per iceberg, not a track. Use
    load_nic_snapshots() over several dated files and
    build_iceberg_tracks() to recover per-iceberg time series.

    Args:
        csv_path: Path to a single NIC Antarctic iceberg CSV export.

    Returns:
        A DataFrame with columns iceberg_id, timestamp, lat, lon,
        area_km2 -- positions only; no environmental data.

    Raises:
        ValueError: If a required column is missing (see
            _resolve_nic_columns) or if no rows survive parsing.
    """
    raw = pd.read_csv(csv_path, encoding="utf-8-sig")
    resolved = _resolve_nic_columns(raw.columns, csv_path)

    df = pd.DataFrame(
        {
            "iceberg_id": raw[resolved["iceberg_id"]].astype(str).str.strip().str.upper(),
            # NIC writes US-style MM/DD/YYYY; be explicit rather than
            # letting pandas guess (it would silently read 05/08 as
            # 8 May in one file and 5 August in another).
            "timestamp": pd.to_datetime(
                raw[resolved["timestamp"]], format="%m/%d/%Y", errors="coerce"
            ),
            "lat": pd.to_numeric(raw[resolved["lat"]], errors="coerce"),
            "lon": pd.to_numeric(raw[resolved["lon"]], errors="coerce"),
            "area_km2": pd.to_numeric(raw[resolved["area_km2"]], errors="coerce"),
        }
    ).dropna(subset=["iceberg_id", "timestamp", "lat", "lon"])

    if df.empty:
        raise ValueError(
            f"load_nic_snapshot: no usable rows parsed from '{csv_path}'. Check that "
            f"the date column is MM/DD/YYYY and that lat/lon are numeric."
        )

    # Normalise longitude to [-180, 180); NIC is already in that
    # convention but a stray 0-360 export would corrupt every geodesic.
    df["lon"] = ((df["lon"] + 180.0) % 360.0) - 180.0
    return df.reset_index(drop=True)


def load_nic_snapshots(
    source: str | Path | Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load and stack every dated NIC snapshot into one long table.

    Args:
        source: A directory to glob for USNIC_CSV_GLOB, an explicit
            sequence of CSV paths, or None to use config.DATA_DIR.

    Returns:
        A DataFrame (iceberg_id, timestamp, lat, lon, area_km2) sorted by
        iceberg then time, with duplicate (iceberg, date) rows collapsed
        to the last occurrence.

    Raises:
        FileNotFoundError: If no snapshot CSVs are found.
        ValueError: If fewer than two distinct dates are present -- a
            single snapshot cannot yield any velocity at all.
    """
    if source is None:
        source = config.DATA_DIR
    if isinstance(source, (str, Path)):
        paths = sorted(glob.glob(str(Path(source) / config.USNIC_CSV_GLOB)))
    else:
        paths = sorted(str(p) for p in source)

    if not paths:
        raise FileNotFoundError(
            f"load_nic_snapshots: no NIC snapshot CSVs found for source={source!r} "
            f"(pattern '{config.USNIC_CSV_GLOB}'). Download dated exports from "
            f"https://usicecenter.gov/Products/AntarcIcebergs into {config.DATA_DIR}."
        )

    frames = [load_nic_snapshot(p) for p in paths]
    combined = pd.concat(frames, ignore_index=True)
    combined = (
        combined.sort_values(["iceberg_id", "timestamp"])
        .drop_duplicates(subset=["iceberg_id", "timestamp"], keep="last")
        .reset_index(drop=True)
    )

    n_dates = combined["timestamp"].nunique()
    if n_dates < 2:
        raise ValueError(
            f"load_nic_snapshots: found only {n_dates} distinct snapshot date(s) across "
            f"{len(paths)} file(s). At least 2 dated snapshots are required to observe "
            f"any iceberg motion."
        )
    return combined


def build_iceberg_tracks(
    snapshots: pd.DataFrame, min_observations: int = 3
) -> dict[str, pd.DataFrame]:
    """Group stacked snapshots into one track DataFrame per iceberg.

    Args:
        snapshots: Long table from load_nic_snapshots().
        min_observations: Icebergs with fewer fixes than this are
            dropped (too short to build lag features and a rollout).

    Returns:
        A dict mapping iceberg_id -> track DataFrame with exactly
        TRACK_SCHEMA_COLUMNS, sorted by timestamp, with the four
        environmental columns present but NaN. Call
        sample_environment_along_segments() next to populate them.
    """
    tracks: dict[str, pd.DataFrame] = {}
    for iceberg_id, group in snapshots.groupby("iceberg_id", sort=True):
        if len(group) < min_observations:
            continue
        track = group.sort_values("timestamp").reset_index(drop=True).copy()
        for col in ("u_wind", "v_wind", "u_current", "v_current"):
            track[col] = np.nan
        tracks[str(iceberg_id)] = track[TRACK_SCHEMA_COLUMNS]
    return tracks


def summarize_iceberg_motion(
    tracks: dict[str, pd.DataFrame],
    grounded_speed_threshold_ms: float = config.GROUNDED_SPEED_THRESHOLD_MS,
) -> pd.DataFrame:
    """Classify each iceberg as drifting or grounded from its own displacement.

    A large fraction of NIC-tracked Antarctic icebergs are grounded on
    the shelf or locked in fast ice; some entries repeat an identical
    position for months because the fix is carried forward rather than
    re-observed. Those bergs are real and worth displaying, but they are
    poison as free-drift training data: the only way to fit them is to
    predict a residual that exactly cancels the physics term, which
    then wrecks forecasts for bergs that genuinely move.

    The discriminator is mean speed over the whole record. The default
    threshold (0.01 m/s ~ 0.9 km/day) sits far below any genuinely
    drifting Southern Ocean berg and comfortably above the ~1.1 km
    quantisation of NIC's 0.01-degree position rounding.

    Args:
        tracks: Mapping of iceberg_id -> track DataFrame, from
            build_iceberg_tracks().
        grounded_speed_threshold_ms: Mean-speed cutoff, m/s.

    Returns:
        A DataFrame indexed by row with columns iceberg_id, n_obs,
        first_seen, last_seen, total_km, mean_speed_ms, max_speed_ms,
        mean_area_km2, is_grounded -- sorted by total_km descending.
    """
    rows: list[dict[str, object]] = []
    for iceberg_id, track in tracks.items():
        lats = track["lat"].to_numpy()
        lons = track["lon"].to_numpy()
        # Geodesic, not flat lat/lon: these are 65 deg S, and two of the
        # bergs (B22A, B22F) straddle the antimeridian.
        _az, _back, dist_m = _GEOD.inv(lons[:-1], lats[:-1], lons[1:], lats[1:])
        dt_s = track["timestamp"].diff().dt.total_seconds().to_numpy()[1:]
        speeds = dist_m / dt_s

        total_m = float(dist_m.sum())
        elapsed_s = float(dt_s.sum())
        rows.append(
            {
                "iceberg_id": iceberg_id,
                "n_obs": len(track),
                "first_seen": track["timestamp"].iloc[0],
                "last_seen": track["timestamp"].iloc[-1],
                "total_km": total_m / 1000.0,
                # Path length over elapsed time, not the mean of the
                # per-segment speeds: that weights each segment by its
                # own duration, which is what we want with uneven gaps.
                "mean_speed_ms": total_m / elapsed_s if elapsed_s > 0 else 0.0,
                "max_speed_ms": float(speeds.max()) if speeds.size else 0.0,
                "mean_area_km2": float(track["area_km2"].mean()),
            }
        )

    summary = pd.DataFrame(rows)
    summary["is_grounded"] = summary["mean_speed_ms"] < grounded_speed_threshold_ms
    return summary.sort_values("total_km", ascending=False).reset_index(drop=True)


# =====================================================================
# ERA5 wind (Copernicus Climate Data Store)
# =====================================================================

# The CDS rejects a request that runs past the end of the archive and
# helpfully names the last available timestamp in the error body. ERA5
# runs ~5 days behind real time, so a request built from "today" will
# routinely overshoot; we parse the date out and retry once, clipped.
_CDS_LATEST_RE = re.compile(
    r"latest date available for this dataset is:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})", re.IGNORECASE
)


def _month_chunks(start_date: str, end_date: str) -> list[tuple[str, str]]:
    """Split an inclusive date range into calendar-month sub-ranges.

    One CDS request per month keeps each job small enough to be queued
    and served promptly, and lets a partially-completed multi-month fetch
    resume from the cache instead of starting over.

    Args:
        start_date: Inclusive ISO start date, "YYYY-MM-DD".
        end_date: Inclusive ISO end date, "YYYY-MM-DD".

    Returns:
        A list of (chunk_start, chunk_end) ISO date-string pairs.
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    chunks: list[tuple[str, str]] = []
    cursor = start
    while cursor <= end:
        month_end = (cursor + pd.offsets.MonthEnd(0)).normalize()
        chunk_end = min(month_end, end)
        chunks.append((cursor.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        cursor = chunk_end + pd.Timedelta(days=1)
    return chunks


def fetch_era5_wind(
    start_date: str,
    end_date: str,
    lat_band: tuple[float, float] = config.ERA5_LAT_BAND,
    grid_deg: float = config.ERA5_GRID_DEG,
    cache_dir: str | Path = config.ERA5_CACHE_DIR,
    force_refresh: bool = False,
) -> list[str]:
    """Fetch ERA5 10 m wind over a circumpolar latitude band, one file per month.

    The band spans all longitudes rather than a per-iceberg box: NIC
    tracks bergs right around the continent (two of them across the
    antimeridian), and one queued CDS request per month is far faster
    than thirty-odd small ones, since CDS cost is dominated by queueing
    rather than by volume. At 0.5 degrees the whole Southern Ocean band
    for four months is only tens of megabytes, and 0.5 degrees is ample
    for synoptic wind feeding a ~2%-of-wind-speed drift term.

    Requires a free CDS API key in ~/.cdsapirc; see
    https://cds.climate.copernicus.eu/how-to-api

    Args:
        start_date: Inclusive ISO start date, "YYYY-MM-DD".
        end_date: Inclusive ISO end date, "YYYY-MM-DD". Automatically
            clipped if it runs past the end of the ERA5 archive (the
            reanalysis lags real time by about five days).
        lat_band: (north, south) latitude limits in degrees.
        grid_deg: Regridding resolution in degrees.
        cache_dir: Directory for the downloaded NetCDF files.
        force_refresh: Re-download even if a cached file exists.

    Returns:
        A sorted list of NetCDF paths covering the requested range.

    Raises:
        RuntimeError: If cdsapi is missing, credentials are not
            configured, or a request fails for any reason other than
            overshooting the end of the archive.
    """
    cache_dir = Path(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    try:
        import cdsapi
    except ImportError as exc:
        raise RuntimeError(
            "fetch_era5_wind: the 'cdsapi' package is not installed. Install it with "
            "`pip install cdsapi`."
        ) from exc

    north, south = lat_band
    client = cdsapi.Client(quiet=True, progress=False)

    def _request(chunk_start: str, chunk_end: str, path: Path) -> None:
        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
                "date": f"{chunk_start}/{chunk_end}",
                "time": [f"{h:02d}:00" for h in range(0, 24, config.SEGMENT_SAMPLE_HOURS)],
                "area": [north, -180, south, 180],
                "grid": [grid_deg, grid_deg],
                "data_format": "netcdf",
                "download_format": "unarchived",
            },
            str(path),
        )

    paths: list[str] = []
    for chunk_start, chunk_end in _month_chunks(start_date, end_date):
        path = cache_dir / f"era5_wind_{chunk_start}_{chunk_end}_{grid_deg}deg.nc"
        if path.exists() and path.stat().st_size > 10_000 and not force_refresh:
            paths.append(str(path))
            continue

        try:
            _request(chunk_start, chunk_end, path)
        except Exception as exc:  # noqa: BLE001 - re-raised with context below
            match = _CDS_LATEST_RE.search(str(exc))
            if match is None:
                raise RuntimeError(
                    "fetch_era5_wind: the CDS request failed. The usual cause is a "
                    "missing or stale ~/.cdsapirc, or not having accepted this "
                    "dataset's licence in the CDS web UI. See "
                    "https://cds.climate.copernicus.eu/how-to-api . "
                    f"Original error: {exc}"
                ) from exc

            # The archive ends inside this chunk: clip and retry once.
            latest = match.group(1)
            if pd.Timestamp(latest) < pd.Timestamp(chunk_start):
                warnings.warn(
                    f"fetch_era5_wind: ERA5 ends at {latest}, entirely before the chunk "
                    f"{chunk_start}..{chunk_end}; skipping it. Track segments after "
                    f"{latest} will be dropped for lack of wind forcing.",
                    stacklevel=2,
                )
                break
            warnings.warn(
                f"fetch_era5_wind: ERA5 currently ends at {latest}; clipping the request "
                f"{chunk_start}..{chunk_end} to {chunk_start}..{latest}. Track segments "
                f"extending past {latest} will be dropped for lack of wind forcing.",
                stacklevel=2,
            )
            path = cache_dir / f"era5_wind_{chunk_start}_{latest}_{grid_deg}deg.nc"
            if not (path.exists() and path.stat().st_size > 10_000 and not force_refresh):
                _request(chunk_start, latest, path)
            paths.append(str(path))
            break

        paths.append(str(path))

    if not paths:
        raise RuntimeError(
            f"fetch_era5_wind: no ERA5 files could be obtained for {start_date}..{end_date}."
        )
    return sorted(paths)


# =====================================================================
# Copernicus Marine surface currents
# =====================================================================


def track_bbox(
    track: pd.DataFrame, pad_deg: float = config.CURRENT_BBOX_PAD_DEG
) -> tuple[float, float, float, float]:
    """Compute a padded lon/lat bounding box around one iceberg's path.

    Handles the antimeridian: two NIC bergs (B22A, B22F) have fixes on
    both sides of 180 deg, whose naive min/max longitude would be
    (-179.7, 178.1) -- a box spanning almost the whole globe the wrong
    way round. When the eastward span is shorter than the naive span, we
    express the box in CONTINUOUS longitude that runs past +/-180 (e.g.
    177.0 to 183.0). The Copernicus Marine toolbox accepts such a range
    and returns a monotonically increasing longitude axis extending past
    180, which _wrap_lon_into() then matches query points against.

    Args:
        track: A track DataFrame for one iceberg.
        pad_deg: Padding added on every side, degrees.

    Returns:
        (min_lon, max_lon, min_lat, max_lat). min_lon may be < -180 or
        max_lon > 180 for an antimeridian-crossing track.
    """
    lats = track["lat"].to_numpy()
    lons = track["lon"].to_numpy()

    min_lat = float(lats.min()) - pad_deg
    max_lat = float(lats.max()) + pad_deg

    naive_span = float(lons.max() - lons.min())
    if naive_span > 180.0:
        # Re-express in [0, 360) and check whether the track is compact
        # there instead; if so, the crossing is at 180, not at 0.
        shifted = lons % 360.0
        if float(shifted.max() - shifted.min()) < naive_span:
            return (
                float(shifted.min()) - pad_deg,
                float(shifted.max()) + pad_deg,
                min_lat,
                max_lat,
            )

    return float(lons.min()) - pad_deg, float(lons.max()) + pad_deg, min_lat, max_lat


def fetch_copernicus_currents(
    bbox: tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    out_path: str | Path,
    dataset_id: str = config.COPERNICUS_DATASET_ID,
    force_refresh: bool = False,
) -> str:
    """Fetch daily-mean surface currents (uo, vo) for one bounding box.

    Currents are fetched per iceberg rather than circumpolar: the CMEMS
    product is 1/12 degree, so a full Southern Ocean band over four
    months would be terabytes, while a padded box around one berg's
    actual path is a few megabytes. The toolbox is a direct download
    with no queue, so many small requests are cheap.

    Requires Copernicus Marine credentials (`copernicusmarine login`,
    stored in ~/.copernicusmarine); see
    https://help.marine.copernicus.eu/en/articles/7970514

    Args:
        bbox: (min_lon, max_lon, min_lat, max_lat), as returned by
            track_bbox() -- longitudes may run past +/-180 for an
            antimeridian-crossing track.
        start_date: Inclusive ISO start date.
        end_date: Inclusive ISO end date.
        out_path: Destination NetCDF path.
        dataset_id: CMEMS dataset identifier.
        force_refresh: Re-download even if the file already exists.

    Returns:
        The out_path as a string.

    Raises:
        RuntimeError: If copernicusmarine is missing, credentials are
            not configured, or the request fails.
    """
    out_path = Path(out_path)
    os.makedirs(out_path.parent, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 10_000 and not force_refresh:
        return str(out_path)

    try:
        import copernicusmarine
    except ImportError as exc:
        raise RuntimeError(
            "fetch_copernicus_currents: the 'copernicusmarine' package is not installed. "
            "Install it with `pip install copernicusmarine`."
        ) from exc

    min_lon, max_lon, min_lat, max_lat = bbox
    try:
        copernicusmarine.subset(
            dataset_id=dataset_id,
            variables=["uo", "vo"],
            minimum_longitude=min_lon,
            maximum_longitude=max_lon,
            minimum_latitude=max(min_lat, -90.0),
            maximum_latitude=min(max_lat, 90.0),
            start_datetime=start_date,
            end_datetime=end_date,
            # Shallowest model level only; the product's first level is
            # ~0.49 m, so this selects a single depth.
            minimum_depth=0.0,
            maximum_depth=1.0,
            output_directory=str(out_path.parent),
            output_filename=out_path.name,
            overwrite=True,
            disable_progress_bar=True,
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise RuntimeError(
            "fetch_copernicus_currents: the Copernicus Marine request failed. The usual "
            "cause is missing credentials -- run `copernicusmarine login`, or see "
            "https://help.marine.copernicus.eu/en/articles/7970514 . "
            f"Requested bbox={bbox}, {start_date}..{end_date}. Original error: {exc}"
        ) from exc

    return str(out_path)


# =====================================================================
# Environmental sampling
# =====================================================================


def _find_variable(dataset, candidates: Sequence[str], role: str, source: str) -> str:
    """Find the first matching variable or coordinate name in an xarray Dataset.

    Args:
        dataset: The xarray Dataset to search.
        candidates: Candidate names, in priority order.
        role: Human-readable description, used in the error message.
        source: Source description, used in the error message.

    Returns:
        The first candidate present in the dataset.

    Raises:
        ValueError: If none of the candidates are present.
    """
    available = set(dataset.variables.keys()) | set(dataset.coords.keys())
    found = next((c for c in candidates if c in available), None)
    if found is None:
        raise ValueError(
            f"Could not find the '{role}' field in {source}. Looked for {list(candidates)}; "
            f"the file has {sorted(available)}."
        )
    return found


def _wrap_lon_into(lons: np.ndarray, axis_min: float, axis_max: float) -> np.ndarray:
    """Shift query longitudes by multiples of 360 to land inside a dataset's axis.

    Needed because an antimeridian-crossing subset comes back on a
    continuous axis such as 177..183, while the NIC fixes inside it are
    reported as, say, -179.7. Adding 360 to that gives 180.3, which is
    the same meridian expressed in the axis's own convention.

    Args:
        lons: Query longitudes, degrees.
        axis_min: Minimum longitude of the dataset's axis.
        axis_max: Maximum longitude of the dataset's axis.

    Returns:
        The query longitudes shifted by whichever multiple of 360 places
        the most of them inside [axis_min, axis_max].
    """
    lons = np.asarray(lons, dtype=float)
    best = lons
    best_inside = int(((lons >= axis_min) & (lons <= axis_max)).sum())
    for shift in (-360.0, 360.0):
        candidate = lons + shift
        inside = int(((candidate >= axis_min) & (candidate <= axis_max)).sum())
        if inside > best_inside:
            best, best_inside = candidate, inside
    return best


def _sample_points(
    dataset,
    times: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    var_names: Sequence[str],
    source: str,
) -> dict[str, np.ndarray]:
    """Nearest-neighbour sample a gridded dataset at arbitrary (t, lat, lon) points.

    Uses xarray's pointwise ("advanced") indexing -- the query arrays
    share a common `points` dimension -- so all lookups happen in one
    vectorized call rather than a Python loop over thousands of points.

    Args:
        dataset: An open xarray Dataset with time/lat/lon coordinates.
        times: Query times, datetime64.
        lats: Query latitudes, degrees.
        lons: Query longitudes, degrees.
        var_names: Names of the data variables to extract.
        source: Description used in error messages.

    Returns:
        A dict mapping each requested variable name to a 1-D array of
        sampled values, one per query point.
    """
    import xarray as xr

    time_name = _find_variable(dataset, ["valid_time", "time"], "time coordinate", source)
    lat_name = _find_variable(dataset, ["latitude", "lat"], "latitude coordinate", source)
    lon_name = _find_variable(dataset, ["longitude", "lon"], "longitude coordinate", source)

    axis = dataset[lon_name].values
    query_lons = _wrap_lon_into(lons, float(axis.min()), float(axis.max()))

    selection = dataset.sel(
        {
            time_name: xr.DataArray(times, dims="points"),
            lat_name: xr.DataArray(np.asarray(lats, dtype=float), dims="points"),
            lon_name: xr.DataArray(query_lons, dims="points"),
        },
        method="nearest",
    )

    out: dict[str, np.ndarray] = {}
    for name in var_names:
        values = selection[name]
        # CMEMS current files carry a length-1 depth dimension; drop any
        # leftover singleton axes so each variable comes back 1-D.
        extra_dims = [d for d in values.dims if d != "points"]
        if extra_dims:
            values = values.isel({d: 0 for d in extra_dims})
        out[name] = np.asarray(values.values, dtype=float)
    return out


def _dataset_time_bounds(dataset, source: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the first and last time coordinate values of a dataset.

    Args:
        dataset: An open xarray Dataset.
        source: Description used in error messages.

    Returns:
        A (first, last) pair of pandas Timestamps.
    """
    time_name = _find_variable(dataset, ["valid_time", "time"], "time coordinate", source)
    values = dataset[time_name].values
    return pd.Timestamp(values.min()), pd.Timestamp(values.max())


def sample_environment_along_segments(
    track: pd.DataFrame,
    wind_dataset,
    current_dataset,
    sample_hours: int = config.SEGMENT_SAMPLE_HOURS,
) -> pd.DataFrame:
    """Populate a track's wind/current columns with SEGMENT-MEAN forcing.

    For each pair of consecutive fixes, this walks the geodesic between
    them at `sample_hours` intervals, samples wind and current at every
    intermediate (time, lat, lon), and averages. The mean is written to
    the LATER row, so row k's forcing describes the interval (t[k-1],
    t[k]] and pairs exactly with the mean velocity features.py derives
    for that same row from the two positions.

    This matters because NIC fixes are ~a week apart. An instantaneous
    ERA5 sample at the moment of the fix is one point drawn from a
    week of storm-scale variability; comparing it against a week-mean
    observed velocity makes the physics baseline close to meaningless.
    The great-circle interpolation is of course only an estimate of the
    path actually taken -- but it is a far better one than assuming the
    berg sat at its start (or end) point for the whole week, and any
    residual path error is exactly what the ML stage is there to absorb.

    Row 0 of the track has no preceding segment; it receives the
    instantaneous value at its own fix and a segment_hours of NaN, and is
    only ever used as lag history, never as a training label.

    Rows whose segment is not fully covered by both datasets are
    returned with NaN forcing (and are dropped by build_real_dataset()),
    rather than being silently filled with an out-of-range edge value.

    Args:
        track: A single iceberg's track DataFrame (TRACK_SCHEMA_COLUMNS).
        wind_dataset: Open xarray Dataset with u10/v10.
        current_dataset: Open xarray Dataset with uo/vo.
        sample_hours: Spacing of the along-segment samples, hours.

    Returns:
        A copy of the track with u_wind/v_wind/u_current/v_current filled
        and a `segment_hours` column added.
    """
    df = track.sort_values("timestamp").reset_index(drop=True).copy()
    n = len(df)

    wind_u = _find_variable(wind_dataset, ["u10", "u_wind"], "10 m u-wind", "the wind dataset")
    wind_v = _find_variable(wind_dataset, ["v10", "v_wind"], "10 m v-wind", "the wind dataset")
    cur_u = _find_variable(current_dataset, ["uo", "u_current"], "eastward current", "the current dataset")
    cur_v = _find_variable(current_dataset, ["vo", "v_current"], "northward current", "the current dataset")

    wind_t0, wind_t1 = _dataset_time_bounds(wind_dataset, "the wind dataset")
    cur_t0, cur_t1 = _dataset_time_bounds(current_dataset, "the current dataset")
    # A daily-mean current field stamped at 00:00 legitimately represents
    # the whole of that day, so allow a day of slack at each end.
    cur_t0 -= pd.Timedelta(hours=12)
    cur_t1 += pd.Timedelta(hours=12)

    timestamps = df["timestamp"].to_numpy()
    lats = df["lat"].to_numpy()
    lons = df["lon"].to_numpy()

    # Build every sample point for every segment up front, tagged with
    # its segment index, so both datasets are queried exactly once.
    sample_times: list[np.datetime64] = []
    sample_lats: list[float] = []
    sample_lons: list[float] = []
    segment_of_sample: list[int] = []
    segment_hours = np.full(n, np.nan)
    covered = np.zeros(n, dtype=bool)

    for k in range(1, n):
        t_start = pd.Timestamp(timestamps[k - 1])
        t_end = pd.Timestamp(timestamps[k])
        hours = (t_end - t_start).total_seconds() / 3600.0
        segment_hours[k] = hours

        in_wind = wind_t0 <= t_start and t_end <= wind_t1
        in_cur = cur_t0 <= t_start and t_end <= cur_t1
        if not (in_wind and in_cur):
            continue
        covered[k] = True

        n_samples = int(np.clip(round(hours / sample_hours), 2, 400))
        # Geod.npts returns the intermediate points only, so asking for
        # n_samples - 2 and adding the endpoints gives an evenly spaced
        # set of n_samples points along the great circle.
        interior = _GEOD.npts(
            lons[k - 1], lats[k - 1], lons[k], lats[k], max(n_samples - 2, 0)
        )
        seg_lonlat = [(lons[k - 1], lats[k - 1]), *interior, (lons[k], lats[k])]
        seg_times = pd.date_range(t_start, t_end, periods=len(seg_lonlat))

        for (lon_s, lat_s), t_s in zip(seg_lonlat, seg_times):
            sample_lons.append(float(lon_s))
            sample_lats.append(float(lat_s))
            sample_times.append(np.datetime64(t_s))
            segment_of_sample.append(k)

    # Row 0 gets the instantaneous value at its own fix, if covered.
    t0 = pd.Timestamp(timestamps[0])
    if wind_t0 <= t0 <= wind_t1 and cur_t0 <= t0 <= cur_t1:
        sample_times.append(np.datetime64(t0))
        sample_lats.append(float(lats[0]))
        sample_lons.append(float(lons[0]))
        segment_of_sample.append(0)
        covered[0] = True

    for col in ("u_wind", "v_wind", "u_current", "v_current"):
        df[col] = np.nan
    df["segment_hours"] = segment_hours

    if not sample_times:
        return df

    times_arr = np.array(sample_times)
    lats_arr = np.array(sample_lats)
    lons_arr = np.array(sample_lons)

    wind_vals = _sample_points(
        wind_dataset, times_arr, lats_arr, lons_arr, [wind_u, wind_v], "the wind dataset"
    )
    cur_vals = _sample_points(
        current_dataset, times_arr, lats_arr, lons_arr, [cur_u, cur_v], "the current dataset"
    )

    # Average each segment's samples. groupby on the segment index is the
    # vectorized equivalent of a per-segment Python loop.
    samples = pd.DataFrame(
        {
            "segment": segment_of_sample,
            "u_wind": wind_vals[wind_u],
            "v_wind": wind_vals[wind_v],
            "u_current": cur_vals[cur_u],
            "v_current": cur_vals[cur_v],
        }
    )
    # CMEMS masks land/ice cells as NaN; skipna keeps a segment that
    # clips a coastal cell usable instead of voiding the whole week.
    means = samples.groupby("segment").mean()

    for col in ("u_wind", "v_wind", "u_current", "v_current"):
        df.loc[means.index, col] = means[col].to_numpy()

    df.loc[~covered, ["u_wind", "v_wind", "u_current", "v_current"]] = np.nan
    return df


def merge_environmental_data(
    track_df: pd.DataFrame, wind_nc_path: str, current_nc_path: str
) -> pd.DataFrame:
    """Populate a track's wind/current columns by INSTANTANEOUS point sampling.

    This is the pointwise counterpart to
    sample_environment_along_segments(): each row gets the value at its
    own timestamp and position, with no averaging. It is the right
    choice for a densely-sampled track (hours between fixes, where the
    instantaneous value is a fine proxy for the interval mean) and for
    ad-hoc lookups; for the ~weekly NIC record, prefer the segment-mean
    version, which build_real_dataset() uses.

    Args:
        track_df: A track DataFrame with timestamp/lat/lon/area_km2 set.
        wind_nc_path: Path to a NetCDF file with 10 m u/v wind.
        current_nc_path: Path to a NetCDF file with uo/vo currents.

    Returns:
        A track DataFrame with all eight schema columns populated.

    Raises:
        RuntimeError: If xarray is not installed.
        ValueError: If a required variable is missing, or if any row
            falls outside a file's coverage (naming the offending row).
    """
    try:
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError(
            "merge_environmental_data: the 'xarray' package is not installed. Install it "
            "with `pip install xarray netCDF4`."
        ) from exc

    df = track_df.sort_values("timestamp").reset_index(drop=True).copy()

    with xr.open_dataset(wind_nc_path) as ds_wind, xr.open_dataset(current_nc_path) as ds_cur:
        for dataset, source in ((ds_wind, wind_nc_path), (ds_cur, current_nc_path)):
            t_min, t_max = _dataset_time_bounds(dataset, source)
            outside = (df["timestamp"] < t_min) | (df["timestamp"] > t_max)
            if outside.any():
                bad = int(np.argmax(outside.to_numpy()))
                raise ValueError(
                    f"merge_environmental_data: row {bad} (timestamp="
                    f"{df['timestamp'].iloc[bad]}, lat={df['lat'].iloc[bad]}, "
                    f"lon={df['lon'].iloc[bad]}) falls outside the time coverage of "
                    f"'{source}' ([{t_min}, {t_max}]). Re-fetch that source over a wider "
                    f"date range, or trim the track to the covered period."
                )

        wind_u = _find_variable(ds_wind, ["u10", "u_wind"], "10 m u-wind", wind_nc_path)
        wind_v = _find_variable(ds_wind, ["v10", "v_wind"], "10 m v-wind", wind_nc_path)
        cur_u = _find_variable(ds_cur, ["uo", "u_current"], "eastward current", current_nc_path)
        cur_v = _find_variable(ds_cur, ["vo", "v_current"], "northward current", current_nc_path)

        times = df["timestamp"].to_numpy()
        lats = df["lat"].to_numpy()
        lons = df["lon"].to_numpy()

        wind_vals = _sample_points(ds_wind, times, lats, lons, [wind_u, wind_v], wind_nc_path)
        cur_vals = _sample_points(ds_cur, times, lats, lons, [cur_u, cur_v], current_nc_path)

    df["u_wind"] = wind_vals[wind_u]
    df["v_wind"] = wind_vals[wind_v]
    df["u_current"] = cur_vals[cur_u]
    df["v_current"] = cur_vals[cur_v]
    return df[TRACK_SCHEMA_COLUMNS]


# =====================================================================
# Orchestration
# =====================================================================


def _cache_key(*parts: object) -> str:
    """Build a short, stable cache key from an arbitrary set of values.

    Args:
        *parts: Values that together identify a cached artifact.

    Returns:
        A 16-character hex digest suitable for use inside a filename.
    """
    return hashlib.sha256("_".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:16]


def _load_cached_pooled(
    cache_dir: str | Path, verbose: bool = True
) -> pd.DataFrame | None:
    """Recover the most recent cached pooled dataset, if one exists.

    Used only as a fallback when the NIC snapshot CSVs are unavailable.
    Because the cache filename is keyed by the inputs that produced it,
    the newest file is taken and the caller is told loudly which one --
    silently serving a stale dataset would be worse than failing.

    Args:
        cache_dir: Directory holding real_track_pooled_*.csv files.
        verbose: Print which cache file was recovered.

    Returns:
        The pooled DataFrame, or None if no cached dataset exists.
    """
    candidates = sorted(
        Path(cache_dir).glob("real_track_pooled_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None

    pooled = pd.read_csv(candidates[0], parse_dates=["timestamp"])
    if verbose:
        warnings.warn(
            f"build_real_dataset: no NIC snapshot CSVs found in the data directory, so "
            f"the cached dataset '{candidates[0].name}' is being used instead "
            f"({len(pooled)} rows, {pooled['iceberg_id'].nunique()} icebergs, "
            f"{pooled['timestamp'].min():%Y-%m-%d}..{pooled['timestamp'].max():%Y-%m-%d}). "
            f"This cannot be refreshed or extended until the snapshots are restored -- "
            f"re-download them from https://usicecenter.gov/Products/AntarcIcebergs into "
            f"{config.DATA_DIR}.",
            stacklevel=3,
        )
    return pooled[POOLED_SCHEMA_COLUMNS]


def build_real_dataset(
    data_dir: str | Path | None = None,
    include_grounded: bool = False,
    max_segment_days: float = config.MAX_SEGMENT_DAYS,
    min_observations: int = 4,
    cache_dir: str | Path = config.CACHE_DIR,
    force_refresh: bool = False,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the pooled, environmentally-forced real training table end to end.

    Pipeline: read every dated NIC snapshot -> group into per-iceberg
    tracks -> classify drifting vs grounded -> fetch circumpolar ERA5
    wind and per-iceberg CMEMS currents -> average both over each
    inter-fix segment -> concatenate into one pooled table.

    Everything expensive is cached under cache_dir, keyed by its own
    inputs, so a re-run costs seconds rather than re-downloading.

    Args:
        data_dir: Directory holding the NIC snapshot CSVs; defaults to
            config.DATA_DIR.
        include_grounded: Keep icebergs classified as grounded. Off by
            default -- see summarize_iceberg_motion() for why they
            corrupt free-drift residual training.
        max_segment_days: Drop segments longer than this; a single mean
            velocity over more than about three weeks is a very weak
            label and free drift is not meaningful at that timescale.
        min_observations: Drop icebergs with fewer fixes than this.
        cache_dir: Root of the NetCDF/track cache.
        force_refresh: Bypass every cache and re-fetch.
        verbose: Print per-stage progress.

    Returns:
        A (pooled_df, motion_summary) tuple. pooled_df has exactly
        POOLED_SCHEMA_COLUMNS, sorted by iceberg then time, with no NaN
        forcing. motion_summary is the full
        summarize_iceberg_motion() table for EVERY iceberg found,
        including those excluded -- so the dashboard can still plot the
        grounded ones and the report can say how many were set aside.

    Raises:
        FileNotFoundError: If no NIC snapshots are found.
        RuntimeError: If no iceberg survives the filters, or if a data
            source cannot be fetched.
    """
    import xarray as xr

    config.ensure_dirs()
    cache_dir = Path(cache_dir)
    data_dir = Path(data_dir) if data_dir is not None else config.DATA_DIR

    try:
        snapshots = load_nic_snapshots(data_dir)
    except FileNotFoundError:
        # The NIC snapshot CSVs are the only input that cannot be
        # regenerated locally -- they are downloaded, not derived. If they
        # are missing but a previously-built pooled dataset survives in
        # the cache, use it: the app and every evaluation still work, and
        # failing outright here would take the whole system down over
        # source files that only a full rebuild actually needs.
        recovered = _load_cached_pooled(cache_dir, verbose=verbose)
        if recovered is None:
            raise
        return recovered, summarize_iceberg_motion(
            {
                str(berg): group.reset_index(drop=True)[TRACK_SCHEMA_COLUMNS]
                for berg, group in recovered.groupby("iceberg_id")
            }
        )
    tracks = build_iceberg_tracks(snapshots, min_observations=min_observations)
    summary = summarize_iceberg_motion(tracks)

    if verbose:
        n_drift = int((~summary["is_grounded"]).sum())
        print(
            f"[data] {len(snapshots)} fixes | {snapshots['timestamp'].nunique()} snapshot dates "
            f"({snapshots['timestamp'].min():%Y-%m-%d} .. {snapshots['timestamp'].max():%Y-%m-%d})"
        )
        print(
            f"[data] {len(tracks)} icebergs with >= {min_observations} fixes: "
            f"{n_drift} drifting, {len(summary) - n_drift} grounded/not re-observed"
        )

    keep_ids = set(
        summary["iceberg_id"] if include_grounded else summary.loc[~summary["is_grounded"], "iceberg_id"]
    )
    tracks = {k: v for k, v in tracks.items() if k in keep_ids}
    if not tracks:
        raise RuntimeError(
            "build_real_dataset: no icebergs survived filtering. Every tracked berg was "
            "classified as grounded (mean speed below "
            f"{config.GROUNDED_SPEED_THRESHOLD_MS} m/s). Pass include_grounded=True to "
            "override, or lower GROUNDED_SPEED_THRESHOLD_MS in config.py."
        )

    start_date = snapshots["timestamp"].min().strftime("%Y-%m-%d")
    end_date = snapshots["timestamp"].max().strftime("%Y-%m-%d")

    pooled_key = _cache_key(
        "pooled", sorted(tracks), start_date, end_date, max_segment_days,
        include_grounded, config.ERA5_GRID_DEG, config.SEGMENT_SAMPLE_HOURS,
    )
    pooled_path = cache_dir / f"real_track_pooled_{pooled_key}.csv"
    if pooled_path.exists() and not force_refresh:
        if verbose:
            print(f"[cache] reusing pooled dataset {pooled_path.name}")
        pooled = pd.read_csv(pooled_path, parse_dates=["timestamp"])
        return pooled[POOLED_SCHEMA_COLUMNS], summary

    # ERA5 needs a little slack at the start so the first segment's
    # samples (which begin at the first fix) are inside the archive.
    wind_start = (pd.Timestamp(start_date) - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    if verbose:
        print(f"[wind] fetching circumpolar ERA5 {wind_start} .. {end_date} "
              f"(lat {config.ERA5_LAT_BAND[0]}..{config.ERA5_LAT_BAND[1]}, "
              f"{config.ERA5_GRID_DEG} deg) -- cached after the first run")
    wind_paths = fetch_era5_wind(wind_start, end_date, force_refresh=force_refresh)

    # combine="by_coords" stitches the monthly files along time; the
    # ERA5 `expver`/`number` scalars differ between final and preliminary
    # months, so they are dropped rather than promoted to a dimension.
    wind_ds = xr.open_mfdataset(
        wind_paths, combine="by_coords", drop_variables=["expver", "number"]
    ).load()
    wind_t0, wind_t1 = _dataset_time_bounds(wind_ds, "the ERA5 wind files")
    if verbose:
        print(f"[wind] coverage {wind_t0:%Y-%m-%d %H:%M} .. {wind_t1:%Y-%m-%d %H:%M}")

    forced: list[pd.DataFrame] = []
    for iceberg_id, track in sorted(tracks.items()):
        bbox = track_bbox(track)
        cur_key = _cache_key("cur", iceberg_id, np.round(bbox, 3).tolist(), start_date, end_date)
        cur_path = config.CURRENTS_CACHE_DIR / f"currents_{iceberg_id}_{cur_key}.nc"
        if verbose and not cur_path.exists():
            print(f"[currents] {iceberg_id}: fetching CMEMS box "
                  f"lon[{bbox[0]:.1f},{bbox[1]:.1f}] lat[{bbox[2]:.1f},{bbox[3]:.1f}]")
        fetch_copernicus_currents(bbox, start_date, end_date, cur_path, force_refresh=force_refresh)

        with xr.open_dataset(cur_path) as cur_ds:
            forced_track = sample_environment_along_segments(track, wind_ds, cur_ds)

        forced_track.insert(0, "iceberg_id", iceberg_id)
        forced.append(forced_track)

    wind_ds.close()

    pooled = pd.concat(forced, ignore_index=True)

    # --- Drop rows that cannot serve as training labels --------------
    env_cols = ["u_wind", "v_wind", "u_current", "v_current"]
    n_before = len(pooled)
    no_forcing = pooled[env_cols].isna().any(axis=1)
    too_long = pooled["segment_hours"] > max_segment_days * 24.0
    # Row 0 of each iceberg has no segment (segment_hours is NaN) but is
    # still needed as lag history, so it is kept as long as it has
    # forcing; features.py never uses it as a label.
    drop = no_forcing | too_long.fillna(False)
    pooled = pooled.loc[~drop].reset_index(drop=True)

    if verbose:
        print(
            f"[filter] dropped {int(no_forcing.sum())} rows without full environmental "
            f"coverage (ERA5 ends {wind_t1:%Y-%m-%d}) and {int(too_long.fillna(False).sum())} "
            f"rows whose segment exceeded {max_segment_days:.0f} days; "
            f"{len(pooled)}/{n_before} rows kept"
        )

    # An iceberg reduced to a single fix contributes no velocity at all.
    counts = pooled.groupby("iceberg_id")["timestamp"].transform("size")
    pooled = pooled.loc[counts >= 2].reset_index(drop=True)

    if pooled.empty:
        raise RuntimeError(
            "build_real_dataset: every row was filtered out. Check that the ERA5 archive "
            "covers the snapshot dates and that the CMEMS boxes cover the tracks."
        )

    pooled = pooled.sort_values(["iceberg_id", "timestamp"]).reset_index(drop=True)
    pooled = pooled[POOLED_SCHEMA_COLUMNS]
    pooled.to_csv(pooled_path, index=False)
    if verbose:
        print(
            f"[data] pooled real dataset: {len(pooled)} rows across "
            f"{pooled['iceberg_id'].nunique()} icebergs -> {pooled_path.name}"
        )
    return pooled, summary


if __name__ == "__main__":
    # Real-data demo. Requires the NIC CSVs in data/ plus CDS and
    # Copernicus Marine credentials; the first run downloads, later runs
    # hit the cache. Falls back to the synthetic generator so the module
    # still self-tests with no network or credentials available.
    try:
        pooled, summary = build_real_dataset()

        print("\nMotion classification (all tracked icebergs):")
        with pd.option_context("display.width", 160, "display.max_rows", 60):
            print(
                summary[
                    ["iceberg_id", "n_obs", "total_km", "mean_speed_ms", "mean_area_km2", "is_grounded"]
                ].to_string(index=False, float_format=lambda v: f"{v:9.4f}")
            )

        assert list(pooled.columns) == POOLED_SCHEMA_COLUMNS, "pooled schema mismatch"
        assert pooled["lat"].between(-90, 90).all(), "lat out of range"
        assert pooled["lon"].between(-180, 180).all(), "lon out of range"
        assert pooled[["u_wind", "v_wind", "u_current", "v_current"]].notna().all().all(), (
            "pooled table still contains unforced rows"
        )

        print(f"\nPooled dataset: {pooled.shape[0]} rows x {pooled.shape[1]} cols")
        print(f"Icebergs: {sorted(pooled['iceberg_id'].unique())}")
        print("\nFirst 5 rows:")
        print(pooled.head(5).to_string(index=False))
        print("\nForcing summary:")
        print(pooled[["u_wind", "v_wind", "u_current", "v_current", "segment_hours"]].describe())

    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Real-data path unavailable ({type(exc).__name__}: {exc})")
        print("Falling back to the synthetic generator.\n")
        track = generate_synthetic_track()
        assert list(track.columns) == TRACK_SCHEMA_COLUMNS, "schema column mismatch"
        os.makedirs(config.DATA_DIR, exist_ok=True)
        track.to_csv(config.DATA_DIR / "synthetic_track.csv", index=False)
        print(track.head(3).to_string(index=False))
