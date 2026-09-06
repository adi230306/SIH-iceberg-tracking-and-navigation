"""
config.py

Single source of truth for filesystem paths and the handful of tunable
constants that the real-data pipeline shares across modules.

Two things motivate this module:

1. PATHS. The project is run from several different working directories
   (``python src/train_on_real_data.py`` from the repo root, ``python
   data_ingest.py`` from inside ``src/``, ``streamlit run app.py``, a
   notebook, ...). Hard-coded relative paths like ``"data/cache"`` break
   under all but one of those. Everything here is resolved once, from
   this file's own location, so every module agrees on where the data
   lives no matter how it was launched.

2. TUNABLES that are genuinely shared. The free-drift wind factor, the
   grounded-iceberg speed threshold, and the environmental-sampling
   cadence all appear in more than one module; keeping one definition
   avoids the classic bug where features are built with one value and
   the forecast rollout uses another.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Filesystem layout -----------------------------------------------
# This file lives at <repo>/src/config.py, alongside app/, assets/,
# data/, models/ and main.py at the repository root. Resolving the root
# from this file's own location (rather than from the working directory)
# is what lets `python main.py`, `python src/train_on_real_data.py` and a
# notebook all agree on where the data lives.
SRC_DIR: Path = Path(__file__).resolve().parent
REPO_ROOT: Path = SRC_DIR.parent
# Kept as an alias: the two were distinct when src/ sat one level deeper.
PROJECT_DIR: Path = REPO_ROOT

DATA_DIR: Path = REPO_ROOT / "data"
CACHE_DIR: Path = DATA_DIR / "cache"
ERA5_CACHE_DIR: Path = CACHE_DIR / "era5"
CURRENTS_CACHE_DIR: Path = CACHE_DIR / "currents"
MODELS_DIR: Path = REPO_ROOT / "models"

# Glob matching the USNIC Antarctic iceberg snapshot exports, e.g.
# data/AntarcticIcebergs_20260904.csv -- one row per iceberg per date.
USNIC_CSV_GLOB: str = "AntarcticIcebergs_*.csv"

# The BYU/NIC consolidated Antarctic iceberg database: one CSV per named
# iceberg, DAILY positions from scatterometer tracking, 1976-present,
# plus the iceberg's length and width in km. This is the primary
# training source -- it gives ~20x more segments than the weekly USNIC
# snapshots, at daily rather than weekly resolution.
BYU_DIR_NAME: str = "byu"
BYU_CSV_GLOB: str = "*.csv"

# Training window for the BYU record. The database is decades long, but
# each month of it needs a month of ERA5 and of ocean-current forcing
# downloaded to be usable, so the window is a deliberate trade between
# dataset size and download budget. Widen it and re-run to train on more.
BYU_START_DATE: str = "2026-01-01"
BYU_END_DATE: str = "2026-04-30"
# Icebergs with fewer daily fixes than this in the window are skipped.
BYU_MIN_FIXES: int = 20

# The BYU fixes are satellite-derived and carry a position error of the
# same order as an iceberg's DAILY displacement (a berg drifting at
# 0.04 m/s moves ~3.5 km/day). Differencing consecutive daily fixes
# therefore measures mostly noise: the raw daily speed distribution has a
# median of 0.037 m/s but an RMS of 0.28 m/s, and the residual's lag-1
# autocorrelation comes out NEGATIVE (-0.31), which is the signature of
# independent per-fix error rather than of real motion.
#
# Binning positions into multi-day windows and taking the MEDIAN
# position per bin fixes both problems at once: it averages the noise
# down and it is robust to the occasional grossly wrong fix. 2 days is
# the shortest bin at which the physics fit stops degrading.
BYU_RESAMPLE_DAYS: int = 2

# Segments implying a speed above this are discarded as bad fixes rather
# than modelled. No Antarctic iceberg sustains 0.5 m/s (43 km/day); the
# fastest in this record averages ~0.27 m/s over months.
MAX_PLAUSIBLE_SPEED_MS: float = 0.5

# Minimum "straightness" -- net displacement divided by total path
# length -- for an iceberg to count as genuinely drifting.
#
# This exists because the mean-speed test alone CANNOT identify a
# grounded iceberg in noisy daily data: position error inflates apparent
# path length without moving the berg anywhere, so a grounded berg
# reports a healthy mean speed. Iceberg C33 in this record walks 410 km
# of path and ends 6 km from where it started -- straightness 0.01. A
# genuinely drifting berg exceeds 0.3. Including these bergs roughly
# halves the fraction of drift the physics can explain, because there is
# no physics that predicts jitter.
MIN_STRAIGHTNESS: float = 0.15


def ensure_dirs() -> None:
    """Create every directory the pipeline writes into, if missing.

    Returns:
        None. Safe to call repeatedly.
    """
    for directory in (DATA_DIR, CACHE_DIR, ERA5_CACHE_DIR, CURRENTS_CACHE_DIR, MODELS_DIR):
        os.makedirs(directory, exist_ok=True)


# --- Physics defaults ------------------------------------------------
# Fraction of the 10 m wind speed transferred into iceberg drift, and the
# Coriolis-driven deflection angle between wind and wind-driven drift.
# These are literature starting points; calibrate_free_drift_params() in
# physics.py refits them (plus a current scaling factor) on the training
# icebergs, and the fitted values are what the pipeline actually uses.
DEFAULT_WIND_FACTOR: float = 0.018
DEFAULT_DEFLECTION_DEG: float = 20.0
# Multiplier on the Copernicus surface current. A large tabular iceberg's
# keel extends 150-300 m down and is dragged by the depth-averaged
# current, which is generally weaker than the 0.5 m surface value the
# satellite/model product reports -- so the physically expected fitted
# value is somewhat below 1.0.
DEFAULT_CURRENT_FACTOR: float = 1.0


# --- Real-data pipeline defaults -------------------------------------
# USNIC snapshots are ~weekly, so a "timestep" here is days, not hours.
# Wind and current are averaged over each inter-observation segment
# rather than sampled instantaneously (see data_ingest.sample_segment_
# environment) -- this is the sampling cadence used along the segment.
SEGMENT_SAMPLE_HOURS: int = 6

# A segment longer than this is dropped from training: a single mean
# velocity across a long gap is a very weak label (the iceberg may have
# looped, stalled, or grounded and released within it), and the
# free-drift baseline is not meaningful at that timescale. The BYU
# record is daily, so most segments are 1 day and this only trims the
# gaps where tracking was lost for a while.
MAX_SEGMENT_DAYS: float = 6.0  # must exceed BYU_RESAMPLE_DAYS by a margin

# Icebergs whose mean speed over the whole record is below this are
# treated as GROUNDED (or simply not re-observed -- several NIC entries
# repeat an identical position for months) and excluded from drift
# training. 0.01 m/s is ~0.86 km/day, i.e. ~6 km/week: comfortably below
# any genuinely drifting Southern Ocean berg and comfortably above
# position-rounding noise (NIC reports lat/lon to 0.01 deg, ~1.1 km).
GROUNDED_SPEED_THRESHOLD_MS: float = 0.01

# Padding, in degrees, added around an iceberg's own lat/lon envelope
# when requesting its ocean-current subset, so that nearest-neighbour
# lookups near the edges (and the interpolated points along each
# segment) stay inside the downloaded box.
CURRENT_BBOX_PAD_DEG: float = 2.0

# ERA5 is fetched once as a circumpolar band rather than per iceberg:
# one queued CDS request instead of thirty-odd. 0.5 deg is ample for
# synoptic Southern Ocean wind driving a ~2% drift term.
ERA5_LAT_BAND: tuple[float, float] = (-45.0, -78.0)  # (north, south)
ERA5_GRID_DEG: float = 0.5

# Copernicus Marine analysis/forecast surface currents, 1/12 deg daily.
COPERNICUS_DATASET_ID: str = "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m"


# --- Feature/model defaults ------------------------------------------
# Number of previous segments of history folded into each feature row.
# With the daily BYU record there are thousands of rows, so lags are
# cheap and 3 days of history is genuinely informative; the weekly USNIC
# record only supported 1.
DEFAULT_N_LAGS: int = 3

# Multi-step rollout horizon used for ADE/FDE. With the daily BYU
# record a step is one day, so this is a 7-day forecast -- the horizon
# that actually matters for routing a vessel.
DEFAULT_HORIZON_STEPS: int = 7
