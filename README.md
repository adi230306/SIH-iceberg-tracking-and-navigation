# Iceberg Tracking and Navigation System

Forecasts Antarctic iceberg drift from real satellite observations and turns it
into a navigation decision: how close an iceberg will come to a vessel, when,
and how much to trust the answer.

Built as a hybrid **physics + machine learning** system. An analytical free-drift
model, with its coefficients fitted to the observed record, predicts the bulk of
the motion; a gradient-boosted model learns the residual the physics cannot
explain. The learned correction is only used if it demonstrably beats the physics
on icebergs it has never seen — see [Results](#results).

---

## Quick start

```bash
pip install -r requirements.txt
python main.py           # http://127.0.0.1:8050
```

No API keys are needed to run it. The iceberg position database and the
pre-built forced dataset are in the repository, and the pipeline reads those
directly. Credentials are only needed to *rebuild* the dataset over a new date
range — see [Refreshing the data](#refreshing-the-data).

---

## What it does

**Landing page** → click the preview to open the live dashboard.

**Dashboard**

- Select any number of the 40 tracked icebergs; each gets its own forecast.
- Physics-only or hybrid (physics + ML) mode.
- Forecast horizon from 6 to 120 hours, driven by a live Open-Meteo wind and
  current forecast (with a graceful fall back to persisted conditions).
- Enter a vessel position for **closest point of approach**, time-to-CPA, and a
  green / amber / red risk call.
- A probability heatmap showing where the iceberg could be, from a bootstrap
  over perturbed environmental forcing.
- SHAP attributions explaining *why* the model adjusted the physics baseline.

---

## Data sources

| Source | Used for | In the repo? |
| --- | --- | --- |
| [BYU/NIC Antarctic iceberg database](https://www.scp.byu.edu/data/iceberg/) | Daily iceberg positions, 1976–2026, plus length/width | Yes — `data/byu/` (647 files) |
| [ERA5](https://cds.climate.copernicus.eu/) | 10 m wind | No — fetched, then cached |
| [Copernicus Marine](https://marine.copernicus.eu/) | Ocean surface currents | No — fetched, then cached |
| [Open-Meteo](https://open-meteo.com/) | Live forward forecast for the dashboard | No — called at runtime, no key |

The current training window is **2026-01-01 → 2026-04-30**: 40 icebergs, 1,510
forced observations, 2-day steps.

---

## Data quality: what the raw record required

The BYU positions are satellite-derived and cannot be differenced naively. Three
problems had to be handled before any modelling, and each materially changes the
result:

**Null-island fixes.** Roughly 3% of rows carry a validity flag of 1 but a
literal `(0.0, 0.0)` position. Latitude 0 is a legal value, so nothing downstream
would reject it — an Antarctic iceberg would simply appear to teleport to the
equator and back, and the resulting velocity would dominate any least-squares
fit. These rows are dropped.

**Position error comparable to daily motion.** A berg drifting at 0.04 m/s moves
~3.5 km/day, which is the same order as the position uncertainty. Differencing
consecutive daily fixes therefore measures mostly noise: the raw daily speed
distribution has a median of 0.037 m/s but an RMS of 0.28 m/s, and the residual's
lag-1 autocorrelation comes out **negative** (−0.31), the signature of
independent per-fix error rather than real motion. Positions are binned into
2-day windows using the **median** — robust to the occasional grossly wrong fix
in a way the mean is not.

**Grounded icebergs that look like they are moving.** A mean-speed test cannot
find them, because position noise inflates apparent path length without moving
the berg anywhere. Iceberg C33 walks 410 km of path and ends **6 km** from where
it started. They are identified by *straightness* — net displacement divided by
path length — and excluded from training. Including them roughly halves the
fraction of drift the physics can explain, since no physics predicts jitter.

Grounded bergs are still **displayed** (they remain hazards) and are marked `•`
in the picker. 22 of the 40 are actively drifting and used for fitting.

---

## Results

Leave-one-iceberg-out evaluation — each iceberg is forecast by a model that never
saw it, which is the honest test for a berg that has just calved. Errors are
displacement in kilometres over a 7-step (~14-day) autoregressive rollout, not
single-step regression error.

| Model | Final displacement error | Movement predicted | Within 50 km |
| --- | --- | --- | --- |
| Persistence (holds last velocity) | 102.6 km | 4% | 31% |
| **Calibrated free-drift physics** | **61.9 km** | **45%** | **58%** |
| Hybrid (physics + XGBoost residual) | 63.6 km | 44% | 57% |

- **Movement predicted** — `100% − error / distance actually travelled`. The
  icebergs walk 108 km of track over a fortnight and the forecast lands 57 km
  from the truth, so it captures about 45% of the movement.
- **Within 50 km** — the share of forecasts landing inside 50 km. The most
  directly useful figure for a navigator, since it is a tolerance rather than a
  ratio. 91% land within 100 km.

Three denominators are defensible and they give very different numbers, so the
label matters more than the value:

| Measured against | Result |
| --- | --- |
| distance actually travelled (108 km of path) | 45% |
| persistence's error (103 km) | 44% |
| assuming the iceberg never moves (67 km net displacement) | 14% |

The last is the harshest and the most revealing: a drifting berg wanders, so its
*net* displacement stays small while error accumulates along the way. Worth
knowing that **persistence is itself worse than assuming no movement** here
(103 km against 67 km) — extrapolating a noisy observed velocity compounds that
noise — so "better than persistence" flatters any model and is not quoted as the
headline.

Both are computed on drifting icebergs only. Scoring against grounded bergs
would flatter the kilometre error and destroy the percentage, since they barely
move: the denominator collapses and the figure stops being about forecast
quality at all.

Two findings worth stating plainly:

**Calibrating the physics is what buys the accuracy.** With literature
coefficients the baseline is *worse than predicting nothing* (residual RMS
0.142 m/s against an observed 0.087 m/s). Fitting `wind_factor`,
`deflection_deg` and `current_factor` to the record turns it into a model that
roughly halves persistence error.

**The ML correction does not currently earn its place.** On held-out icebergs it
is a statistical tie with physics, so `select_forecast_mode()` ships physics-only
and records that decision in `models/model_meta.json`. The model is still trained
and saved, so the comparison re-runs automatically as the record grows. Several
individual bergs *do* improve under the hybrid; the aggregate does not clear the
bar, and reporting it as a win would be dishonest.

A related result the calibration surfaced: the fitted **wind factor is
essentially zero**. That is not a bug. The Copernicus product reports the total
modelled current, which already contains the wind-driven Ekman response, so the
textbook 1.8%-of-wind term double-counts it. The fitted current factor of ~0.29
is consistent with a deep-keeled berg feeling less than the surface current.

---

## Repository layout

```
main.py                 Wires the backend to the Dash frontend; the only file
                        that knows about both halves
src/
  config.py             Paths, training window, and every tunable constant
  physics.py            Free-drift model, geodesic helpers, coefficient fitting
  data_ingest.py        BYU/USNIC loaders, ERA5 + Copernicus fetch, forcing
  features.py           Observed velocity, physics residual, feature table
  train_model.py        Training, rollout ADE/FDE, leave-one-iceberg-out
  decision_support.py   Forecast rollout, uncertainty cone, CPA, risk grading
  explain.py            SHAP attributions for the ML correction
  weather_api.py        Live Open-Meteo forecast for the dashboard
  train_on_real_data.py End-to-end train + evaluate report
app/                    Dash layout, callbacks, figures, landing shell
assets/                 Stylesheets
data/byu/               BYU iceberg position database
data/cache/             Pooled dataset (committed); NetCDF cache (not committed)
models/                 Trained residual models and metadata
```

Every module runs standalone as a self-test:

```bash
python src/physics.py            # geodesic and hemisphere checks
python src/features.py           # observed-velocity round trip
python src/decision_support.py   # CPA and risk grading
python src/explain.py            # SHAP values reconstruct the predictions
python src/train_on_real_data.py # full train + evaluation report
```

---

## Configuration

Common knobs, all in `src/config.py` unless noted:

| Setting | Meaning |
| --- | --- |
| `DEFAULT_ICEBERGS` (`main.py`) | Icebergs ticked on load, e.g. `["D33C", "A81"]`. Empty means "choose the tightest cluster automatically" |
| `BYU_START_DATE` / `BYU_END_DATE` | Training window. Widening it needs a data refresh |
| `BYU_RESAMPLE_DAYS` | Position binning width, in days |
| `MIN_STRAIGHTNESS` | Grounded-iceberg cutoff |
| `DEFAULT_HORIZON_STEPS` | Rollout horizon used for ADE/FDE |

---

## Refreshing the data

Only needed to change the training window. Requires two free accounts:

```bash
# ERA5 -- https://cds.climate.copernicus.eu/how-to-api  (writes ~/.cdsapirc)
# Copernicus Marine:
copernicusmarine login
```

Then widen the window in `src/config.py` and rebuild:

```python
from data_ingest import build_real_dataset
build_real_dataset(force_refresh=True)
```

The download is roughly 200 MB of ERA5 per four months plus a small current box
per iceberg. Both are cached under `data/cache/` and ignored by git.

---

## Hosting

The app needs no credentials to serve, so it deploys as-is. See
[DEPLOY.md](DEPLOY.md) for step-by-step instructions.

```bash
gunicorn main:server --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 180
```

Startup builds the dataset, fits the physics and trains the residual model in
memory: about **10 seconds and 290 MB**, which fits every major free tier.

---

## Known limitations

- **Forecast timestamps.** The BYU record ends 2026-04-30 while Open-Meteo
  forecasts from today, so the dashboard rolls an April position forward with
  current weather. Reasonable for a demo ("where would it go from here?"), but
  the two clocks are not the same clock.
- **Fixed forecast cadence.** The dashboard steps at 6 hours while the training
  record is 2-day; the calibrated physics integrates at any step size, but the
  model-error estimate is expressed per *day* of lead time for that reason.
- **The physics explains ~7% of drift magnitude on this berg set.** The modelled
  surface current correlates only ~0.4 with actual drift here — many of these
  are coastal bergs in summer fast ice, where a 1/12° surface current is a poor
  guide and sea-ice drag dominates. More data did not fix this; it is an
  observational limit, not a tuning problem.
