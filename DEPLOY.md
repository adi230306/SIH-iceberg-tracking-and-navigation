# Hosting this dashboard for free

The app serves with **no API keys**: the iceberg database (`data/byu/`) and the
pre-built forced dataset (`data/cache/real_track_pooled_byu_*.csv`) are both in
the repository, and the pipeline reads those before it would reach for ERA5 or
Copernicus. So any host that can run a Python web process will work.

What it needs at runtime:

| | |
| --- | --- |
| Memory | ~290 MB peak (measured) |
| Startup | ~10 s — builds the dataset, fits the physics, trains the residual model |
| Disk | ~30 MB |
| Network | Outbound to `open-meteo.com` for live forecasts (optional; it falls back) |
| Secrets | none |

The 10-second startup is why every option below sets a generous worker timeout.

---

## Option 1 — Hugging Face Spaces (recommended)

Free, 16 GB RAM, and it does **not** sleep after a few minutes of inactivity —
which matters if you are sending the link to judges who may open it cold.

1. Create a Space at <https://huggingface.co/new-space>
   - **SDK: Docker** (not Gradio or Streamlit)
   - Hardware: *CPU basic* (free)
2. Push this repository to the Space:

```bash
git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
git push space main
```

3. Add this to the very top of `README.md` (Spaces reads it as configuration):

```yaml
---
title: Iceberg Tracking and Navigation System
emoji: 🧊
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
---
```

The included `Dockerfile` already listens on 7860, which is the port Spaces
routes to. First build takes a few minutes; after that it is live at
`https://huggingface.co/spaces/<your-username>/<space-name>`.

> If `git push space main` is rejected for size, it is the `data/byu/` directory
> (25 MB, 647 files). That is under every limit, but Spaces wants large binary
> files via LFS — run `git lfs install && git lfs track "data/byu/*.csv"` if you
> hit it.

---

## Option 2 — Render

Free web service, straightforward, gives you a `*.onrender.com` URL.

1. <https://render.com> → **New** → **Web Service** → connect the GitHub repo
2. Settings:
   - **Runtime:** Python 3
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:**
     ```
     gunicorn main:server --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 180
     ```
   - **Instance type:** Free

The included `Procfile` carries the same command, so Render will usually pick it
up without you typing it.

**The catch:** the free tier sleeps after 15 minutes of inactivity, and because
of the 10-second startup a cold request takes roughly a minute to answer. Fine
for a link you are actively demoing, poor for one someone opens unannounced.
Render's free tier is also capped at 512 MB — the app fits at ~290 MB, but there
is not a lot of headroom.

---

## Option 3 — Fly.io

Free allowance, no sleep, and the `Dockerfile` is already there.

```bash
fly launch --no-deploy      # accept the detected Dockerfile
fly deploy
```

Then in `fly.toml` set `internal_port = 7860` to match the image, and give the
machine 512 MB:

```toml
[[vm]]
  memory = "512mb"
```

---

## Verifying before you deploy

Run exactly what the host will run:

```bash
pip install -r requirements.txt
PORT=8050 HOST=0.0.0.0 DASH_DEBUG=0 \
  gunicorn main:server --bind 0.0.0.0:8050 --workers 1 --threads 4 --timeout 180
```

Then open <http://127.0.0.1:8050>. If that works, the deployment will too — this
is the same entry point (`main:server`) every option above uses.

To check it truly needs no credentials, run it from a clean copy:

```bash
git clone <this-repo> /tmp/deploycheck && cd /tmp/deploycheck
pip install -r requirements.txt
HOME=/nonexistent python main.py
```

`HOME=/nonexistent` hides `~/.cdsapirc` and `~/.copernicusmarine`, so if it
starts, nothing is secretly depending on your local credentials.

---

## Notes

**One worker, on purpose.** Startup trains a model in memory; a second worker
would repeat all of it and double the memory for no benefit. Threads handle
concurrency instead, which is right for a read-mostly dashboard.

**Do not deploy with `python main.py`.** That is Dash's development server —
single-threaded and explicitly not for hosting. Use `gunicorn main:server`.

**The NetCDF cache is not in the repository** (~750 MB) and is not needed. It is
only used when rebuilding the dataset over a new date range, which is a local
task. `.dockerignore` excludes it so it cannot creep into an image.
