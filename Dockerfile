# Container image for hosting the dashboard (Hugging Face Spaces, Fly.io,
# Render, Cloud Run -- anything that takes a Dockerfile).
#
# The image needs NO CDS or Copernicus credentials. data/byu (the iceberg
# position database) and the small pooled dataset under data/cache are in
# the repository, and the pipeline short-circuits to that pooled file
# before it would reach for either API. Refreshing the dataset from the
# APIs is a local task; see the README.
FROM python:3.12-slim

# pyproj needs PROJ, and netCDF4/xarray need the HDF5/netCDF runtime.
# Installed as runtime libraries only -- the Python wheels ship their own
# compiled extensions, so no build toolchain is required.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libproj25 libgeos-c1v5 libhdf5-103-1 libnetcdf19 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Requirements first, so a code change does not invalidate the (slow)
# dependency layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces routes to 7860; every other host sets $PORT.
ENV PORT=7860 \
    HOST=0.0.0.0 \
    DASH_DEBUG=0 \
    MPLCONFIGDIR=/tmp/mpl \
    XDG_CACHE_HOME=/tmp/cache
EXPOSE 7860

# One worker on purpose. Startup builds the dataset, calibrates the drift
# physics and trains the residual model in memory (~290 MB, ~10 s), and a
# second worker would repeat all of it for no benefit -- this is a
# read-mostly dashboard, not a write-heavy service. --timeout covers that
# startup; --preload would run it before forking, which with one worker
# saves nothing.
CMD gunicorn main:server \
    --bind "$HOST:$PORT" \
    --workers 1 \
    --threads 4 \
    --timeout 180
