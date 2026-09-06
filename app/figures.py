"""Pure Plotly figure-building functions — themed to match the app's
palette. No Dash-specific code, easy to test standalone."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

INK = "#0B0F0D"
GREEN = "#1B7A3D"
RED = "#B23A3A"
PANEL = "#FFFFFF"

# Track colors, tuned for contrast against the dark oceanic basemap.
ICE_BLUE_PALE = "#274156"
ICE_BLUE = "#8FD3E8"
FORECAST_PALE = "#1B4A2E"
FORECAST_GREEN = "#3ED67C"
NOW_COLOR = "#FFFFFF"
END_COLOR = "#F2B84B"
MAP_BG = "#0B1620"

_COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Approximate initial compass bearing (degrees, 0=N) from point 1 to point 2.

    A simple spherical bearing formula, deliberately not the full WGS84
    geodesic in physics.py -- this is presentation-only (hover text),
    not used for any forecasting math.
    """
    lat1r, lat2r = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2r)
    y = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _compass_label(bearing: float) -> str:
    if not math.isfinite(bearing):
        return ""
    idx = int((bearing / 22.5) + 0.5) % 16
    return _COMPASS[idx]


def _heading_customdata(lats: pd.Series, lons: pd.Series) -> np.ndarray:
    """Per-point heading strings ('042 deg NE'), bearing from the previous
    point. The first point repeats the second point's heading since
    there's no 'previous' point to bear from."""
    lats_arr = np.asarray(lats, dtype=float)
    lons_arr = np.asarray(lons, dtype=float)
    headings = []
    for i in range(len(lats_arr)):
        # Separator rows (used to break the line between icebergs and at
        # date-line crossings) are NaN, and a bearing to or from one is
        # undefined -- so blank the label rather than letting int(nan)
        # raise inside _compass_label.
        if not (np.isfinite(lats_arr[i]) and np.isfinite(lons_arr[i])):
            headings.append("")
            continue
        if i == 0 and len(lats_arr) > 1 and np.isfinite(lats_arr[1]) and np.isfinite(lons_arr[1]):
            bearing = _bearing_deg(lats_arr[0], lons_arr[0], lats_arr[1], lons_arr[1])
        elif i > 0 and np.isfinite(lats_arr[i - 1]) and np.isfinite(lons_arr[i - 1]):
            bearing = _bearing_deg(lats_arr[i - 1], lons_arr[i - 1], lats_arr[i], lons_arr[i])
        else:
            headings.append("")
            continue
        headings.append(f"{bearing:03.0f}\u00b0 {_compass_label(bearing)}")
    return np.array(headings)


def _wake_glow(scatter_map, lats, lons, color: str):
    """A wide, low-opacity line beneath the main trail -- a subtle 'glow'
    that reads as motion/wake on a dark basemap rather than a flat line."""
    return scatter_map(
        lat=lats, lon=lons, mode="lines", showlegend=False,
        line=dict(width=9, color=color),
        opacity=0.18, hoverinfo="skip",
    )


def _gradient_trail(scatter_map, lats, lons, pale: str, bright: str, name: str, size: int = 7):
    """A line+marker trace that fades from `pale` (oldest) to `bright`
    (most recent), so direction of travel reads visually without
    needing rotated arrow icons (map-layer symbol rotation needs custom
    sprite tiles, which is out of scope here)."""
    n = len(lats)
    idx = list(range(n))
    return scatter_map(
        lat=lats, lon=lons, mode="lines+markers", name=name,
        line=dict(width=3, color=bright),
        marker=dict(
            size=size, color=idx,
            colorscale=[[0, pale], [1, bright]],
            showscale=False,
        ),
        hovertemplate="%{lat:.4f}, %{lon:.4f}<br>heading %{customdata}<extra>" + name + "</extra>",
        customdata=_heading_customdata(pd.Series(lats), pd.Series(lons)),
    )


# Nominal pixel size of the map viewport, used to work out how far to
# zoom out to fit a set of points. The graph is responsive so the real
# width varies with the browser window; these are deliberate
# underestimates of the common case, which errs toward zooming out
# slightly too far. Showing a little more ocean than necessary is a much
# cheaper mistake than cropping an iceberg off the edge.
_MAP_VIEWPORT_PX = {"normal": (820.0, 340.0), "expanded": (1200.0, 620.0)}

# Fraction of the viewport left as margin around the fitted points, so
# tracks do not run right up against the panel edge (and clear the
# legend in the bottom-left).
_FIT_PADDING = 0.82

# Bounds for the computed zoom. The lower bound keeps a globe-spanning
# selection readable; the upper bound stops a single stationary iceberg
# zooming to street level, where the basemap is meaningless at sea.
_MIN_ZOOM, _MAX_ZOOM = 1.2, 7.5


def _mercator_y(lat_deg: float) -> float:
    """Project a latitude to its Web Mercator y coordinate.

    Args:
        lat_deg: Latitude in degrees.

    Returns:
        The Mercator y value, in radians-equivalent units where the full
        world spans 2*pi.
    """
    # Clamped just short of the poles, where the projection diverges.
    lat = max(min(lat_deg, 85.05), -85.05)
    return math.log(math.tan(math.pi / 4.0 + math.radians(lat) / 2.0))


def _fit_view(lats, lons, expanded: bool) -> tuple[float, float, float]:
    """Compute the map centre and zoom that frame every supplied point.

    Plotly's map subplot has no `fitbounds` (unlike its geo subplot), so
    the Web Mercator zoom is derived directly: at zoom z the world is
    256 * 2**z pixels across, so the largest z that still fits the
    point spread inside the viewport is what we want, taking the tighter
    of the horizontal and vertical constraints.

    ANTIMERIDIAN: longitudes are also evaluated shifted into [0, 360),
    and whichever convention gives the SMALLER spread wins. Without this
    a pair of icebergs either side of 180 deg -- which this record has --
    looks 359 degrees apart and the map zooms out to the whole planet
    instead of framing the few kilometres actually between them.

    Args:
        lats: Latitudes of every point to frame.
        lons: Longitudes of every point to frame.
        expanded: Whether the map is in the full-screen layout.

    Returns:
        A (center_lat, center_lon, zoom) tuple.
    """
    width_px, height_px = _MAP_VIEWPORT_PX["expanded" if expanded else "normal"]

    lat_min, lat_max = min(lats), max(lats)

    # Pick the longitude convention that keeps the points closest
    # together, so an antimeridian-crossing selection is framed tightly.
    raw_min, raw_max = min(lons), max(lons)
    shifted = [lon % 360.0 for lon in lons]
    shift_min, shift_max = min(shifted), max(shifted)
    if (shift_max - shift_min) < (raw_max - raw_min):
        lon_min, lon_max = shift_min, shift_max
    else:
        lon_min, lon_max = raw_min, raw_max

    center_lat = (lat_min + lat_max) / 2.0
    # Normalise the centre back into [-180, 180] for Plotly.
    center_lon = ((lon_min + lon_max) / 2.0 + 180.0) % 360.0 - 180.0

    lon_span = max(lon_max - lon_min, 1e-4)
    merc_span = max(abs(_mercator_y(lat_max) - _mercator_y(lat_min)), 1e-6)

    zoom_lon = math.log2(width_px * _FIT_PADDING * 360.0 / (256.0 * lon_span))
    zoom_lat = math.log2(height_px * _FIT_PADDING * 2.0 * math.pi / (256.0 * merc_span))

    zoom = max(_MIN_ZOOM, min(_MAX_ZOOM, min(zoom_lon, zoom_lat)))
    return center_lat, center_lon, zoom




def split_antimeridian(lats, lons):
    """Insert a break wherever a track crosses the +/-180 deg meridian.

    Longitude is cyclic but a polyline is not: an iceberg stepping from
    179.95 to -179.90 has moved about five kilometres, yet drawing a
    straight segment between those two numbers sweeps 359.85 degrees and
    paints a line right across the map. Three icebergs in this record do
    it (B22F, B22H, B22I), all of them circling the continent past the
    date line.

    The fix is to cut the line at the crossing rather than to unwrap the
    longitudes: unwrapping would push points beyond +/-180, which the map
    projection will not display.

    Args:
        lats: Latitudes, in track order.
        lons: Longitudes, in track order.

    Returns:
        An (lats, lons) pair of lists with NaN inserted at each crossing.
        NaN breaks a Plotly line, so the track resumes on the far side
        instead of streaking back across the map.
    """
    lat_values = list(pd.Series(lats).to_numpy(dtype=float))
    lon_values = list(pd.Series(lons).to_numpy(dtype=float))

    out_lat: list[float] = []
    out_lon: list[float] = []
    for index, (lat, lon) in enumerate(zip(lat_values, lon_values)):
        if index > 0:
            previous = lon_values[index - 1]
            # A jump beyond half the globe between consecutive fixes is
            # always a date-line wrap, never real motion: it would need
            # an iceberg to cross an ocean in one timestep.
            if np.isfinite(previous) and np.isfinite(lon) and abs(lon - previous) > 180.0:
                out_lat.append(np.nan)
                out_lon.append(np.nan)
        out_lat.append(lat)
        out_lon.append(lon)
    return out_lat, out_lon



def _endpoints_per_iceberg(df) -> tuple[list[float], list[float], list[str]]:
    """Find the last drawn position of EACH iceberg in a combined frame.

    A multi-iceberg track or forecast arrives here as several tracks
    concatenated with NaN separator rows. Taking `.iloc[-1]` therefore
    marks only the final iceberg in the frame, which is why the NOW and
    forecast-end dots used to appear for one berg out of however many
    were selected.

    Args:
        df: A track or forecast frame, possibly holding several
            icebergs separated by NaN rows, with an optional
            `iceberg_id` column.

    Returns:
        An (lats, lons, labels) triple with one entry per iceberg.
        Falls back to the single final row when the frame carries no
        iceberg_id, which is the single-iceberg case.
    """
    if df is None or not len(df):
        return [], [], []

    valid = df.dropna(subset=["lat", "lon"])
    if not len(valid):
        return [], [], []

    if "iceberg_id" in valid.columns and valid["iceberg_id"].notna().any():
        named = valid[valid["iceberg_id"].notna()]
        # sort=False keeps the drawing order the caller chose, so the
        # markers line up with the trail colours.
        groups = named.groupby("iceberg_id", sort=False)
        lats = [float(g["lat"].iloc[-1]) for _, g in groups]
        lons = [float(g["lon"].iloc[-1]) for _, g in groups]
        labels = [str(name) for name, _ in groups]
        return lats, lons, labels

    last = valid.iloc[-1]
    return [float(last["lat"])], [float(last["lon"])], [""]


def build_map_figure(track_df: pd.DataFrame, forecast_df: pd.DataFrame, expanded: bool = False) -> go.Figure:
    """Build the map figure: observed track + forecast, on a dark oceanic
    basemap, with a fading trail to show direction of travel and
    distinct NOW / forecast-end markers.

    Args:
        expanded: when True (the "map only" popout view), render with
            autosize on and a touch more zoom, since the map is the
            only thing on screen rather than sharing space below it.
    """
    fig = go.Figure()
    scatter_map = go.Scattermap if hasattr(go, "Scattermap") else go.Scattermapbox

    # Break the polyline at any date-line crossing before drawing, or a
    # berg stepping from +179.9 to -179.9 streaks across the whole map.
    track_lat, track_lon = split_antimeridian(track_df.lat, track_df.lon)

    fig.add_trace(_wake_glow(scatter_map, track_lat, track_lon, ICE_BLUE))
    fig.add_trace(_gradient_trail(
        scatter_map, track_lat, track_lon, ICE_BLUE_PALE, ICE_BLUE, "Observed"
    ))

    if len(forecast_df):
        fc_lat, fc_lon = split_antimeridian(forecast_df.lat, forecast_df.lon)
        fig.add_trace(_wake_glow(scatter_map, fc_lat, fc_lon, FORECAST_GREEN))
        fig.add_trace(_gradient_trail(
            scatter_map, fc_lat, fc_lon, FORECAST_PALE, FORECAST_GREEN, "Forecast"
        ))

        if "uncertainty_km" in forecast_df.columns:
            fig.add_trace(scatter_map(
                lat=forecast_df.lat, lon=forecast_df.lon, mode="markers",
                name="Uncertainty", showlegend=True,
                marker=dict(size=(forecast_df["uncertainty_km"] * 1.5).clip(upper=40),
                            color=FORECAST_GREEN, opacity=0.15),
                hoverinfo="skip",
            ))

        end_lats, end_lons, end_labels = _endpoints_per_iceberg(forecast_df)
        if end_lats:
            fig.add_trace(scatter_map(
                lat=end_lats, lon=end_lons, mode="markers",
                name="Forecast end",
                marker=dict(size=13, color=END_COLOR, symbol="circle"),
                customdata=end_labels,
                hovertemplate=(
                    "%{customdata}<br>Forecast end"
                    "<br>%{lat:.4f}, %{lon:.4f}<extra></extra>"
                ),
            ))

    now_lats, now_lons, now_labels = _endpoints_per_iceberg(track_df)
    if now_lats:
        fig.add_trace(scatter_map(
            lat=now_lats, lon=now_lons, mode="markers",
            name="Now",
            marker=dict(size=13, color=NOW_COLOR, symbol="circle"),
            customdata=now_labels,
            hovertemplate=(
                "%{customdata}<br>Last observed"
                "<br>%{lat:.4f}, %{lon:.4f}<extra></extra>"
            ),
        ))

    # Frame everything drawn -- observed track(s) AND forecast -- instead
    # of centring on the last fix at a fixed zoom. With several icebergs
    # shown at once a fixed zoom leaves most of them off-screen, and a
    # multi-iceberg map is only useful if it shows the icebergs.
    points = pd.concat(
        [
            track_df[["lat", "lon"]],
            forecast_df[["lat", "lon"]] if len(forecast_df) else track_df.iloc[0:0][["lat", "lon"]],
        ]
    ).dropna()
    if len(points):
        center_lat, center_lon, zoom = _fit_view(
            points["lat"].tolist(), points["lon"].tolist(), expanded
        )
    else:
        center_lat, center_lon, zoom = 0.0, 0.0, 1.5

    map_layout_key = "map" if hasattr(go, "Scattermap") else "mapbox"
    fig.update_layout(
        **{
            f"{map_layout_key}_style": "carto-darkmatter",
            f"{map_layout_key}_zoom": zoom,
            f"{map_layout_key}_center": {"lat": center_lat, "lon": center_lon},
        },
        paper_bgcolor=MAP_BG,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="left", x=0.01,
                    font=dict(family="Inter, sans-serif", size=11, color="#EDEFEF"),
                    bgcolor="rgba(11,22,32,0.65)", bordercolor="rgba(255,255,255,0.08)", borderwidth=1),
        height=340 if not expanded else None,
        autosize=expanded,
        font=dict(family="Inter, sans-serif"),
        hoverlabel=dict(bgcolor="#111C27", bordercolor="rgba(255,255,255,0.1)",
                         font=dict(family="JetBrains Mono, monospace", size=12, color="#EDEFEF")),
    )
    return fig


def build_heatmap_figure(track_df: pd.DataFrame, samples: list, expanded: bool = False) -> go.Figure:
    """Build a density heatmap of the iceberg's possible future positions,
    from a list of bootstrapped forecast rollouts (see
    decision_support.bootstrap_uncertainty_cone / the physics-only
    equivalent in callbacks.py). Every (lat, lon) visited by any sample,
    at any future step, contributes one point to the density -- so
    areas where many samples agree read as a hot (bright) core, and
    areas only a few outlier samples reach read as a faint edge.
    """
    fig = go.Figure()
    density_map = go.Densitymap if hasattr(go, "Densitymap") else go.Densitymapbox

    all_lats, all_lons = [], []
    for sample_df in samples:
        all_lats.extend(sample_df["lat"].tolist())
        all_lons.extend(sample_df["lon"].tolist())

    if all_lats:
        fig.add_trace(density_map(
            lat=all_lats, lon=all_lons,
            z=[1] * len(all_lats), radius=22,
            colorscale=[
                [0.0, "rgba(11,22,32,0)"],
                [0.25, "#1B4A2E"],
                [0.55, "#3ED67C"],
                [0.8, "#F2B84B"],
                [1.0, "#E85D3D"],
            ],
            showscale=False,
            name="Possible positions",
            hoverinfo="skip",
        ))

    scatter_map = go.Scattermap if hasattr(go, "Scattermap") else go.Scattermapbox
    _hm_track_lat, _hm_track_lon = split_antimeridian(track_df.lat, track_df.lon)
    fig.add_trace(scatter_map(
        lat=_hm_track_lat, lon=_hm_track_lon, mode="lines",
        name="Observed", line=dict(width=2, color="#8FD3E8"),
        opacity=0.6, hoverinfo="skip",
    ))

    now_lats, now_lons, now_labels = _endpoints_per_iceberg(track_df)
    if now_lats:
        fig.add_trace(scatter_map(
            lat=now_lats, lon=now_lons, mode="markers",
            name="Now",
            marker=dict(size=13, color=NOW_COLOR, symbol="circle"),
            customdata=now_labels,
            hovertemplate=(
                "%{customdata}<br>Last observed"
                "<br>%{lat:.4f}, %{lon:.4f}<extra></extra>"
            ),
        ))

    # Fit to the observed track plus every bootstrap sample, so the whole
    # spread is visible -- the point of the heatmap is its extent.
    fit_lats = list(track_df["lat"].dropna()) + all_lats
    fit_lons = list(track_df["lon"].dropna()) + all_lons
    if fit_lats:
        center_lat, center_lon, zoom = _fit_view(fit_lats, fit_lons, expanded)
    elif now_lats:
        center_lat, center_lon, zoom = now_lats[0], now_lons[0], 5.0
    else:
        center_lat, center_lon, zoom = 0.0, 0.0, 1.5
    map_layout_key = "map" if hasattr(go, "Scattermap") else "mapbox"
    fig.update_layout(
        **{
            f"{map_layout_key}_style": "carto-darkmatter",
            f"{map_layout_key}_zoom": zoom,
            f"{map_layout_key}_center": {"lat": center_lat, "lon": center_lon},
        },
        paper_bgcolor=MAP_BG,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        height=340 if not expanded else None,
        autosize=expanded,
        font=dict(family="Inter, sans-serif"),
        hoverlabel=dict(bgcolor="#111C27", bordercolor="rgba(255,255,255,0.1)",
                         font=dict(family="JetBrains Mono, monospace", size=12, color="#EDEFEF")),
    )
    return fig


def build_diagnostics_figure(metrics: dict) -> go.Figure:
    """Build the physics-only vs hybrid FDE comparison bar chart, with an
    improvement-percentage annotation between the two bars."""
    labels = {"physics_only": "Physics only", "hybrid": "Hybrid (Physics + ML)"}
    x_labels = [labels.get(k, k) for k in metrics.keys()]
    values = list(metrics.values())
    colors = [INK if k != "hybrid" else GREEN for k in metrics.keys()]

    fig = go.Figure(go.Bar(
        x=x_labels, y=values,
        marker_color=colors,
        marker_line_width=0,
        text=[f"{v:.1f} km" for v in values], textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace", size=13, color=INK),
        width=0.45,
    ))

    annotations = []
    if "physics_only" in metrics and "hybrid" in metrics and metrics["physics_only"] > 0:
        improvement_pct = 100.0 * (1.0 - metrics["hybrid"] / metrics["physics_only"])
        annotations.append(dict(
            x=0.5, xref="paper", y=max(values) * 1.18, yref="y",
            text=f"{improvement_pct:+.0f}% FDE",
            showarrow=False,
            font=dict(family="JetBrains Mono, monospace", size=12, color=GREEN),
        ))

    fig.update_layout(
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        yaxis_title="Final displacement error (km)",
        font=dict(family="Inter, sans-serif", color=INK),
        margin=dict(l=50, r=20, t=40, b=40),
        height=320,
        bargap=0.5,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#E4E1D8", zeroline=False,
                    range=[0, max(values) * 1.35] if values else None),
        annotations=annotations,
    )
    return fig


if __name__ == "__main__":
    dummy_track = pd.DataFrame({"lat": [-65.0, -64.9], "lon": [-55.0, -54.8]})
    dummy_forecast = pd.DataFrame({"lat": [-64.8, -64.7], "lon": [-54.6, -54.4], "uncertainty_km": [3, 6]})
    f1 = build_map_figure(dummy_track, dummy_forecast)
    f2 = build_diagnostics_figure({"physics_only": 18.4, "hybrid": 6.7})
    assert f1.data and f2.data
    print("figures OK")