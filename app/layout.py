"""Static UI layout — live forecast panel + model diagnostics only.
No data logic here, only structure."""
from __future__ import annotations

from dash import dcc, html


def _map_toolbar() -> html.Div:
    """Icon buttons floating over the top-right corner of the map itself
    (not the panel header), so they travel with the map whether it's at
    normal size or already popped out to fullscreen."""
    return html.Div(className="iw-map-toolbar", children=[
        html.Button("\u2338", id="heatmap-btn", n_clicks=0,
                    className="iw-map-tool-btn", title="Toggle probability heatmap"),
        html.Button("\u26f6", id="fullscreen-btn", n_clicks=0,
                    className="iw-map-tool-btn", title="Toggle fullscreen"),
    ])


def _live_panel() -> html.Div:
    return html.Div(className="iw-panel", id="live-panel", children=[
        html.Div(className="iw-panel-head", children=[
            html.Div(className="iw-panel-head-left", children=[
                html.Span(className="iw-dot pulse"),
                html.Span(id="panel-title", children="LIVE FORECAST"),
            ]),
        ]),

        html.Div(className="iw-map-wrap", children=[
            _map_toolbar(),
            html.Div(id="heatmap-meta", className="iw-heatmap-meta", style={"display": "none"}),
            dcc.Loading(
                dcc.Graph(id="map-graph", config={"displayModeBar": False, "responsive": True},
                          style={"height": "100%"}),
                type="circle", parent_style={"height": "100%"},
            ),
        ]),

        html.Div(id="iw-details", children=[
            html.Div(className="iw-controls", children=[
                html.Div(className="iw-control-row", children=[
                    html.Label("Iceberg"),
                    dcc.Dropdown(
                        id="iceberg-select",
                        options=[{"label": "Synthetic track", "value": "synthetic_track"}],
                        value="synthetic_track", clearable=False,
                    ),
                ]),
                html.Div(className="iw-control-row", children=[
                    html.Label("Forecast horizon (hours)"),
                    dcc.Slider(id="horizon-slider", min=6, max=120, step=6, value=48,
                               marks={h: str(h) for h in range(0, 121, 24)}),
                ]),
                html.Div(className="iw-control-row", children=[
                    html.Label("Mode"),
                    dcc.RadioItems(
                        id="mode-select",
                        options=[{"label": " Physics only", "value": "physics"},
                                 {"label": " Hybrid (physics + ML)", "value": "hybrid"}],
                        value="hybrid", labelStyle={"display": "inline-block", "marginRight": "16px"},
                    ),
                ]),
                html.Div(className="iw-control-row", style={"display": "flex", "gap": "10px"}, children=[
                    html.Div(style={"flex": 1}, children=[
                        html.Label("Vessel lat"),
                        dcc.Input(id="vessel-lat", type="number", placeholder="lat", debounce=True,
                                  style={"width": "100%"}),
                    ]),
                    html.Div(style={"flex": 1}, children=[
                        html.Label("Vessel lon"),
                        dcc.Input(id="vessel-lon", type="number", placeholder="lon", debounce=True,
                                  style={"width": "100%"}),
                    ]),
                ]),
                html.Div(className="iw-control-row", children=[
                    html.Label("Forecast step"),
                    dcc.Slider(id="timestep-slider", min=0, max=1, step=1, value=0),
                ]),
            ]),

            html.Div(id="forecast-list", className="iw-list"),

            html.Div(className="iw-panel-foot", children=[
                html.Span(id="cpa-text", children="Enter a vessel position to compute CPA."),
                html.Span(id="risk-pill"),
            ]),
        ]),
    ])


def build_layout() -> html.Div:
    """Build the top-level app layout."""
    return html.Div(className="iw-page", children=[
        html.Div(className="iw-section", children=[
            _live_panel(),
        ]),
        html.Div(className="iw-section", children=[
            html.Div("Model diagnostics", className="iw-section-title"),
            html.Div(className="iw-panel", children=[
                dcc.Graph(id="diagnostics-graph", config={"displayModeBar": False}),
            ]),
        ]),
        dcc.Store(id="forecast-store"),
        dcc.Store(id="fullscreen-state", data=False),
        dcc.Store(id="heatmap-state", data=False),
    ])