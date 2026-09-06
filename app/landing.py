"""Landing shell that wraps the live dashboard.

The dashboard from app/layout.py is not duplicated or re-rendered here.
It is mounted ONCE, inside `.ib-stage`, and the shell simply changes the
class on its container:

    landing view  -- the hero occupies the left column and the stage sits
                     in the right one as a preview card, with a
                     transparent overlay catching the click.
    open view     -- the hero collapses and the stage expands to the full
                     width of the page.

Doing it with one mounted instance, rather than a mock preview plus a
real dashboard, matters for more than tidiness: Dash forbids duplicate
component ids, so a second copy of the map would need a parallel set of
ids and callbacks, and the "preview" would drift out of sync with the
thing it is previewing. Here the preview IS the dashboard, at a
different size.
"""

from __future__ import annotations

from dash import dcc, html


def _nav() -> html.Div:
    """Build the top navigation bar.

    Returns:
        The nav element, carrying the project title as the brand.
    """
    return html.Div(
        className="ib-nav",
        children=[
            html.Div(
                className="ib-brand",
                children=[
                    html.Span("✦", className="ib-brand-mark"),
                    html.Span("Iceberg Tracking & Navigation", className="ib-brand-name"),
                ],
            ),
            html.Div(
                className="ib-nav-actions",
                children=[
                    html.Button("Back to overview", id="ib-back-btn", n_clicks=0,
                                className="ib-btn ib-btn-ghost", style={"display": "none"}),
                    html.Button("Open dashboard →", id="ib-open-btn", n_clicks=0,
                                className="ib-btn ib-btn-primary"),
                ],
            ),
        ],
    )


def _stat(value: str, label: str) -> html.Div:
    """Build one figure in the hero's statistics strip.

    Args:
        value: The large number, already formatted.
        label: The small caption beneath it.

    Returns:
        A stat block.
    """
    return html.Div(
        className="ib-stat",
        children=[
            html.Div(value, className="ib-stat-value"),
            html.Div(label, className="ib-stat-label"),
        ],
    )


def _hero_copy(stats: list[tuple[str, str]], live_note: str) -> html.Div:
    """Build the left-hand hero column.

    Args:
        stats: (value, label) pairs for the statistics strip.
        live_note: Text for the small live pill above the headline.

    Returns:
        The hero copy column.
    """
    return html.Div(
        className="ib-hero-copy",
        children=[
            html.Div(
                className="ib-pill-live",
                children=[html.Span(className="ib-dot"), html.Span(live_note)],
            ),
            html.H1(
                className="ib-headline",
                children=[
                    "Track it. Forecast it. ",
                    html.Span("Steer clear.", className="ib-mark"),
                ],
            ),
            html.P(
                "Iceberg Tracking and Navigation System — daily satellite fixes fused "
                "with ERA5 winds and Copernicus currents through a calibrated drift "
                "model, then turned into a closest-approach distance and a graded "
                "risk call. Built so a watch officer decides, not the algorithm.",
                className="ib-sub",
            ),
            html.Div(
                className="ib-cta-row",
                children=[
                    html.Button("Open live dashboard →", id="ib-hero-open-btn",
                                n_clicks=0, className="ib-btn ib-btn-primary ib-btn-lg"),
                    html.A("How it works", href="#ib-how", className="ib-btn ib-btn-outline ib-btn-lg"),
                ],
            ),
            html.Div(className="ib-stats", children=[_stat(v, l) for v, l in stats]),
        ],
    )


def _how_it_works() -> html.Div:
    """Build the short 'how it works' strip below the hero.

    Returns:
        A section explaining the pipeline in four steps.
    """
    steps = [
        ("01", "Observe",
         "Daily scatterometer fixes from the BYU/NIC Antarctic database, "
         "screened for bad positions and for bergs that are aground rather than adrift."),
        ("02", "Force",
         "ERA5 10 m winds and Copernicus Marine surface currents, averaged over "
         "each interval between fixes rather than sampled at an instant."),
        ("03", "Predict",
         "Free-drift physics with coefficients fitted to the observed record, "
         "plus a gradient-boosted residual that is only used when it earns its place."),
        ("04", "Decide",
         "Closest point of approach to a vessel, graded green / amber / red, "
         "escalated when the forecast spread is wide rather than presenting false confidence."),
    ]
    return html.Div(
        id="ib-how",
        className="ib-how",
        children=[
            html.Div("How it works", className="ib-how-title"),
            html.Div(
                className="ib-how-grid",
                children=[
                    html.Div(
                        className="ib-how-card",
                        children=[
                            html.Div(num, className="ib-how-num"),
                            html.Div(name, className="ib-how-name"),
                            html.Div(body, className="ib-how-body"),
                        ],
                    )
                    for num, name, body in steps
                ],
            ),
        ],
    )


def build_shell(dashboard, stats: list[tuple[str, str]], live_note: str) -> html.Div:
    """Wrap the dashboard in the landing shell.

    Args:
        dashboard: The layout returned by app.layout.build_layout().
        stats: (value, label) pairs for the hero statistics strip.
        live_note: Text for the live pill above the headline.

    Returns:
        The full page layout.
    """
    return html.Div(
        id="ib-shell",
        className="ib-shell ib-view-landing",
        children=[
            _nav(),
            html.Div(
                className="ib-hero",
                children=[
                    _hero_copy(stats, live_note),
                    html.Div(
                        className="ib-stage",
                        children=[
                            # Transparent hit area, active only in the
                            # landing view. It sits ABOVE the dashboard so
                            # a click anywhere on the preview opens it
                            # instead of being swallowed by the map's own
                            # pan/zoom handlers.
                            html.Button(
                                id="ib-stage-overlay",
                                n_clicks=0,
                                className="ib-stage-overlay",
                                children=html.Span(
                                    "Click to open the live dashboard ↗",
                                    className="ib-stage-hint",
                                ),
                            ),
                            html.Div(className="ib-stage-inner", children=dashboard),
                        ],
                    ),
                ],
            ),
            _how_it_works(),
            dcc.Store(id="ib-view", data=False),
        ],
    )
