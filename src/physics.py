"""
physics.py

Analytical "free drift" physics baseline for iceberg motion, plus the
geodesic math helpers used throughout the rest of the project.

This module is intentionally dependency-light (numpy + pyproj only) and
does no file I/O or network access, so it can be imported safely from
anywhere in the codebase (data_ingest.py, features.py, decision_support.py)
without pulling in heavier dependencies or side effects.

Physical background: an untethered iceberg (or ice floe) is pushed by
both the ocean current directly beneath it and by wind drag on its
above-water surface. Because the wind-driven component travels through
a turbulent, Coriolis-influenced boundary layer, it does not simply add
in the wind's own direction -- it is deflected by a roughly constant
angle relative to the wind, to the right of the wind in the Northern
Hemisphere and to the left in the Southern Hemisphere (the same sense
of deflection seen in classical Ekman transport). This "free drift"
approximation ignores iceberg-specific effects like draft, keel shape,
and sea-ice interaction -- those are exactly what the ML residual model
elsewhere in this project is trained to correct for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from pyproj import Geod
from scipy.optimize import lsq_linear

# Earth's angular velocity, rad/s. Used to compute the Coriolis
# parameter f = 2 * Omega * sin(lat).
EARTH_ANGULAR_VELOCITY: float = 7.2921e-5

# WGS84 geodesic calculator, reused by every geodesic helper in this
# module. Using a proper ellipsoidal geodesic (rather than flat
# lat/lon arithmetic) matters a lot here because icebergs live at high
# latitudes, where a degree of longitude can be a tiny fraction of its
# equatorial distance, and naive Euclidean lat/lon math would badly
# distort both distances and bearings.
_GEOD = Geod(ellps="WGS84")


# The drift coefficients used when a caller does not pass its own.
# main.py sets these to the values calibrated on the real record at
# startup, so callers that cannot pass parameters explicitly -- notably
# the frozen Dash frontend, which calls free_drift_velocity() with only
# the five physical arguments -- still get the calibrated physics rather
# than the literature defaults, which on this dataset are worse than
# predicting nothing. Every training and evaluation path passes its
# coefficients explicitly and is therefore unaffected by this global.
_DEFAULT_PARAMS: dict[str, float] = {
    "wind_factor": 0.018,
    "deflection_deg": 20.0,
    "current_factor": 1.0,
}


def set_default_drift_params(params: "DriftParams") -> None:
    """Set the drift coefficients used when a caller passes none.

    Args:
        params: The calibrated coefficients to adopt as defaults.

    Returns:
        None.
    """
    _DEFAULT_PARAMS.update(params.as_kwargs())


def get_default_drift_params() -> "DriftParams":
    """Return the drift coefficients currently used as defaults.

    Returns:
        A DriftParams holding the current module-level defaults.
    """
    return DriftParams(**_DEFAULT_PARAMS)


def coriolis_parameter(lat_deg: float) -> float:
    """Compute the Coriolis parameter at a given latitude.

    Args:
        lat_deg: Latitude in degrees (-90 to 90).

    Returns:
        The Coriolis parameter f = 2 * Omega * sin(lat), in rad/s.
        Positive in the Northern Hemisphere, negative in the Southern
        Hemisphere, zero at the equator.
    """
    lat_rad = math.radians(lat_deg)
    return 2.0 * EARTH_ANGULAR_VELOCITY * math.sin(lat_rad)


def free_drift_velocity(
    u_wind: float,
    v_wind: float,
    u_current: float,
    v_current: float,
    lat_deg: float,
    wind_factor: float | None = None,
    deflection_deg: float | None = None,
    current_factor: float | None = None,
) -> tuple[float, float]:
    """Estimate iceberg drift velocity using the standard "free drift" approximation.

    iceberg velocity ~= wind_factor * R(deflection_angle) @ wind_vector
                         + current_factor * ocean_current_vector

    wind_factor (~0.018, i.e. ~1.8% of wind speed) is an empirical
    constant widely used in sea-ice/iceberg drift studies: wind drag on
    the above-water sail area is a small fraction of the momentum
    transferred by the ocean current directly, since water is ~800x
    denser than air but the relevant drag areas and velocities differ.
    It is deliberately left as a tunable parameter rather than derived
    from first principles, since the "correct" value depends on
    iceberg draft/shape that we don't observe directly -- exactly the
    gap the ML residual model is trained to fill.

    Args:
        u_wind: Eastward 10m wind component, m/s.
        v_wind: Northward 10m wind component, m/s.
        u_current: Eastward ocean surface current component, m/s.
        v_current: Northward ocean surface current component, m/s.
        lat_deg: Latitude in degrees, used only to determine the sign
            of the Coriolis deflection (not to scale its magnitude).
        wind_factor: Fraction of wind speed transferred to drift speed.
            None uses the module default (see set_default_drift_params).
        deflection_deg: Magnitude of the Coriolis-driven deflection
            angle (degrees) between the wind vector and the resulting
            wind-driven drift component. None uses the module default.
        current_factor: Multiplier on the ocean current term. None uses
            the module default. The
            Copernicus product reports the current at ~0.5 m depth, but
            a tabular iceberg's keel reaches 150-300 m down and is
            dragged by the depth-averaged current, which is usually
            weaker -- so a fitted value below 1.0 is physically
            expected. See calibrate_free_drift_params().

    Returns:
        A (u_drift, v_drift) tuple in m/s, using the same
        eastward/northward convention as the wind/current inputs.
    """
    # CRITICAL: the sense of Coriolis deflection is opposite in the two
    # hemispheres -- wind-driven drift deflects to the RIGHT of the wind
    # in the Northern Hemisphere (clockwise, i.e. a *negative* rotation
    # in the standard mathematical/counterclockwise-positive
    # convention) and to the LEFT of the wind in the Southern
    # Hemisphere (counterclockwise, positive rotation). We capture this
    # by flipping the sign of the rotation angle using the sign of
    # lat_deg. At the equator the Coriolis effect vanishes, so
    # np.sign(0) == 0 correctly yields zero deflection.
    if wind_factor is None:
        wind_factor = _DEFAULT_PARAMS["wind_factor"]
    if deflection_deg is None:
        deflection_deg = _DEFAULT_PARAMS["deflection_deg"]
    if current_factor is None:
        current_factor = _DEFAULT_PARAMS["current_factor"]

    hemisphere_sign = np.sign(lat_deg)
    theta = math.radians(deflection_deg) * hemisphere_sign

    cos_t, sin_t = math.cos(theta), math.sin(theta)
    # Standard 2D rotation matrix R(theta) = [[cos, -sin], [sin, cos]],
    # applied to the wind vector (u_wind, v_wind).
    u_wind_rot = cos_t * u_wind - sin_t * v_wind
    v_wind_rot = sin_t * u_wind + cos_t * v_wind

    u_drift = wind_factor * u_wind_rot + current_factor * u_current
    v_drift = wind_factor * v_wind_rot + current_factor * v_current
    return u_drift, v_drift


def step_position(
    lat: float,
    lon: float,
    u_ms: float,
    v_ms: float,
    dt_seconds: float,
) -> tuple[float, float]:
    """Advance a lat/lon position forward by one timestep using geodesic projection.

    Uses pyproj's WGS84 geodesic forward projection rather than flat
    lat/lon addition, which is essential near the poles (where icebergs
    live) since a degree of longitude there covers a tiny fraction of
    the distance it does at the equator -- naive addition would either
    barely move a high-latitude iceberg or wildly overshoot it,
    depending on which axis you naively scaled.

    Args:
        lat: Current latitude, degrees.
        lon: Current longitude, degrees.
        u_ms: Eastward velocity component, m/s.
        v_ms: Northward velocity component, m/s.
        dt_seconds: Timestep duration, seconds.

    Returns:
        The (new_lat, new_lon) position after moving at the given
        velocity for dt_seconds, in degrees.
    """
    speed = math.hypot(u_ms, v_ms)
    if speed == 0.0:
        # No motion: return the input position unchanged rather than
        # calling geod.fwd() with an undefined bearing (atan2(0, 0)).
        return lat, lon

    # Bearing convention: 0 deg = due north, 90 deg = due east,
    # measured clockwise -- so atan2(eastward, northward), not the
    # mathematical atan2(y, x).
    bearing_deg = math.degrees(math.atan2(u_ms, v_ms))
    distance_m = speed * dt_seconds

    new_lon, new_lat, _back_azimuth = _GEOD.fwd(lon, lat, bearing_deg, distance_m)
    return new_lat, new_lon


def geodesic_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute the geodesic (WGS84 ellipsoidal) distance between two points.

    Args:
        lat1: Latitude of the first point, degrees.
        lon1: Longitude of the first point, degrees.
        lat2: Latitude of the second point, degrees.
        lon2: Longitude of the second point, degrees.

    Returns:
        Distance between the two points in kilometers. Correctly
        handles antimeridian wraparound (e.g. lon1=179, lon2=-179 is a
        short distance, not a ~358-degree-of-longitude one), since
        pyproj.Geod.inv() operates on the ellipsoid rather than raw
        lon differences.
    """
    _forward_azimuth, _back_azimuth, distance_m = _GEOD.inv(lon1, lat1, lon2, lat2)
    return distance_m / 1000.0



@dataclass(frozen=True)
class DriftParams:
    """The three fitted coefficients of the free-drift baseline.

    Attributes:
        wind_factor: Fraction of 10 m wind speed transferred to drift.
        deflection_deg: Coriolis deflection angle magnitude, degrees.
        current_factor: Multiplier on the reported surface current.
    """

    wind_factor: float = 0.018
    deflection_deg: float = 20.0
    current_factor: float = 1.0

    def as_kwargs(self) -> dict[str, float]:
        """Return the parameters as keyword arguments for free_drift_velocity().

        Returns:
            A dict with wind_factor / deflection_deg / current_factor keys.
        """
        return {
            "wind_factor": self.wind_factor,
            "deflection_deg": self.deflection_deg,
            "current_factor": self.current_factor,
        }


def free_drift_velocity_array(
    u_wind: np.ndarray,
    v_wind: np.ndarray,
    u_current: np.ndarray,
    v_current: np.ndarray,
    lat_deg: np.ndarray,
    wind_factor: float = 0.018,
    deflection_deg: float = 20.0,
    current_factor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized free_drift_velocity() over whole arrays of observations.

    Identical physics to the scalar free_drift_velocity(), including the
    hemisphere-dependent sign of the deflection, but evaluated with numpy
    over every row at once. The real-data pipeline calls this on the full
    pooled multi-iceberg table (and inside the calibration least-squares
    loop, which evaluates it ~80 times), where a per-row .apply() would
    dominate the runtime for no reason.

    Args:
        u_wind: Eastward 10 m wind components, m/s.
        v_wind: Northward 10 m wind components, m/s.
        u_current: Eastward surface current components, m/s.
        v_current: Northward surface current components, m/s.
        lat_deg: Latitudes in degrees, used only for the sign of the
            deflection (broadcast against the other arrays).
        wind_factor: Fraction of wind speed transferred to drift speed.
        deflection_deg: Deflection angle magnitude, degrees.
        current_factor: Multiplier on the ocean current term.

    Returns:
        A (u_drift, v_drift) tuple of numpy arrays in m/s.
    """
    # Same hemisphere flip as the scalar version: clockwise (negative
    # rotation) in the Northern Hemisphere, counterclockwise in the
    # Southern. np.sign gives 0 at the equator, where Coriolis vanishes.
    theta = np.radians(deflection_deg) * np.sign(np.asarray(lat_deg, dtype=float))
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    u_wind_rot = cos_t * u_wind - sin_t * v_wind
    v_wind_rot = sin_t * u_wind + cos_t * v_wind

    u_drift = wind_factor * u_wind_rot + current_factor * u_current
    v_drift = wind_factor * v_wind_rot + current_factor * v_current
    return u_drift, v_drift


def calibrate_free_drift_params(
    u_wind: np.ndarray,
    v_wind: np.ndarray,
    u_current: np.ndarray,
    v_current: np.ndarray,
    lat_deg: np.ndarray,
    obs_u: np.ndarray,
    obs_v: np.ndarray,
    deflection_grid_deg: np.ndarray | None = None,
    wind_factor_bounds: tuple[float, float] = (0.0, 0.05),
    current_factor_bounds: tuple[float, float] = (0.0, 1.5),
    tie_tolerance: float = 1e-2,
) -> DriftParams:
    """Fit wind_factor, current_factor and deflection_deg to observed drift.

    The textbook values (1.8% of wind, 20 deg deflection, current taken
    at face value) are generic sea-ice numbers. Refitting them on the
    actual training icebergs is nearly free and moves a substantial
    chunk of error out of the ML residual and into the physics term,
    which is where we would much rather have it: the physics term
    extrapolates to unseen icebergs and unseen regions, the tree
    ensemble does not.

    A WARNING THE ANTARCTIC DATA MAKES CONCRETE: the textbook 1.8%
    wind factor assumes the current term is a GEOSTROPHIC current, which
    contains no wind-driven flow. The Copernicus analysis product used
    here reports the total modelled current at ~0.5 m depth, which
    already includes the Ekman and Stokes response to the wind. Adding a
    separate free-drift wind term on top therefore DOUBLE-COUNTS the
    wind. On the real NIC record this shows up unmistakably: the
    unconstrained wind factor comes out negative, and forcing it up to
    even 0.005 makes the fit worse. Fitting the coefficients rather than
    assuming them is what surfaces this instead of burying it in the ML
    residual.

    The fit exploits the fact that, for a FIXED deflection angle, the
    free-drift model is *linear* in its two remaining coefficients:

        [obs_u; obs_v] = wind_factor * R(theta) @ [u_wind; v_wind]
                       + current_factor * [u_current; v_current]

    So we grid-search only the one nonlinear parameter (theta) and solve
    the other two exactly by ordinary least squares at each candidate,
    stacking the u- and v-equations of every observation into one tall
    system. That is both faster and far more robust than throwing all
    three at a general-purpose optimizer with no good initial guess.

    Args:
        u_wind: Eastward wind components for each observation, m/s.
        v_wind: Northward wind components, m/s.
        u_current: Eastward current components, m/s.
        v_current: Northward current components, m/s.
        lat_deg: Latitudes, degrees (sets the deflection sign per row).
        obs_u: Observed eastward drift velocities, m/s.
        obs_v: Observed northward drift velocities, m/s.
        deflection_grid_deg: Candidate deflection magnitudes to search.
            Defaults to 0..40 degrees in 1-degree steps.
        wind_factor_bounds: Physically admissible range for the wind
            factor. Bounded below at 0 because an unconstrained fit can
            return a NEGATIVE wind factor -- drift moving into the wind
            -- when the wind term is weakly identified, which is
            nonsense as physics and extrapolates disastrously.
        current_factor_bounds: Admissible range for the current factor.
        tie_tolerance: Relative error tolerance for the deflection
            tie-break. When the wind factor fits near zero the deflection
            angle is unidentifiable (rotating a near-zero vector changes
            nothing), and an argmin over a flat surface returns an
            arbitrary angle -- often pinned to the edge of the grid. Any
            deflection within this relative tolerance of the best is
            treated as tied, and the one closest to zero is chosen, so
            the reported parameters are stable and honest about what the
            data actually constrains. 1% of the sum of squares is well
            inside the noise floor of a ~150-observation fit.

    Returns:
        The best-fitting DriftParams (lowest total squared velocity
        error over the supplied observations, subject to the bounds).

    Raises:
        ValueError: If fewer than 3 finite observations are supplied
            (two free coefficients cannot be fit from fewer points), or
            if the inputs have mismatched lengths.
    """
    arrays = [np.asarray(a, dtype=float) for a in
              (u_wind, v_wind, u_current, v_current, lat_deg, obs_u, obs_v)]
    lengths = {a.shape for a in arrays}
    if len(lengths) != 1:
        raise ValueError(
            f"calibrate_free_drift_params: all inputs must have the same shape, got {lengths}."
        )
    u_w, v_w, u_c, v_c, lat, o_u, o_v = arrays

    finite = np.isfinite(np.column_stack(arrays)).all(axis=1)
    if finite.sum() < 3:
        raise ValueError(
            f"calibrate_free_drift_params: need at least 3 finite observations to fit "
            f"the two linear coefficients, got {int(finite.sum())}."
        )
    u_w, v_w, u_c, v_c, lat, o_u, o_v = (a[finite] for a in (u_w, v_w, u_c, v_c, lat, o_u, o_v))

    if deflection_grid_deg is None:
        # Symmetric by default: let the data choose the SIGN of the
        # deflection rather than assuming our hemisphere convention is
        # right. A fit that only makes sense at a negative angle is a
        # signal that the convention is inverted somewhere.
        deflection_grid_deg = np.arange(-40.0, 41.0, 1.0)

    sign = np.sign(lat)
    # Stacked target: every observation contributes two rows (u then v).
    target = np.concatenate([o_u, o_v])

    lower = np.array([wind_factor_bounds[0], current_factor_bounds[0]])
    upper = np.array([wind_factor_bounds[1], current_factor_bounds[1]])

    candidates: list[tuple[float, DriftParams]] = []
    for deflection in np.asarray(deflection_grid_deg, dtype=float):
        theta = np.radians(deflection) * sign
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        wind_rot_u = cos_t * u_w - sin_t * v_w
        wind_rot_v = sin_t * u_w + cos_t * v_w

        # Design matrix: column 0 multiplies the rotated wind, column 1
        # the reported current; the u- and v-equations are stacked.
        design = np.column_stack(
            [np.concatenate([wind_rot_u, wind_rot_v]), np.concatenate([u_c, v_c])]
        )
        result = lsq_linear(design, target, bounds=(lower, upper), method="bvls")
        sse = float(np.sum((design @ result.x - target) ** 2))
        candidates.append(
            (
                sse,
                DriftParams(
                    wind_factor=float(result.x[0]),
                    deflection_deg=float(deflection),
                    current_factor=float(result.x[1]),
                ),
            )
        )

    best_sse = min(sse for sse, _ in candidates)
    # Among deflections the data cannot distinguish, prefer the smallest
    # -- see tie_tolerance in the Args section. The threshold needs an
    # absolute floor as well as a relative one: when the wind term fits
    # to exactly zero every deflection gives the same fit, and a purely
    # relative tolerance around a near-zero SSE would separate them on
    # floating-point noise and hand back an arbitrary angle again.
    scale = float(np.sum(target**2)) or 1.0
    threshold = best_sse * (1.0 + tie_tolerance) + 1e-9 * scale
    tied = [params for sse, params in candidates if sse <= threshold]
    return min(tied, key=lambda p: abs(p.deflection_deg))


if __name__ == "__main__":
    # --- coriolis_parameter sign check ---
    f_north = coriolis_parameter(45.0)
    f_south = coriolis_parameter(-45.0)
    print(f"coriolis_parameter(45)  = {f_north:.6e} rad/s")
    print(f"coriolis_parameter(-45) = {f_south:.6e} rad/s")
    assert f_north > 0, "Coriolis parameter should be positive in the Northern Hemisphere"
    assert f_south < 0, "Coriolis parameter should be negative in the Southern Hemisphere"

    # --- free_drift_velocity hemisphere-flip check ---
    # Same wind/current sample vector, evaluated at a Northern and a
    # Southern Hemisphere latitude. The wind is due east (u_wind>0,
    # v_wind=0), so the wind-driven component's deflection direction is
    # directly visible in the sign of v_drift (current's v-component is
    # 0 here, isolating the wind term).
    sample_u_wind, sample_v_wind = 10.0, 0.0
    sample_u_current, sample_v_current = 0.2, 0.0

    u_drift_n, v_drift_n = free_drift_velocity(
        sample_u_wind, sample_v_wind, sample_u_current, sample_v_current, lat_deg=60.0
    )
    u_drift_s, v_drift_s = free_drift_velocity(
        sample_u_wind, sample_v_wind, sample_u_current, sample_v_current, lat_deg=-60.0
    )
    print(f"\nNH (lat=60) drift:  u={u_drift_n:.4f}, v={v_drift_n:.4f} m/s")
    print(f"SH (lat=-60) drift: u={u_drift_s:.4f}, v={v_drift_s:.4f} m/s")

    # Due-east wind rotated by +/-20 degrees deflects the v-component
    # (north/south) in opposite directions between hemispheres; u should
    # match in sign (still mostly eastward) but v must flip sign.
    assert not np.isclose(v_drift_n, v_drift_s), "hemispheres should not produce identical deflection"
    assert np.sign(v_drift_n) != np.sign(v_drift_s), (
        "wind-driven deflection should flip sign between hemispheres"
    )

    # --- chained free_drift_velocity -> step_position sanity check ---
    lat, lon = -65.0, -60.0  # Antarctic-like start
    dt_seconds = 6 * 3600  # 6-hour step, matching the project's track schema
    track = [(lat, lon)]
    for _ in range(10):
        u_drift, v_drift = free_drift_velocity(
            u_wind=8.0, v_wind=3.0, u_current=0.2, v_current=0.1, lat_deg=lat
        )
        lat, lon = step_position(lat, lon, u_drift, v_drift, dt_seconds)
        assert not (math.isnan(lat) or math.isnan(lon)), "position went NaN"
        track.append((lat, lon))

    print("\n10-step synthetic track (constant moderate wind/current):")
    for step_idx, (t_lat, t_lon) in enumerate(track):
        print(f"  step {step_idx}: lat={t_lat:.5f}, lon={t_lon:.5f}")

    # Sanity-check step sizes: moderate wind/current over 6 hours should
    # not produce implausible jumps.
    for i in range(len(track) - 1):
        step_km = geodesic_distance_km(*track[i], *track[i + 1])
        assert step_km < 50.0, f"implausible single-step jump of {step_km:.1f} km at step {i}"
    print("\nAll single-step distances are under the 50 km plausibility threshold.")

    # --- geodesic_distance_km sanity check ---
    one_degree_lon_km = geodesic_distance_km(0.0, 0.0, 0.0, 1.0)
    print(f"\ngeodesic_distance_km(0,0, 0,1) = {one_degree_lon_km:.3f} km")
    assert math.isclose(one_degree_lon_km, 111.0, abs_tol=1.0), (
        "1 degree of longitude at the equator should be ~111 km"
    )

    print("\nAll physics.py sanity checks passed.")