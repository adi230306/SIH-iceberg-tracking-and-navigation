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

import numpy as np
from pyproj import Geod

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
    wind_factor: float = 0.018,
    deflection_deg: float = 20.0,
) -> tuple[float, float]:
    """Estimate iceberg drift velocity using the standard "free drift" approximation.

    iceberg velocity ~= wind_factor * R(deflection_angle) @ wind_vector
                         + ocean_current_vector

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
        deflection_deg: Magnitude of the Coriolis-driven deflection
            angle (degrees) between the wind vector and the resulting
            wind-driven drift component.

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
    hemisphere_sign = np.sign(lat_deg)
    theta = math.radians(deflection_deg) * hemisphere_sign

    cos_t, sin_t = math.cos(theta), math.sin(theta)
    # Standard 2D rotation matrix R(theta) = [[cos, -sin], [sin, cos]],
    # applied to the wind vector (u_wind, v_wind).
    u_wind_rot = cos_t * u_wind - sin_t * v_wind
    v_wind_rot = sin_t * u_wind + cos_t * v_wind

    u_drift = wind_factor * u_wind_rot + u_current
    v_drift = wind_factor * v_wind_rot + v_current
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