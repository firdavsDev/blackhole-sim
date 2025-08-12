"""Very simple emissivity & redshift tools for a thin Keplerian disk.

This is intentionally simple for the prototype: emissivity ~ r^{-q} inside [r_in, r_out]
and emitter 4-velocity assumes locally circular Keplerian orbits (no pressure, test-particle).
"""

import numpy as np

from sim.metrics import schwarzschild_metric


def disk_emissivity(
    r: float, r_in: float = 6.0, r_out: float = 40.0, q: float = 3.0
) -> float:
    if (r < r_in) or (r > r_out):
        return 0.0
    return r ** (-q)


def emitter_four_velocity(r: float, M: float = 1.0) -> np.ndarray:
    """Circular Keplerian test-particle 4-velocity in Schwarzschild (equatorial).
    Valid for r > 3M (photon orbit 3M)."""
    if r <= 3.0 * M:
        return np.array([np.nan, 0.0, 0.0, np.nan])
    denom = np.sqrt(1.0 - 3.0 * M / r)
    ut = 1.0 / denom
    uphi = np.sqrt(M / (r**3)) / denom
    return np.array([ut, 0.0, 0.0, uphi])


def redshift_factor(k_contra: np.ndarray, x: np.ndarray, M: float = 1.0) -> float:
    # covariant momentum
    g_cov = schwarzschild_metric(x[1], x[2], M)
    k_cov = g_cov.dot(k_contra)

    u_emit = emitter_four_velocity(x[1], M)
    if np.isnan(u_emit[0]):
        return 0.0

    # observer at infinity u_obs = (1,0,0,0) (approx)
    omega_obs = -k_cov[0] * 1.0
    omega_emit = -float(np.dot(k_cov, u_emit))
    if omega_emit == 0.0:
        return 0.0
    return float(omega_obs / omega_emit)


def shade_disk(
    hit_info: dict, r_in: float = 6.0, r_out: float = 40.0, q: float = 3.0
) -> np.ndarray:
    x = hit_info["state"][:4]
    k_contra = hit_info["k_contra"]

    r = x[1]
    emiss = disk_emissivity(r, r_in, r_out, q)
    if emiss == 0.0:
        return np.array([0.0, 0.0, 0.0])

    g = redshift_factor(k_contra, x)
    intensity = emiss * (g**3 if g > 0.0 else 0.0)  # invariant I/nu^3

    # simple warm colormap
    v = np.clip(intensity * 8.0, 0.0, 1.0)
    return np.array([v, v**0.7, v**0.3])
