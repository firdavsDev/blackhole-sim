"""Geodesic ODE system and a simple trace_ray helper.

We integrate the first-order system for null geodesics:
    dx^mu/dlambda = k^mu
    dk^mu/dlambda = - Gamma^mu_{alpha beta} k^alpha k^beta

State vector y = [t, r, theta, phi, k^t, k^r, k^theta, k^phi]
"""

import numpy as np
from scipy.integrate import solve_ivp

from sim.metrics import christoffel_symbols, schwarzschild_metric


def geodesic_ode(lam: float, y: np.ndarray, M: float = 1.0) -> np.ndarray:
    x = y[:4]
    k = y[4:]
    # compute Christoffel at the current spacetime point
    Gamma = christoffel_symbols((x[0], x[1], x[2], x[3]), M)

    dx = k.copy()
    dk = np.zeros(4, dtype=float)

    # dk^mu/dlambda = -Gamma^mu_{alpha beta} k^alpha k^beta
    for mu in range(4):
        s = 0.0
        for alpha in range(4):
            for beta in range(4):
                s += Gamma[mu, alpha, beta] * k[alpha] * k[beta]
        dk[mu] = -s

    return np.concatenate([dx, dk])


def trace_ray(
    x0: np.ndarray,
    k0: np.ndarray,
    *,
    M: float = 1.0,
    r_disk_inner: float = 6.0,
    r_disk_outer: float = 40.0,
    r_max: float = 2000.0,
    lam_max: float = 1e5,
    rtol: float = 1e-6,
    atol: float = 1e-9
) -> dict:
    """Integrate one geodesic and return a hit-summary dict.

    x0: [t, r, theta, phi]
    k0: contravariant 4-momentum k^mu at x0
    """
    y0 = np.concatenate([x0, k0]).astype(float)

    # event: horizon crossing r - 2M = 0 (terminal)
    def horizon_event(lam, y):
        r = y[1]
        return r - 2.0 * M

    horizon_event.terminal = True
    horizon_event.direction = -1

    # event: crossing equatorial plane theta - pi/2 = 0 (non-terminal)
    def theta_cross_event(lam, y):
        theta = y[2]
        return theta - (np.pi / 2.0)

    theta_cross_event.terminal = False
    theta_cross_event.direction = 0

    # event: escaping far away
    def escape_event(lam, y):
        r = y[1]
        return r - r_max

    escape_event.terminal = True
    escape_event.direction = 1

    sol = solve_ivp(
        lambda t, y: geodesic_ode(t, y, M),
        (0.0, lam_max),
        y0,
        events=[horizon_event, theta_cross_event, escape_event],
        rtol=rtol,
        atol=atol,
        max_step=1.0,
    )

    hit_type = "none"
    hit_state = sol.y[:, -1]

    # 1) did horizon event happen?
    if sol.t_events[0].size > 0:
        hit_type = "horizon"
        hit_state = sol.y_events[0][0]

    # 2) check equatorial crossings for disk intersection
    # sol.y_events[1] is a list of states at theta crossings (may be empty)
    elif len(sol.y_events) > 1 and len(sol.y_events[1]) > 0:
        # find the earliest crossing which has r within disk bounds
        disk_hit = None
        for st in sol.y_events[1]:
            r_val = st[1]
            if r_disk_inner <= r_val <= r_disk_outer:
                disk_hit = st
                break
        if disk_hit is not None:
            hit_type = "disk"
            hit_state = disk_hit

    # 3) escape
    if hit_type == "none" and sol.t_events[2].size > 0:
        hit_type = "escape"
        hit_state = sol.y_events[2][0]

    # compute covariant momentum at intersection (if available)
    x_hit = hit_state[:4]
    k_contra = hit_state[4:]
    g_cov = schwarzschild_metric(x_hit[1], x_hit[2], M)
    k_cov = g_cov.dot(k_contra)

    return {
        "type": hit_type,
        "state": hit_state,
        "k_contra": k_contra,
        "k_cov": k_cov,
        "sol": sol,
    }
