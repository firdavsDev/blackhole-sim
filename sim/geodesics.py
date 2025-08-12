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
    lam_max: float = 1e5
) -> dict:
    """Integrate one geodesic and return a small hit-summary dict.

    x0: array-like [t,r,theta,phi]
    k0: contravariant 4-momentum k^mu at x0
    """
    y0 = np.concatenate([x0, k0]).astype(float)

    # events (closures so they can see r_disk_inner / r_disk_outer)
    def horizon_event(lam, y):
        r = y[1]
        return r - 2.0 * M

    horizon_event.terminal = True
    horizon_event.direction = -1

    def disk_event(lam, y):
        r = y[1]
        theta = y[2]
        # only trigger if radius is inside disk band
        if r_disk_inner <= r <= r_disk_outer:
            return theta - (np.pi / 2.0)
        return 1.0  # no zero crossing while outside disk

    disk_event.terminal = True
    disk_event.direction = 0

    def escape_event(lam, y):
        r = y[1]
        return r - r_max

    escape_event.terminal = True
    escape_event.direction = 1

    sol = solve_ivp(
        lambda t, y: geodesic_ode(t, y, M),
        (0.0, lam_max),
        y0,
        events=[horizon_event, disk_event, escape_event],
        rtol=1e-6,
        atol=1e-9,
        max_step=1.0,
    )

    # determine which event fired (if any)
    hit_type = "none"
    hit_state = sol.y[:, -1]

    if sol.t_events[0].size > 0:
        hit_type = "horizon"
        hit_state = sol.y_events[0][0]
    elif sol.t_events[1].size > 0:
        hit_type = "disk"
        hit_state = sol.y_events[1][0]
    elif sol.t_events[2].size > 0:
        hit_type = "escape"
        hit_state = sol.y_events[2][0]

    # covariant momentum at intersection
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


def integrate_null_geodesic(
    init_state, metric_func, r_disk=6.0, r_sing=2.0, max_tau=200
):
    """
    Integrate null geodesic until it hits the disk or escapes.
    Returns:
        hit (bool), r_hit, phi_hit, g_factor
    """

    def geodesic_equations(tau, y):
        r, theta, phi, pr, ptheta, pphi = y
        g = metric_func(r, theta)

        # Simplified: radial derivative from momentum
        dr_dtau = pr
        dtheta_dtau = ptheta / r**2
        dphi_dtau = pphi / (r**2 * np.sin(theta) ** 2)

        # No full Christoffel here yet — minimal test version
        dpr_dtau = -(1 - 2 / r) * (pphi**2 / (r**3 * np.sin(theta) ** 2))
        dptheta_dtau = 0.0
        dpphi_dtau = 0.0

        return [dr_dtau, dtheta_dtau, dphi_dtau, dpr_dtau, dptheta_dtau, dpphi_dtau]

    def stop_conditions(tau, y):
        r = y[0]
        if r <= r_sing:
            return 0
        if r <= r_disk and abs(y[1] - np.pi / 2) < 1e-3:
            return 0
        if r > 100:
            return 0
        return 1

    sol = solve_ivp(geodesic_equations, [0, max_tau], init_state, max_step=0.1)

    for i in range(sol.y.shape[1]):
        r = sol.y[0, i]
        theta = sol.y[1, i]
        phi = sol.y[2, i]
        if r <= r_disk and abs(theta - np.pi / 2) < 1e-3:
            return True, r, phi, 1.0  # TODO: real g_factor
    return False, None, None, 1.0
