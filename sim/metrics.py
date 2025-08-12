"""Schwarzschild metric and Christoffel symbols.
Coordinates: (t, r, theta, phi)
Units: geometric (G = c = 1), mass M is input (default M = 1).
This file provides:  g_{mu nu}, inverse metric, and an explicit (safe) implementation
of the non-zero Christoffel symbols for the Schwarzschild metric.
"""

import numpy as np


def schwarzschild_metric(r: float, theta: float, M: float = 1.0) -> np.ndarray:
    """Return the covariant metric g_{mu nu} in Schwarzschild coordinates.

    Order: t=0, r=1, theta=2, phi=3
    """
    A = 1.0 - 2.0 * M / r
    g = np.zeros((4, 4), dtype=float)
    g[0, 0] = -A
    g[1, 1] = 1.0 / A
    g[2, 2] = r * r
    g[3, 3] = r * r * (np.sin(theta) ** 2)
    return g


def inverse_metric(r: float, theta: float, M: float = 1.0) -> np.ndarray:
    g = schwarzschild_metric(r, theta, M)
    return np.linalg.inv(g)


def christoffel_symbols(coords: tuple, M: float = 1.0) -> np.ndarray:
    """Return Christoffel symbols Gamma^mu_{alpha beta} as a (4,4,4) array.

    Uses closed-form non-zero components for Schwarzschild in standard coords.
    """
    t, r, theta, phi = coords
    eps = 1e-12
    A = 1.0 - 2.0 * M / r

    Gamma = np.zeros((4, 4, 4), dtype=float)

    # Non-zero components (symmetric in lower indices)
    # Gamma^t_{tr} = Gamma^t_{rt} = M / (r(r-2M))
    Gamma[0, 1, 0] = Gamma[0, 0, 1] = M / (r * (r - 2.0 * M) + eps)

    # Gamma^r_{tt} = M(2M - r)/r^3
    Gamma[1, 0, 0] = M * (2.0 * M - r) / (r**3 + eps)

    # Gamma^r_{rr} = -M/(r(r-2M))
    Gamma[1, 1, 1] = -M / (r * (r - 2.0 * M) + eps)

    # Gamma^r_{theta theta} = -(r-2M)
    Gamma[1, 2, 2] = -(r - 2.0 * M)

    # Gamma^r_{phi phi} = -(r-2M) * sin^2(theta)
    Gamma[1, 3, 3] = -(r - 2.0 * M) * (np.sin(theta) ** 2)

    # Gamma^theta_{r theta} = Gamma^theta_{theta r} = 1/r
    Gamma[2, 1, 2] = Gamma[2, 2, 1] = 1.0 / r

    # Gamma^theta_{phi phi} = -sin(theta) cos(theta)
    Gamma[2, 3, 3] = -np.sin(theta) * np.cos(theta)

    # Gamma^phi_{r phi} = Gamma^phi_{phi r} = 1/r
    Gamma[3, 1, 3] = Gamma[3, 3, 1] = 1.0 / r

    # Gamma^phi_{theta phi} = Gamma^phi_{phi theta} = cot(theta)
    Gamma[3, 2, 3] = Gamma[3, 3, 2] = np.cos(theta) / (max(np.sin(theta), eps))

    return Gamma
