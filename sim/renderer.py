"""Simple CPU renderer (very slow but conceptually clear).

This module provides a Camera class that builds an orthonormal tetrad for a
static observer at the camera position and a simple ray loop that performs
backward ray-tracing by sending one geodesic per pixel.

Notes:
 - This is intentionally unoptimized (pure python loops). We'll later replace
   the heavy parts with numba/Cython/GPU shaders.
 - Coordinates: place camera at (t=0, r=r_cam, theta=theta_cam, phi=phi_cam).
"""

import numpy as np

from sim.emission import disk_emissivity, shade_disk
from sim.geodesics import integrate_null_geodesic, trace_ray
from sim.metrics import schwarzschild_metric


class Camera:
    def __init__(self, r=20.0, theta=np.pi / 2.0, phi=0.0, fov_deg=20.0, M=1.0):
        self.r = float(r)
        self.theta = float(theta)
        self.phi = float(phi)
        self.fov = float(np.deg2rad(fov_deg))
        self.M = float(M)

    def tetrad(self):
        """Return orthonormal tetrad vectors e_a^mu (a=0..3, mu=0..3).

        e_0 = static observer 4-velocity (normalised)
        e_1 = radial unit vector (points outward)
        e_2 = polar (theta) unit vector
        e_3 = azimuthal (phi) unit vector
        """
        r = self.r
        theta = self.theta
        A = 1.0 - 2.0 * self.M / r

        # e_0^mu = (1/sqrt(A), 0, 0, 0)
        e0 = np.array([1.0 / np.sqrt(max(A, 1e-12)), 0.0, 0.0, 0.0])
        # e_1^mu = (0, sqrt(A), 0, 0)
        e1 = np.array([0.0, np.sqrt(max(A, 1e-12)), 0.0, 0.0])
        # e_2^mu = (0, 0, 1/r, 0)
        e2 = np.array([0.0, 0.0, 1.0 / r, 0.0])
        # e_3^mu = (0, 0, 0, 1/(r*sin(theta)))
        e3 = np.array([0.0, 0.0, 0.0, 1.0 / max(r * np.sin(theta), 1e-12)])

        # Return as array shape (4,4): e[a, mu]
        return np.vstack([e0, e1, e2, e3])


def render(
    width: int,
    height: int,
    camera: Camera,
    r_disk_inner: float = 6.0,
    r_disk_outer: float = 40.0,
    q: float = 3.0,
) -> np.ndarray:
    """Render a low-res image (width x height) and return an RGB float image.

    This function loops over pixels and traces one geodesic per pixel.
    """
    aspect = float(width) / float(height)
    image = np.zeros((height, width, 3), dtype=float)

    e = camera.tetrad()  # e[a, mu]
    # define camera local axes: forward points toward BH center = -e1
    forward = -e[1]
    right = e[3]
    up = -e[2]

    # field plane scaling
    half_width = np.tan(camera.fov / 2.0)

    for j in range(height):
        py = (1.0 - 2.0 * (j + 0.5) / height) * half_width  # vertical normalized
        for i in range(width):
            px = (2.0 * (i + 0.5) / width - 1.0) * half_width * aspect

            # direction in camera's local spatial frame (3-vector wrt tetrad spatial axes)
            local_dir = forward[1:] + px * right[1:] + py * up[1:]
            local_dir /= np.linalg.norm(local_dir) + 1e-12

            # assemble tetrad components k^(a): choose k^(0)=1 (photon energy at camera)
            k_tetrad = np.array([1.0, local_dir[0], local_dir[1], local_dir[2]])

            # convert to coordinate basis: k^mu = sum_a k^(a) e_a^mu
            k_coord = np.zeros(4, dtype=float)
            for a in range(4):
                k_coord += k_tetrad[a] * e[a]

            # initial position x0 (t=0)
            x0 = np.array([0.0, camera.r, camera.theta, camera.phi])

            # trace geodesic
            hit = trace_ray(
                x0,
                k_coord,
                M=camera.M,
                r_disk_inner=r_disk_inner,
                r_disk_outer=r_disk_outer,
            )

            if hit["type"] == "disk":
                color = shade_disk(hit, r_disk_inner, r_disk_outer, q)
            elif hit["type"] == "horizon":
                color = np.array([0.0, 0.0, 0.0])
            else:  # escape -> sky background (simple gradient)
                # use direction polar angle to make a subtle sky
                theta_dir = np.arccos(np.clip(local_dir[2], -1.0, 1.0))
                t = theta_dir / np.pi
                color = np.array(
                    [0.1 + 0.4 * (1 - t), 0.1 + 0.5 * (1 - t), 0.2 + 0.6 * (1 - t)]
                )

            image[j, i, :] = np.clip(color, 0.0, 1.0)

        # simple progress print
        if (j + 1) % max(1, height // 8) == 0:
            print(f"render: {j+1}/{height} rows done")

    return image


def render_image(width=200, height=200, fov=60, r_obs=20.0):
    """
    Ray-trace the black hole image.
    width, height: output resolution
    fov: field of view in degrees
    r_obs: observer distance in M units
    """
    img = np.zeros((height, width, 3), dtype=np.float32)

    # Camera parameters
    aspect_ratio = width / height
    scale = np.tan(np.radians(fov) / 2)

    # Loop over pixels
    for y in range(height):
        if y % 25 == 0:
            print(f"render: {y}/{height} rows done")

        for x in range(width):
            # Normalized pixel coords [-1, 1]
            px = (2 * (x + 0.5) / width - 1) * aspect_ratio * scale
            py = (1 - 2 * (y + 0.5) / height) * scale

            # Ray initial conditions
            L = np.sqrt(px**2 + py**2) * r_obs
            E = 1.0
            init_state = [r_obs, np.pi / 2, 0.0, -1.0, 0.0, L]  # r, θ, φ, p_r, p_θ, p_φ

            hit, r_hit, phi_hit, g_factor = integrate_null_geodesic(
                init_state, schwarzschild_metric
            )

            if hit:
                intensity = disk_emissivity(r_hit, phi_hit) * g_factor**4
                img[y, x] = [
                    intensity,
                    intensity * 0.6,
                    0.2 * intensity,
                ]  # warm disk colors
            else:
                img[y, x] = [0.02, 0.02, 0.08]  # deep space blue background

    return np.clip(img, 0, 1)
