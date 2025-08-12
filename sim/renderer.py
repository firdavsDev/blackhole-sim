"""Simple CPU renderer (very slow but conceptually clear).

This module provides a Camera class that builds an orthonormal tetrad for a
static observer at the camera position and a simple ray loop that performs
backward ray-tracing by sending one geodesic per pixel.

Notes:
 - This is intentionally unoptimized (pure python loops). We'll later replace
   the heavy parts with numba/Cython/GPU shaders.
 - Coordinates: place camera at (t=0, r=r_cam, theta=theta_cam, phi=phi_cam).
"""

import os

import numpy as np

from sim.emission import shade_disk
from sim.geodesics import trace_ray

# optional background image (equirectangular)
try:
    import matplotlib.image as mpimg
except Exception:
    mpimg = None


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
        e_1 = radial unit vector
        e_2 = polar unit vector
        e_3 = azimuthal unit vector
        """
        r = self.r
        theta = self.theta
        A = 1.0 - 2.0 * self.M / r

        e0 = np.array([1.0 / np.sqrt(max(A, 1e-12)), 0.0, 0.0, 0.0])
        e1 = np.array([0.0, np.sqrt(max(A, 1e-12)), 0.0, 0.0])
        e2 = np.array([0.0, 0.0, 1.0 / r, 0.0])
        e3 = np.array([0.0, 0.0, 0.0, 1.0 / max(r * np.sin(theta), 1e-12)])

        return np.vstack([e0, e1, e2, e3])


def _load_sky_image():
    path = os.path.join("assets", "sky_equirect.jpg")
    if mpimg is None:
        return None
    if os.path.exists(path):
        try:
            img = mpimg.imread(path)
            # convert grayscale to RGB if needed
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            return img
        except Exception:
            return None
    return None


def render(
    width: int,
    height: int,
    camera: Camera,
    r_disk_inner: float = 6.0,
    r_disk_outer: float = 40.0,
    q: float = 3.0,
) -> np.ndarray:
    aspect = float(width) / float(height)
    image = np.zeros((height, width, 3), dtype=float)

    e = camera.tetrad()  # e[a, mu]
    forward = -e[1]
    right = e[3]
    up = -e[2]

    forward_sp = forward[1:]
    right_sp = right[1:]
    up_sp = up[1:]

    half_width = np.tan(camera.fov / 2.0)

    sky_img = _load_sky_image()

    for j in range(height):
        py = (1.0 - 2.0 * (j + 0.5) / height) * half_width
        for i in range(width):
            px = (2.0 * (i + 0.5) / width - 1.0) * half_width * aspect

            local_dir = forward_sp + px * right_sp + py * up_sp
            ln = np.linalg.norm(local_dir) + 1e-15
            local_dir = local_dir / ln

            # tetrad components: choose k^(0)=1 and spatial = local_dir (unit)
            k_tetrad = np.array([1.0, local_dir[0], local_dir[1], local_dir[2]])

            # build coordinate-basis k^mu = sum_a k^(a) e_a^mu
            k_coord = np.zeros(4, dtype=float)
            for a in range(4):
                k_coord += k_tetrad[a] * e[a]

            x0 = np.array([0.0, camera.r, camera.theta, camera.phi])

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
            else:  # escape -> sky
                if sky_img is not None:
                    # map local_dir to spherical coords (approx)
                    theta_dir = np.arccos(np.clip(local_dir[2], -1.0, 1.0))
                    phi_dir = np.arctan2(local_dir[1], local_dir[0])
                    u = (phi_dir + np.pi) / (2.0 * np.pi)
                    v = 1.0 - (theta_dir / np.pi)
                    h, w = sky_img.shape[0], sky_img.shape[1]
                    iu = int(np.clip(u * w, 0, w - 1))
                    iv = int(np.clip(v * h, 0, h - 1))
                    color = sky_img[iv, iu, :3]
                else:
                    theta_dir = np.arccos(np.clip(local_dir[2], -1.0, 1.0))
                    t = theta_dir / np.pi
                    color = np.array(
                        [0.1 + 0.5 * (1 - t), 0.1 + 0.6 * (1 - t), 0.2 + 0.7 * (1 - t)]
                    )

            image[j, i, :] = np.clip(color, 0.0, 1.0)

        if (j + 1) % max(1, height // 8) == 0:
            print(f"render: {j+1}/{height} rows done")

    return image
