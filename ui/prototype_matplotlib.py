"""Small runnable script: low-res prototype render using matplotlib.

Run:
    python -m ui.prototype_matplotlib

This will produce a low-res image (200x200) and display it. Expect it to be slow because we are integrating many ODEs in python.
"""

import matplotlib.pyplot as plt
import numpy as np

from sim.renderer import Camera, render, render_image


def main():
    # cam = Camera(r=20.0, theta=np.pi / 2.0, phi=0.0, fov_deg=18.0, M=1.0)
    print("Starting low-res render (200x200). This can take ~minutes on a slow CPU.")
    # img = render(200, 200, cam, r_disk_inner=6.0, r_disk_outer=40.0, q=3.0)
    img = render_image(width=200, height=200, fov=18, r_obs=20.0)

    plt.figure(figsize=(6, 6))
    plt.imshow(img, origin="lower")
    plt.axis("off")
    plt.title("blackhole-sim prototype (Schwarzschild, low-res)")
    plt.show()


if __name__ == "__main__":
    main()
