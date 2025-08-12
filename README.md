# Blackhole-sim: Scaffold + Core Physics (Schwarzschild)

A Python-based black hole visualizer and ray tracer. This project simulates photon paths around a Schwarzschild black hole, including gravitational lensing and an accretion disk with relativistic effects.

## Features (current stage)
- Schwarzschild metric implementation
- Geodesic integration for light rays
- Thin Keplerian disk emissivity model
- Redshift & Doppler effects
- Matplotlib-based prototype renderer (slow but educational)

## Project Structure
```
blackhole-sim/
├─ sim/
│  ├─ metrics.py          # Schwarzschild metric, Christoffel symbols
│  ├─ geodesics.py        # ODE system for null geodesics
│  ├─ emission.py         # Disk emissivity, Doppler, redshift
│  └─ renderer.py         # Ray loop, mapping pixel->ray
├─ ui/
│  ├─ prototype_matplotlib.py   # Proof-of-concept UI
│  └─ vispy_app.py              # Planned interactive UI (future)
├─ assets/
│  └─ sky_equirect.jpg          # Background sky texture
└─ README.md
```

## Installation
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
pip install --upgrade pip
pip install numpy scipy matplotlib
```

## Usage
Run the prototype renderer:
```bash
python -m ui.prototype_matplotlib
```

The script will produce a low-resolution (200x200) image showing gravitational lensing of the accretion disk.

Run the GPU-based renderer:
```bash
python -m ui.vispy_app
```

The script will create an interactive 3D visualization of the black hole and its surroundings.

## Units & Physics Notes
- Units: geometric (G = c = 1), M=1.0 by default
- Schwarzschild radius: r = 2M
- Photon sphere: r = 3M
- ISCO: r = 6M
- Camera: static observer on equatorial plane (default r=20M)

## Limitations
- Pure Python, CPU-bound — slow
- Only Schwarzschild (non-rotating) metric implemented
- Very basic disk model & shading

## Roadmap
1. Improve camera orientation & positioning
2. Add pixel-inspector (trace visualization)
3. Optimize with `numba` / vectorization
4. Port renderer to GPU (GLSL or compute shader)
5. Implement Kerr (rotating) metric
