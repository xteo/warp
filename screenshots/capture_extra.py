#!/usr/bin/env python3
"""Capture screenshots from examples that need manual rendering."""

import os
import sys

os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

SCREENSHOT_DIR = os.path.dirname(os.path.abspath(__file__))

import warp as wp
wp.init()


def save_fig(name):
    for i in plt.get_fignums():
        fig = plt.figure(i)
        outpath = os.path.join(SCREENSHOT_DIR, f"{name}.png")
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        print(f"[SCREENSHOT] Saved: {outpath}")
    plt.close("all")


# 1. Fluid simulation - render a frame
print("\n=== example_fluid ===")
try:
    from warp.examples.core.example_fluid import Example as FluidExample
    with wp.ScopedDevice("cuda:0"):
        ex = FluidExample()
        for _ in range(50):
            ex.step()
        pixels = ex.pixels.numpy()
        plt.figure(figsize=(10, 5))
        plt.imshow(pixels, cmap="gray", origin="lower")
        plt.title("Fluid Simulation (50 steps)")
        plt.colorbar()
        save_fig("example_fluid_sim")
        print("[OK] Fluid simulation captured")
except Exception as e:
    print(f"[FAIL] {e}")


# 2. Graph capture - render a frame
print("\n=== example_graph_capture ===")
try:
    from warp.examples.core.example_graph_capture import Example as GraphExample
    with wp.ScopedDevice("cuda:0"):
        ex = GraphExample()
        for _ in range(20):
            ex.step()
        pixels = ex.pixel_values.numpy()
        plt.figure(figsize=(8, 8))
        plt.imshow(pixels, cmap="gray", origin="lower")
        plt.title("CUDA Graph Capture (20 steps)")
        plt.colorbar()
        save_fig("example_graph_capture_sim")
        print("[OK] Graph capture captured")
except Exception as e:
    print(f"[FAIL] {e}")


# 3. Fluid checkpoint optimization
print("\n=== example_fluid_checkpoint ===")
try:
    from warp.examples.optim.example_fluid_checkpoint import Example as FluidCheckpointExample
    with wp.ScopedDevice("cuda:0"):
        ex = FluidCheckpointExample()
        ex.solve()
        plt.figure(figsize=(10, 5))
        plt.plot(ex.loss_history)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Fluid Checkpoint Optimization Loss")
        plt.grid(True)
        save_fig("example_fluid_checkpoint_loss")
        print("[OK] Fluid checkpoint captured")
except Exception as e:
    print(f"[FAIL] {e}")


# 4. Particle repulsion
print("\n=== example_particle_repulsion ===")
try:
    from warp.examples.optim.example_particle_repulsion import Example as ParticleExample
    with wp.ScopedDevice("cuda:0"):
        ex = ParticleExample()
        ex.step()
        # Get particle positions
        pos = ex.positions.numpy()
        plt.figure(figsize=(8, 8))
        plt.scatter(pos[:, 0], pos[:, 1], s=10, alpha=0.6)
        plt.title("Particle Repulsion (after optimization)")
        plt.axis("equal")
        plt.grid(True)
        save_fig("example_particle_repulsion_result")
        print("[OK] Particle repulsion captured")
except Exception as e:
    print(f"[FAIL] {e}")


# 5. OpenGL renderer - capture pixels
print("\n=== example_render_opengl ===")
try:
    from warp.render import OpenGLRenderer
    with wp.ScopedDevice("cuda:0"):
        renderer = OpenGLRenderer(headless=True)
        renderer.begin_frame(0.0)
        renderer.render_ground()
        renderer.render_sphere(name="sphere1", pos=(0, 1, 0), rot=(0, 0, 0, 1), radius=0.5)
        renderer.render_box(name="box1", pos=(2, 0.5, 0), rot=(0, 0, 0, 1), extents=(0.5, 0.5, 0.5))
        renderer.render_capsule(name="cap1", pos=(-2, 0.5, 0), rot=(0, 0, 0, 1), radius=0.3, half_height=0.7)
        renderer.end_frame()
        pixels = renderer.get_pixels(mode="rgb").numpy()
        if pixels is not None and pixels.size > 0:
            h, w = renderer.screen_height, renderer.screen_width
            img = pixels.reshape(h, w, 3)
            plt.figure(figsize=(12, 8))
            plt.imshow(img, origin="lower")
            plt.title("OpenGL Renderer (Headless)")
            plt.axis("off")
            save_fig("example_opengl_renderer")
            print("[OK] OpenGL renderer captured")
        else:
            print("[SKIP] No pixels returned from headless renderer")
        renderer.clear()
except Exception as e:
    print(f"[FAIL] OpenGL: {e}")


# 6. Wave simulation
print("\n=== example_wave ===")
try:
    from warp.examples.core.example_wave import Example as WaveExample
    with wp.ScopedDevice("cuda:0"):
        ex = WaveExample()
        for _ in range(100):
            ex.step()
        # Try to access height field
        if hasattr(ex, "heights"):
            h = ex.heights.numpy()
            n = int(np.sqrt(len(h)))
            plt.figure(figsize=(8, 8))
            plt.imshow(h.reshape(n, n), cmap="coolwarm", origin="lower")
            plt.title("Wave Simulation (100 steps)")
            plt.colorbar()
            save_fig("example_wave_sim")
            print("[OK] Wave sim captured")
        elif hasattr(ex, "hmap"):
            h = ex.hmap.numpy()
            n = int(np.sqrt(len(h)))
            plt.figure(figsize=(8, 8))
            plt.imshow(h.reshape(n, n), cmap="coolwarm", origin="lower")
            plt.title("Wave Simulation (100 steps)")
            plt.colorbar()
            save_fig("example_wave_sim")
            print("[OK] Wave sim captured")
        else:
            attrs = [a for a in dir(ex) if not a.startswith('_')]
            print(f"[SKIP] Wave attributes: {attrs}")
except Exception as e:
    print(f"[FAIL] {e}")


# 7. SPH simulation
print("\n=== example_sph ===")
try:
    from warp.examples.core.example_sph import Example as SphExample
    with wp.ScopedDevice("cuda:0"):
        ex = SphExample()
        for _ in range(50):
            ex.step()
        if hasattr(ex, "x"):
            pos = ex.x.numpy()
            plt.figure(figsize=(8, 8))
            ax = plt.axes(projection='3d')
            ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=1, alpha=0.5)
            ax.set_title("SPH Simulation (50 steps)")
            save_fig("example_sph_sim")
            print("[OK] SPH captured")
        else:
            attrs = [a for a in dir(ex) if not a.startswith('_')]
            print(f"[SKIP] SPH attributes: {attrs}")
except Exception as e:
    print(f"[FAIL] {e}")


# 8. FFT Poisson Navier-Stokes 2D - better capture
print("\n=== example_fft_poisson_navier_stokes_2d (more steps) ===")
try:
    from warp.examples.core.example_fft_poisson_navier_stokes_2d import Example as FFTExample
    with wp.ScopedDevice("cuda:0"):
        ex = FFTExample()
        for _ in range(100):
            ex.step()
        if hasattr(ex, "vorticity"):
            v = ex.vorticity.numpy()
            n = int(np.sqrt(len(v)))
            plt.figure(figsize=(8, 8))
            plt.imshow(v.reshape(n, n), cmap="RdBu_r", origin="lower")
            plt.title("2D Turbulence (FFT Navier-Stokes, 100 steps)")
            plt.colorbar(label="Vorticity")
            save_fig("example_fft_turbulence")
            print("[OK] FFT NS captured")
        else:
            attrs = [a for a in dir(ex) if not a.startswith('_')]
            print(f"[SKIP] FFT attributes: {attrs}")
except Exception as e:
    print(f"[FAIL] {e}")


# 9. Streamlines
print("\n=== example_streamlines ===")
try:
    from warp.examples.fem.example_streamlines import Example as StreamlinesExample
    with wp.ScopedDevice("cuda:0"):
        ex = StreamlinesExample()
        if hasattr(ex, "plot"):
            ex.plot()
            save_fig("example_streamlines_result")
            print("[OK] Streamlines captured")
        else:
            print("[SKIP] Streamlines has no plot method")
except Exception as e:
    print(f"[FAIL] {e}")


print(f"\n{'='*60}")
print("Extra capture complete!")
print(f"{'='*60}")

# List new screenshots
new_files = sorted(f for f in os.listdir(SCREENSHOT_DIR)
                   if f.endswith(('.png', '.jpg', '.gif'))
                   and ('_sim' in f or '_result' in f or '_loss' in f or 'opengl' in f or 'turbulence' in f))
print(f"New screenshots: {len(new_files)}")
for f in new_files:
    sz = os.path.getsize(os.path.join(SCREENSHOT_DIR, f))
    print(f"  {f} ({sz:,} bytes)")
