#!/usr/bin/env python3
"""Capture screenshots from OpenGL-based Warp examples."""

import os
import sys

os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCREENSHOT_DIR = os.path.dirname(os.path.abspath(__file__))

import warp as wp
import warp.render  # explicit import needed
wp.init()


def save_fig(name):
    for i in plt.get_fignums():
        fig = plt.figure(i)
        outpath = os.path.join(SCREENSHOT_DIR, f"{name}.png")
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        print(f"[SCREENSHOT] Saved: {outpath}")
    plt.close("all")


# 1. OpenGL Renderer - scene with capsules, cylinders, cones
print("\n=== example_render_opengl ===")
try:
    renderer = wp.render.OpenGLRenderer(vsync=False, headless=True)

    # Render scene objects (no tiled rendering, simpler approach)
    time = 0.5
    renderer.begin_frame(time)
    renderer.render_ground()
    for i in range(10):
        renderer.render_capsule(
            f"capsule_{i}",
            [i - 5.0, np.sin(time + i * 0.2), -3.0],
            [0.0, 0.0, 0.0, 1.0],
            radius=0.5,
            half_height=0.8,
        )
    renderer.render_cylinder(
        "cylinder",
        [3.2, 1.0, np.sin(time + 0.5)],
        np.array(wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.sin(time + 0.5))),
        radius=0.5,
        half_height=0.8,
    )
    renderer.render_cone(
        "cone",
        [-1.2, 1.0, 0.0],
        np.array(wp.quat_from_axis_angle(wp.vec3(0.707, 0.707, 0.0), time)),
        radius=0.5,
        half_height=0.8,
    )
    renderer.end_frame()

    # Get RGB pixels - full screen (no tiled mode)
    h, w = renderer.screen_height, renderer.screen_width
    pixels = wp.zeros((h, w, 3), dtype=wp.float32)
    renderer.get_pixels(pixels, split_up_tiles=False, mode="rgb")
    pixels_np = pixels.numpy()

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(pixels_np)
    ax.set_title(f"OpenGL Renderer - RGB ({w}x{h}, 10 capsules, cylinder, cone)")
    ax.axis("off")
    save_fig("example_render_opengl_rgb")

    # Get depth pixels
    depth_pixels = wp.zeros((h, w, 1), dtype=wp.float32)
    renderer.get_pixels(depth_pixels, split_up_tiles=False, mode="depth")
    depth_np = depth_pixels.numpy().squeeze()

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    im = ax.imshow(depth_np, vmin=renderer.camera_near_plane, vmax=renderer.camera_far_plane, cmap="viridis")
    ax.set_title("OpenGL Renderer - Depth Map")
    ax.axis("off")
    plt.colorbar(im, ax=ax, label="Depth")
    save_fig("example_render_opengl_depth")

    # Now set up tiled rendering (4 tiles)
    num_tiles = 4
    instance_ids = []
    for i in range(num_tiles):
        instances = list(range(13))  # all instances
        instance_ids.append(instances)
    renderer.setup_tiled_rendering(instance_ids)

    # Render another frame with tiles
    renderer.begin_frame(time)
    renderer.render_ground()
    for i in range(10):
        renderer.render_capsule(
            f"capsule_{i}",
            [i - 5.0, np.sin(time + i * 0.2), -3.0],
            [0.0, 0.0, 0.0, 1.0],
            radius=0.5,
            half_height=0.8,
        )
    renderer.render_cylinder(
        "cylinder",
        [3.2, 1.0, np.sin(time + 0.5)],
        np.array(wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.sin(time + 0.5))),
        radius=0.5,
        half_height=0.8,
    )
    renderer.render_cone(
        "cone",
        [-1.2, 1.0, 0.0],
        np.array(wp.quat_from_axis_angle(wp.vec3(0.707, 0.707, 0.0), time)),
        radius=0.5,
        half_height=0.8,
    )
    renderer.end_frame()

    # Get individual tiles
    tile_pixels = wp.zeros(
        (num_tiles, renderer.tile_height, renderer.tile_width, 3),
        dtype=wp.float32,
    )
    renderer.get_pixels(tile_pixels, split_up_tiles=True, mode="rgb")
    tile_np = tile_pixels.numpy()

    ncols = 2
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 10))
    for i in range(num_tiles):
        ax = axes[i // ncols, i % ncols]
        ax.imshow(tile_np[i])
        ax.set_title(f"Tile {i+1}")
        ax.axis("off")
    fig.suptitle("OpenGL Tiled Rendering (4 viewports)", fontsize=14)
    plt.tight_layout()
    save_fig("example_render_opengl_tiles")

    renderer.clear()
    print("[OK] OpenGL renderer captured (RGB, depth, tiles)")
except Exception as e:
    print(f"[FAIL] OpenGL renderer: {e}")
    import traceback
    traceback.print_exc()


# 2. APIC Fluid with OpenGL rendering
print("\n=== example_apic_fluid (OpenGL) ===")
try:
    renderer2 = wp.render.OpenGLRenderer(vsync=False, headless=True)

    from warp.examples.fem.example_apic_fluid import Example as ApicExample
    with wp.ScopedDevice("cuda:0"):
        ex = ApicExample(stage_path=None)
        # Assign OpenGL renderer
        ex.renderer = renderer2

        # Run simulation steps
        for i in range(30):
            ex.step()
            ex.render()

        # Capture
        h, w = renderer2.screen_height, renderer2.screen_width
        pixels = wp.zeros((h, w, 3), dtype=wp.float32)
        renderer2.get_pixels(pixels, split_up_tiles=False, mode="rgb")
        pixels_np = pixels.numpy()

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(pixels_np)
        ax.set_title("APIC Fluid Simulation (OpenGL, 30 steps)")
        ax.axis("off")
        save_fig("example_apic_fluid_opengl")

        renderer2.clear()
        print("[OK] APIC fluid with OpenGL captured")
except Exception as e:
    print(f"[FAIL] APIC fluid OpenGL: {e}")
    import traceback
    traceback.print_exc()


print(f"\n{'='*60}")
print("OpenGL capture complete!")
print(f"{'='*60}")

new_files = sorted(f for f in os.listdir(SCREENSHOT_DIR)
                   if f.endswith('.png') and 'opengl' in f)
print(f"OpenGL screenshots: {len(new_files)}")
for f in new_files:
    sz = os.path.getsize(os.path.join(SCREENSHOT_DIR, f))
    print(f"  {f} ({sz:,} bytes)")
