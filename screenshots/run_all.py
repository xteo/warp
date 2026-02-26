#!/usr/bin/env python3
"""Run all visual Warp examples and capture screenshots."""

import json
import os
import subprocess
import sys
import time

SCREENSHOT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(SCREENSHOT_DIR, "run_example.py")

# All examples to run, with extra args if needed
EXAMPLES = [
    # Core - matplotlib/visual
    ("warp.examples.core.example_raymarch", []),
    ("warp.examples.core.example_raycast", []),
    ("warp.examples.core.example_fluid", ["--num-frames", "30"]),
    ("warp.examples.core.example_graph_capture", ["--num-frames", "20"]),
    ("warp.examples.core.example_fft_poisson_navier_stokes_2d", ["--num-frames", "30"]),
    # Core - USD-based
    ("warp.examples.core.example_dem", ["--num-frames", "30"]),
    ("warp.examples.core.example_marching_cubes", ["--num-frames", "10"]),
    ("warp.examples.core.example_wave", []),
    ("warp.examples.core.example_mesh", []),
    ("warp.examples.core.example_mesh_intersect", []),
    ("warp.examples.core.example_nvdb", []),
    ("warp.examples.core.example_sample_mesh", []),
    ("warp.examples.core.example_sph", []),
    # Core - compute-only
    ("warp.examples.core.example_spin_lock", []),
    ("warp.examples.core.example_work_queue", []),
    # Optimization
    ("warp.examples.optim.example_diffray", ["--train-iters", "50"]),
    ("warp.examples.optim.example_fluid_checkpoint", ["--num-frames", "20"]),
    ("warp.examples.optim.example_particle_repulsion", []),
    # Tile
    ("warp.examples.tile.example_tile_mlp", ["--train-iters", "50"]),
    ("warp.examples.tile.example_tile_nbody", []),
    ("warp.examples.tile.example_tile_fft", []),
    ("warp.examples.tile.example_tile_filtering", []),
    ("warp.examples.tile.example_tile_matmul", []),
    ("warp.examples.tile.example_tile_convolution", []),
    ("warp.examples.tile.example_tile_mcgp", []),
    ("warp.examples.tile.example_tile_cholesky", []),
    ("warp.examples.tile.example_tile_block_cholesky", []),
    ("warp.examples.tile.example_tile_stream_compaction", []),
    # FEM
    ("warp.examples.fem.example_diffusion", []),
    ("warp.examples.fem.example_diffusion_3d", []),
    ("warp.examples.fem.example_stokes", []),
    ("warp.examples.fem.example_navier_stokes", []),
    ("warp.examples.fem.example_mixed_elasticity", []),
    ("warp.examples.fem.example_convection_diffusion", []),
    ("warp.examples.fem.example_convection_diffusion_dg", []),
    ("warp.examples.fem.example_burgers", []),
    ("warp.examples.fem.example_deformed_geometry", []),
    ("warp.examples.fem.example_streamlines", []),
    ("warp.examples.fem.example_stokes_transfer", []),
    ("warp.examples.fem.example_nonconforming_contact", []),
    ("warp.examples.fem.example_elastic_shape_optimization", []),
    ("warp.examples.fem.example_adaptive_grid", []),
    ("warp.examples.fem.example_apic_fluid", []),
    ("warp.examples.fem.example_magnetostatics", []),
    ("warp.examples.fem.example_distortion_energy", []),
    ("warp.examples.fem.example_darcy_ls_optimization", []),
]

results = {}

for module_path, extra_args in EXAMPLES:
    example_name = module_path.split(".")[-1]
    print(f"\n{'='*70}")
    print(f"[{len(results)+1}/{len(EXAMPLES)}] Running: {module_path}")
    print(f"{'='*70}")

    # Count screenshots before
    before = set(os.listdir(SCREENSHOT_DIR))

    cmd = [
        sys.executable, RUNNER, module_path
    ] + extra_args

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"

    start_time = time.time()
    try:
        proc = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min max per example
        )
        elapsed = time.time() - start_time

        # Count new screenshots
        after = set(os.listdir(SCREENSHOT_DIR))
        new_files = sorted(after - before)
        new_screenshots = [f for f in new_files if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]

        success = proc.returncode == 0
        output_lines = (proc.stdout + proc.stderr).strip().split('\n')
        # Get last few relevant lines
        tail = '\n'.join(output_lines[-5:])

        results[module_path] = {
            "success": success,
            "screenshots": new_screenshots,
            "time_s": round(elapsed, 1),
            "tail": tail,
        }

        status = "OK" if success else "FAIL"
        print(f"  [{status}] {elapsed:.1f}s | Screenshots: {new_screenshots}")
        if not success:
            print(f"  Error: {tail}")

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        results[module_path] = {
            "success": False,
            "screenshots": [],
            "time_s": round(elapsed, 1),
            "tail": "TIMEOUT (>300s)",
        }
        print(f"  [TIMEOUT] {elapsed:.1f}s")
    except Exception as e:
        elapsed = time.time() - start_time
        results[module_path] = {
            "success": False,
            "screenshots": [],
            "time_s": round(elapsed, 1),
            "tail": str(e),
        }
        print(f"  [ERROR] {e}")

# Save results
report_path = os.path.join(SCREENSHOT_DIR, "results.json")
with open(report_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {report_path}")

# Print summary
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
succeeded = sum(1 for r in results.values() if r["success"])
with_screenshots = sum(1 for r in results.values() if r["screenshots"])
total_screenshots = sum(len(r["screenshots"]) for r in results.values())
print(f"Total examples: {len(results)}")
print(f"Succeeded: {succeeded}")
print(f"Failed: {len(results) - succeeded}")
print(f"With screenshots: {with_screenshots}")
print(f"Total screenshot files: {total_screenshots}")

# List all screenshots
all_screenshots = sorted(
    f for f in os.listdir(SCREENSHOT_DIR)
    if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))
)
print(f"\nAll screenshot files:")
for f in all_screenshots:
    size = os.path.getsize(os.path.join(SCREENSHOT_DIR, f))
    print(f"  {f} ({size:,} bytes)")
