# Core Examples

These are the best first examples for learning Warp patterns. All 18 core examples
have been verified to build and run successfully on CUDA (NVIDIA GB10, sm_121).

## Browsing and running examples

Run any core example:

```sh
uv run python -m warp.examples.core.<example_name>
```

For example:

```sh
uv run python -m warp.examples.core.example_mesh
```

Most visual examples accept `--headless` to skip GUI, `--device` to choose GPU/CPU,
and `--num-frames` to limit simulation steps.

## `example_mesh.py`

- Kernel-based particle simulation loop.
- Deforming mesh with kernel.
- `wp.mesh_query_ray()`, `wp.mesh_query_point_sign_normal()` for collision.
- `mesh.refit()` after topology-preserving vertex changes.

Relevant file: `warp/examples/core/example_mesh.py`

## `example_sample_mesh.py`

- Triangle-area-based CDF sampling over mesh surface.
- Kernel pipeline: triangle areas, probabilities, prefix sum.
- `wp.lower_bound` plus sampling in kernel space.

Relevant file: `warp/examples/core/example_sample_mesh.py`

## `example_marching_cubes.py`

- Signed distance field kernel.
- `wp.MarchingCubes.surface` extraction.
- OpenGL/USD style rendering integration pattern.

Relevant file: `warp/examples/core/example_marching_cubes.py`

## `example_fluid.py`

- Multi-kernel time step loop (field init, divergence, pressure solve, velocity advection).
- Standard simulation loop pattern for PDE/CFD pipelines.

Relevant file: `warp/examples/core/example_fluid.py`

## `example_sph.py`

- Smoothed Particle Hydrodynamics (SPH) simulation.
- `wp.HashGrid` for particle neighbor queries.
- Density/pressure computation kernels and integration.

Relevant file: `warp/examples/core/example_sph.py`

## `example_dem.py`

- Discrete Element Method for granular material simulation.
- `wp.HashGrid` for particle contact detection.
- Contact force computation and explicit time integration.

Relevant file: `warp/examples/core/example_dem.py`

## `example_wave.py`

- Grid-based wave equation solver.
- Explicit time stepping with stencil computation.

Relevant file: `warp/examples/core/example_wave.py`

## `example_fft_poisson_navier_stokes_2d.py`

- Spectral Navier-Stokes solver using tile-based FFT.
- Uses `wp.tile_fft()` and `wp.tile_ifft()`.

Relevant file: `warp/examples/core/example_fft_poisson_navier_stokes_2d.py`

## `example_spin_lock.py`

- Demonstrates atomic operations (`wp.atomic_add()`, etc.).
- Spin lock pattern for thread coordination.

Relevant file: `warp/examples/core/example_spin_lock.py`

## `example_graph_capture.py`

- CUDA graph capture using `wp.ScopedCapture()`.
- Records FBM kernel launches, replays via captured graph.
- Reduces kernel launch overhead for repeated patterns.

Relevant file: `warp/examples/core/example_graph_capture.py`

## `example_torch.py`

- Direct PyTorch integration with `wp.from_torch()` / `wp.to_torch()`.

Relevant file: `warp/examples/core/example_torch.py`

## `example_cupy.py`

- CuPy interoperability via `__cuda_array_interface__`.

Relevant file: `warp/examples/core/example_cupy.py`

## `example_work_queue.py`

- Dynamic work distribution pattern.
- Producer-consumer with atomic operations.

Relevant file: `warp/examples/core/example_work_queue.py`

## `example_raymarch.py`

- GPU ray marching of signed distance field (spheres, planes).
- Renders a 2048x1024 image using a single kernel.
- Output: matplotlib `plt.imshow()` display.
- Args: `--headless`, `--device`, `--width`, `--height`

Relevant file: `warp/examples/core/example_raymarch.py`

## `example_raycast.py`

- Ray casting on the Stanford Bunny mesh.
- Uses `wp.mesh_query_ray()` for per-pixel intersection.
- Requires: `pxr` (USD) for mesh loading.
- Output: matplotlib display of rendered image.
- Args: `--headless`, `--device`, `--width`, `--height`

Relevant file: `warp/examples/core/example_raycast.py`

## `example_nvdb.py`

- NanoVDB volume operations — sparse volumetric data.
- Uses `wp.Volume` for voxel reads/writes.
- Output: USD file (`example_nvdb.usd`).

Relevant file: `warp/examples/core/example_nvdb.py`

## `example_mesh_intersect.py`

- Mesh intersection queries between two meshes.
- Output: USD file (`example_mesh_intersect.usd`).

Relevant file: `warp/examples/core/example_mesh_intersect.py`

## `example_render_opengl.py`

- Interactive OpenGL rendering with `warp.render.OpenGLRenderer`.
- Tiled rendering with up to N viewports in a single frame.
- Pixel readback via `renderer.get_pixels()` (RGB and depth modes).
- Requires: `pyglet>=2.0`. Optional: `imgui[pyglet]` for UI overlays.
- Args: `--num-tiles`, `--show-plot`, `--render-mode`, `--use-imgui`
- For headless: use `xvfb-run -a` and `OpenGLRenderer(headless=True)`.

Relevant file: `warp/examples/core/example_render_opengl.py`

## Example output types

| Example | Visual output | Dependencies |
|---------|--------------|-------------|
| `example_raymarch` | matplotlib (2D image) | matplotlib |
| `example_raycast` | matplotlib (2D image) | matplotlib, pxr |
| `example_fluid` | matplotlib (animation) | matplotlib |
| `example_graph_capture` | matplotlib (animation) | matplotlib |
| `example_fft_poisson_navier_stokes_2d` | matplotlib (vorticity plot) | matplotlib |
| `example_render_opengl` | OpenGL window (RGB/depth) | pyglet>=2.0 |
| `example_mesh` | USD file | pxr |
| `example_wave` | USD file | pxr |
| `example_dem` | USD file | pxr |
| `example_sph` | USD file | pxr |
| `example_marching_cubes` | USD file | pxr |
| `example_nvdb` | USD file | pxr |
| `example_sample_mesh` | USD file | pxr |
| `example_mesh_intersect` | USD file | pxr |
| `example_spin_lock` | console only | (none) |
| `example_work_queue` | console only | (none) |
| `example_torch` | matplotlib (optimization) | torch |
| `example_cupy` | console only | cupy |

## Recommended extension path

1. Run and read `example_raymarch.py` — simplest visual output, single kernel.
2. Move to `example_sample_mesh.py` — multi-kernel pipeline with mesh operations.
3. Try `example_marching_cubes.py` — 3D field kernels + iso-surface extraction.
4. Explore `example_fluid.py` — multi-stage PDE solver pattern.
5. Advance to `warp/examples/optim/example_diffray.py` for differentiable training.

## Docs links

- https://nvidia.github.io/warp/user_guide/runtime.html
- https://nvidia.github.io/warp/user_guide/basics.html
