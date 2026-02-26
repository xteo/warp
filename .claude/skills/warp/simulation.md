# Simulation Workflows

Warp targets simulation and optimization workflows across fluids, soft bodies, FEM, and differentiable pipelines.

## What this page covers

- Core simulation examples in `warp/examples/core`
- FEM workflows in `warp/examples/fem`
- Differentiable optimization in `warp/examples/optim`
- Performance tooling and domain modules

## Core simulation examples

### Fluid advection (`warp/examples/core/example_fluid.py`)

This example shows a standard simulation loop with multiple kernels:

- Field initialization
- Divergence computation
- Pressure solve
- Velocity advection
- Pressure apply

Use this structure for your own PDE/CFD pipeline:

1. allocate `wp.array` state
2. run stage kernels in order
3. write outputs

### SPH particle simulation (`warp/examples/core/example_sph.py`)

Smoothed Particle Hydrodynamics using `wp.HashGrid` for neighbor queries,
density/pressure computation kernels, and integration steps.

### DEM granular simulation (`warp/examples/core/example_dem.py`)

Discrete Element Method with `wp.HashGrid`, contact force computation,
and explicit time integration for granular materials.

### Wave equation solver (`warp/examples/core/example_wave.py`)

Grid-based stencil computation with explicit time stepping for the wave equation.

### FFT-based simulation (`warp/examples/core/example_fft_poisson_navier_stokes_2d.py`)

Spectral Navier-Stokes solver using tile-based FFT via `wp.tile_fft()`/`wp.tile_ifft()`.

### Mesh interaction (`warp/examples/core/example_mesh.py`)

- Demonstrates `wp.Mesh` collision queries.
- Shows vertex deformation via kernel.
- Calls `mesh.refit()` to keep BVH valid after geometry updates.

## Differentiable simulation

`wp.Tape` enables gradient-based optimization of kernel programs.

```python
with wp.Tape() as tape:
    wp.launch(loss_kernel, dim=n, inputs=[state], outputs=[loss])

tape.backward(loss=loss)
```

Set arrays with `requires_grad=True` and run optimizer updates in Python. For richer examples see:

- `warp/examples/optim/example_fluid_checkpoint.py`
- `warp/examples/tile/example_tile_mlp.py`

These examples show checkpointing, batching, and multi-stage optimizers.

## FEM in Warp

18 FEM examples in `warp/examples/fem/`, all verified to build and run. See [examples/fem.md](examples/fem.md) for the complete list with output types and descriptions.

Categories:
- **Introductory PDEs** (5): diffusion, diffusion_3d, convection_diffusion, convection_diffusion_dg, burgers
- **Fluid dynamics** (4): stokes, navier_stokes, apic_fluid, streamlines
- **Elasticity** (3): mixed_elasticity, nonconforming_contact, distortion_energy
- **Optimization/advanced** (6): elastic_shape_optimization, darcy_ls_optimization, magnetostatics, adaptive_grid, deformed_geometry, stokes_transfer

The core API page for setup is in `docs/domain_modules/fem.rst`.

FEM workflow pattern:

- Create geometry (`Grid2D`, `Grid3D`, `TriangleMesh`, etc.)
- Define function spaces and domains
- Build forms with test/trial fields
- Assemble and solve with linear solvers

Dependencies: `matplotlib`, `scipy` (most examples), `pxr` (apic_fluid USD), `pyglet>=2.0` (streamlines, apic_fluid OpenGL).

## Sparse and linear algebra

See `docs/domain_modules/sparse.rst` and FEM examples for solver usage such as conjugate gradients, CG-like workflows, and custom operators.

## Distributed simulation

`warp/examples/distributed/example_jacobi_mpi.py` shows:

- MPI-aware rank decomposition
- Explicit stream and event usage
- Multi-device iterative solves

Useful when scaling CPU/GPU workflows across nodes.

## Tile-based GPU compute

10 tile examples in `warp/examples/tile/`, all verified. See [examples/tile.md](examples/tile.md) for details.

Highlights:
- `example_tile_mlp` — neural network training entirely on GPU (30s, produces image)
- `example_tile_nbody` — N-body gravitational simulation with tile-based force accumulation
- `example_tile_matmul` — basic tile matrix multiplication pattern
- `example_tile_fft` — FFT operations backed by cuFFTDx
- `example_tile_cholesky` — linear system solving with Cholesky factorization

Tile operations require CUDA 12.6.3+ and MathDx. Use `wp.launch_tiled()` instead of `wp.launch()`.

## Performance and capture

Timing:

```python
with wp.ScopedTimer("step", synchronize=True):
    wp.launch(kernel, dim=n, inputs=[state])
```

The `synchronize=True` parameter ensures GPU work completes before measuring wall-clock time.
For GPU-only timing, use CUDA events. For profiling with NVIDIA Nsight Systems, use NVTX
markers which Warp emits automatically when `wp.config.enable_nvtx = True`.

Graph capture for repeated loops:

- Use `wp.ScopedCapture()` to record, `wp.capture_launch()` to replay.
- See `warp/examples/core/example_graph_capture.py`.

Use tile APIs for block-level data-parallel operations.

## Where to read next

- [examples/fem.md](examples/fem.md)
- [kernels.md](kernels.md)
- `docs/user_guide/differentiability.rst`
- `docs/domain_modules/fem.rst`
- `docs/domain_modules/sparse.rst`

## Web references

- https://nvidia.github.io/warp/domain_modules/fem.html
- https://nvidia.github.io/warp/domain_modules/sparse.html
- https://nvidia.github.io/warp/user_guide/differentiability.html
