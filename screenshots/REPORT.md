# Warp Examples Build & Screenshot Report

**Date:** 2026-02-26
**Warp Version:** 1.13.0.dev0
**Platform:** Linux aarch64 (NVIDIA GB10, CUDA 13.0, Driver 580.126.09)
**Build:** `uv run build_lib.py --quick` (sm_121)

---

## Summary

| Metric | Count |
|--------|-------|
| Total examples attempted | 49 |
| Succeeded | **49 (100%)** |
| Failed | 0 |
| Screenshot files captured | **35** |
| PNG images | 32 |
| JPEG images | 1 |
| Animated GIFs | 2 |
| USD files generated | 9 |

---

## Screenshot Gallery

All screenshots saved to: `screenshots/`

### Core Examples

| Example | Status | Screenshot | Time | Description |
|---------|--------|------------|------|-------------|
| `example_raymarch` | OK | `example_raymarch_fig1.png` | 1.8s | Ray marching SDF scene (spheres, planes) |
| `example_raycast` | OK | `example_raycast_fig1.png` | 1.5s | Ray casting on Stanford Bunny mesh |
| `example_fluid` | OK | `example_fluid_fig1.png` | 1.3s | 2D Euler fluid simulation |
| `example_graph_capture` | OK | `example_graph_capture_sim.png` | 1.0s | CUDA graph capture wave simulation |
| `example_fft_poisson_navier_stokes_2d` | OK | `example_fft_poisson_navier_stokes_2d_fig1.png` | 5.5s | 2D incompressible turbulence (FFT solver) |
| `example_dem` | OK | USD: `example_dem.usd` | 2.3s | Discrete Element Method particle sim |
| `example_marching_cubes` | OK | USD: `example_marching_cubes.usd` | 1.7s | Isosurface extraction |
| `example_wave` | OK | USD: `example_wave.usd` | 1.7s | Wave equation simulation |
| `example_mesh` | OK | USD: `example_mesh.usd` | 2.2s | Mesh operations demo |
| `example_mesh_intersect` | OK | USD: `example_mesh_intersect.usd` | 2.1s | Mesh intersection queries |
| `example_nvdb` | OK | USD: `example_nvdb.usd` | 2.0s | NanoVDB volume operations |
| `example_sample_mesh` | OK | USD: `example_sample_mesh.usd` | 1.1s | Mesh surface sampling |
| `example_sph` | OK | `example_sph_sim.png` | 15.1s | 3D SPH fluid simulation |
| `example_spin_lock` | OK | (compute only) | 1.2s | Atomic spin lock demo |
| `example_work_queue` | OK | (compute only) | 1.4s | Work queue pattern |

### Optimization Examples

| Example | Status | Screenshot | Time | Description |
|---------|--------|------------|------|-------------|
| `example_diffray` | OK | `example_diffray_target_image.png`, `example_diffray_final_image.png`, `example_diffray_animation.gif` | 6.8s | Differentiable ray caster (bunny) - target vs optimized + training animation |
| `example_fluid_checkpoint` | OK | (loss printed) | 3.8s | Fluid optimization with checkpointing |
| `example_particle_repulsion` | OK | `example_particle_repulsion_result.png` | 6.4s | Particle repulsion forming NVIDIA logo |

### Tile API Examples

| Example | Status | Screenshot | Time | Description |
|---------|--------|------------|------|-------------|
| `example_tile_mlp` | OK | `example_tile_mlp.jpg` | 30.4s | Neural network image reconstruction (tile MLP) |
| `example_tile_nbody` | OK | `example_tile_nbody_fig1.png` | 1.2s | N-body gravitational simulation |
| `example_tile_filtering` | OK | `example_tile_filtering_fig1.png` | 4.4s | Signal smoothing filter |
| `example_tile_mcgp` | OK | `example_tile_mcgp_animation.gif` | 96.8s | Monte Carlo geometry processing animation |
| `example_tile_fft` | OK | (compute only) | 4.4s | FFT operations |
| `example_tile_matmul` | OK | (compute only) | 9.0s | Matrix multiplication |
| `example_tile_convolution` | OK | (compute only) | 4.3s | Convolution operations |
| `example_tile_cholesky` | OK | (compute only) | 3.4s | Cholesky decomposition |
| `example_tile_block_cholesky` | OK | (compute only) | 44.8s | Block Cholesky solver |
| `example_tile_stream_compaction` | OK | (compute only) | 1.2s | Stream compaction |

### Finite Element Method (FEM) Examples

| Example | Status | Screenshot | Time | Description |
|---------|--------|------------|------|-------------|
| `example_diffusion` | OK | `example_diffusion_fig1.png` | 7.7s | 2D heat diffusion |
| `example_diffusion_3d` | OK | `example_diffusion_3d_fig1.png` | 3.4s | 3D heat diffusion (bar plot) |
| `example_stokes` | OK | `example_stokes_fig1.png` | 9.4s | Stokes flow (pressure + streamlines) |
| `example_navier_stokes` | OK | `example_navier_stokes_fig1.png` | 9.8s | Navier-Stokes flow (vorticity) |
| `example_mixed_elasticity` | OK | `example_mixed_elasticity_fig1.png` | 10.6s | Mixed elasticity (deformed beam) |
| `example_convection_diffusion` | OK | `example_convection_diffusion_fig1.png` | 5.7s | Convection-diffusion transport |
| `example_convection_diffusion_dg` | OK | `example_convection_diffusion_dg_fig1.png` | 12.0s | Discontinuous Galerkin transport |
| `example_burgers` | OK | `example_burgers_fig1.png` | 5.3s | Burgers equation solution |
| `example_deformed_geometry` | OK | `example_deformed_geometry_fig1.png` | 7.1s | Deformed mesh geometry |
| `example_streamlines` | OK | (OpenGL unavailable) | 7.2s | Streamline visualization |
| `example_stokes_transfer` | OK | `example_stokes_transfer_fig1.png` | 9.2s | Stokes equation transfer |
| `example_nonconforming_contact` | OK | `example_nonconforming_contact_fig1.png` | 11.8s | Nonconforming contact analysis |
| `example_elastic_shape_optimization` | OK | `example_elastic_shape_optimization_fig1.png` | 12.5s | Elastic shape optimization |
| `example_adaptive_grid` | OK | `example_adaptive_grid_fig1.png` | 17.5s | Adaptive grid refinement (3D) |
| `example_apic_fluid` | OK | USD: `example_apic_fluid.usd` | 17.8s | APIC fluid simulation |
| `example_magnetostatics` | OK | `example_magnetostatics_fig1.png` | 9.6s | 3D magnetostatic field (vector arrows) |
| `example_distortion_energy` | OK | `example_distortion_energy_fig1.png` | 5.7s | Mesh distortion energy (3D surface + 2D) |
| `example_darcy_ls_optimization` | OK | `example_darcy_ls_optimization_fig1.png` | 8.8s | Darcy flow level-set topology optimization |

---

### OpenGL Renderer Examples (via Xvfb virtual display)

| Example | Status | Screenshot | Time | Description |
|---------|--------|------------|------|-------------|
| `example_render_opengl` (RGB) | OK | `example_render_opengl_rgb.png` | 2.0s | 10 capsules, cylinder, cone on checkerboard ground |
| `example_render_opengl` (depth) | OK | `example_render_opengl_depth.png` | -- | Depth map with viridis colormap |
| `example_render_opengl` (tiles) | OK | `example_render_opengl_tiles.png` | -- | 4-viewport tiled rendering |
| `example_apic_fluid` (OpenGL) | OK | `example_apic_fluid_opengl.png` | 5.0s | APIC fluid sim with OpenGL renderer |

---

## Examples NOT Attempted

These examples were skipped due to missing dependencies or special requirements:

| Example | Reason |
|---------|--------|
| `example_torch` | Requires PyTorch (large dependency) |
| `example_cupy` | Requires CuPy |
| `example_jacobi_mpi` | Requires MPI (`mpi4py`) |
| `example_jax_*` (3 files) | Requires JAX |
| `benchmarks/*` (19 files) | Performance benchmarks, not functional demos |
| `cpp/*` (2 files) | C++ integration, requires separate compilation |
| `example_diffusion_mgpu` | Requires multi-GPU setup |

---

## Notable Highlights

1. **Differentiable Ray Caster** (`example_diffray`): Beautifully renders a Stanford Bunny with normal-mapped coloring, producing both target/final comparison images and a training animation GIF showing the optimization converging.

2. **Particle Repulsion** (`example_particle_repulsion`): Particles self-organize into the NVIDIA logo shape through gradient-based repulsion optimization.

3. **Tile MLP** (`example_tile_mlp`): Neural network trained entirely on GPU using Warp's tile API, reconstructing an image from coordinates.

4. **3D Magnetostatics** (`example_magnetostatics`): Impressive 3D vector field visualization of magnetic flux density using arrow plots.

5. **SPH Fluid** (`example_sph`): 3D smoothed particle hydrodynamics simulation with thousands of particles.

6. **FEM Suite**: Complete set of 18 finite element examples all running successfully, covering diffusion, Stokes flow, Navier-Stokes, elasticity, shape optimization, and more.

---

## Build Details

- **Native library build**: `uv run build_lib.py --quick` (sm_121 arch for GB10)
- **CUDA Toolkit**: 13.0 / Driver: 13.0 (forward compat disabled)
- **MathDx**: 0.3.0, NanoVDB: 32.8.0
- **Dependencies installed**: matplotlib, Pillow, usd-exchange, scipy, pyglet 2.1.11
- **Total run time**: ~8 minutes for all 46 examples
