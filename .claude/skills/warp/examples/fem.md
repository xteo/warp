# FEM Examples

Warp FEM is covered under `warp.fem` and includes PDEs, elasticity, optimization, and
transfer workflows. All 18 FEM examples have been verified to build and run successfully.

## Run examples

```sh
uv run --with matplotlib --with scipy python -m warp.examples.fem.example_diffusion
uv run --with matplotlib --with scipy python -m warp.examples.fem.example_navier_stokes
uv run --with matplotlib --with scipy python -m warp.examples.fem.example_elastic_shape_optimization
```

Most FEM examples require `matplotlib` and `scipy`. Some also require `pxr` (USD).

## Complete example list (18 examples verified)

### Introductory PDE examples

| Example | Output | Description |
|---------|--------|-------------|
| `example_diffusion` | matplotlib (2D) | Scalar heat diffusion on 2D grid |
| `example_diffusion_3d` | matplotlib (3D bar) | 3D heat diffusion with bar plot |
| `example_convection_diffusion` | matplotlib | Advection-diffusion transport |
| `example_convection_diffusion_dg` | matplotlib | Discontinuous Galerkin transport |
| `example_burgers` | matplotlib | Burgers equation (non-linear PDE) |

### Fluid dynamics examples

| Example | Output | Description |
|---------|--------|-------------|
| `example_stokes` | matplotlib (pressure + streamlines) | Stokes flow |
| `example_navier_stokes` | matplotlib (vorticity) | Incompressible Navier-Stokes |
| `example_apic_fluid` | USD file / OpenGL | APIC fluid simulation (particles) |
| `example_streamlines` | OpenGL (requires pyglet) | Streamline visualization |

### Elasticity and mechanics

| Example | Output | Description |
|---------|--------|-------------|
| `example_mixed_elasticity` | matplotlib (deformed beam) | Mixed elasticity formulation |
| `example_nonconforming_contact` | matplotlib | Contact analysis between bodies |
| `example_distortion_energy` | matplotlib (3D surface + 2D) | Mesh distortion energy visualization |

### Optimization and advanced

| Example | Output | Description |
|---------|--------|-------------|
| `example_elastic_shape_optimization` | matplotlib | Shape optimization of elastic structure |
| `example_darcy_ls_optimization` | matplotlib | Level-set topology optimization for Darcy flow |
| `example_magnetostatics` | matplotlib (3D vector arrows) | 3D magnetostatic field computation |
| `example_adaptive_grid` | matplotlib (3D) | Adaptive grid refinement |
| `example_deformed_geometry` | matplotlib | Deformed mesh geometry |
| `example_stokes_transfer` | matplotlib | Stokes equation transfer between meshes |

## FEM pipeline pattern

From `example_diffusion.py` — the typical workflow:

1. Create geometry (`Grid2D`, `Grid3D`, `TriangleMesh`, etc.)
2. Define function spaces and domains (`wp.fem.make_trial`, `wp.fem.make_test`)
3. Build bilinear and linear forms with test/trial fields
4. Assemble system matrix and RHS
5. Apply boundary conditions
6. Solve linear system

This pattern appears in `docs/domain_modules/fem.rst` and maps directly to source.

## Dependencies

| Dependency | Required for |
|-----------|-------------|
| `matplotlib` | All FEM examples (visualization) |
| `scipy` | Most FEM examples (sparse solvers) |
| `pxr` (USD) | `example_apic_fluid` (USD output) |
| `pyglet>=2.0` | `example_streamlines`, `example_apic_fluid` (OpenGL mode) |

## Local code points

- `warp/examples/fem/` — all 18 example files
- `warp/fem/` — FEM API implementation
- `docs/domain_modules/fem.rst` — FEM documentation

## Web references

- https://nvidia.github.io/warp/domain_modules/fem.html
- https://nvidia.github.io/warp/user_guide/differentiability.html
