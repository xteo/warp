# FEM Examples

Warp FEM is covered under `warp.fem` and includes PDEs, elasticity, and transfer workflows.

## Run examples

```sh
uv run python -m warp.examples.fem.example_diffusion
uv run python -m warp.examples.fem.example_navier_stokes
uv run python -m warp.examples.fem.example_elastic_shape_optimization
```

## Core learning examples

- `example_diffusion.py`: scalar diffusion solve.
- `example_diffusion_3d.py`: 3D variant.
- `example_convection_diffusion.py`: advection/transport behavior.
- `example_navier_stokes.py`: fluid incompressible flow forms.
- `example_burgers.py`: DG-style non-linear PDE sample.
- `example_mixed_elasticity.py`: mixed formulation.

## Advanced FEM examples

- `example_distortion_energy.py`
- `example_magnetostatics.py`
- `example_adaptive_grid.py`
- `example_nonconforming_contact.py`
- `example_streamlines.py`
- `example_stokes_transfer.py`
- `example_darcy_ls_optimization.py`
- `example_elastic_shape_optimization.py`

## Pattern from `example_diffusion.py`

Typical FEM pipeline:

1. create geometry
2. define domains
3. build trial/test spaces
4. integrate linear form / bilinear form
5. solve and optionally apply BC constraints

This pattern appears in `docs/domain_modules/fem.rst` and maps almost directly to source.

## Local code points

- `warp/examples/fem/example_diffusion.py`
- `docs/domain_modules/fem.rst`
- `warp/fem` APIs in `docs/api_reference/`

## Web references

- https://nvidia.github.io/warp/domain_modules/fem.html
- https://nvidia.github.io/warp/user_guide/differentiability.html
