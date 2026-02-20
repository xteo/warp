# Warp Onboarding README

## Repository map at a glance

`skills/warp/` is the training skill area.
`warp/` is the package, with runtime and examples in this checkout.

## What to read first

1. `skills/warp/basics.md` for core concepts
2. `skills/warp/kernels.md` for kernel patterns
3. `skills/warp/examples/core.md` for hands-on scripts
4. `skills/warp/meshing.md` and `skills/warp/simulation.md`
5. `skills/warp/api/overview.md` for API lookup flow
6. `skills/warp/troubleshooting.md` when issues appear

## Repository entry points you should know

### Package entry

- `warp/__init__.py` exposes user-facing API symbols and module namespaces.
- `warp/_src/` holds implementation internals.

### Examples grouped by domain

- `warp/examples/core/`
- `warp/examples/fem/`
- `warp/examples/interop/`
- `warp/examples/tile/`
- `warp/examples/optim/`
- `warp/examples/distributed/`

### Documentation in-repo

- `docs/index.rst`
- `docs/user_guide/`
- `docs/domain_modules/`
- `docs/api_reference/`
- `docs/language_reference/builtins.rst`

## How this skill maps concepts to files

- Kernel launch and data model: `skills/warp/basics.md`, `skills/warp/kernels.md`
- Core simulation examples: `skills/warp/examples/core.md`
- FEM and PDE workflows: `skills/warp/examples/fem.md`
- Interop and ML frameworks: `skills/warp/examples/interop.md`
- Tile-based GPU kernels: `skills/warp/examples/tile.md`
- Optimized pipelines and differentiable training: `skills/warp/examples/optim.md`
- Multi-GPU and MPI: `skills/warp/examples/distributed.md`

## Recommended onboarding sprint

Start by running one tiny example, then move to the same task in a structured example folder.

```sh
uv run python -m warp.examples.core.example_sample_mesh --num_frames 2
```

Then inspect and adapt:

- `warp/examples/core/example_mesh.py`
- `warp/examples/core/example_marching_cubes.py`

## External links from this skill

- Official docs: https://nvidia.github.io/warp/
- GitHub project home: https://github.com/NVIDIA/warp
- Warp package on PyPI: https://pypi.org/project/warp-lang/

