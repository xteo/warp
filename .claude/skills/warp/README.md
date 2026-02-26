# Warp Onboarding README

## Repository map at a glance

`.claude/skills/warp/` is the training skill area.
`warp/` is the package, with runtime and examples in this checkout.

## What to read first

1. [basics.md](basics.md) for core concepts
2. [kernels.md](kernels.md) for kernel patterns
3. [examples/core.md](examples/core.md) for hands-on scripts
4. [meshing.md](meshing.md) and [simulation.md](simulation.md)
5. [api/overview.md](api/overview.md) for API lookup flow
6. [troubleshooting.md](troubleshooting.md) when issues appear

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

- Kernel launch and data model: [basics.md](basics.md), [kernels.md](kernels.md)
- Core simulation examples: [examples/core.md](examples/core.md)
- FEM and PDE workflows: [examples/fem.md](examples/fem.md)
- Interop and ML frameworks: [examples/interop.md](examples/interop.md)
- Tile-based GPU kernels: [examples/tile.md](examples/tile.md)
- Optimized pipelines and differentiable training: [examples/optim.md](examples/optim.md)
- Multi-GPU and MPI: [examples/distributed.md](examples/distributed.md)

## Recommended onboarding sprint

Start by running one tiny example, then move to the same task in a structured example folder.

```sh
uv run python -m warp.examples.core.example_sample_mesh --num-frames 2
```

Then inspect and adapt:

- `warp/examples/core/example_mesh.py`
- `warp/examples/core/example_marching_cubes.py`

## External links from this skill

- Official docs: https://nvidia.github.io/warp/
- GitHub project home: https://github.com/NVIDIA/warp
- Warp package on PyPI: https://pypi.org/project/warp-lang/

