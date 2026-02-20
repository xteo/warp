# API Reference: Experimental and Domain Modules

## Scope

Experimental APIs are changing; use caution and pin versions for production.

## jax_experimental

- `warp.jax_experimental.jax_kernel`
- Works with `jax.jit`, scalar launch defaults, output arity
- Supports kernel launch from JAX and input/output wrappers

Example file: `warp/examples/interop/example_jax_kernel.py`.

## New APIs in domain modules

### `warp.fem`

- Finite-element geometry and integration.
- Key concepts: `Geometry`, `FunctionSpace`, `integrate`, `interpolant`/`integrand`.

Examples:

- `warp/examples/fem/example_diffusion.py`
- `warp/examples/fem/example_navier_stokes.py`

### `warp.sparse`

- Block sparse row matrices, solvers, and direct access.

See `docs/domain_modules/sparse.rst`.

### `warp.render`

- `UsdRenderer`, `OpenGLRenderer`, CUDA texture interop.

## Optimization module

`warp.optim` includes optimizers and iterative linear operators often used with `wp.Tape`.

- `warp.optim.Adam`, `warp.optim.SGD` classes
- `warp.optim.linear` linear solvers

Examples:

- `warp/examples/optim/example_fluid_checkpoint.py`
- `warp/examples/tile/example_tile_mlp.py`

## Why this matters

These modules are where Warp becomes domain-specific. Start in examples first, then use docs API pages.

## Documentation links

- https://nvidia.github.io/warp/modules/warp_fem.html
- https://nvidia.github.io/warp/modules/warp_sparse.html
- https://nvidia.github.io/warp/modules/warp_optim.html
- https://nvidia.github.io/warp/modules/warp_render.html
- https://nvidia.github.io/warp/modules/warp_jax_experimental.html
