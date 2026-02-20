# API Overview

## Use this file first

Start here to discover what module to read for a task.

## Public modules

- `warp` (always available on `import warp`).
- `warp.autograd`
- `warp.config`
- `warp.fem`
- `warp.jax_experimental`
- `warp.optim`
- `warp.render`
- `warp.sparse`
- `warp.types`
- `warp.utils`

See `warp/__init__.py` for import behavior and auto-loaded names.

## How to locate APIs quickly

1. Start from symbol name.
2. If symbol is kernel-level and math-like, use runtime/function docs.
3. If symbol is a high-level domain concept, use module docs.

## Core API families

### Runtime and launch

- Device helpers
- Array creation and transfer
- Streams, events, graphs
- Launch helpers and timers

Path: `docs/user_guide/runtime.rst`, `docs/user_guide/runtime` entries.

### Language built-ins

- Math primitives
- Vector/matrix operators
- Atomic operations
- Random and geometry helpers

Path: `docs/language_reference/builtins.rst`.

### Interop

- NumPy / CuPy / JAX / PyTorch / Paddle conversion APIs
- Shared-memory interfaces

Path: `docs/user_guide/interoperability.rst` and `docs/api_reference/warp.rst` sections.

### Domain modules

- FEM: `docs/domain_modules/fem.rst`
- Sparse: `docs/domain_modules/sparse.rst`
- Rendering: `docs/domain_modules/render.rst`

## Local module map to files

- API docs generated pages are in `docs/api_reference/`.
- Runtime behavior and examples often connect to source in:
  - `warp/_src/context.py`
  - `warp/_src/types.py`
  - `warp/_src/tape.py`
  - `warp/_src/build.py`

## Web reference

- https://nvidia.github.io/warp/
- https://nvidia.github.io/warp/modules/runtime.html
- https://nvidia.github.io/warp/modules/functions.html
- https://nvidia.github.io/warp/modules/configuration.html
