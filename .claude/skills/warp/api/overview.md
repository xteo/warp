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

## Configuration (`wp.config.*`)

### Key global options

| Option | Type | Default | Purpose |
|--------|------|---------|---------|
| `mode` | `str` | `"release"` | `"debug"` enables bounds checking, `wp.breakpoint()` |
| `verbose` | `bool` | `False` | Verbose compilation output |
| `verify_fp` | `bool` | `False` | Check for NaN/Inf in kernel inputs/outputs |
| `verify_cuda` | `bool` | `False` | CUDA error checking after launches |
| `verify_autograd_array_access` | `bool` | `False` | Flag array overwrites affecting gradients |
| `enable_backward` | `bool` | `True` | Generate backward (adjoint) kernels |
| `fast_math` | `bool` | `False` | Fast math optimizations (less precise) |
| `cache_kernels` | `bool` | `True` | Persist compiled kernels between runs |
| `kernel_cache_dir` | `str?` | `None` | Override cache directory (env: `WARP_CACHE_PATH`) |
| `enable_mathdx_gemm` | `bool` | `True` | Use cuBLASDx for `tile_matmul` (disable for faster compilation) |

### Module-level settings

Override global config per-module:

```python
wp.set_module_options({"fast_math": True, "block_dim": 128})
```

### Kernel-level settings

Override per-kernel via decorator:

```python
@wp.kernel(enable_backward=False)
def forward_only_kernel(...):
    ...
```

Default block_dim is 256. For tile kernels, set via `wp.launch_tiled(..., block_dim=TILE_THREADS)`.

### AOT (ahead-of-time) compilation

Pre-compile kernels for deployment without JIT:

```python
wp.compile_aot_module(module, device="cuda")
```

Works without a GPU (generates PTX/CUBIN during Docker builds).

### Kernel cache management

Cache directory is printed by `wp.init()`. Override with:

- `wp.config.kernel_cache_dir = "/path/to/cache"`
- Environment variable: `WARP_CACHE_PATH`
- Clear cache: `wp.clear_kernel_cache()` (NEVER in multi-process context)

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

### Generics and compile-time features

- `typing.Any` as type placeholder for generic kernels (auto-specialization)
- `@wp.overload` for explicit type instantiation
- `wp.static(expr)` — evaluate Python expression at compile time (changes trigger recompilation)
- `wp.constant(value)` — module-level constant (WARNING: changes after first compilation do NOT trigger recompilation)

### Interop

- NumPy / CuPy / JAX / PyTorch / Paddle conversion APIs
- Shared-memory interfaces

Path: `docs/user_guide/interoperability.rst` and `docs/api_reference/warp.rst` sections.

### Domain modules

- FEM: `docs/domain_modules/fem.rst`
- Sparse: `docs/domain_modules/sparse.rst`
- Rendering: `docs/domain_modules/render.rst`

### Special array types

- `wp.indexedarray(source, indices)` — zero-copy virtual view with index-based access
- `wp.fabricarray` — zero-copy access to Omniverse Runtime Fabric data

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
