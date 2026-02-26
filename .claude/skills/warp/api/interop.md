# API Reference: Interoperability

## Scope

How Warp integrates with NumPy, PyTorch, JAX, and related array frameworks.

## CPU/NumPy interoperability

- `wp.array(np_data, dtype=float)` — create from NumPy array
- `wp.from_numpy(np_data)` — explicit conversion
- `array.numpy()` — readback (zero-copy view for CPU, data transfer for CUDA)
- Use explicit `device` when converting interface arrays.

## PyTorch

### Basic conversion

- `wp.from_torch(tensor)` — zero-copy on same device, preserves `requires_grad`
- `wp.to_torch(warp_array)` — zero-copy on same device
- `wp.dtype_from_torch(torch_dtype)`, `wp.dtype_to_torch(wp_dtype)`
- `wp.stream_from_torch(torch_stream)`, `wp.stream_to_torch(wp_stream)`

### Performance optimization

Use `return_ctype=True` for ~4.3x speedup when converting frequently:

```python
arr = wp.from_torch(tensor, return_ctype=True)  # Returns low-level descriptor
```

Convert once outside loops and pre-allocate gradients to avoid sync overhead.

### PyTorch custom op integration

For PyTorch <= 2.3, wrap Warp operations with `torch.autograd.Function`:

```python
class WarpOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        wp_x = wp.from_torch(x)
        # ... run Warp kernels ...
        return wp.to_torch(result)
```

For PyTorch >= 2.4, use `@torch.library.custom_op` (compatible with `torch.compile()`):

```python
@torch.library.custom_op("warp::my_op", mutates_args=())
def my_op(x: torch.Tensor) -> torch.Tensor:
    wp_x = wp.from_torch(x)
    # ... run Warp kernels ...
    return wp.to_torch(result)
```

### End-to-end differentiable pipeline

Pattern: PyTorch network -> `wp.from_torch()` -> Warp simulation under `wp.Tape()` ->
`wp.to_torch()` -> PyTorch backward. See `warp/examples/core/example_torch.py`.

## JAX

- `wp.from_jax(jax_array)`, `wp.to_jax(warp_array)`
- `warp.jax_experimental.jax_kernel()` for `@jax.jit` wrappers
- Scalar launch args must be static (use `static_argnums` in JAX)

### JAX vmap support

```python
from warp.jax_experimental import jax_kernel
jax_fn = jax_kernel(wp_kernel, vmap_method="broadcast_all")
```

Options for `vmap_method`: `"broadcast_all"`, `"sequential"`.

Examples: `warp/examples/interop/example_jax_kernel.py`

## CuPy

Zero-copy sharing via `__cuda_array_interface__`:

```python
import cupy as cp
wp_array = wp.array(cupy_array)  # Uses CUDA array interface
```

See `warp/examples/core/example_cupy.py` for a complete example.

## DLPack protocol

- `wp.from_dlpack(dl_tensor)` — import from any DLPack-compatible framework
- `wp.to_dlpack(warp_array)` — export to DLPack format

Used internally by JAX conversion.

## Paddle support

- `wp.from_paddle(tensor)`, `wp.to_paddle(warp_array)`
- `wp.dtype_from_paddle()`, `wp.dtype_to_paddle()`
- `wp.stream_from_paddle()` — stream coordination

Same zero-copy patterns as PyTorch.

## Where to look in docs

- https://nvidia.github.io/warp/user_guide/interoperability.html
- `docs/user_guide/interoperability.rst`

## Internal source entry points

- `warp/_src/torch.py`
- `warp/_src/jax.py`
- `warp/_src/dlpack.py`
- `warp/_src/interop utilities in _src/context.py`
