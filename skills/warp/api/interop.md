# API Reference: Interoperability

## Scope

How Warp integrates with NumPy, PyTorch, JAX, and related array frameworks.

## CPU/NumPy interoperability

- `wp.array(...)` accepts objects that expose array-like protocols.
- `array.numpy()` transfers or views to NumPy.
- Use explicit `device` when converting interface arrays.

## PyTorch

- `wp.from_torch(tensor)`
- `wp.to_torch(warp_array)`
- `dtype_from_torch`, `dtype_to_torch`
- `stream_from_torch`, `stream_to_torch`

Common pattern for optimization:

- Allocate Warp state with `requires_grad=True`.
- Convert with `wp.to_torch(...)`.
- Call `tape.backward(loss)` or custom optimizer steps.

## JAX

- Convert with `wp.from_jax`, `wp.to_jax`.
- Use `warp.jax_experimental.jax_kernel` for `jax.jit` wrappers.
- Optional static argument patterns apply for scalar launch args.

Examples:

- `warp/examples/interop/example_jax_kernel.py`

## CuPy / DLPack / CUDA array interface

- Supports zero-copy where interfaces are compatible.
- Confirm device alignment and dtype support.

## Paddle support

- Conversion helpers mirror PyTorch/JAX conventions:
  - `from_paddle`, `to_paddle`, `dtype_*`
  - stream mapping helpers

## Where to look in docs

- https://nvidia.github.io/warp/user_guide/interoperability.html
- `docs/user_guide/interoperability.rst`

## Internal source entry points

- `warp/_src/torch.py`
- `warp/_src/jax.py`
- `warp/_src/dlpack.py`
- `warp/_src/interop utilities in _src/context.py`
