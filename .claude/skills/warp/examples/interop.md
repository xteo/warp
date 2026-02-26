# Interop Examples

Interoperability lets Warp sit inside ML frameworks and array ecosystems.
These examples require their respective framework dependencies to be installed.

## PyTorch (`example_torch.py`)

Located in `warp/examples/core/example_torch.py`:

- Direct integration with `wp.from_torch()` / `wp.to_torch()`
- Gradient flow between Warp kernels and PyTorch autograd
- Uses `requires_grad=True` tensors with `wp.Tape`
- Output: matplotlib optimization plot
- Dependency: `torch`

```sh
uv run --with torch python -m warp.examples.core.example_torch
```

## CuPy (`example_cupy.py`)

Located in `warp/examples/core/example_cupy.py`:

- Zero-copy sharing via `__cuda_array_interface__`
- Console-only output
- Dependency: `cupy`

```sh
uv run --with cupy python -m warp.examples.core.example_cupy
```

## JAX (3 examples)

Located in `warp/examples/interop/`:

| Example | Description |
|---------|-------------|
| `example_jax_kernel.py` | Wrapping Warp kernels for JAX, multiple outputs, launch dimension overrides |
| `example_jax_fk.py` | Forward kinematics with JAX integration |
| `example_jax_primitives.py` | JAX custom primitives with Warp |

```sh
uv run --with jax python -m warp.examples.interop.example_jax_kernel
```

All JAX examples demonstrate:
- Wrapping kernels with `jax_kernel`
- Returning multiple outputs
- Launch dimension defaults and overrides
- Scalar launch parameters as static args

## NumPy (always available)

NumPy interop is built in — no extra dependencies:

- `wp.from_numpy(array)` — create Warp array from NumPy
- `array.numpy()` — readback to NumPy (implicitly synchronizes for CUDA arrays)
- For CPU arrays, `.numpy()` returns a zero-copy view

## Web references

- https://nvidia.github.io/warp/user_guide/interoperability.html
