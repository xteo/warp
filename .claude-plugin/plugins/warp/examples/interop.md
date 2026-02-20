# Interop Examples

Interoperability lets Warp sit inside ML frameworks and array ecosystems.

## JAX

```sh
uv run python -m warp.examples.interop.example_jax_kernel
```

This demonstrates:

- wrapping kernels with `jax_kernel`
- returning multiple outputs
- launch dimension defaults and overrides
- scalar launch parameters as static args

Relevant file: `warp/examples/interop/example_jax_kernel.py`

## PyTorch and conversion workflows

See docs and core examples for direct `from_torch` and `to_torch` flows.

- Convert tensors to warp arrays and back.
- Use `requires_grad=True` tensors with `wp.Tape`.
- Avoid excessive re-conversion each step in training loops.

## NumPy/CuPy/PyTorch one-liners

For fast prototyping:

- Many examples accept NumPy arrays directly when launched on CPU.
- On CUDA, arrays must align with selected device.

## Where to read in docs

- https://nvidia.github.io/warp/user_guide/interoperability.html
- https://nvidia.github.io/warp/user_guide/interoperability.html
