# GitHub Copilot Instructions for NVIDIA Warp

## What is Warp?

Warp is a Python framework that JIT-compiles code to run on CUDA GPUs or CPUs.
It's designed for high-performance simulation, graphics, and differentiable programming.

## Key Decorators

- `@wp.kernel` - Parallel code entry point (like CUDA `__global__`)
- `@wp.func` - Device function callable from kernels (like CUDA `__device__`)
- `@wp.struct` - Custom data structures usable in kernels

## Core Patterns

### Kernel Definition

```python
import warp as wp

@wp.kernel
def compute(input: wp.array(dtype=wp.float32),
            output: wp.array(dtype=wp.float32)):
    i = wp.tid()  # Get thread index
    output[i] = input[i] * 2.0
```

### Kernel Launch

```python
wp.launch(kernel=compute, dim=n, inputs=[input_arr, output_arr], device="cuda")
```

### Array Creation

```python
# From NumPy
arr = wp.array(np_array, dtype=wp.float32, device="cuda")

# Empty allocation
arr = wp.zeros(n, dtype=wp.float32, device="cuda")
```

### Gradient Computation

```python
tape = wp.Tape()
with tape:
    wp.launch(forward_kernel, dim=n, inputs=[...])

tape.backward(loss)
gradients = tape.gradients[input_array]
```

## Testing

- **Framework**: unittest (NOT pytest)
- **Run single test**: `uv run warp/tests/test_<module>.py`
- **Run all tests**: `uv run --extra dev -m warp.tests -s autodetect`
- **New tests**: Add to `warp/tests/`, register in `unittest_suites.py`

## Python Execution

Always use `uv run` for Python execution:

```bash
uv run script.py
uv run -m module_name
uv run --extra dev -m warp.tests
```

Never use bare `python` or `python3`.

## Code Style

- **Docstrings**: Google-style
- **Formatting**: Run `uvx pre-commit run -a`
- **Imports**:
  - Public code: `import warp as wp`
  - Internal code: `from warp._src.module import X`

## Key Constraints

- Kernels cannot return values - write to output arrays
- Kernels cannot use Python objects (list, dict, etc.)
- Use `wp.tid()` for thread index, not Python range
- Never call `wp.clear_kernel_cache()` in test files (only in `if __name__ == "__main__":`)

## Project Structure

- `warp/` - Main Python package
- `warp/_src/` - Internal implementation
- `warp/native/` - C++/CUDA native code
- `warp/tests/` - Test suite (unittest)
- `warp/examples/` - Example code

## Building

- **Build native libs**: `uv run build_lib.py`
- **Build docs**: `uv run --extra docs build_docs.py`

## Common Warp Types

- Scalars: `wp.float32`, `wp.float64`, `wp.int32`, `wp.int64`
- Vectors: `wp.vec2`, `wp.vec3`, `wp.vec4`
- Matrices: `wp.mat22`, `wp.mat33`, `wp.mat44`
- Quaternions: `wp.quat`
- Transforms: `wp.transform`
