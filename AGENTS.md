# AGENTS.md

## Project Overview

Warp is a Python framework for writing high-performance simulation and graphics code. It JIT compiles Python functions to efficient kernel code that runs on CPU or CUDA GPUs.

### Architecture

The native library (`build_lib.py`) statically embeds:

- NVRTC — CUDA runtime compiler for GPU kernels (Linux/Windows only)
- LLVM/Clang — Compiler for CPU kernels (warp-clang library)
- libmathdx — NVIDIA math library (cuBLASDx/cuFFTDx) for tile operations (Linux/Windows only)

### Kernel Cache

JIT compilation artifacts live in a cache directory (see `warp/_src/build.py:init_kernel_cache`). This includes generated `.cpp`/`.cu` files from `codegen.py` and compiled binaries (`.cubin`, `.ptx`, `.o`).

---

## Quick Reference

### Essential Commands

| Task | Command |
|------|---------|
| Run script | `uv run script.py` |
| Run module | `uv run -m module_name` |
| Run single test | `uv run warp/tests/test_<name>.py` |
| Run all tests | `uv run --extra dev -m warp.tests -s autodetect` |
| Format code | `uvx pre-commit run -a` |
| Build native libs | `uv run build_lib.py` |
| Build docs | `uv run --extra docs build_docs.py` |

### Key Patterns

```python
import warp as wp

# Define a parallel kernel
@wp.kernel
def my_kernel(input: wp.array(dtype=wp.float32),
              output: wp.array(dtype=wp.float32)):
    i = wp.tid()  # Get thread index (0 to dim-1)
    output[i] = input[i] * 2.0

# Create arrays
data = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device="cuda")
result = wp.zeros(3, dtype=wp.float32, device="cuda")

# Launch kernel
wp.launch(my_kernel, dim=3, inputs=[data, result])

# Read back to CPU
print(result.numpy())  # [2.0, 4.0, 6.0]
```

### Helper Functions in Kernels

```python
@wp.func
def square(x: float) -> float:
    return x * x

@wp.kernel
def compute(arr: wp.array(dtype=wp.float32)):
    i = wp.tid()
    arr[i] = square(arr[i])  # Can call @wp.func from @wp.kernel
```

### Automatic Differentiation

```python
tape = wp.Tape()
with tape:
    wp.launch(forward_kernel, dim=n, inputs=[x, loss])

tape.backward(loss)
gradient = tape.gradients[x]
```

---

## Boundaries

### Always Do

- Use `uv run` for all Python execution (never bare `python`)
- Use `unittest` framework for tests (never pytest)
- Run `uvx pre-commit run -a` before committing
- Specify explicit `dtype` and `device` for arrays
- Use Google-style docstrings
- Import public API as `import warp as wp`

### Ask First

- Changes to `warp/native/` C++/CUDA code (requires rebuild)
- Changes to `warp/_src/builtins.py` (affects code generation, requires doc rebuild)
- Adding new dependencies to `pyproject.toml`
- Modifying CI/CD pipelines (`.github/workflows/`, `.gitlab-ci.yml`)
- Database schema or major architectural changes

### Never Do

- Call `wp.clear_kernel_cache()` outside `if __name__ == "__main__":` blocks
- Use pytest (project uses unittest exclusively)
- Import from `warp._src` in public-facing code or examples
- Modify auto-generated files (`warp/__init__.pyi`)
- Use Python objects (list, dict) inside kernel code
- Return values from kernels (write to output arrays instead)

---

## Core Concepts

### Decorators

| Decorator | Purpose | Constraints |
|-----------|---------|-------------|
| `@wp.kernel` | Parallel GPU/CPU entry point | No return, use `wp.tid()` |
| `@wp.func` | Reusable device function | Callable from kernels only |
| `@wp.struct` | Custom composite type | Fields must be Warp types |

### Data Types

```python
# Scalars
wp.float32, wp.float64, wp.int32, wp.int64, wp.bool

# Vectors
wp.vec2, wp.vec3, wp.vec4  # float32 vectors
wp.vec3d  # float64 vector

# Matrices
wp.mat22, wp.mat33, wp.mat44  # float32 matrices
wp.mat33d  # float64 matrix

# Special
wp.quat      # Quaternion
wp.transform # Rigid transform (quat + vec3)
```

### Array Operations

```python
# Creation
arr = wp.array(numpy_data, dtype=wp.float32, device="cuda")
arr = wp.zeros(n, dtype=wp.float32)
arr = wp.empty(n, dtype=wp.vec3)

# Device transfer
arr_gpu = arr.to("cuda:0")
arr_cpu = arr.to("cpu")

# NumPy interop (implicit sync)
np_data = arr.numpy()
```

---

## Python Execution

- Use `uv run` for all Python execution (matches CI/CD): `uv run script.py` or `uv run -m module_name`
- With extras: `uv run --extra dev -m warp.tests`
- With additional packages: `uv run --with rich script.py`
- Never run bare `python`/`python3`—always use `uv run` or activate `.venv` first.

---

## Testing

IMPORTANT: Warp uses `unittest`, not pytest.

- Rebuild native libraries (only needed after changes to `warp/native/` C++/CUDA code) with `build_lib.py`
- Run all tests (~10-20 min): `uv run --extra dev -m warp.tests -s autodetect`
- Run specific test file (preferred): `uv run warp/tests/test_modules_lite.py`
- New test modules should be added to `default_suite` in `warp/tests/unittest_suites.py`.
- NEVER call `wp.clear_kernel_cache()` outside `if __name__ == "__main__":` blocks—parallel test runners will conflict.

### Test File Template

```python
import unittest
import warp as wp
from warp.tests.unittest_utils import *

class TestMyFeature(unittest.TestCase):
    def test_basic_case(self):
        arr = wp.array([1.0, 2.0], dtype=wp.float32)
        self.assertEqual(arr.shape[0], 2)

if __name__ == "__main__":
    wp.clear_kernel_cache()  # OK here, in main block
    unittest.main(verbosity=2)
```

---

## Code Style

### Formatting

Run `uvx pre-commit run -a` to format code (config in `.pre-commit-config.yaml`).

### Docstrings

Follow Google-style docstrings with these Warp-specific guidelines:

- Document `__init__` parameters in the class docstring, not the `__init__` method
- Don't repeat default values from signatures—Sphinx autodoc shows them automatically
- Use double backticks for code elements (RST syntax): ``` ``.nvdb`` ```, ``` ``"none"`` ```
- Use Sphinx roles for cross-references: `:class:`warp.array``, `:func:`warp.launch``, `:mod:`warp.render``
- In `builtins.py`, use `Args:` and `Returns:` (Google style), not `:param:` and `:returns:` (RST style)

---

## Key File Locations

### Core Implementation

| File | Purpose | Lines |
|------|---------|-------|
| `warp/_src/context.py` | Runtime, device management, `wp.launch()` | ~8000 |
| `warp/_src/codegen.py` | Kernel code generation | ~4500 |
| `warp/_src/types.py` | Type system, `wp.array` | ~6000 |
| `warp/_src/builtins.py` | Built-in kernel functions | ~11000 |
| `warp/_src/autograd.py` | Gradient computation | ~1200 |

### Native Code

| File | Purpose |
|------|---------|
| `warp/native/warp.h` | Core C++ declarations |
| `warp/native/warp.cpp` | CPU implementation |
| `warp/native/warp.cu` | CUDA implementation |
| `warp/native/builtins.h` | Built-in function declarations |
| `warp/native/vec.h` | Vector math |
| `warp/native/mat.h` | Matrix math |
| `warp/native/tile.h` | Tile operations |

### Tests & Examples

| Location | Purpose |
|----------|---------|
| `warp/tests/unittest_utils.py` | Test utilities and helpers |
| `warp/tests/unittest_suites.py` | Test suite definitions |
| `warp/examples/core/` | Core concept examples |
| `warp/examples/fem/` | Finite element examples |
| `warp/examples/optim/` | Optimization examples |

---

## Domain Glossary

| Term | Meaning |
|------|---------|
| **Kernel** | Parallel function that runs on GPU/CPU, one thread per element |
| **Thread ID** | `wp.tid()` returns index 0 to dim-1 for current thread |
| **Device** | Execution target: `"cpu"`, `"cuda"`, `"cuda:0"`, `"cuda:1"` |
| **Tape** | Records operations for automatic differentiation |
| **Adjoint** | Gradient with respect to a kernel's inputs |
| **Tile** | Block of data processed cooperatively by thread group |
| **HashGrid** | Spatial data structure for O(1) neighbor queries |
| **BVH** | Bounding Volume Hierarchy for mesh queries |
| **FEM** | Finite Element Method module for PDE solving |
| **Graph Capture** | Record kernel launches for replay (reduces overhead) |

---

## CHANGELOG.md

- Use imperative present tense ("Add X", not "Added X" or "This adds X").
- Include issue refs: `([GH-XXX](https://github.com/NVIDIA/warp/issues/XXX))`.
- Avoid internal implementation details users wouldn't understand.

---

## Commit Messages

Use imperative mood ("Fix X", not "Fixed X"), ~50 char subject, reference issues as `(GH-XXX)`. Body explains *why*, not what.

---

## Documentation

- Build docs with `uv run --extra docs build_docs.py` (not `make`/`sphinx-build`).
- Use doctest for code examples where practical.

---

## Codebase Internals

- `warp/_src/` contains internal implementation, re-exported through `warp/__init__.py`. Public-facing code should import from `warp`, not `warp._src`. Internal code should import directly from `warp/_src/` modules.
- Native bindings use ctypes; function signatures are registered in `Runtime.__init__` in `warp/_src/context.py`.
- `warp/_src/builtins.py` defines kernel-callable functions. After modifying, run `build_docs.py` to regenerate `warp/__init__.pyi`.

---

## CI/CD and GitHub

Dual pipelines exist for GitLab (`.gitlab-ci.yml`) and GitHub (`.github/workflows/`)—changes may need updating in both. Follow templates in `.github/` for issues and PRs.

---

## Common Pitfalls

### Kernel Errors

```python
# WRONG: Cannot return from kernels
@wp.kernel
def bad_kernel(arr: wp.array(dtype=float)):
    return arr[0]  # Error!

# CORRECT: Write to output array
@wp.kernel
def good_kernel(input: wp.array(dtype=float), output: wp.array(dtype=float)):
    i = wp.tid()
    output[i] = input[i] * 2.0
```

### Python Objects in Kernels

```python
# WRONG: Python list not allowed
@wp.kernel
def bad_kernel():
    my_list = [1, 2, 3]  # Error!

# CORRECT: Use Warp arrays passed as arguments
@wp.kernel
def good_kernel(data: wp.array(dtype=wp.int32)):
    i = wp.tid()
    val = data[i]  # OK
```

### Cache Clearing

```python
# WRONG: At module level
wp.clear_kernel_cache()
class TestFoo(unittest.TestCase):
    ...

# CORRECT: In main block only
class TestFoo(unittest.TestCase):
    ...

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main()
```
