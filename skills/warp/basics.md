# Warp Basics

## What this file covers

If you can complete this page, you can read and modify most Warp examples.

## Core model

Warp is kernel-first:

- Kernels are Python functions marked with `@wp.kernel`.
- Kernels compile to C++/CUDA on first launch.
- Kernel execution uses `wp.launch` over a threaded grid.

## First working example

```python
import warp as wp

@wp.kernel
def add(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    i = wp.tid()
    c[i] = a[i] + b[i]

n = 8
a = wp.arange(0, n, dtype=float)
b = wp.ones(n, dtype=float)
c = wp.zeros(n, dtype=float)

wp.launch(kernel=add, dim=n, inputs=[a, b, c], outputs=[c])
print(c.numpy())
```

## Initialization and devices

Warp initializes on first call and prints device info.

```python
import warp as wp

wp.init()  # optional explicit init
print(wp.get_devices())
```

Default device behavior:

- Uses `cuda:0` when available, else `cpu`.
- Target other devices with `device` argument per call.
- Use `wp.ScopedDevice("cuda:1")` for temporary overrides.

Relevant docs and local files:

- `docs/user_guide/basics.rst`
- `docs/user_guide/devices.rst`
- `skills/warp/installation.md`

## Arrays and transfer

- Arrays are typed allocations: scalar/struct/vector/matrix values.
- You can allocate with `wp.zeros`, `wp.ones`, `wp.full`, `wp.empty`, `wp.from_numpy`.
- `array.numpy()` is the canonical readback path.

```python
import numpy as np
import warp as wp

host = np.arange(16, dtype=np.float32)
a = wp.array(host, dtype=float, device="cuda")
view = a.numpy()  # sync to CPU
print(view.shape)
```

## Thread indexing

- 1D indexing: `tid = wp.tid()`
- 2D indexing: `i, j = wp.tid()`
- 3D/4D follows tuple unpacking from `wp.tid()`.

Grid launch example:

```python
@wp.kernel
def image_kernel(img: wp.array2d(dtype=wp.vec3)):
    i, j = wp.tid()
    img[i, j] = wp.vec3(0.0)

wp.launch(image_kernel, dim=(1024, 1024), inputs=[img])
```

## User functions and structs

`@wp.func` creates reusable functions that can be called by kernels.

```python
@wp.func
def dot2(a: wp.vec2, b: wp.vec2):
    return wp.dot(a, b)
```

`@wp.struct` defines custom fielded types.

```python
@wp.struct
class Particle:
    pos: wp.vec3
    vel: wp.vec3
    active: int
```

## Python scope vs kernel scope

Kernel scope: no arbitrary Python state, only whitelisted runtime/kernel functions.
Python scope: orchestration and integration with other Python libraries.

## Common mistakes to avoid

- Mixing CPU/GPU arrays in the same launch.
- Assigning many elements directly in Python loops on CUDA arrays.
- Expecting implicit type conversion for mismatched types.

## Local examples to read

- `warp/examples/core/example_fluid.py` shows kernel chaining and multiple launches.
- `warp/examples/core/example_mesh.py` combines mesh build + simulation.
- `warp/examples/core/example_sample_mesh.py` shows typed arrays and launch structure.
- `warp/examples/core/example_marching_cubes.py` shows typed field kernels and `wp.MarchingCubes`.

## Web references

- https://nvidia.github.io/warp/user_guide/basics.html
- https://nvidia.github.io/warp/modules/functions.html
- https://nvidia.github.io/warp/user_guide/runtime.html
