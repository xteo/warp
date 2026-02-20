# API Reference: Runtime Scope

## What this section is

Python-scope APIs used for memory, launch orchestration, devices, and execution control.

## High-value entry points

- `wp.init()`
- `wp.launch()`
- `wp.launch_tiled()`
- `wp.capture_begin()`, `wp.capture_end()`, `wp.capture_launch()`
- `wp.synchronize()`, `wp.synchronize_device()`
- `wp.Stream`, `wp.Event`, `wp.ScopedCapture`, `wp.TimedTimer`

## Device controls

- `wp.get_devices()`
- `wp.get_cuda_device_count()`
- `wp.get_cuda_devices()`
- `wp.get_device()`
- `wp.set_device()`
- `wp.ScopedDevice`
- `wp.get_preferred_device()`

## Array constructors

- `wp.zeros`, `wp.ones`, `wp.full`, `wp.empty`
- `wp.array`, `wp.array2d`, `wp.array3d`, `wp.array4d`
- `wp.indexedarray*` for gather-like subsets

Type conversion helpers:

- `wp.from_numpy`, `wp.from_torch`, `wp.from_jax`, etc.
- `wp.to_numpy`, `wp.to_torch`, `wp.to_jax`.

## Launch execution

`wp.launch()` and `wp.launch_tiled()` accept:

- `kernel`
- `dim`
- `inputs`
- `outputs`
- `device`
- `block_dim` (for tiled launches)
- profiling / debug flags via config and context

## Stream, graph, and timing

Useful for throughput workloads:

- Record kernel bursts with CUDA graph capture when repeated.
- Use `wp.capture_if`, `wp.capture_while` for control-flow graphs.
- Add timers via `wp.ScopedTimer` and `wp.timing_*` APIs.

## Spatial primitives in runtime scope

- `wp.Mesh`, `wp.HashGrid`, `wp.Bvh`, `wp.Volume`, `wp.Texture*`

Objects are device-aware and often passed to kernels through ids.

## Autograd interaction

- Use `wp.Tape` in Python scope for differentiation.
- Pair with kernel launches and `requires_grad=True` tensors.
- Review gradient buffers and zero/reset semantics per optimization loop.

## Local code pointers

- `warp/_src/context.py`
- `warp/_src/types.py`
- `warp/_src/utils.py`
- `docs/user_guide/runtime.rst`

## Web references

- https://nvidia.github.io/warp/user_guide/runtime.html
- https://nvidia.github.io/warp/modules/runtime.html
