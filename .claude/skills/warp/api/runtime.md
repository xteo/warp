# API Reference: Runtime Scope

## What this section is

Python-scope APIs used for memory, launch orchestration, devices, and execution control.

## High-value entry points

- `wp.init()`
- `wp.launch()`
- `wp.launch_tiled()`
- `wp.ScopedCapture()` — CUDA graph capture context manager
- `wp.capture_launch(graph)` — replay captured graph
- `wp.synchronize()`, `wp.synchronize_device()`
- `wp.Stream`, `wp.Event`

## Device controls

- `wp.get_devices()` — all available devices
- `wp.get_cuda_device_count()` — number of CUDA devices
- `wp.get_cuda_devices()` — list CUDA devices
- `wp.get_device()` — current default device
- `wp.set_device(device)` — set default device
- `wp.ScopedDevice("cuda:N")` — temporary device context manager
- `wp.get_preferred_device()` — best available device

## Multi-GPU

### Enumerate and scope devices

```python
devices = wp.get_cuda_devices()
for device in devices:
    with wp.ScopedDevice(device):
        a = wp.zeros(n, dtype=float)  # on this device
        wp.launch(kernel, dim=n, inputs=[a])
```

### Cross-device data transfer

```python
a = wp.zeros(n, dtype=float, device="cuda:0")
b = a.to("cuda:1")           # Copy to another device
wp.copy(dest_array, src_array)  # Explicit copy
```

### Peer access (direct GPU-to-GPU memory access)

```python
wp.is_peer_access_supported("cuda:0", "cuda:1")   # Check support
wp.set_peer_access_enabled("cuda:0", "cuda:1", True)  # Enable

# Or use context manager
with wp.ScopedPeerAccess("cuda:0", "cuda:1"):
    # Direct memory access between devices
    wp.launch(kernel, dim=n, inputs=[array_on_gpu0], device="cuda:1")
```

### IPC array sharing (inter-process, Linux only)

```python
# Process A: export
handle = array.ipc_handle()

# Process B: import
array = wp.from_ipc_handle(handle, dtype=float, shape=(n,))
```

## Array constructors

- `wp.zeros`, `wp.ones`, `wp.full`, `wp.empty`
- `wp.array`, `wp.array2d`, `wp.array3d`, `wp.array4d`
- `wp.indexedarray(source, indices)` — zero-copy virtual view
- `wp.fabricarray` — Omniverse Runtime Fabric zero-copy access

Type conversion helpers:

- `wp.from_numpy`, `wp.from_torch`, `wp.from_jax`, etc.
- `wp.to_numpy`, `wp.to_torch`, `wp.to_jax`.
- `wp.from_dlpack`, `wp.to_dlpack` — DLPack protocol

## Launch execution

`wp.launch()` and `wp.launch_tiled()` accept:

- `kernel`
- `dim`
- `inputs` — read-only arguments
- `outputs` — written arguments (matters for autodiff)
- `device`
- `block_dim` (for tiled launches, default 256)
- profiling / debug flags via config and context

## Stream, graph, and timing

Useful for throughput workloads:

- Record kernel bursts with `wp.ScopedCapture()`, replay with `wp.capture_launch(graph)`.
- Use `wp.capture_if`, `wp.capture_while` for control-flow graphs.
- Add timers via `wp.ScopedTimer("name", synchronize=True)`.

## Spatial primitives in runtime scope

- `wp.Mesh`, `wp.HashGrid`, `wp.Bvh`, `wp.Volume`, `wp.Texture*`

Objects are device-aware and often passed to kernels through ids.

## Autograd interaction

- Use `wp.Tape` in Python scope for differentiation.
- Pair with kernel launches and `requires_grad=True` arrays.
- Access gradients via `array.grad`.
- Review gradient buffers and zero/reset semantics per optimization loop.

## Local code pointers

- `warp/_src/context.py`
- `warp/_src/types.py`
- `warp/_src/utils.py`
- `docs/user_guide/runtime.rst`

## Web references

- https://nvidia.github.io/warp/user_guide/runtime.html
- https://nvidia.github.io/warp/modules/runtime.html
