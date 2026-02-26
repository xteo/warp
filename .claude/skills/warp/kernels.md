# Writing Warp Kernels

## What you should learn here

This is the practical layer used by all examples:

- kernel signatures and type rules
- launch configuration
- execution options and optimization helpers
- differentiable kernels
- cache and graph capture patterns

## Function signatures

Kernel arguments must be typed.

```python
@wp.kernel
def update_positions(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    dt: float,
):
    i = wp.tid()
    positions[i] = positions[i] + velocities[i] * dt
```

Kernel arguments support:

- arrays and structs
- scalar primitives
- integer/float constants
- optional parameters with default Python defaults used in kernel call dispatch

## Launch options

Basic launch:

```python
wp.launch(
    kernel=update_positions,
    dim=len(positions),
    inputs=[positions, velocities, 1.0 / 60.0],
    device="cuda:0",
)
```

You can also provide explicit outputs list for readability.

```python
wp.launch(
    kernel=compute_loss,
    dim=n,
    inputs=[positions, velocities],
    outputs=[loss],
    device="cuda",
)
```

## Multi-dimensional kernels

```python
@wp.kernel
def apply_field(field: wp.array3d(dtype=float), scale: float):
    i, j, k = wp.tid()
    field[i, j, k] *= scale

wp.launch(apply_field, dim=field.shape, inputs=[field, 0.5])
```

## Tile launch path

For tile-based kernels use `wp.launch_tiled`.

```python
wp.launch_tiled(
    kernel=tiled_kernel,
    dim=[num_rows],
    inputs=[A, B],
    block_dim=64,
)
```

Tile launch is used by high-performance kernels in `warp/examples/tile/example_tile_mlp.py`.

## Conditional logic and control flow

Standard kernel control flow applies:

```python
if i < n and velocity > 0.0:
    vel[i] = wp.max(vel[i], 0.0)
```

## Differentiation in kernels

A differentiable workflow uses `requires_grad` arrays and a tape:

```python
x = wp.array([0.0, 1.0], dtype=float, requires_grad=True)
y = wp.zeros(1, dtype=float, requires_grad=True)

@wp.kernel
def loss_fn(x: wp.array(dtype=float), y: wp.array(dtype=float)):
    i = wp.tid()
    if i == 0:
        wp.atomic_add(y, 0, x[0] * x[0] + x[1] * x[1])

with wp.Tape() as tape:
    wp.launch(loss_fn, dim=2, inputs=[x], outputs=[y])

tape.backward(loss=y)
print(x.grad.numpy())  # Access gradients via .grad attribute
```

### Inputs vs outputs in `wp.launch()`

The `inputs` list contains read-only arguments; `outputs` contains written arrays.
This separation matters for autodiff: Warp uses it to determine which arrays need
gradient tracking and adjoint computation during `tape.backward()`.

### Differentiation constraints

- Do NOT overwrite intermediate arrays between tape-recorded launches
- `*=` and `/=` operators are NOT supported in backward pass
- Dynamic loop trip counts are not replayed during backward
- Vector component writes must be single-assignment (no read-modify-write per component)
- Use `wp.config.verify_autograd_array_access = True` to detect problematic overwrites

### Custom gradient functions

```python
@wp.func
def my_func(x: float) -> float:
    return wp.sin(x)

@wp.func_grad(my_func)
def my_func_grad(x: float, adj_ret: float):
    wp.adjoint[x] += wp.cos(x) * adj_ret

@wp.func_replay(my_func)
def my_func_replay(x: float) -> float:
    return wp.sin(x)  # custom forward replay during backward
```

### Jacobian and gradient verification

```python
import warp.autograd

# Full Jacobian computation
J = warp.autograd.jacobian(tape, loss, param)

# Finite-difference Jacobian for comparison
J_fd = warp.autograd.jacobian_fd(func, param)

# Verify autodiff matches finite differences
warp.autograd.gradcheck(tape, loss, param)
```

### Tape visualization

```python
with wp.Tape() as tape:
    wp.launch(kernel, dim=n, inputs=[x], outputs=[y])

tape.visualize()  # Outputs GraphViz dot format showing kernel launches and array deps
```

For optimized differentiable training workflows see:

- `warp/examples/optim/example_fluid_checkpoint.py`
- `warp/examples/tile/example_tile_mlp.py`
- `docs/user_guide/differentiability.rst`

## CUDA graph capture

Record and replay kernel sequences to reduce launch overhead:

```python
with wp.ScopedCapture() as capture:
    wp.launch(kernel_a, dim=n, inputs=[a], device="cuda")
    wp.launch(kernel_b, dim=n, inputs=[b], device="cuda")

# Replay the recorded graph
wp.capture_launch(capture.graph)
```

See `warp/examples/core/example_graph_capture.py` for a complete example.

## Caching and launch reuse

- `wp.clear_kernel_cache()` clears compiled module artifacts.
- Use compile-friendly, stable signatures to avoid recompilation churn.
- `Launch` objects can reduce launch overhead.

## Object and parameter lifecycle

Any spatial primitive must stay alive on the Python side while kernels use its id.

- Keep `wp.Mesh`, `wp.Bvh`, `wp.HashGrid`, `wp.Volume` references alive.
- Update mesh data with `Mesh.refit()` after vertex movement.

## Useful local examples

- `warp/examples/core/example_mesh.py` for deformed meshes + refit.
- `warp/examples/core/example_sample_mesh.py` for multi-kernel mesh workflow.
- `warp/examples/core/example_marching_cubes.py` for `wp.MarchingCubes`.
- `warp/examples/distributed/example_jacobi_mpi.py` for device-scope workflows.

## API references

- https://nvidia.github.io/warp/user_guide/runtime.html#kernels
- https://nvidia.github.io/warp/modules/runtime.html#kernels
- https://nvidia.github.io/warp/profiling.html
