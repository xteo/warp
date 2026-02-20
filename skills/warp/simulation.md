# Simulation Workflows

Warp targets simulation and optimization workflows across fluids, soft bodies, FEM, and differentiable pipelines.

## What this page covers

- Core simulation examples in `warp/examples/core`
- FEM workflows in `warp/examples/fem`
- Differentiable optimization in `warp/examples/optim`
- Performance tooling and domain modules

## Core simulation examples

### Fluid advection (`warp/examples/core/example_fluid.py`)

This example shows a standard simulation loop with multiple kernels:

- Field initialization
- Divergence computation
- Pressure solve
- Velocity advection
- Pressure apply

Use this structure for your own PDE/CFD pipeline:

1. allocate `wp.array` state
2. run stage kernels in order
3. write outputs

### Mesh interaction (`warp/examples/core/example_mesh.py`)

- Demonstrates `wp.Mesh` collision queries.
- Shows vertex deformation via kernel.
- Calls `mesh.refit()` to keep BVH valid after geometry updates.

## Differentiable simulation

`wp.Tape` enables gradient-based optimization of kernel programs.

```python
with wp.Tape() as tape:
    wp.launch(loss_kernel, dim=n, inputs=[state], outputs=[loss])

tape.backward(loss=loss)
```

Set arrays with `requires_grad=True` and run optimizer updates in Python. For richer examples see:

- `warp/examples/optim/example_fluid_checkpoint.py`
- `warp/examples/tile/example_tile_mlp.py`

These examples show checkpointing, batching, and multi-stage optimizers.

## FEM in Warp

FEM examples are in `warp/examples/fem`.

- `example_diffusion.py` and `example_diffusion_3d.py`
- `example_navier_stokes.py`
- `example_burgers.py`
- `example_elastic_shape_optimization.py`
- `example_mixed_elasticity.py`

The core API page for setup is in `docs/domain_modules/fem.rst`.

FEM workflow pattern:

- Create geometry (`Grid2D`, `Grid3D`, `TriangleMesh`, etc.)
- Define function spaces and domains
- Build forms with test/trial fields
- Assemble and solve with linear solvers

## Sparse and linear algebra

See `docs/domain_modules/sparse.rst` and FEM examples for solver usage such as conjugate gradients, CG-like workflows, and custom operators.

## Distributed simulation

`warp/examples/distributed/example_jacobi_mpi.py` shows:

- MPI-aware rank decomposition
- Explicit stream and event usage
- Multi-device iterative solves

Useful when scaling CPU/GPU workflows across nodes.

## Performance and capture

- Use `wp.ScopedTimer` around step/solve sections.
- Use `wp.capture_begin()` + `capture_launch()` for repeated training loops.
- Use tile APIs for block-level data-parallel operations.

## Where to read next

- `skills/warp/examples/fem.md`
- `skills/warp/kernels.md`
- `docs/user_guide/differentiability.rst`
- `docs/domain_modules/fem.rst`
- `docs/domain_modules/sparse.rst`

## Web references

- https://nvidia.github.io/warp/domain_modules/fem.html
- https://nvidia.github.io/warp/domain_modules/sparse.html
- https://nvidia.github.io/warp/user_guide/differentiability.html
