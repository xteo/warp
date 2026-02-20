# Core Examples

These are the best first examples for learning Warp patterns.

## Run command pattern

Most examples run with:

```sh
uv run python -m warp.examples.core.<example_name>
```

For example:

```sh
uv run python -m warp.examples.core.example_mesh
```

## `example_mesh.py`

What it demonstrates:

- Kernel-based particle simulation loop.
- Deforming mesh with kernel.
- `wp.Mesh.query` + collision response in kernel.
- `mesh.refit()` after topology-preserving vertex changes.

Relevant file: `warp/examples/core/example_mesh.py`

## `example_sample_mesh.py`

What it demonstrates:

- Triangle-area-based CDF sampling over mesh surface.
- Kernel pipeline: triangle areas, probabilities, prefix sum.
- `wp.lower_bound` plus sampling in kernel space.

Relevant file: `warp/examples/core/example_sample_mesh.py`

## `example_marching_cubes.py`

What it demonstrates:

- Signed distance field kernel.
- `wp.MarchingCubes.surface` extraction.
- `OpenGL/USD` style rendering integration pattern.

Relevant file: `warp/examples/core/example_marching_cubes.py`

## `example_fluid.py`

What it demonstrates:

- Multi-kernel time step loop.
- Dimensioned arrays in CUDA kernels.
- Pressure-solve pipeline style for fluid constraints.

Relevant file: `warp/examples/core/example_fluid.py`

## `example_torch.py`

Shows direct integration with torch launch and shared workflows.

Relevant file: `warp/examples/core/example_torch.py`

## `example_nvdb.py` and `example_raycast.py`

Useful for volume and ray tracing workflow patterns.

Relevant files:

- `warp/examples/core/example_nvdb.py`
- `warp/examples/core/example_raycast.py`

## Recommended extension path

1. Run and read `example_sample_mesh.py`.
2. Modify kernel and print small outputs.
3. Move to `example_marching_cubes.py` to add mesh extraction.
4. Repeat with differentiable training flow from `optim` examples.

## Docs links

- https://nvidia.github.io/warp/user_guide/runtime.html
- https://nvidia.github.io/warp/user_guide/basics.html
