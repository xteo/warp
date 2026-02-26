# Optimization Examples

All 3 optimization examples demonstrate differentiable simulation and gradient-driven
workflows using `wp.Tape` for automatic differentiation. All verified to build and run.

## Run

```sh
uv run --with matplotlib --with Pillow python -m warp.examples.optim.example_diffray
uv run python -m warp.examples.optim.example_fluid_checkpoint
uv run --with matplotlib python -m warp.examples.optim.example_particle_repulsion
```

## `example_diffray.py`

Differentiable ray caster that optimizes mesh vertex positions to match a target image.

- Renders the Stanford Bunny mesh with normal-mapped coloring
- Uses `wp.Tape` to compute gradients of image loss w.r.t. vertex positions
- Produces target image, optimized final image, and training animation GIF
- Output: `example_diffray_target_image.png`, `example_diffray_final_image.png`, `example_diffray_animation.gif`
- Dependencies: matplotlib, Pillow
- This is the best example for learning differentiable rendering pipelines

Relevant file: `warp/examples/optim/example_diffray.py`

## `example_fluid_checkpoint.py`

Demonstrates checkpointed differentiable fluid simulation:

- Checkpointed rollout loop (saves memory for long time horizons)
- `wp.Tape` capture and backward pass through simulation steps
- Optimizer integration with `warp.optim` (Adam optimizer)
- Prints loss values to console — no visual output
- This is the best example for learning differentiable physics optimization

Relevant file: `warp/examples/optim/example_fluid_checkpoint.py`

## `example_particle_repulsion.py`

Gradient-based particle arrangement that self-organizes into the NVIDIA logo:

- Particles start random and converge to target shape via repulsion + attraction
- Uses `wp.Tape` for gradient computation
- Output: `example_particle_repulsion_result.png` showing final particle positions
- Dependency: matplotlib

Relevant file: `warp/examples/optim/example_particle_repulsion.py`

## Pairing with other examples

- Use `tile/example_tile_mlp.py` to experiment with neural network training using tile APIs
- Combine with `core/example_fluid.py` for understanding the forward simulation before optimizing
- See `fem/example_elastic_shape_optimization.py` and `fem/example_darcy_ls_optimization.py` for FEM-based optimization

## Key API connections

- `wp.Tape` — records kernel launches for automatic differentiation
- `wp.optim` module — optimizers (Adam, SGD)
- `requires_grad=True` on arrays to track gradients
- `tape.backward(loss=loss_array)` to compute gradients
- `wp.ScopedCapture()` + graph capture for batched learning loops

## Web references

- https://nvidia.github.io/warp/modules/warp_optim.html
- https://nvidia.github.io/warp/user_guide/differentiability.html
