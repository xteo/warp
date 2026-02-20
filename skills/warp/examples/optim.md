# Optimization Examples

Use these for differentiable simulation and gradient-driven workflows.

## Run

```sh
uv run python -m warp.examples.optim.example_fluid_checkpoint
uv run python -m warp.examples.optim.example_diffray
uv run python -m warp.examples.optim.example_particle_repulsion
```

## `example_fluid_checkpoint.py`

Demonstrates:

- checkpointed rollout loop
- Warp Tape capture and backward pass
- optimizer integration with `warp.optim`
- stable differentiable solver design

Relevant file: `warp/examples/optim/example_fluid_checkpoint.py`

## `example_diffray.py`

A differentiable ray simulation pattern (domain-specific example).

## `example_particle_repulsion.py`

Particle-based objective and regularization pattern.

## Pairing with other examples

- Use `tile_mlp` to experiment with mixed optimization + custom kernels.
- Combine with `core/example_fluid.py` if you need stable baseline constraints.

## API connections

- `wp.Tape`
- `wp.optim` module
- `Launch` + graph capture for batched learning loops

Relevant docs:

- https://nvidia.github.io/warp/modules/warp_optim.html
- https://nvidia.github.io/warp/user_guide/differentiability.html
