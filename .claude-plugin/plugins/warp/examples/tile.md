# Tile Examples

Tile APIs allow block-structured data movement and compute, especially for dense operations.

## Run

```sh
uv run python -m warp.examples.tile.example_tile_mlp
uv run python -m warp.examples.tile.example_tile_nbody
uv run python -m warp.examples.tile.example_tile_mcgp
```

Note: script names depend on available files in this checkout.

## Core tile example to inspect

`warp/examples/tile/example_tile_mlp.py`

- Uses `wp.tile` types for batched matrix math.
- Uses `wp.tile_load`, `wp.tile_matmul`, `wp.tile_store`.
- Captures training graphs with `wp.capture` for repeatability.

## Other tile examples

- `warp/examples/tile/example_tile_cholesky.py`
- `warp/examples/tile/example_tile_convolution.py`
- `warp/examples/tile/example_tile_fft.py`
- `warp/examples/tile/example_tile_filtering.py`
- `warp/examples/tile/example_tile_block_cholesky.py`

## Mapping to docs

- https://nvidia.github.io/warp/user_guide/tiles.html
- https://nvidia.github.io/warp/domain_modules/sparse.html
- https://nvidia.github.io/warp/user_guide/runtime.html#spatial-computing-primitives

## Practical advice

- Keep `block_dim` aligned with tile shape assumptions.
- Prefer smaller tiles for shared-memory-constrained kernels.
- Validate output shape mathematically with small random data before scaling.
