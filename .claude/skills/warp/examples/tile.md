# Tile Examples

All 10 tile examples have been verified to build and run successfully.

Tile APIs allow block-structured data movement and compute, especially for dense operations.
Tile operations require CUDA 12.6.3+ and leverage Tensor Cores and shared memory.

## When to use tiles vs regular SIMT kernels

**Use tile operations for:**
- Dense linear algebra (matmul, Cholesky, etc.)
- Reductions (tile_sum is ~52x faster than atomic operations)
- FFT operations
- Operations that benefit from shared memory and Tensor Cores

**Use regular SIMT kernels for:**
- Irregular access patterns
- Sparse operations
- Heavy conditional logic
- When CUDA 12.6.3+ is not available

## Key tile APIs

- `wp.tile_load(array, offset, shape)` — load data into tile
- `wp.tile_store(array, offset, tile)` — store tile back
- `wp.tile_matmul(a, b, out)` — matrix multiplication
- `wp.tile_sum(tile)` — reduction sum
- `wp.tile_fft(tile)` / `wp.tile_ifft(tile)` — FFT (backed by cuFFTDx)
- `wp.tile_cholesky(tile)` — Cholesky factorization
- `wp.tile_lower_solve(L, b)` / `wp.tile_upper_solve(U, b)` — triangular solves
- `wp.tile_scan_inclusive(tile)` — parallel prefix sum / stream compaction

## Run

```sh
uv run python -m warp.examples.tile.example_tile_mlp
uv run python -m warp.examples.tile.example_tile_matmul
uv run python -m warp.examples.tile.example_tile_nbody
```

## `example_tile_matmul.py`

- Basic tile matrix multiplication with `TILE_M/N/K` constants.
- Uses `wp.tile_load()`, `wp.tile_matmul()`, `wp.tile_store()`.
- Launched with `wp.launch_tiled()`.

Relevant file: `warp/examples/tile/example_tile_matmul.py`

## `example_tile_mlp.py`

- Neural network training using tile APIs.
- Fused matmul + bias + activation in shared memory.
- Captures training graphs with `wp.ScopedCapture()`.

Relevant file: `warp/examples/tile/example_tile_mlp.py`

## `example_tile_fft.py`

- FFT operations using `wp.tile_fft()` and `wp.tile_ifft()`.
- Backed by cuFFTDx library.

Relevant file: `warp/examples/tile/example_tile_fft.py`

## `example_tile_cholesky.py`

- Linear system solving with `wp.tile_cholesky()`.
- Forward/backward substitution with `wp.tile_lower_solve()` / `wp.tile_upper_solve()`.

Relevant file: `warp/examples/tile/example_tile_cholesky.py`

## `example_tile_nbody.py`

- N-body simulation using tile-based force accumulation.

Relevant file: `warp/examples/tile/example_tile_nbody.py`

## `example_tile_stream_compaction.py`

- Parallel prefix sum using `wp.tile_scan_inclusive()`.
- Stream compaction pattern.

Relevant file: `warp/examples/tile/example_tile_stream_compaction.py`

## `example_tile_filtering.py`

- Signal smoothing using tile-based filter operations.
- Output: matplotlib plot of filtered vs original signal.

Relevant file: `warp/examples/tile/example_tile_filtering.py`

## `example_tile_convolution.py`

- Tile-based convolution operations.
- Compute-only output (verification printed to console).

Relevant file: `warp/examples/tile/example_tile_convolution.py`

## `example_tile_block_cholesky.py`

- Block Cholesky decomposition for larger systems.
- Compute-only output (takes ~45s due to compilation).

Relevant file: `warp/examples/tile/example_tile_block_cholesky.py`

## `example_tile_mcgp.py`

- Monte Carlo Geometry Processing using tile operations.
- Produces an animated GIF of the simulation.
- Longest-running tile example (~97s due to many iterations).
- Dependencies: matplotlib, Pillow

Relevant file: `warp/examples/tile/example_tile_mcgp.py`

## Output summary

| Example | Output | Time |
|---------|--------|------|
| `example_tile_mlp` | JPEG image (neural reconstruction) | ~30s |
| `example_tile_nbody` | matplotlib (particle positions) | ~1s |
| `example_tile_filtering` | matplotlib (filtered signal) | ~4s |
| `example_tile_mcgp` | animated GIF | ~97s |
| `example_tile_fft` | console only | ~4s |
| `example_tile_matmul` | console only | ~9s |
| `example_tile_convolution` | console only | ~4s |
| `example_tile_cholesky` | console only | ~3s |
| `example_tile_block_cholesky` | console only | ~45s |
| `example_tile_stream_compaction` | console only | ~1s |

## Mapping to docs

- https://nvidia.github.io/warp/user_guide/tiles.html
- https://nvidia.github.io/warp/domain_modules/sparse.html

## Practical advice

- Keep `block_dim` aligned with tile shape assumptions.
- Default `block_dim` is 256; set per-launch with `wp.launch_tiled(..., block_dim=TILE_THREADS)`.
- Configure per-module with `wp.set_module_options({"block_dim": 128})`.
- Set `wp.config.enable_mathdx_gemm = False` to speed up compilation during development (disables cuBLASDx, falls back to scalar GEMM).
- Prefer smaller tiles for shared-memory-constrained kernels.
- Validate output shape mathematically with small random data before scaling.
