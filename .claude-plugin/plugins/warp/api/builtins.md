# API Reference: Kernel Built-Ins

## What belongs here

All functions/types available inside Warp kernels and selected helper functions used both in kernels and Python scope.

## Main categories

- Math: `sin`, `cos`, `sqrt`, `exp`, `pow`, `clamp`, `max`, `min`, `abs`, etc.
- Vector/matrix ops: dot products, cross products, swizzles, constructors
- Geometry queries and sampling helpers
- Atomics and synchronization primitives
- Random numbers and control functions
- Tile APIs and distributed reductions

## Types and constructors you will use daily

- Scalars: `wp.int32`, `wp.float32`, `wp.float16`, etc.
- Vectors: `wp.vec2`, `wp.vec3`, `wp.vec4`
- Matrices: `wp.mat22`, `wp.mat33`, `wp.mat44`
- Quaternions: `wp.quat`
- Spatial types: `wp.spatial_vector`, `wp.spatial_matrix`

## Kernel-only concepts

- `wp.tid()` thread index (1D/2D/3D/4D unpacking)
- `wp.atomic_add`, `wp.atomic_max`, etc.
- `wp.mesh_query_*`, `wp.hash_grid_query*`, `wp.bvh_query*`
- `wp.volume_store`, `wp.texture_sample`

## Built-in constants

- `wp.inf`, `wp.nan`, `wp.pi`, etc.

## Local examples

- `warp/examples/core/example_fluid.py` for arithmetic and vector math.
- `warp/examples/core/example_mesh.py` for geometry queries.
- `warp/examples/core/example_sample_mesh.py` for random sampling helpers.
- `warp/examples/tile/example_tile_mlp.py` for tile math and reduction.

## External references

- https://nvidia.github.io/warp/modules/functions.html
- https://nvidia.github.io/warp/language_reference/builtins.html
