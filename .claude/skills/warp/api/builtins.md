# API Reference: Kernel Built-Ins

## What belongs here

All functions/types available inside Warp kernels and selected helper functions used both in kernels and Python scope.

## Main categories

- Math: `wp.sin`, `wp.cos`, `wp.sqrt`, `wp.exp`, `wp.pow`, `wp.clamp`, `wp.max`, `wp.min`, `wp.abs`, etc.
- Vector ops: `wp.dot()`, `wp.cross()`, `wp.normalize()`, `wp.length()`, `wp.length_sq()`
- Matrix ops: `wp.mul()`, `wp.transpose()`, `wp.determinant()`, `wp.inverse()`
- Geometry queries: `wp.mesh_query_*`, `wp.hash_grid_query*`, `wp.bvh_query*`, `wp.volume_sample_*`
- Atomics: `wp.atomic_add()`, `wp.atomic_sub()`, `wp.atomic_max()`, `wp.atomic_min()`
- Random: `wp.rand_init()`, `wp.randf()`, `wp.randi()`, `wp.sample_unit_sphere()`
- Tile APIs: `wp.tile_load()`, `wp.tile_store()`, `wp.tile_matmul()`, `wp.tile_sum()`, `wp.tile_fft()`, `wp.tile_ifft()`, `wp.tile_cholesky()`, `wp.tile_lower_solve()`, `wp.tile_upper_solve()`, `wp.tile_scan_inclusive()`

## Types and constructors you will use daily

- Scalars: `wp.int32`, `wp.float32`, `wp.float16`, etc.
- Vectors: `wp.vec2`, `wp.vec3`, `wp.vec4`
- Matrices: `wp.mat22`, `wp.mat33`, `wp.mat44`
- Quaternions: `wp.quat`
- Spatial types: `wp.spatial_vector`, `wp.spatial_matrix`

## Quaternion and transform helpers

- `wp.quat(i, j, k, w)` — quaternion constructor (layout: i, j, k, w)
- `wp.quat_from_axis_angle(axis, angle)` — create from axis-angle
- `wp.quat_from_matrix(mat)` — create from rotation matrix
- `wp.quat_to_matrix(q)` — convert to 3x3 rotation matrix
- `wp.quat_rotate(q, v)` — rotate vector by quaternion
- `wp.quat_inverse(q)` — conjugate/inverse
- `wp.quat_slerp(a, b, t)` — spherical linear interpolation
- `wp.transform_point(xform, p)` — apply rigid transform to point
- `wp.transform_vector(xform, v)` — apply rotation only

## Kernel-only concepts

- `wp.tid()` thread index (1D/2D/3D/4D unpacking)
- `wp.atomic_add()`, `wp.atomic_sub()`, `wp.atomic_max()`, `wp.atomic_min()`
- `wp.mesh_query_ray()`, `wp.mesh_query_point_sign_normal()`, `wp.mesh_query_aabb()`
- `wp.hash_grid_query()`, `wp.hash_grid_query_next()`
- `wp.bvh_query_ray()`, `wp.bvh_query_aabb()`, `wp.bvh_query_next()`
- `wp.volume_sample_f()`, `wp.volume_lookup_f()`, `wp.volume_store_f()`
- `wp.printf()` — C-style formatted printing in kernels
- `wp.breakpoint()` — trigger debugger breakpoint (debug mode only)

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
