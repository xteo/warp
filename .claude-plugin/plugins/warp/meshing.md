# Meshing and Spatial Geometry in Warp

Warp includes geometry-focused primitives to avoid rewriting BVH/hash/data structure code.

## What this page covers

- `wp.Mesh` creation and mesh updates
- Triangle sampling and CDF workflows
- `wp.MarchingCubes`
- Spatial primitive queries (`HashGrid`, `Bvh`, `Volume`, `Texture`)

## Meshes for collision and querying

A mesh is created from points + indices.

```python
mesh = wp.Mesh(
    points=wp.array(verts, dtype=wp.vec3),
    indices=wp.array(faces, dtype=wp.int32),
)
```

Query points in kernels with `mesh_query_point_sign_normal` or point/ ray queries.

```python
@wp.kernel
def collide(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    mesh: wp.uint64,
):
    i = wp.tid()
    p = positions[i]
    q = wp.mesh_query_point_sign_normal(mesh, p, 2.0)
```

If you move mesh vertices, call `mesh.refit()`.

## Meshing example files

- `warp/examples/core/example_mesh.py`
  - Demonstrates deformed mesh + `mesh.refit()` + collision sampling.
- `warp/examples/core/example_mesh_intersect.py`
  - Additional collision geometry patterns.

## Sampling points on mesh surface

`example_sample_mesh.py` is a good starter for mesh-based sampling pipelines:

- `compute_tri_areas` kernel accumulates triangle areas.
- `compute_probability_distribution` builds PDF.
- `accumulate_cdf` builds cumulative distribution.
- `sample_mesh` selects triangles and barycentric samples via `wp.lower_bound`.

This is directly reusable for particle initialization and surface synthesis tasks.

## Marching cubes

`wp.MarchingCubes` extracts iso-surfaces from dense 3D scalar fields.

```python
mc = wp.MarchingCubes(nx, ny, nz, max_verts, max_tris)
...
mc.surface(field, 0.0)
```

- Input: scalar field `wp.array3d`.
- Output arrays accessed via `mc.verts`, `mc.indices`.

See `warp/examples/core/example_marching_cubes.py` for end-to-end flow.

## Hash grids and nearest-neighbor workflows

Use `wp.HashGrid` for particle search tasks.

```python
grid = wp.HashGrid(dim_x=64, dim_y=64, dim_z=64, device="cuda")
grid.build(points=positions, radius=support_radius)
```

In kernels call `wp.hash_grid_query` and `wp.hash_grid_query_next`.

## BVH and ray/overlap queries

Use `wp.Bvh` for intersection-heavy workloads.

- `wp.bvh_query_ray`
- `wp.bvh_query_next`
- `wp.bvh_query_aabb`

Useful for ray tracing, collision, and broad-phase intersection.

## Volumes and textures

`warp/examples/core/example_marching_cubes.py` uses SDF fields.

- `wp.Volume` for NanoVDB/implicit fields.
- `wp.Texture2D`, `wp.Texture3D` for interpolated reads.

## Pitfall: object lifetime

Keep references to spatial objects alive. Storing only `.id` often causes crashes.

This is explicitly covered in `docs/user_guide/runtime.rst` under Object Lifetime Pitfall.

## Where to read in-repo

- `docs/user_guide/runtime.rst` (Spatial Computing Primitives)
- `docs/domain_modules/render.rst`
- `docs/user_guide/runtime.rst` (Meshes, HashGrid, BVH, Volumes)
- `warp/examples/core/example_sample_mesh.py`
- `warp/examples/core/example_marching_cubes.py`

## Web references

- https://nvidia.github.io/warp/user_guide/runtime.html#spatial-computing-primitives
- https://nvidia.github.io/warp/api_reference/_generated/warp.Mesh.html
- https://nvidia.github.io/warp/modules/runtime.html
