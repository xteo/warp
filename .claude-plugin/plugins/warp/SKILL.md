---
name: warp
description: NVIDIA Warp onboarding skill - high-performance Python framework for GPU kernels, differentiable simulation, and spatial computing. Use when working with Warp for: (1) Installing and setting up Warp, (2) Writing custom kernels with @wp.kernel, (3) Learning core concepts like arrays, launches, and thread indexing, (4) Working with mesh, volume, and BVH spatial primitives, (5) Building differentiable simulations with wp.Tape, (6) Exploring examples in core, FEM, optimization, and interop domains.
---

# Warp Onboarding Skill

This skill is a practical entry point for learning NVIDIA Warp in this repository.

## What this covers

Warp is a Python framework for high-performance CPU/GPU kernels, differentiable simulation, and spatial computing.
This skill teaches how to set up Warp, write and launch kernels, move data, use core example patterns, and connect to surrounding ecosystems.

## Recommended reading order

1. `README.md`
2. `installation.md`
3. `basics.md`
4. `kernels.md`
5. `examples/core.md`
6. `meshing.md`
7. `simulation.md`
8. `troubleshooting.md`

## Core concepts introduced

- `warp` imports and module layout in `warp/__init__.py`
- kernel programming with `@wp.kernel`
- thread indexing with `wp.tid()`
- data management with `wp.array`, `wp.zeros`, `wp.from_numpy`, and `wp.to_numpy`
- `launch`, `launch_tiled`, and capture/graph execution
- spatial primitives such as `wp.Mesh`, `wp.HashGrid`, `wp.Bvh`, `wp.Volume`
- auto-differentiation with `wp.Tape`

## Local source anchors

- Top-level API exports: `warp/__init__.py`
- Examples: `warp/examples/`
- Official docs in-repo: `docs/`
- Core API references: `docs/api_reference/`
- Tutorials and deep docs: `docs/user_guide/`

## Web resources (official)

- Docs home: https://nvidia.github.io/warp/
- Installation guide: https://nvidia.github.io/warp/user_guide/installation.html
- Basics guide: https://nvidia.github.io/warp/user_guide/basics.html
- Runtime API: https://nvidia.github.io/warp/user_guide/runtime.html
- API reference root: https://nvidia.github.io/warp/modules/runtime.html
- Interoperability guide: https://nvidia.github.io/warp/user_guide/interoperability.html
- GitHub repo: https://github.com/NVIDIA/warp
- Notebooks: https://github.com/NVIDIA/warp/tree/main/notebooks

## Practical first steps

Use `uv run` for any commands in this repository.

```sh
uv run python -m warp.examples.core.example_fluid
```

Use `wp.init()` when you need to force deterministic startup output and initialization timing.

