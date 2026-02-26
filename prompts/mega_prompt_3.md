# Mega Prompt 3: Differentiable Scene Optimizer (Image-to-3D Primitives)

**Inspiration:** `example_diffray.py` + `example_particle_repulsion.py` + PhysGaussian research + 3DGS-Warp-Scratch (254 stars)

**What it builds:** A differentiable renderer that optimizes 3D primitive parameters (positions, radii, colors) to match a target image. Uses `wp.Tape` for automatic differentiation and gradient descent. The system iteratively adjusts scene parameters to minimize pixel loss, producing a training animation.

**Difficulty:** Medium | **Est. Lines:** ~200 | **Key APIs:** `wp.Tape`, `warp.optim.Adam`, `@wp.kernel`

---

## Prompt

Using the Warp skills, build a differentiable scene optimizer that fits 3D primitives to a target image.

## Architecture

Create `diffscene.py` that:
1. Renders a scene of parametric 3D primitives using a differentiable ray casting kernel
2. Computes pixel-wise loss against a target image
3. Uses `wp.Tape` to backpropagate gradients to primitive parameters
4. Optimizes positions, radii, and colors using Adam optimizer from `warp.optim`

## Scene Representation

Parameterize the scene as N spheres (start with N=20):
- `sphere_positions: wp.array(dtype=wp.vec3, requires_grad=True)` — (N,) centers
- `sphere_radii: wp.array(dtype=float, requires_grad=True)` — (N,) radii
- `sphere_colors: wp.array(dtype=wp.vec3, requires_grad=True)` — (N,) RGB colors

Initialize randomly within a bounding box, radii in [0.1, 0.5].

## Differentiable Ray Caster Kernel

Write a `@wp.kernel` that for each pixel:
- Computes ray origin and direction from a fixed camera
- Tests intersection with each sphere (analytic ray-sphere intersection)
- Finds the nearest hit, computes surface normal
- Applies simple diffuse lighting: color = sphere_color * max(0, dot(normal, light_dir))
- For missed rays, returns background color
- Writes to `rendered_image: wp.array(dtype=wp.vec3)` with shape (H*W,)

All operations must be differentiable (use `wp.func` helpers, avoid non-differentiable branches).

## Loss Kernel

Write a `@wp.kernel` that computes per-pixel squared error:
```
loss[0] += wp.length_sq(rendered[i] - target[i])
```
Use `wp.atomic_add()` to accumulate into a scalar loss array.

## Training Loop (based on example_diffray.py pattern)

```python
optimizer = warp.optim.Adam([sphere_positions, sphere_radii, sphere_colors], lr=0.01)

for step in range(500):
    tape = wp.Tape()
    with tape:
        wp.launch(render_kernel, dim=H*W, inputs=[...], outputs=[rendered, loss])
    tape.backward(loss)
    optimizer.step([sphere_positions.grad, sphere_radii.grad, sphere_colors.grad])
    tape.zero()

    if step % 50 == 0:
        save_progress_image(rendered, step)
```

## Target Image

Generate a synthetic target by:
1. Placing 5 spheres at known positions with known colors
2. Rendering at 512x512 with the same ray caster
3. Saving as `target.png`

Then initialize 20 random spheres and optimize to match.

## Output

- `target.png` — the ground truth image
- `optimized_step_XXX.png` — progress images every 50 steps
- `optimization.gif` — animated GIF of the optimization process (using Pillow)
- Print loss value each step

## Dependencies

- warp-lang, matplotlib, Pillow
- Run: `uv run --with matplotlib --with Pillow python diffscene.py`
- Headless safe (no display needed, saves to files)

## Reference Examples

- `warp/examples/optim/example_diffray.py` — differentiable ray casting, wp.Tape, optimization loop, animation GIF
- `warp/examples/optim/example_particle_repulsion.py` — gradient-based particle optimization with wp.Tape
- `warp/examples/optim/example_fluid_checkpoint.py` — checkpointed differentiable simulation, warp.optim
