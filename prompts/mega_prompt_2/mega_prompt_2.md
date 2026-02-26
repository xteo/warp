# Mega Prompt 2: Real-time GPU Particle Fluid with Interactive Forces

**Inspiration:** `example_sph.py` + `example_dem.py` + Warbler Blender integration + community interest in interactive physics

**What it builds:** A real-time 2D/3D SPH fluid simulator where the user can click to add force fields, spawn particles, and create obstacles. Particles are rendered as colored points with velocity-based coloring. Uses HashGrid for neighbor queries.

**Difficulty:** Medium-Hard | **Est. Lines:** ~350 | **Key APIs:** `wp.HashGrid`, `@wp.kernel`, `OpenGLRenderer`

---

## Prompt

Using the Warp skills, build a real-time GPU particle fluid simulator with interactive force fields.

## Architecture

Create `interactive_fluid.py` — a 2D particle-based fluid simulator with:
1. SPH (Smoothed Particle Hydrodynamics) kernels based on `example_sph.py` patterns
2. `wp.HashGrid` for efficient neighbor queries (see `example_sph.py` and `example_dem.py`)
3. OpenGL renderer for real-time particle visualization
4. Mouse interaction for adding forces and spawning particles

## Particle System (5,000-20,000 particles)

State arrays:
- `positions: wp.array(dtype=wp.vec3)` — particle positions (z=0 for 2D)
- `velocities: wp.array(dtype=wp.vec3)` — particle velocities
- `densities: wp.array(dtype=float)` — computed densities
- `pressures: wp.array(dtype=float)` — computed pressures
- `forces: wp.array(dtype=wp.vec3)` — accumulated forces

## SPH Kernels

Write these `@wp.kernel` functions:
1. `compute_density` — for each particle, query HashGrid neighbors within smoothing radius h,
   accumulate density using cubic spline kernel W(r, h)
2. `compute_pressure_force` — pressure gradient + viscosity from neighbors
3. `apply_external_forces` — gravity + user-applied force fields
4. `integrate` — symplectic Euler: velocity += dt * force/density, position += dt * velocity
5. `enforce_boundaries` — keep particles in domain [0, domain_size], reflect velocities

Use `@wp.func` for:
- SPH kernel function W(r, h) — cubic spline
- SPH kernel gradient grad_W(r, h)
- Equation of state: pressure = k * (density/rest_density - 1)

## HashGrid Update Pattern

Each timestep:
```python
grid.build(positions, grid_cell_size)  # rebuild spatial hash
wp.launch(compute_density, dim=n, inputs=[positions, densities, grid.id, ...])
wp.launch(compute_pressure_force, dim=n, inputs=[...])
wp.launch(apply_external_forces, dim=n, inputs=[..., mouse_pos, mouse_force])
wp.launch(integrate, dim=n, inputs=[...])
```

## Interactive Forces

- Pass mouse position as a `wp.vec3` constant to the external force kernel
- When mouse is clicked, apply a radial force field: F = strength * (particle_pos - mouse_pos) / |r|^2
- Left click = attract, right click = repel
- Keyboard: Space = pause/resume, R = reset, +/- = adjust particle count

## Visualization

Use `warp.render.OpenGLRenderer`:
- Render particles as small spheres via `renderer.render_points()`
- Color particles by velocity magnitude (blue=slow, red=fast) using a transfer kernel
- Display frame rate and particle count

For headless mode:
- Run 200 simulation steps
- Save final particle state as matplotlib scatter plot
- Save intermediate frames as PNG sequence

## Simulation Parameters

- Smoothing radius h = 0.04
- Rest density = 1000
- Gas constant k = 50
- Viscosity = 0.1
- Gravity = (0, -9.81, 0)
- Time step dt = 0.0005
- Domain = 1.0 x 1.0 (2D, z clamped to 0)

## Run Commands

- Interactive: `uv run --with "pyglet>=2.0" python interactive_fluid.py`
- Headless: `uv run --with matplotlib python interactive_fluid.py --headless --num-frames 200`

## Reference Examples

- `warp/examples/core/example_sph.py` — SPH with HashGrid, density/pressure kernels, integration
- `warp/examples/core/example_dem.py` — HashGrid for contact detection, force computation
- `warp/examples/core/example_render_opengl.py` — OpenGL renderer, pixel readback, headless
