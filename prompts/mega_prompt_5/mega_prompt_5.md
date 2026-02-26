# Mega Prompt 5: Mini 2D Lattice Boltzmann CFD Solver

**Inspiration:** Autodesk XLB (8x speedup over JAX, matching C++ solvers) + `example_fluid.py` + `example_fft_poisson_navier_stokes_2d.py` + community interest in CFD

**What it builds:** A compact Lattice Boltzmann Method (LBM) fluid solver in ~300 lines of Warp. Simulates 2D incompressible flow around obstacles with real-time visualization. Demonstrates how Warp can match specialized CFD codes while staying pure Python. Produces beautiful von Karman vortex streets.

**Difficulty:** Hard | **Est. Lines:** ~400 | **Key APIs:** `wp.array3d`, `@wp.kernel` (2D grid), streaming patterns

---

## Prompt

Using the Warp skills, build a mini 2D Lattice Boltzmann CFD solver with obstacle interaction.

## Architecture

Create `lbm_solver.py` — a D2Q9 Lattice Boltzmann solver with:
1. Warp kernels for streaming, collision, and boundary conditions
2. Real-time visualization of velocity magnitude and vorticity
3. Configurable obstacles (cylinder, airfoil, custom shapes)
4. Von Karman vortex street as the showcase simulation

## D2Q9 Lattice Boltzmann

Use the standard D2Q9 velocity set with 9 discrete velocities:
```
e = [(0,0), (1,0), (0,1), (-1,0), (0,-1), (1,1), (-1,1), (-1,-1), (1,-1)]
w = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]
```

State: `f: wp.array3d(dtype=float)` with shape (9, Ny, Nx) — distribution functions.

## Kernels

Write these `@wp.kernel` functions:

1. `equilibrium_kernel` — compute equilibrium distribution:
   f_eq[q][j][i] = w[q] * rho * (1 + 3*(e_q . u) + 4.5*(e_q . u)^2 - 1.5*|u|^2)

2. `collision_kernel` — BGK collision operator:
   f_out[q] = f[q] - (f[q] - f_eq[q]) / tau

3. `streaming_kernel` — propagate distributions to neighbor cells:
   f_new[q][j][i] = f_out[q][j - ey[q]][i - ex[q]]

4. `boundary_kernel` — bounce-back on solid nodes (obstacles and walls):
   Reverse distribution direction at solid boundaries

5. `inlet_boundary_kernel` — Zou-He velocity inlet on left boundary

6. `macroscopic_kernel` — compute density (rho) and velocity (ux, uy) from distributions:
   rho = sum(f[q]), rho*ux = sum(f[q]*ex[q]), rho*uy = sum(f[q]*ey[q])

7. `vorticity_kernel` — compute curl of velocity field for visualization

## Obstacle Setup

Create a `@wp.kernel` that marks solid cells:
- Cylinder: circle at (Nx/5, Ny/2) with radius R
- Support loading custom obstacle masks from numpy arrays

## Simulation Loop

```python
for step in range(num_steps):
    wp.launch(collision_kernel, dim=(Ny, Nx), ...)
    wp.launch(streaming_kernel, dim=(Ny, Nx), ...)
    wp.launch(boundary_kernel, dim=(Ny, Nx), ...)
    wp.launch(inlet_boundary_kernel, dim=(Ny,), ...)
    wp.launch(macroscopic_kernel, dim=(Ny, Nx), ...)

    if step % plot_interval == 0:
        wp.launch(vorticity_kernel, dim=(Ny-2, Nx-2), ...)
        save_frame(vorticity, step)
```

## Visualization

Use matplotlib to render:
- Vorticity field with `coolwarm` colormap (shows von Karman vortex street beautifully)
- Velocity magnitude with `viridis` colormap
- Obstacle boundary overlay in black
- Optionally: streamlines using matplotlib `streamplot`

Save frames as PNGs and compile to GIF.

## Parameters

- Grid: 800 x 200 (Nx x Ny)
- Reynolds number Re = 200 (produces clear vortex shedding)
- Inlet velocity u_in = 0.04
- Relaxation time tau = 3 * nu + 0.5, where nu = u_in * D / Re
- Cylinder diameter D = 40 cells
- Run for 20,000 steps, save every 100 steps

## Output

- `lbm_vorticity.gif` — animated vorticity field showing vortex street
- `lbm_velocity_XXX.png` — velocity magnitude snapshots
- `lbm_vorticity_XXX.png` — vorticity snapshots
- Print Reynolds number, grid size, performance (MLUPS — million lattice updates per second)

## Performance Target

At 800x200 grid, aim for >100 MLUPS on GPU (comparable to specialized LBM codes).
Print wall-clock time per 1000 steps for benchmarking.

## Run

- `uv run --with matplotlib --with Pillow python lbm_solver.py --num-steps 20000`
- `uv run --with matplotlib --with Pillow python lbm_solver.py --Re 400 --grid 1600x400`

## Reference Examples

- `warp/examples/core/example_fluid.py` — multi-kernel simulation loop pattern (field init, divergence, pressure, advection)
- `warp/examples/core/example_fft_poisson_navier_stokes_2d.py` — spectral Navier-Stokes, tile FFT
- `warp/examples/core/example_wave.py` — grid-based stencil computation, explicit time stepping
