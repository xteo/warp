# Mega Prompt 4: GPU N-body Galaxy Simulator with Tile Acceleration

**Inspiration:** `example_tile_nbody.py` + `example_graph_capture.py` + community interest in beautiful physics simulations

**What it builds:** A gravitational N-body simulator that creates galaxy-like structures. Uses tile-based force computation for performance, CUDA graph capture for reduced launch overhead, and renders particles with velocity-mapped colors and motion trails.

**Difficulty:** Medium | **Est. Lines:** ~300 | **Key APIs:** `wp.launch_tiled`, `wp.ScopedCapture`, `@wp.kernel`

---

## Prompt

Using the Warp skills, build a GPU-accelerated N-body galaxy simulator with tile-based acceleration.

## Architecture

Create `galaxy_sim.py` with:
1. Tile-based gravitational force computation (based on `example_tile_nbody.py`)
2. CUDA graph capture for the simulation loop (based on `example_graph_capture.py`)
3. Matplotlib visualization with motion trails and velocity coloring
4. Multiple initial conditions (spiral galaxy, collision, random cluster)

## Particle State (10,000-50,000 bodies)

- `positions: wp.array(dtype=wp.vec3)` — particle positions
- `velocities: wp.array(dtype=wp.vec3)` — particle velocities
- `masses: wp.array(dtype=float)` — particle masses
- `trail_positions: wp.array(dtype=wp.vec3)` — previous N positions for trails

## Tile-Based Force Kernel

Use `wp.launch_tiled()` with tile operations:
```python
@wp.kernel
def compute_forces(
    positions: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=float),
    forces: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    # Load tiles of particle data for efficient shared memory access
    # Accumulate gravitational force: F = G * m1 * m2 * (r / |r|^3)
    # Use softening parameter epsilon to prevent singularities
```

Compare performance with a naive O(N^2) SIMT kernel to demonstrate tile speedup.

## Initial Conditions

Implement 3 galaxy presets:
1. **Spiral Galaxy**: Particles in a disk with tangential velocity, density falling off with radius,
   slight vertical dispersion. Central massive body.
2. **Galaxy Collision**: Two spiral galaxies approaching each other with relative velocity.
3. **Random Cluster**: Uniform random positions with low random velocities, watching gravitational collapse.

Use `@wp.kernel` to initialize each pattern.

## CUDA Graph Capture

After warm-up, capture the simulation loop:
```python
with wp.ScopedCapture() as capture:
    wp.launch(compute_forces, ...)
    wp.launch(integrate, ...)
graph = capture.graph

for frame in range(num_frames):
    wp.capture_launch(graph)
```

## Visualization

Render with matplotlib:
- 2D projection (x-y plane) with semi-transparent particles
- Color particles by velocity magnitude (viridis colormap)
- Particle size proportional to mass (log scale)
- Optional: motion trails using previous positions (fading alpha)
- Save each frame as PNG, compile to GIF at the end

For headless: save animation frames and create GIF with Pillow.

## Output

- `galaxy_spiral.gif` — spiral galaxy evolution (200 frames)
- `galaxy_collision.gif` — two galaxies merging
- `galaxy_frame_XXX.png` — individual high-res frames
- Print timing comparison: tile kernel vs naive kernel

## Parameters

- G = 1.0 (normalized gravitational constant)
- Softening epsilon = 0.01
- dt = 0.001
- Spiral galaxy: 10,000 particles, central mass = 1000
- Each galaxy in collision: 5,000 particles

## Run

- `uv run --with matplotlib --with Pillow python galaxy_sim.py --preset spiral --num-frames 200`
- `uv run --with matplotlib --with Pillow python galaxy_sim.py --preset collision --num-frames 300`

## Reference Examples

- `warp/examples/tile/example_tile_nbody.py` — tile-based N-body force computation
- `warp/examples/core/example_graph_capture.py` — CUDA graph capture with wp.ScopedCapture
- `warp/examples/core/example_sph.py` — particle simulation loop pattern, HashGrid
