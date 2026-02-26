# Mega Prompt 4: GPU N-body Galaxy Simulator

Gravitational N-body galaxy simulator with tile-based acceleration, CUDA graph capture, and velocity-mapped particle rendering using NVIDIA Warp.

![Screenshot](mega_prompt_4.png)

## Simulation

- 10,240 gravitational bodies simulated on GPU
- Tile-based force computation (64-particle tiles via `wp.tile_load`) for ~2x speedup over naive O(N^2)
- CUDA graph capture (`wp.ScopedCapture`) eliminates kernel launch overhead
- Leapfrog integration with softening parameter to prevent singularities
- Motion trails via circular buffer of previous positions

## Presets

- **Spiral Galaxy**: Disk of particles orbiting a massive central body with spiral arm modulation and tangential velocities
- **Galaxy Collision**: Two spiral galaxies on a collision course, merging under mutual gravity
- **Random Cluster**: Uniform spherical distribution with low velocities, showing gravitational collapse

## How to Run

**Spiral galaxy** (200 frames):

```sh
uv run --with matplotlib --with Pillow python prompts/mega_prompt_4/galaxy_sim.py --preset spiral --num-frames 200
```

**Galaxy collision** (300 frames):

```sh
uv run --with matplotlib --with Pillow python prompts/mega_prompt_4/galaxy_sim.py --preset collision --num-frames 300
```

**Random cluster** (200 frames):

```sh
uv run --with matplotlib --with Pillow python prompts/mega_prompt_4/galaxy_sim.py --preset cluster --num-frames 200
```

**With benchmark** (compare tile vs naive kernel):

```sh
uv run --with matplotlib --with Pillow python prompts/mega_prompt_4/galaxy_sim.py --preset spiral --num-frames 100 --benchmark
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--device` | auto | Warp device (`cpu`, `cuda:0`, etc.) |
| `--preset` | `spiral` | Initial conditions: `spiral`, `collision`, `cluster` |
| `--num-frames` | 200 | Number of animation frames |
| `--num-bodies` | 10240 | Number of particles (rounded to multiple of 64) |
| `--dt` | 0.001 | Simulation timestep |
| `--G` | 1.0 | Gravitational constant |
| `--output-dir` | script dir | Output directory for frames and GIFs |
| `--benchmark` | off | Run tile vs naive kernel timing comparison |
| `--no-gif` | off | Skip GIF creation |

## Output

- `galaxy_spiral.gif` / `galaxy_collision.gif` / `galaxy_cluster.gif` -- animated GIF of the simulation
- `galaxy_frame_XXXX.png` -- high-res key frames at select timesteps
- `mega_prompt_4.png` -- representative screenshot
- Console prints: timing comparison (with `--benchmark`), frame rate, simulation stats

## Dependencies

- `warp-lang` (already installed in this repo)
- `matplotlib` (rendering)
- `Pillow` (GIF creation)
