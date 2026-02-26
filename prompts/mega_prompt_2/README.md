# Mega Prompt 2: Real-time GPU Particle Fluid

Interactive 2D SPH fluid simulator with mouse-driven forces, rendered on GPU using NVIDIA Warp.

![Screenshot](interactive_fluid.png)

## Simulation

- 10,000 particles (adjustable 2k-30k at runtime)
- SPH (Smoothed Particle Hydrodynamics) with cubic spline kernel
- `wp.HashGrid` for O(n) neighbor queries
- Weakly compressible equation of state for pressure
- Viscosity diffusion and gravity
- Dam-break initial configuration
- Velocity-based particle coloring (blue=slow, red=fast)

## How to Run

**Interactive** (real-time with mouse interaction):

```sh
uv run --with "pyglet>=2.0" --with matplotlib python prompts/mega_prompt_2/interactive_fluid.py
```

**Headless** (runs simulation and saves PNG):

```sh
uv run --with matplotlib python prompts/mega_prompt_2/interactive_fluid.py --headless --num-frames 200
```

**Custom particle count:**

```sh
uv run --with "pyglet>=2.0" python prompts/mega_prompt_2/interactive_fluid.py --num-particles 20000
```

## Controls

| Key / Mouse | Action |
|-------------|--------|
| Left click + drag | Attract particles toward mouse |
| Right click + drag | Repel particles from mouse |
| Space | Pause / resume simulation |
| R | Reset simulation |
| +/= | Increase particle count (+2000) |
| - | Decrease particle count (-2000) |
| ESC | Quit |

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--device` | auto | Warp device (`cpu`, `cuda:0`, etc.) |
| `--headless` | off | Run simulation and save PNG |
| `--num-particles` | 10000 | Number of SPH particles |
| `--num-frames` | 200 | Simulation frames (headless only) |
| `--width` | 900 | Window width (interactive only) |
| `--height` | 900 | Window height (interactive only) |
| `--output` | `interactive_fluid.png` | Output path (headless only) |

## Dependencies

- `warp-lang` (already installed in this repo)
- `pyglet>=2.0` (interactive mode)
- `matplotlib` (headless mode)
