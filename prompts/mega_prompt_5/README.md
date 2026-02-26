# Mega Prompt 5: Mini 2D Lattice Boltzmann CFD Solver

GPU-accelerated D2Q9 Lattice Boltzmann fluid solver using NVIDIA Warp.

![Vorticity](lbm_vorticity_20000.png)

## Simulation

- D2Q9 lattice Boltzmann method for 2D incompressible flow
- BGK collision operator with configurable relaxation time
- Zou-He velocity inlet, zero-gradient outlet
- Bounce-back boundary conditions for solid walls and obstacles
- Cylinder obstacle producing von Karman vortex street at Re=200
- Asymmetric perturbation to trigger vortex shedding
- Velocity ramp-up for stability

## How to Run

**Default** (800x200 grid, Re=200, 20000 steps):

```sh
uv run --with matplotlib --with Pillow python prompts/mega_prompt_5/lbm_solver.py
```

**Custom Reynolds number and grid:**

```sh
uv run --with matplotlib --with Pillow python prompts/mega_prompt_5/lbm_solver.py --Re 400 --grid 1600x400
```

**Fewer steps for quick test:**

```sh
uv run --with matplotlib --with Pillow python prompts/mega_prompt_5/lbm_solver.py --num-steps 5000
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--device` | auto | Warp device (`cpu`, `cuda:0`, etc.) |
| `--num-steps` | 20000 | Number of simulation steps |
| `--Re` | 200.0 | Reynolds number |
| `--u-in` | 0.1 | Inlet velocity (lattice units) |
| `--cylinder-d` | 40 | Cylinder diameter in cells |
| `--grid` | 800x200 | Grid size NxM |
| `--save-interval` | 100 | Save vorticity frame every N steps |
| `--output-dir` | script directory | Output directory for images and GIF |

## Output

- `lbm_vorticity_NNNNN.png` -- vorticity field snapshots (coolwarm colormap)
- `lbm_velocity_NNNNN.png` -- velocity magnitude snapshots (viridis colormap)
- `lbm_vorticity.gif` -- animated vorticity field showing vortex street development

## Performance

At 800x200 grid on NVIDIA GB10 GPU: ~179 MLUPS (million lattice updates per second).

## Dependencies

- `warp-lang` (already installed in this repo)
- `matplotlib` (visualization)
- `Pillow` (GIF generation)
