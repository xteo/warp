# Mega Prompt 3: Differentiable Scene Optimizer

Optimizes 3D sphere parameters to match a target image using differentiable rendering and automatic differentiation, powered by NVIDIA Warp.

![Screenshot](mega_prompt_3.png)

*Left: target image (5 spheres) | Center: initial random state (20 spheres) | Right: optimized result (step 150)*

## How It Works

- A soft differentiable renderer computes per-pixel color from sphere parameters using front-to-back alpha compositing with smooth sigmoid-based opacity
- `wp.Tape` records the forward pass (render + loss) and backpropagates gradients to sphere positions, radii, and colors
- The Adam optimizer (`warp.optim.Adam`) updates all parameters each step
- Loss drops from ~11,300 to ~200 (56x reduction) in 150 steps

## How to Run

```sh
uv run --with matplotlib --with Pillow python prompts/mega_prompt_3/diffscene.py
```

**Custom settings:**

```sh
uv run --with matplotlib --with Pillow python prompts/mega_prompt_3/diffscene.py --steps 300 --num-spheres 30 --lr 0.005
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--device` | auto | Warp device (`cpu`, `cuda:0`, etc.) |
| `--width` | 512 | Image width in pixels |
| `--height` | 512 | Image height in pixels |
| `--num-spheres` | 20 | Number of spheres to optimize |
| `--steps` | 500 | Optimization steps |
| `--lr` | 0.01 | Learning rate for Adam optimizer |
| `--save-every` | 50 | Save progress image every N steps |

## Output

- `target.png` -- ground truth image (5 spheres at known positions/colors)
- `optimized_step_XXX.png` -- progress images at regular intervals
- `optimization.gif` -- animated GIF of the optimization process
- `loss_curve.png` -- convergence plot (log scale)

## Dependencies

- `warp-lang` (already installed in this repo)
- `matplotlib` (loss curve plot)
- `Pillow` (image I/O and GIF generation)
