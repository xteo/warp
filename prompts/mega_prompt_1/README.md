# Mega Prompt 1: Interactive GPU Ray Marcher

Real-time SDF scene explorer rendered entirely on GPU using NVIDIA Warp.

![Screenshot](mega_prompt_1.png)

## Scene

- Ground plane with checkerboard pattern
- 3 spheres: reflective silver, matte red, emissive gold
- 1 torus (metallic blue)
- 1 rounded box (green)
- Smooth min blending between nearby objects
- Soft shadows, ambient occlusion, Phong lighting, sky gradient

## How to Run

**Interactive** (real-time with camera controls):

```sh
uv run --with "pyglet>=2.0" --with matplotlib python prompts/mega_prompt_1/interactive_raymarch.py
```

**Headless** (renders a single high-res PNG):

```sh
uv run --with matplotlib python prompts/mega_prompt_1/interactive_raymarch.py --headless
```

**Custom resolution:**

```sh
uv run --with "pyglet>=2.0" python prompts/mega_prompt_1/interactive_raymarch.py --width 1920 --height 1080
```

## Controls

| Key | Action |
|-----|--------|
| W/S | Move forward / back |
| A/D | Strafe left / right |
| Q/E | Rotate camera left / right |
| Arrow keys | Rotate camera |
| R/F | Move up / down |
| ESC | Quit |

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--device` | auto | Warp device (`cpu`, `cuda:0`, etc.) |
| `--headless` | off | Render single frame to PNG |
| `--width` | 1024 (interactive) / 2048 (headless) | Image width |
| `--height` | 512 (interactive) / 1024 (headless) | Image height |
| `--output` | `interactive_raymarch.png` | Output path (headless only) |

## Dependencies

- `warp-lang` (already installed in this repo)
- `pyglet>=2.0` (interactive mode)
- `matplotlib` (headless mode)
