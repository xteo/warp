# Mega Prompt 1: Interactive GPU Ray Marcher with Camera Navigation

**Inspiration:** `example_raymarch.py` + `example_render_opengl.py` + community demand for interactive real-time tools

**What it builds:** A real-time SDF scene explorer rendered entirely on GPU. The user navigates a 3D scene of procedural objects (spheres, toruses, fractal geometry) using keyboard controls. The ray marching kernel computes lighting, shadows, and ambient occlusion in real time.

**Difficulty:** Medium | **Est. Lines:** ~250 | **Key APIs:** `@wp.kernel` (2D), `@wp.func`, `OpenGLRenderer`

---

## Prompt

Using the Warp skills, build an interactive GPU ray marcher with real-time camera navigation.

## Architecture

Create a single Python file `interactive_raymarch.py` that combines:
1. A Warp kernel for SDF ray marching (based on the pattern in `example_raymarch.py`)
2. The OpenGL renderer from `warp.render.OpenGLRenderer` for display (based on `example_render_opengl.py`)
3. pyglet keyboard/mouse event handling for camera control

## Scene Definition

Define a scene with these SDF primitives computed in the kernel:
- Ground plane with checkerboard pattern
- 3 spheres at different positions with different materials (reflective, matte, emissive)
- 1 torus (shows off SDF composition)
- 1 rounded box
- Use smooth min (`smin`) for organic blending between nearby objects

## Ray Marching Kernel

Write a `@wp.kernel` that:
- Takes camera position (wp.vec3), camera forward/right/up vectors, and image dimensions
- For each pixel (2D thread grid), computes a ray direction
- Marches the ray through the SDF scene (max 128 steps, min distance 0.001)
- Computes surface normal via central differences on the SDF
- Applies Phong lighting with one directional light + ambient
- Computes soft shadows by marching a secondary ray toward the light
- Writes RGB color to a `wp.array(dtype=wp.vec3)` output image

## Camera System

Implement a simple FPS-style camera:
- WASD keys move forward/back/left/right relative to camera orientation
- Arrow keys or Q/E rotate the camera yaw
- R/F move up/down
- Store camera state in Python (position, yaw, pitch) and pass to kernel each frame
- Use `wp.quat_from_axis_angle()` to compute camera orientation

## Display Loop

Use `warp.render.OpenGLRenderer` with `headless=False`:
- Create the renderer with `vsync=False` for maximum framerate
- Each frame: update camera from input -> launch ray march kernel -> display result
- Use `renderer.get_pixels()` for the render-to-screen pipeline
- Alternatively, render to a wp.array and use matplotlib animation for headless mode
- Target 30+ FPS at 1024x512 resolution

## Headless Fallback

If no display is available (headless environment), fall back to:
- Render a single high-res frame (2048x1024) with a preset camera path
- Save as PNG via matplotlib
- Accept `--headless` flag

## Dependencies

- warp-lang (already installed)
- pyglet>=2.0 (for OpenGL display)
- matplotlib (for headless fallback)
- Run with: `uv run --with "pyglet>=2.0" --with matplotlib python interactive_raymarch.py`

## Key Warp Patterns to Use

- `@wp.kernel` with 2D `wp.tid()` for pixel grid
- `@wp.func` for SDF primitives, lighting, shadow computation
- `wp.vec3` math throughout (dot, cross, normalize, length)
- `wp.array` for image buffer on GPU
- `wp.launch(kernel, dim=(width, height), ...)` for 2D grid launch

## Reference Examples

- `warp/examples/core/example_raymarch.py` — SDF ray marching pattern, single kernel rendering
- `warp/examples/core/example_render_opengl.py` — OpenGL renderer setup, pixel readback, headless mode
