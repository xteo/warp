# Rendering and Graphics Examples

Warp provides two rendering backends and several GPU ray tracing/marching kernels.

## USD renderer

USD-based examples output `.usd` files for inspection in Omniverse or USD viewers.
Requires `pxr` module (`usd-core` on x86-64, `usd-exchange` on aarch64).

| Example | Output file |
|---------|------------|
| `example_mesh` | `example_mesh.usd` |
| `example_wave` | `example_wave.usd` |
| `example_dem` | `example_dem.usd` |
| `example_sph` | `example_sph.usd` |
| `example_marching_cubes` | `example_marching_cubes.usd` |
| `example_nvdb` | `example_nvdb.usd` |
| `example_sample_mesh` | `example_sample_mesh.usd` |
| `example_mesh_intersect` | `example_mesh_intersect.usd` |
| `example_apic_fluid` | `example_apic_fluid.usd` |

## OpenGL renderer

Interactive real-time rendering via `warp.render.OpenGLRenderer`. Requires `pyglet>=2.0`.

```sh
uv run --with "pyglet>=2.0" python -m warp.examples.core.example_render_opengl
```

Key features:
- Tiled rendering with multiple viewports (`setup_tiled_rendering()`)
- Pixel readback via `renderer.get_pixels()` (RGB and depth modes)
- Headless mode: `OpenGLRenderer(vsync=False, headless=True)` with `xvfb-run`
- When using `get_pixels()` without tiled rendering, pass `split_up_tiles=False`

Headless OpenGL example:

```sh
xvfb-run -a --server-args="-screen 0 1920x1080x24" \
  uv run --with "pyglet>=2.0" python -m warp.examples.core.example_render_opengl
```

CUDA-GL interop may fail with software OpenGL (Xvfb); Warp falls back to CPU copy mode automatically.

**Important**: `import warp.render` must be called explicitly before using the renderer — it is not auto-imported by `import warp`.

## GPU ray tracing kernels (matplotlib output)

These examples render images entirely on the GPU using kernel-based ray tracing:

| Example | Description | Dependencies |
|---------|------------|-------------|
| `example_raymarch` | SDF ray marching (spheres, planes), single kernel | matplotlib |
| `example_raycast` | Ray casting on Stanford Bunny mesh | matplotlib, pxr |

Both accept `--headless`, `--device`, `--width`, `--height`.

## Rendering integration pattern

To add rendering to a simulation:

1. Create renderer: `renderer = wp.render.UsdRenderer(stage_path)` or `OpenGLRenderer()`
2. In simulation loop: `renderer.begin_frame(time)` → render objects → `renderer.end_frame()`
3. For USD: `renderer.save()` to write file
4. For OpenGL: `renderer.get_pixels()` for programmatic readback

## Interactive pyglet apps with custom Warp kernel rendering

For custom pixel-level rendering (ray marching, custom shaders, etc.) where you write
pixels from a Warp kernel and display them in a window, use pyglet directly with a
**manual event loop**. Do NOT use `pyglet.app.run()` with `schedule_interval` — it does
not reliably update `KeyStateHandler` for continuous key input.

**Critical pattern** — manual event loop (matches `warp.render.OpenGLRenderer` internals):

```python
import time
import pyglet
from pyglet.window import key
import numpy as np
import warp as wp

window = pyglet.window.Window(width, height, vsync=False)
key_handler = key.KeyStateHandler()
window.push_handlers(key_handler)

pixels = wp.zeros(width * height, dtype=wp.vec3)
running = [True]

@window.event
def on_close():
    running[0] = False

@window.event
def on_key_press(symbol, modifiers):
    if symbol == key.ESCAPE:
        running[0] = False

last_time = time.perf_counter()

while running[0]:
    # 1. Process OS events — updates KeyStateHandler
    window.dispatch_events()

    # 2. Compute dt
    now = time.perf_counter()
    dt = now - last_time
    last_time = now

    # 3. Read key states and update simulation/camera
    if key_handler[key.W]:
        # move forward ...
        pass

    # 4. Launch Warp kernel
    wp.launch(my_kernel, dim=(width, height), inputs=[..., pixels])

    # 5. Read pixels (.numpy() implicitly synchronizes GPU)
    pixels_np = pixels.numpy().reshape(height, width, 3)
    pixels_uint8 = (pixels_np * 255).astype(np.uint8)

    # 6. Draw and flip
    window.switch_to()
    window.clear()
    image = pyglet.image.ImageData(width, height, "RGB",
                                    pixels_uint8.tobytes(), pitch=width * 3)
    image.blit(0, 0)
    window.flip()

window.close()
```

**Why manual loop?** The `OpenGLRenderer` in `warp/_src/render/render_opengl.py` uses
`platform_event_loop.step()` + `_redraw_windows()` — never `pyglet.app.run()`. The
manual pattern ensures `dispatch_events()` processes all pending input before you read
`KeyStateHandler`, then you render and flip in the same iteration. Using `pyglet.app.run()`
with `schedule_interval` splits event processing and rendering across separate callbacks,
which causes `KeyStateHandler` to appear unresponsive.

**Headless fallback**: For headless environments, skip pyglet entirely. Launch the kernel
once and save via matplotlib:

```python
wp.launch(kernel, dim=(width, height), inputs=[..., pixels])
pixels_np = pixels.numpy().reshape(height, width, 3)
plt.imsave("output.png", pixels_np, origin="lower")
```

## Runtime API reference

- `warp.render.UsdRenderer` — USD stage output
- `warp.render.OpenGLRenderer` — real-time OpenGL with pixel readback
- See `docs/domain_modules/render.rst`

## Web references

- https://nvidia.github.io/warp/domain_modules/render.html
