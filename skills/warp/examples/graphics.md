# Rendering and Graphics Examples

Render and geometry examples use `warp.render` and spatial outputs.

## USD renderer

`warp/render` examples output USD sequences for inspection in Omniverse / USD viewers.

- `warp/examples/core/example_mesh.py`
- `warp/examples/core/example_marching_cubes.py`
- `warp/examples/core/example_sample_mesh.py`

## OpenGL path

`warp/render/` and `OpenGLRenderer` examples exist for interactive previews.

- `warp/examples/core/example_render_opengl.py`

## How-to for visualization

1. Add/enable renderer creation in example constructor.
2. Run step and render loop.
3. Export `.usd` and review in compatible tool.

## Runtime API reference

- `warp.render.UsdRenderer`
- `warp.render.OpenGLRenderer`
- `warp.render.Texture` paths
- See `docs/domain_modules/render.rst`
