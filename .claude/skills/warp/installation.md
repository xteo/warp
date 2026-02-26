# Warp Installation

This page summarizes practical install and runtime options from official docs.

## Supported platforms and package targets

Warp supports:

- x86-64 and ARMv8 on Linux and Windows
- Apple Silicon macOS (ARM64) for supported versions

GPU support requires CUDA-capable NVIDIA hardware and compatible driver/toolkit.

## CUDA driver requirements

GPU support requires NVIDIA driver version 525.60.13+ (Linux) or 528.33+ (Windows).
Check your driver with `nvidia-smi`. The driver must match or exceed the CUDA toolkit
version used to build the Warp wheel.

## Quick install (recommended)

From PyPI with default driver/runtime compatibility:

```sh
pip install warp-lang
```

If you need example dependencies:

```sh
pip install warp-lang[examples]
```

For extra packages used by many examples:

```sh
pip install matplotlib pillow scipy
pip install usd-core         # x86-64 Linux/Windows
pip install usd-exchange     # aarch64 Linux (e.g., Jetson, Grace Hopper)
pip install "pyglet>=2.0"    # OpenGL renderer (example_render_opengl, streamlines)
```

When using `uv` for ad-hoc runs without installing:

```sh
uv run --with matplotlib --with Pillow --with usd-exchange --with scipy python -m warp.examples.fem.example_diffusion
```

## Nightly builds

From NVIDIA package index:

```sh
uv run pip install -U --pre warp-lang --extra-index-url=https://pypi.nvidia.com/
```

## Conda

Community package:

```sh
uv run conda install conda-forge::warp-lang
```

Pin a CUDA variant when needed:

```sh
uv run conda install conda-forge::warp-lang=*=*cuda126*
```

## GitHub releases wheels

Use the wheel matching OS/arch and CUDA variant in `warp-lang-X.Y.Z+cu13-...`.
Install with pip and the direct URL from the releases page:

```sh
uv run pip install https://github.com/NVIDIA/warp/releases/download/v1.11.0/warp_lang-1.11.0+cu13-py3-none-manylinux_2_28_x86_64.whl
```

## Build from source

Useful for development and custom local changes. This project uses `uv` as the
standard runner (matches CI/CD).

```sh
uv run build_lib.py
```

Quick build for faster iteration (~2-4 minutes vs 10-20):

```sh
uv run build_lib.py --quick
```

The `--quick` flag compiles for minimal GPU architectures and disables CUDA forward
compatibility. Only use if your CUDA driver version >= the CUDA Toolkit version used
for the build. Check with `nvidia-smi` (driver) and the CUDA Toolkit path (set via
`WARP_CUDA_PATH`, `CUDA_HOME`, `CUDA_PATH`, or `which nvcc`).

Build requirements:

- Visual Studio 2019+ (Windows)
- GCC 9.4+
- CUDA Toolkit 12.0+
- Git LFS

## CPU-only usage

Warp works without an NVIDIA GPU. Pass `device="cpu"` to array constructors and
`wp.launch()` calls. GPU-specific features (CUDA graphs, tile operations, NanoVDB)
are unavailable in CPU mode.

```python
a = wp.zeros(n, dtype=float, device="cpu")
wp.launch(kernel, dim=n, inputs=[a], device="cpu")
```

## Example dependency matrix

| Category | Required packages |
|----------|------------------|
| Core (mesh, fluid, wave, DEM, SPH) | `usd-core` or `usd-exchange` (for `pxr`) |
| Visualization (raymarch, raycast, FFT) | `matplotlib` |
| Optimization (diffray, tile_mlp) | `matplotlib`, `pillow` |
| FEM examples | `matplotlib`, `scipy` |
| OpenGL renderer | `pyglet>=2.0` |
| Interop | `torch`, `jax`, or `cupy` (per example) |

## Headless rendering

For running examples without a display (CI, SSH, containers):

```sh
MPLBACKEND=Agg uv run python -m warp.examples.core.example_raymarch
```

For OpenGL examples, use `xvfb-run` to provide a virtual framebuffer:

```sh
xvfb-run -a uv run --with "pyglet>=2.0" python -m warp.examples.core.example_render_opengl
```

## Installation decision checklist

- GPU available? If no, use CPU mode with `device="cpu"`.
- Driver age. Pre-built packages require NVIDIA driver 525+.
- Do you need tile/math primitives using MathDx? Requires CUDA 12.6.3+.
- Architecture? Use `usd-exchange` instead of `usd-core` on aarch64 Linux.
- If using source builds, ensure build and runtime C++ runtimes are compatible.

## First-run sanity checks

Import and initialize once.

```python
import warp as wp

wp.init()
print(wp.get_device())
```

## Diagnostics and environment info

```python
wp.print_diagnostics()       # Full environment report
wp.get_devices()             # List all available devices
wp.get_cuda_devices()        # List CUDA devices only
wp.get_cuda_toolkit_version()  # Returns (major, minor) or None
wp.get_cuda_driver_version()   # Returns (major, minor) or None
```

Check `uv run python -m warp.tests` for test baseline if needed.

## Official references

- https://nvidia.github.io/warp/user_guide/installation.html
- https://nvidia.github.io/warp/
- https://github.com/NVIDIA/warp/releases

