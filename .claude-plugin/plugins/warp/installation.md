# Warp Installation

This page summarizes practical install and runtime options from official docs.

## Supported platforms and package targets

Warp supports:

- x86-64 and ARMv8 on Linux and Windows
- Apple Silicon macOS (ARM64) for supported versions

GPU support requires CUDA-capable NVIDIA hardware and compatible driver/toolkit.

## Quick install (recommended)

From PyPI with default driver/runtime compatibility:

```sh
uv run pip install warp-lang
```

If you need example dependencies:

```sh
uv run pip install warp-lang[examples]
```

For extra packages used by many examples:

```sh
uv run pip install usd-core matplotlib pyglet
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

Useful for development and custom local changes.

```sh
uv run python build_lib.py
uv run pip install -e .
```

Build requirements:

- Visual Studio 2019+ (Windows)
- GCC 9.4+
- CUDA Toolkit 12.0+
- Git LFS

## Installation decision checklist

- GPU available? If no, use CPU mode only.
- Driver age. Pre-built packages require modern NVIDIA drivers.
- Do you need tile/math primitives using MathDx? Requires CUDA 12.6.3+.
- If using source builds, ensure build and runtime C++ runtimes are compatible.

## First-run sanity checks

Import and initialize once.

```python
import warp as wp

wp.init()
print(wp.get_device())
```

Check `uv run python -m warp.tests` for test baseline if needed.

## Official references

- https://nvidia.github.io/warp/user_guide/installation.html
- https://nvidia.github.io/warp/
- https://github.com/NVIDIA/warp/releases

