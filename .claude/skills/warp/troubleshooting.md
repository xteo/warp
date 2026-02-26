# Warp Troubleshooting

## Common first issues

### 1) CUDA driver mismatch

Symptom:

- `Warp UserWarning: Insufficient CUDA driver version`

Fix:

- Check your current driver version with `nvidia-smi`
- Update driver to 525.60.13+ (Linux) or 528.33+ (Windows)
- Or install a wheel matching your driver/runtime matrix
- Or build from source using a compatible CUDA Toolkit

See docs: `docs/user_guide/installation.rst` CUDA Requirements.

### 2) Device mismatch errors

Symptom:

- runtime exception about arrays on different devices.

Fix:

- Ensure all kernel arguments live on same device.
- Keep source and destination array intent explicit.

### 3) Wrong thread-space assumptions

Symptom:

- stale values or out-of-range indexing.

Fix:

- Validate `dim` against array length.
- Include boundary checks in kernels for 2D/3D launch bounds.

### 4) Mesh/bvh object crash

Symptom:

- random crashes after several frames using mesh queries.

Fix:

- Keep references to `wp.Mesh`, `wp.HashGrid`, `wp.Bvh`, `wp.Volume` alive in Python scope.
- Avoid storing only ids (`mesh.id`) in lists; store object instances.

### 5) Empty/uninitialized outputs in kernels

Symptom:

- outputs remain zero or NaN.

Fix:

- Check kernel launch `dim` and pointer ordering in `inputs`/`outputs`.
- Confirm all arrays are writable where expected.

### 6) Build fails on custom environment

Symptom:

- compile failures during import.

Fix:

- confirm `python build_lib.py` prerequisites
- verify C++ runtime compatibility for runtime Conda env
- on Linux compare `libstdc++` version constraints in installation docs

### 7) Missing `pxr` module (USD)

Symptom:

- `ModuleNotFoundError: No module named 'pxr'`
- `ImportError: Failed to import pxr`

Fix:

- On x86-64: `pip install usd-core`
- On aarch64 Linux: `pip install usd-exchange` (usd-core is not available for ARM)
- Many core examples (mesh, wave, DEM, SPH, marching_cubes, nvdb, sample_mesh) require `pxr` for USD rendering output
- Pass `--stage-path None` to disable USD output when possible

### 8) OpenGL renderer fails

Symptom:

- `OpenGLRenderer requires pyglet (version >= 2.0) to be installed`
- `Cannot connect to "None"` (no display)
- `CUDA error 999` in `wp_cuda_graphics_register_gl_buffer`

Fix:

- Install pyglet: `pip install "pyglet>=2.0"`
- For headless environments, use `xvfb-run -a --server-args="-screen 0 1920x1080x24"` to provide a virtual display
- CUDA-GL interop may fail with software OpenGL (Xvfb); Warp falls back to CPU copy mode automatically
- When calling `renderer.get_pixels()` without tiled rendering, pass `split_up_tiles=False`
- The headless OpenGL path creates a real pyglet window in the virtual framebuffer

### 9) Matplotlib not showing plots

Symptom:

- `plt.show()` hangs or does nothing in headless/SSH environments

Fix:

- Set `MPLBACKEND=Agg` environment variable before running
- Or in Python: `import matplotlib; matplotlib.use("Agg")` before importing pyplot
- Use `plt.savefig()` to save to file instead of displaying
- Most examples accept `--headless` to skip visualization entirely

### 10) Async timing confusion

Symptom:

- program appears to hang or stale results after launch.

Fix:

- `.numpy()` implicitly synchronizes — no need to sync before readback.
- Use `wp.synchronize_device()` (not `wp.synchronize()`) for explicit sync.
- `wp.synchronize()` syncs ALL devices which is rarely desired.
- Use `wp.synchronize_stream()` for stream-specific debugging.

## Debug configuration flags

### Debug mode with bounds checking

```python
wp.config.mode = "debug"  # Enables array bounds checking, wp.tid() overflow warnings
```

Also enables `wp.breakpoint()` support inside kernels for attaching debuggers.

### NaN/Inf detection

```python
wp.config.verify_fp = True  # Checks floating-point values before/after operations
```

### CUDA error checking

```python
wp.config.verify_cuda = True  # Checks for CUDA errors after each kernel launch
```

### Autograd array access verification

```python
wp.config.verify_autograd_array_access = True  # Flags array overwrites that break gradients
```

Use `warp.autograd.gradcheck()` to compare autodiff results against finite differences.

### Kernel print debugging

`print()` and `wp.printf()` work inside kernels:

```python
@wp.kernel
def debug_kernel(data: wp.array(dtype=float)):
    i = wp.tid()
    if i == 0:
        wp.printf("Value at 0: %f\n", data[0])
```

## Debugging playbook

1. Start with simple kernel in `example_torch.py`-style minimal script.
2. Add `wp.config.quiet = False` for startup prints.
3. Set `wp.config.mode = "debug"` for bounds checking.
4. Set `wp.config.verify_fp = True` to catch NaN/Inf.
5. Set `wp.config.verify_cuda = True` for CUDA error checking.
6. Use `wp.ScopedTimer` for hot sections.
7. Inspect generated code in kernel cache (path shown by `wp.init()`).

## Debugging references

- `docs/user_guide/debugging.rst`
- `docs/user_guide/faq.rst`
- `docs/user_guide/limitations.rst`

## Quick fixes checklist

- Use matching Python/NumPy versions.
- Prefer `wp.zeros`/`wp.empty_like` for buffers.
- If stale compiled binaries are suspected, clear the kernel cache directory shown
  by `wp.init()` output. **Do NOT call `wp.clear_kernel_cache()` in test files** —
  it is not multi-process-safe and causes LLVM crashes in parallel test runs.
  The test runner and `build_lib.py` clear caches from a single process at the right times.
- Rebuild with minimal reproducible script.
- NEVER define `@wp.kernel` in `python -c "..."` — Warp's codegen uses
  `inspect.getsourcelines()` which fails for code not in a file.

## Web references

- https://nvidia.github.io/warp/user_guide/debugging.html
- https://nvidia.github.io/warp/user_guide/faq.html
- https://github.com/NVIDIA/warp/issues
