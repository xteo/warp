# Warp Troubleshooting

## Common first issues

### 1) CUDA driver mismatch

Symptom:

- `Warp UserWarning: Insufficient CUDA driver version`

Fix:

- Update driver
- Install wheel matching your driver/runtime matrix
- Build from source using compatible CUDA Toolkit

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

### 7) Async timing confusion

Symptom:

- program appears to hang or stale results after launch.

Fix:

- call `wp.synchronize_device()` around timing-critical sections.
- prefer `wp.synchronize_stream()` for stream-specific debugging.

## Debugging playbook

1. Start with simple kernel in `example_torch.py`-style minimal script.
2. Add `wp.config.quiet = False` for startup prints.
3. Enable debug-mode checks in `wp.config` when diagnosing memory issues.
4. Use `wp.ScopedTimer` for hot sections.
5. Inspect generated code in kernel cache (path shown by `wp.init()`).

## Debugging references

- `docs/user_guide/debugging.rst`
- `docs/user_guide/faq.rst`
- `docs/user_guide/limitations.rst`

## Quick fixes checklist

- Use matching Python/Numpy versions.
- Prefer `wp.zeros`/`wp.empty_like` for buffers.
- Reinitialize caches with `wp.clear_kernel_cache()` if stale binaries are suspected.
- Rebuild with minimal reproducible script.

## Web references

- https://nvidia.github.io/warp/user_guide/debugging.html
- https://nvidia.github.io/warp/user_guide/faq.html
- https://github.com/NVIDIA/warp/issues
