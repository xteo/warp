"""
Experiment 4-3: Skip adjoint generation for forward-only kernels
Measures NVRTC compile time reduction when enable_backward=False.
"""

import os
import sys
import time
import json
import shutil
import numpy as np

os.environ["CUDA_HOME"] = os.path.expanduser("~/.local/cuda-12.8")

import warp as wp

wp.init()

N = 10000


# ============================================================
# Non-trivial kernel: cloth spring force evaluation
# Two copies: one with backward, one forward-only
# ============================================================

@wp.kernel(enable_backward=True, module="unique")
def cloth_springs_backward(
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    rest_length: wp.array(dtype=float),
    spring_idx_a: wp.array(dtype=int),
    spring_idx_b: wp.array(dtype=int),
    ks: float,
    kd: float,
    dt: float,
    forces: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    ia = spring_idx_a[tid]
    ib = spring_idx_b[tid]
    pa = pos[ia]
    pb = pos[ib]
    va = vel[ia]
    vb = vel[ib]
    delta = pb - pa
    length = wp.length(delta)
    rest = rest_length[tid]
    if length > 1.0e-6:
        direction = delta / length
        stretch = length - rest
        f_spring = direction * (ks * stretch)
        rel_vel = vb - va
        damping = wp.dot(rel_vel, direction)
        f_damp = direction * (kd * damping)
        f_total = f_spring + f_damp
        wp.atomic_add(forces, ia, f_total)
        wp.atomic_sub(forces, ib, f_total)


@wp.kernel(enable_backward=False, module="unique")
def cloth_springs_forward_only(
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    rest_length: wp.array(dtype=float),
    spring_idx_a: wp.array(dtype=int),
    spring_idx_b: wp.array(dtype=int),
    ks: float,
    kd: float,
    dt: float,
    forces: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    ia = spring_idx_a[tid]
    ib = spring_idx_b[tid]
    pa = pos[ia]
    pb = pos[ib]
    va = vel[ia]
    vb = vel[ib]
    delta = pb - pa
    length = wp.length(delta)
    rest = rest_length[tid]
    if length > 1.0e-6:
        direction = delta / length
        stretch = length - rest
        f_spring = direction * (ks * stretch)
        rel_vel = vb - va
        damping = wp.dot(rel_vel, direction)
        f_damp = direction * (kd * damping)
        f_total = f_spring + f_damp
        wp.atomic_add(forces, ia, f_total)
        wp.atomic_sub(forces, ib, f_total)


# Deterministic per-element kernel for correctness check (no atomics)
@wp.kernel(enable_backward=True, module="unique")
def spring_force_per_element_bwd(
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    rest_length: wp.array(dtype=float),
    spring_idx_a: wp.array(dtype=int),
    spring_idx_b: wp.array(dtype=int),
    ks: float,
    kd: float,
    out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    ia = spring_idx_a[tid]
    ib = spring_idx_b[tid]
    pa = pos[ia]
    pb = pos[ib]
    va = vel[ia]
    vb = vel[ib]
    delta = pb - pa
    length = wp.length(delta)
    rest = rest_length[tid]
    f = wp.vec3(0.0, 0.0, 0.0)
    if length > 1.0e-6:
        direction = delta / length
        stretch = length - rest
        f_spring = direction * (ks * stretch)
        rel_vel = vb - va
        damping = wp.dot(rel_vel, direction)
        f_damp = direction * (kd * damping)
        f = f_spring + f_damp
    out[tid] = f


@wp.kernel(enable_backward=False, module="unique")
def spring_force_per_element_fwd(
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    rest_length: wp.array(dtype=float),
    spring_idx_a: wp.array(dtype=int),
    spring_idx_b: wp.array(dtype=int),
    ks: float,
    kd: float,
    out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    ia = spring_idx_a[tid]
    ib = spring_idx_b[tid]
    pa = pos[ia]
    pb = pos[ib]
    va = vel[ia]
    vb = vel[ib]
    delta = pb - pa
    length = wp.length(delta)
    rest = rest_length[tid]
    f = wp.vec3(0.0, 0.0, 0.0)
    if length > 1.0e-6:
        direction = delta / length
        stretch = length - rest
        f_spring = direction * (ks * stretch)
        rel_vel = vb - va
        damping = wp.dot(rel_vel, direction)
        f_damp = direction * (kd * damping)
        f = f_spring + f_damp
    out[tid] = f


def setup_data(device):
    rng = np.random.default_rng(42)
    num_particles = 1000
    num_springs = N
    positions = rng.random((num_particles, 3)).astype(np.float32)
    velocities = rng.random((num_particles, 3)).astype(np.float32) * 0.1
    idx_a = rng.integers(0, num_particles, size=num_springs).astype(np.int32)
    idx_b = rng.integers(0, num_particles, size=num_springs).astype(np.int32)
    rest_lengths = rng.random(num_springs).astype(np.float32) * 0.5 + 0.1
    pos = wp.array(positions, dtype=wp.vec3, device=device)
    vel = wp.array(velocities, dtype=wp.vec3, device=device)
    rest_len = wp.array(rest_lengths, dtype=float, device=device)
    spring_a = wp.array(idx_a, dtype=int, device=device)
    spring_b = wp.array(idx_b, dtype=int, device=device)
    return pos, vel, rest_len, spring_a, spring_b


def clear_module_cache(kernel):
    module = kernel.module
    module.loaded_modules = {}
    module.cuda_modules = {}
    module.hash_module = None


def measure_compile_time(kernel_fn, pos, vel, rest_len, spring_a, spring_b, device, label):
    clear_module_cache(kernel_fn)
    forces = wp.zeros(pos.shape[0], dtype=wp.vec3, device=device)
    wp.synchronize()
    t0 = time.perf_counter()
    wp.launch(
        kernel_fn, dim=N,
        inputs=[pos, vel, rest_len, spring_a, spring_b, 100.0, 1.0, 0.01],
        outputs=[forces], device=device,
    )
    wp.synchronize()
    t1 = time.perf_counter()
    compile_ms = (t1 - t0) * 1000.0
    print(f"  {label}: compile+first-launch = {compile_ms:.2f} ms")
    return forces, compile_ms


def measure_runtime(kernel_fn, pos, vel, rest_len, spring_a, spring_b, device, label, iterations=200):
    """Measure runtime performance (kernel already compiled from compile phase)."""
    forces = wp.zeros(pos.shape[0], dtype=wp.vec3, device=device)
    wp.synchronize()

    # Warmup (kernel already compiled, just warming GPU caches)
    for _ in range(10):
        forces.zero_()
        wp.launch(
            kernel_fn, dim=N,
            inputs=[pos, vel, rest_len, spring_a, spring_b, 100.0, 1.0, 0.01],
            outputs=[forces], device=device,
        )
    wp.synchronize()

    # Timed
    t0 = time.perf_counter()
    for _ in range(iterations):
        forces.zero_()
        wp.launch(
            kernel_fn, dim=N,
            inputs=[pos, vel, rest_len, spring_a, spring_b, 100.0, 1.0, 0.01],
            outputs=[forces], device=device,
        )
    wp.synchronize()
    t1 = time.perf_counter()
    runtime_us = (t1 - t0) * 1e6 / iterations
    print(f"  {label}: runtime = {runtime_us:.2f} us/iter")
    return forces, runtime_us


def measure_correctness(pos, vel, rest_len, spring_a, spring_b, device):
    """Use deterministic per-element kernels to verify correctness."""
    out_bwd = wp.zeros(N, dtype=wp.vec3, device=device)
    out_fwd = wp.zeros(N, dtype=wp.vec3, device=device)

    wp.launch(
        spring_force_per_element_bwd, dim=N,
        inputs=[pos, vel, rest_len, spring_a, spring_b, 100.0, 1.0],
        outputs=[out_bwd], device=device,
    )
    wp.launch(
        spring_force_per_element_fwd, dim=N,
        inputs=[pos, vel, rest_len, spring_a, spring_b, 100.0, 1.0],
        outputs=[out_fwd], device=device,
    )
    wp.synchronize()

    a = out_bwd.numpy()
    b = out_fwd.numpy()
    max_diff = float(np.max(np.abs(a - b)))
    match = max_diff < 1e-5
    checksum_a = float(np.sum(a))
    checksum_b = float(np.sum(b))
    print(f"  Max diff: {max_diff:.2e}, checksums: {checksum_a:.6f} vs {checksum_b:.6f}, match: {match}")
    return match


def main():
    device = "cuda:0"
    print(f"Warp version: {wp.__version__}")
    print(f"Device: {device}")
    print(f"Springs: {N}")
    print()

    pos, vel, rest_len, spring_a, spring_b = setup_data(device)

    # === Phase 1: Cold compile time ===
    print("=== Phase 1: Cold Compilation Time ===")
    forces_bwd, compile_bwd_ms = measure_compile_time(
        cloth_springs_backward, pos, vel, rest_len, spring_a, spring_b, device, "enable_backward=True"
    )
    forces_fwd, compile_fwd_ms = measure_compile_time(
        cloth_springs_forward_only, pos, vel, rest_len, spring_a, spring_b, device, "enable_backward=False"
    )

    # === Phase 2: Runtime Performance ===
    print("\n=== Phase 2: Runtime Performance ===")
    _, runtime_bwd_us = measure_runtime(
        cloth_springs_backward, pos, vel, rest_len, spring_a, spring_b, device, "enable_backward=True"
    )
    _, runtime_fwd_us = measure_runtime(
        cloth_springs_forward_only, pos, vel, rest_len, spring_a, spring_b, device, "enable_backward=False"
    )

    # === Phase 3: Correctness ===
    print("\n=== Phase 3: Correctness Check (deterministic per-element kernel) ===")
    correct = measure_correctness(pos, vel, rest_len, spring_a, spring_b, device)

    # === Summary ===
    print("\n" + "=" * 60)
    print("EXPERIMENT 4-3: Skip Adjoint for Forward-Only Kernels")
    print("=" * 60)

    compile_speedup = ((compile_bwd_ms - compile_fwd_ms) / compile_bwd_ms) * 100.0 if compile_bwd_ms > 0 else 0
    runtime_diff = ((runtime_bwd_us - runtime_fwd_us) / runtime_bwd_us) * 100.0 if runtime_bwd_us > 0 else 0

    print(f"  Baseline compile (backward=True):  {compile_bwd_ms:.2f} ms")
    print(f"  Optimized compile (backward=False): {compile_fwd_ms:.2f} ms")
    print(f"  Compile speedup: {compile_speedup:.1f}%")
    print()
    print(f"  Baseline runtime:  {runtime_bwd_us:.2f} us/iter")
    print(f"  Optimized runtime: {runtime_fwd_us:.2f} us/iter")
    print(f"  Runtime diff: {runtime_diff:.1f}% (expected ~0, forward code is identical)")
    print()
    print(f"  Correctness: {'PASS' if correct else 'FAIL'}")

    status = "winner" if compile_speedup > 5.0 and correct else "negative"
    print(f"  Status: {status}")
    print()

    result = {
        "experiment_name": "exp4-3-skip-adjoint",
        "baseline_compile_ms": round(float(compile_bwd_ms), 2),
        "optimized_compile_ms": round(float(compile_fwd_ms), 2),
        "compile_speedup_pct": round(float(compile_speedup), 1),
        "baseline_runtime_us": round(float(runtime_bwd_us), 2),
        "optimized_runtime_us": round(float(runtime_fwd_us), 2),
        "runtime_diff_pct": round(float(runtime_diff), 1),
        "checksums_ok": bool(correct),
        "status": status,
        "description": "Skipping adjoint code generation for forward-only kernels reduces NVRTC compile time by avoiding backward pass codegen and compilation.",
    }
    print("RESULT_JSON:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
