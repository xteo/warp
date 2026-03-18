import warp as wp
import numpy as np
import time

wp.init()
wp.clear_kernel_cache()

device = "cuda:0"
N = 200_000
NUM_STEPS = 500
WARM_UP = 10

@wp.kernel
def compute_forces(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    forces: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    p = positions[i]
    v = velocities[i]
    # Gravity + simple drag
    f = wp.vec3(0.0, -9.81, 0.0) - v * 0.1
    # Repulsion from ground plane
    if p[1] < 0.5:
        f = f + wp.vec3(0.0, (0.5 - p[1]) * 500.0, 0.0)
    forces[i] = f

@wp.kernel
def integrate(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    forces: wp.array(dtype=wp.vec3),
    dt: float,
):
    i = wp.tid()
    v = velocities[i] + forces[i] * dt
    p = positions[i] + v * dt
    velocities[i] = v
    positions[i] = p

@wp.kernel
def apply_bounds(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    p = positions[i]
    v = velocities[i]
    # Floor bounce
    if p[1] < 0.0:
        positions[i] = wp.vec3(p[0], 0.0, p[2])
        velocities[i] = wp.vec3(v[0], wp.abs(v[1]) * 0.5, v[2])
    # Box bounds [-10, 10]
    for d in range(3):
        if p[d] < -10.0:
            p = wp.vec3(wp.max(p[0], -10.0), wp.max(p[1], -10.0), wp.max(p[2], -10.0))
            positions[i] = p
            velocities[i] = wp.vec3()

DT = 0.001

# Initialize
np.random.seed(42)
pos_np = np.random.uniform(-5, 5, (N, 3)).astype(np.float32)
pos_np[:, 1] = np.abs(pos_np[:, 1]) + 1.0  # Start above ground
vel_np = np.random.uniform(-1, 1, (N, 3)).astype(np.float32)

# === BASELINE ===
positions = wp.array(pos_np, dtype=wp.vec3, device=device)
velocities = wp.array(vel_np, dtype=wp.vec3, device=device)
forces = wp.zeros(N, dtype=wp.vec3, device=device)

# Warm up
for _ in range(WARM_UP):
    wp.launch(compute_forces, dim=N, inputs=[positions, velocities, forces], device=device)
    wp.launch(integrate, dim=N, inputs=[positions, velocities, forces, DT], device=device)
    wp.launch(apply_bounds, dim=N, inputs=[positions, velocities], device=device)
wp.synchronize()

# Reset
positions = wp.array(pos_np, dtype=wp.vec3, device=device)
velocities = wp.array(vel_np, dtype=wp.vec3, device=device)

start = time.perf_counter()
for _ in range(NUM_STEPS):
    wp.launch(compute_forces, dim=N, inputs=[positions, velocities, forces], device=device)
    wp.launch(integrate, dim=N, inputs=[positions, velocities, forces, DT], device=device)
    wp.launch(apply_bounds, dim=N, inputs=[positions, velocities], device=device)
wp.synchronize()
baseline_time = time.perf_counter() - start
baseline_checksum = positions.numpy().sum()
print(f"Baseline: {baseline_time*1000:.1f} ms ({NUM_STEPS} steps, {N} particles)")
print(f"Baseline checksum: {baseline_checksum:.6f}")

# === GRAPH CAPTURED ===
positions_g = wp.array(pos_np, dtype=wp.vec3, device=device)
velocities_g = wp.array(vel_np, dtype=wp.vec3, device=device)
forces_g = wp.zeros(N, dtype=wp.vec3, device=device)

# Warm up
for _ in range(WARM_UP):
    wp.launch(compute_forces, dim=N, inputs=[positions_g, velocities_g, forces_g], device=device)
    wp.launch(integrate, dim=N, inputs=[positions_g, velocities_g, forces_g, DT], device=device)
    wp.launch(apply_bounds, dim=N, inputs=[positions_g, velocities_g], device=device)
wp.synchronize()

# Reset
positions_g = wp.array(pos_np, dtype=wp.vec3, device=device)
velocities_g = wp.array(vel_np, dtype=wp.vec3, device=device)
forces_g = wp.zeros(N, dtype=wp.vec3, device=device)

# Capture
wp.capture_begin(device=device)
wp.launch(compute_forces, dim=N, inputs=[positions_g, velocities_g, forces_g], device=device)
wp.launch(integrate, dim=N, inputs=[positions_g, velocities_g, forces_g, DT], device=device)
wp.launch(apply_bounds, dim=N, inputs=[positions_g, velocities_g], device=device)
graph = wp.capture_end(device=device)

start = time.perf_counter()
for _ in range(NUM_STEPS):
    wp.capture_launch(graph)
wp.synchronize()
graph_time = time.perf_counter() - start
graph_checksum = positions_g.numpy().sum()
print(f"\nGraph: {graph_time*1000:.1f} ms ({NUM_STEPS} steps)")
print(f"Graph checksum: {graph_checksum:.6f}")

speedup = (baseline_time - graph_time) / baseline_time * 100
checksums_ok = abs(baseline_checksum - graph_checksum) < abs(baseline_checksum) * 0.01
print(f"\nSpeedup: {speedup:.1f}%")
print(f"Checksums match: {checksums_ok}")

# === MULTI-STEP GRAPH (10 steps per graph) ===
STEPS_PER_GRAPH = 10
positions_m = wp.array(pos_np, dtype=wp.vec3, device=device)
velocities_m = wp.array(vel_np, dtype=wp.vec3, device=device)
forces_m = wp.zeros(N, dtype=wp.vec3, device=device)

# Warm up
for _ in range(WARM_UP):
    wp.launch(compute_forces, dim=N, inputs=[positions_m, velocities_m, forces_m], device=device)
    wp.launch(integrate, dim=N, inputs=[positions_m, velocities_m, forces_m, DT], device=device)
    wp.launch(apply_bounds, dim=N, inputs=[positions_m, velocities_m], device=device)
wp.synchronize()

# Reset
positions_m = wp.array(pos_np, dtype=wp.vec3, device=device)
velocities_m = wp.array(vel_np, dtype=wp.vec3, device=device)
forces_m = wp.zeros(N, dtype=wp.vec3, device=device)

# Capture 10 steps in one graph
wp.capture_begin(device=device)
for _ in range(STEPS_PER_GRAPH):
    wp.launch(compute_forces, dim=N, inputs=[positions_m, velocities_m, forces_m], device=device)
    wp.launch(integrate, dim=N, inputs=[positions_m, velocities_m, forces_m, DT], device=device)
    wp.launch(apply_bounds, dim=N, inputs=[positions_m, velocities_m], device=device)
multi_graph = wp.capture_end(device=device)

num_replays = NUM_STEPS // STEPS_PER_GRAPH
start = time.perf_counter()
for _ in range(num_replays):
    wp.capture_launch(multi_graph)
wp.synchronize()
multi_time = time.perf_counter() - start
multi_checksum = positions_m.numpy().sum()
print(f"\nMulti-step Graph ({STEPS_PER_GRAPH} steps/graph): {multi_time*1000:.1f} ms ({NUM_STEPS} steps)")
print(f"Multi-step checksum: {multi_checksum:.6f}")

multi_speedup = (baseline_time - multi_time) / baseline_time * 100
multi_checksums_ok = abs(baseline_checksum - multi_checksum) < abs(baseline_checksum) * 0.01
print(f"Multi-step speedup: {multi_speedup:.1f}%")
print(f"Multi-step checksums match: {multi_checksums_ok}")

print("\n=== SUMMARY ===")
print(f"Baseline:          {baseline_time*1000:.1f} ms")
print(f"Single-step graph: {graph_time*1000:.1f} ms  ({speedup:.1f}% speedup)")
print(f"Multi-step graph:  {multi_time*1000:.1f} ms  ({multi_speedup:.1f}% speedup)")
