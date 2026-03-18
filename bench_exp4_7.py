import warp as wp
import numpy as np
import time


class Cloth:
    def __init__(self, lower, dx, dy, radius, stretch_stiffness, bend_stiffness, shear_stiffness, mass, fix_corners=True):
        self.triangles = []
        self.positions = []
        self.velocities = []
        self.inv_masses = []
        self.spring_indices = []
        self.spring_lengths = []
        self.spring_stiffness = []
        self.spring_damping = []

        def grid(x, y, stride):
            return y * stride + x

        def create_spring(i, j, stiffness, damp=10.0):
            length = np.linalg.norm(np.array(self.positions[i]) - np.array(self.positions[j]))
            self.spring_indices.append(i)
            self.spring_indices.append(j)
            self.spring_lengths.append(length)
            self.spring_stiffness.append(stiffness)
            self.spring_damping.append(damp)

        for y in range(dy):
            for x in range(dx):
                p = np.array(lower) + radius * np.array((float(x), float(0.0), float(y)))
                self.positions.append(p)
                self.velocities.append(np.zeros(3))
                if x > 0 and y > 0:
                    self.triangles.append(grid(x - 1, y - 1, dx))
                    self.triangles.append(grid(x, y - 1, dx))
                    self.triangles.append(grid(x, y, dx))
                    self.triangles.append(grid(x - 1, y - 1, dx))
                    self.triangles.append(grid(x, y, dx))
                    self.triangles.append(grid(x - 1, y, dx))
                if fix_corners and y == 0 and (x == 0 or x == dx - 1):
                    w = 0.0
                else:
                    w = 1.0 / mass
                self.inv_masses.append(w)

        for y in range(dy):
            for x in range(dx):
                index0 = y * dx + x
                if x > 0:
                    index1 = y * dx + x - 1
                    create_spring(index0, index1, stretch_stiffness)
                if x > 1 and bend_stiffness > 0.0:
                    index2 = y * dx + x - 2
                    create_spring(index0, index2, bend_stiffness)
                if y > 0 and x < dx - 1 and shear_stiffness > 0.0:
                    indexDiag = (y - 1) * dx + x + 1
                    create_spring(index0, indexDiag, shear_stiffness)
                if y > 0 and x > 0 and shear_stiffness > 0.0:
                    indexDiag = (y - 1) * dx + x - 1
                    create_spring(index0, indexDiag, shear_stiffness)

        for x in range(dx):
            for y in range(dy):
                index0 = y * dx + x
                if y > 0:
                    index1 = (y - 1) * dx + x
                    create_spring(index0, index1, stretch_stiffness)
                if y > 1 and bend_stiffness > 0.0:
                    index2 = (y - 2) * dx + x
                    create_spring(index0, index2, bend_stiffness)

        self.positions = np.array(self.positions, dtype=np.float32)
        self.velocities = np.array(self.velocities, dtype=np.float32)
        self.inv_masses = np.array(self.inv_masses, dtype=np.float32)
        self.spring_lengths = np.array(self.spring_lengths, dtype=np.float32)
        self.spring_indices = np.array(self.spring_indices, dtype=np.int32)
        self.spring_stiffness = np.array(self.spring_stiffness, dtype=np.float32)
        self.spring_damping = np.array(self.spring_damping, dtype=np.float32)
        self.num_particles = len(self.positions)
        self.num_springs = len(self.spring_lengths)

wp.init()
wp.clear_kernel_cache()

device = "cuda:0"

@wp.kernel
def eval_springs(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    spring_indices: wp.array(dtype=wp.vec2i),
    spring_params: wp.array(dtype=wp.vec3),
    f: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    idx = spring_indices[tid]
    i = idx[0]
    j = idx[1]
    params = spring_params[tid]
    rest = params[0]
    ke = params[1]
    kd = params[2]
    xi = x[i]
    xj = x[j]
    xij = xi - xj
    l = wp.length(xij)
    l_inv = 1.0 / l
    dir = xij * l_inv
    c = l - rest
    vi = v[i]
    vj = v[j]
    dcdt = wp.dot(dir, vi - vj)
    fs = dir * (ke * c + kd * dcdt)
    wp.atomic_sub(f, i, fs)
    wp.atomic_add(f, j, fs)

@wp.kernel
def integrate_particles(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    f: wp.array(dtype=wp.vec3),
    w: wp.array(dtype=float),
    dt: float,
):
    tid = wp.tid()
    x0 = x[tid]
    v0 = v[tid]
    f0 = f[tid]
    inv_mass = w[tid]
    g = wp.vec3()
    if inv_mass > 0.0:
        g = wp.vec3(0.0, 0.0 - 9.81, 0.0)
    v1 = v0 + (f0 * inv_mass + g) * dt
    x1 = x0 + v1 * dt
    x[tid] = x1
    v[tid] = v1
    f[tid] = wp.vec3()

# Setup cloth
cloth = Cloth(
    lower=(-1.0, 2.0, -1.0),
    dx=128, dy=128,
    radius=0.02,
    stretch_stiffness=10000.0,
    bend_stiffness=100.0,
    shear_stiffness=100.0,
    mass=0.1,
)

positions = wp.array(cloth.positions, dtype=wp.vec3, device=device)
velocities = wp.zeros(cloth.num_particles, dtype=wp.vec3, device=device)
forces = wp.zeros(cloth.num_particles, dtype=wp.vec3, device=device)
inv_masses = wp.array(cloth.inv_masses, dtype=float, device=device)

# Pack springs
indices_packed = cloth.spring_indices.reshape(-1, 2)
spring_indices = wp.array(indices_packed, dtype=wp.vec2i, device=device)
params_packed = np.stack([cloth.spring_lengths, cloth.spring_stiffness, cloth.spring_damping], axis=1).astype(np.float32)
spring_params = wp.array(params_packed, dtype=wp.vec3, device=device)

num_springs = len(cloth.spring_lengths)
num_particles = cloth.num_particles
sim_dt = 1.0/60.0 / 32  # 32 substeps
NUM_SUBSTEPS = 32
NUM_FRAMES = 50

# === BASELINE: individual launches ===
positions_baseline = wp.array(cloth.positions, dtype=wp.vec3, device=device)
velocities_baseline = wp.zeros(num_particles, dtype=wp.vec3, device=device)
forces_baseline = wp.zeros(num_particles, dtype=wp.vec3, device=device)

# Warm up
for _ in range(3):
    for s in range(NUM_SUBSTEPS):
        wp.launch(eval_springs, dim=num_springs, inputs=[positions_baseline, velocities_baseline, spring_indices, spring_params, forces_baseline], device=device)
        wp.launch(integrate_particles, dim=num_particles, inputs=[positions_baseline, velocities_baseline, forces_baseline, inv_masses, sim_dt], device=device)
wp.synchronize()

# Reset state after warmup
positions_baseline = wp.array(cloth.positions, dtype=wp.vec3, device=device)
velocities_baseline = wp.zeros(num_particles, dtype=wp.vec3, device=device)
forces_baseline = wp.zeros(num_particles, dtype=wp.vec3, device=device)

start = time.perf_counter()
for frame in range(NUM_FRAMES):
    for s in range(NUM_SUBSTEPS):
        wp.launch(eval_springs, dim=num_springs, inputs=[positions_baseline, velocities_baseline, spring_indices, spring_params, forces_baseline], device=device)
        wp.launch(integrate_particles, dim=num_particles, inputs=[positions_baseline, velocities_baseline, forces_baseline, inv_masses, sim_dt], device=device)
wp.synchronize()
baseline_time = time.perf_counter() - start
baseline_checksum = positions_baseline.numpy().sum()
print(f"Baseline: {baseline_time*1000:.1f} ms ({NUM_FRAMES} frames x {NUM_SUBSTEPS} substeps)")
print(f"Baseline checksum: {baseline_checksum:.6f}")

# === OPTIMIZED: CUDA graph capture ===
positions_graph = wp.array(cloth.positions, dtype=wp.vec3, device=device)
velocities_graph = wp.zeros(num_particles, dtype=wp.vec3, device=device)
forces_graph = wp.zeros(num_particles, dtype=wp.vec3, device=device)

# Warm up (ensures kernels are compiled)
for _ in range(3):
    for s in range(NUM_SUBSTEPS):
        wp.launch(eval_springs, dim=num_springs, inputs=[positions_graph, velocities_graph, spring_indices, spring_params, forces_graph], device=device)
        wp.launch(integrate_particles, dim=num_particles, inputs=[positions_graph, velocities_graph, forces_graph, inv_masses, sim_dt], device=device)
wp.synchronize()

# Reset for graph capture
positions_graph = wp.array(cloth.positions, dtype=wp.vec3, device=device)
velocities_graph = wp.zeros(num_particles, dtype=wp.vec3, device=device)
forces_graph = wp.zeros(num_particles, dtype=wp.vec3, device=device)

# Capture one frame (all substeps)
wp.capture_begin(device=device)
for s in range(NUM_SUBSTEPS):
    wp.launch(eval_springs, dim=num_springs, inputs=[positions_graph, velocities_graph, spring_indices, spring_params, forces_graph], device=device)
    wp.launch(integrate_particles, dim=num_particles, inputs=[positions_graph, velocities_graph, forces_graph, inv_masses, sim_dt], device=device)
graph = wp.capture_end(device=device)

start = time.perf_counter()
for frame in range(NUM_FRAMES):
    wp.capture_launch(graph)
wp.synchronize()
graph_time = time.perf_counter() - start
graph_checksum = positions_graph.numpy().sum()
print(f"\nGraph: {graph_time*1000:.1f} ms ({NUM_FRAMES} frames)")
print(f"Graph checksum: {graph_checksum:.6f}")

speedup = (baseline_time - graph_time) / baseline_time * 100
checksums_match = abs(baseline_checksum - graph_checksum) < 1e-2
print(f"\nSpeedup: {speedup:.1f}%")
print(f"Checksums match: {checksums_match}")
