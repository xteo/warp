"""GPU N-body Galaxy Simulator with Tile Acceleration.

Gravitational N-body simulator that creates galaxy-like structures using
NVIDIA Warp. Uses tile-based force computation for performance, CUDA graph
capture for reduced launch overhead, and renders particles with velocity-mapped
colors and motion trails.

Usage:
    uv run --with matplotlib --with Pillow python galaxy_sim.py --preset spiral --num-frames 200
    uv run --with matplotlib --with Pillow python galaxy_sim.py --preset collision --num-frames 300
    uv run --with matplotlib --with Pillow python galaxy_sim.py --preset cluster --num-frames 200
"""

import argparse
import math
import os
import time

import numpy as np

import warp as wp

wp.init()

# ── Simulation Constants ──────────────────────────────────────────────────

TILE_SIZE = wp.constant(64)
SOFTENING_SQ = wp.constant(0.01)  # epsilon^2 to prevent singularities


# ── Warp Kernels ──────────────────────────────────────────────────────────


@wp.func
def body_body_interaction(p0: wp.vec3, m0: float, pi: wp.vec3, mi: float, G: float):
    """Gravitational acceleration on body at p0 due to body at pi."""
    r = pi - p0
    dist_sq = wp.length_sq(r) + SOFTENING_SQ
    inv_dist = 1.0 / wp.sqrt(dist_sq)
    inv_dist_cubed = inv_dist * inv_dist * inv_dist
    return G * mi * inv_dist_cubed * r


@wp.kernel
def compute_forces_tiled(
    positions: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=float),
    forces: wp.array(dtype=wp.vec3),
    num_bodies: int,
    G: float,
):
    """Tile-based O(N^2) gravitational force computation."""
    i = wp.tid()
    p0 = positions[i]
    m0 = masses[i]
    accel = wp.vec3(0.0, 0.0, 0.0)

    num_tiles = num_bodies / TILE_SIZE
    for k in range(num_tiles):
        pos_tile = wp.tile_load(positions, shape=TILE_SIZE, offset=k * TILE_SIZE)
        mass_tile = wp.tile_load(masses, shape=TILE_SIZE, offset=k * TILE_SIZE)
        for idx in range(TILE_SIZE):
            pi = pos_tile[idx]
            mi = mass_tile[idx]
            accel += body_body_interaction(p0, m0, pi, mi, G)

    forces[i] = accel


@wp.kernel
def compute_forces_naive(
    positions: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=float),
    forces: wp.array(dtype=wp.vec3),
    num_bodies: int,
    G: float,
):
    """Naive O(N^2) gravitational force computation for benchmarking."""
    i = wp.tid()
    p0 = positions[i]
    accel = wp.vec3(0.0, 0.0, 0.0)

    for j in range(num_bodies):
        r = positions[j] - p0
        dist_sq = wp.length_sq(r) + SOFTENING_SQ
        inv_dist = 1.0 / wp.sqrt(dist_sq)
        inv_dist_cubed = inv_dist * inv_dist * inv_dist
        accel += G * masses[j] * inv_dist_cubed * r

    forces[i] = accel


@wp.kernel
def integrate(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    forces: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=float),
    dt: float,
):
    """Leapfrog (kick-drift) integration."""
    i = wp.tid()
    v = velocities[i] + forces[i] * dt
    velocities[i] = v
    positions[i] = positions[i] + v * dt


@wp.kernel
def store_trail(
    positions: wp.array(dtype=wp.vec3),
    trail_buf: wp.array2d(dtype=wp.vec3),
    trail_idx: int,
):
    """Copy current positions into the circular trail buffer."""
    i = wp.tid()
    trail_buf[trail_idx, i] = positions[i]


# ── Initial Condition Generators ──────────────────────────────────────────


def init_spiral_galaxy(num_bodies, rng, center=np.zeros(3), bulk_vel=np.zeros(3),
                       central_mass=1000.0, disk_radius=5.0):
    """Generate a spiral galaxy disk with tangential velocities."""
    # Ensure num_bodies is multiple of TILE_SIZE
    n = num_bodies

    # Central massive body at index 0
    positions = np.zeros((n, 3), dtype=np.float32)
    velocities = np.zeros((n, 3), dtype=np.float32)
    masses = np.ones(n, dtype=np.float32)

    positions[0] = center
    masses[0] = central_mass
    velocities[0] = bulk_vel

    # Disk particles (indices 1..n-1)
    # Radial distribution: r ~ sqrt(uniform) for uniform surface density
    r = disk_radius * np.sqrt(rng.uniform(0.05, 1.0, size=n - 1))
    theta = rng.uniform(0, 2 * np.pi, n - 1)

    # Add spiral arm modulation
    num_arms = 2
    arm_phase = theta * num_arms + r * 1.5
    r_mod = r * (1.0 + 0.3 * np.sin(arm_phase))

    x = r_mod * np.cos(theta)
    y = r_mod * np.sin(theta)
    z = rng.normal(0, 0.05 * disk_radius, n - 1)  # slight vertical dispersion

    positions[1:, 0] = x + center[0]
    positions[1:, 1] = y + center[1]
    positions[1:, 2] = z + center[2]

    # Tangential velocity for circular orbits: v = sqrt(G * M_enc / r)
    G = 1.0
    v_circ = np.sqrt(G * central_mass / (r + 0.1))
    velocities[1:, 0] = -v_circ * np.sin(theta) + bulk_vel[0]
    velocities[1:, 1] = v_circ * np.cos(theta) + bulk_vel[1]
    velocities[1:, 2] = bulk_vel[2]

    # Add small random velocity dispersion
    velocities[1:] += rng.normal(0, 0.05, (n - 1, 3)).astype(np.float32)

    return positions, velocities, masses


def init_collision(num_bodies, rng):
    """Two spiral galaxies on a collision course."""
    n_each = num_bodies // 2
    # Round each to multiple of TILE_SIZE
    n_each = max((n_each // 64) * 64, 64)

    total = n_each * 2
    sep = 8.0

    pos1, vel1, mass1 = init_spiral_galaxy(
        n_each, rng,
        center=np.array([-sep / 2, 0.0, 0.0]),
        bulk_vel=np.array([2.0, 0.5, 0.0]),
        central_mass=500.0,
        disk_radius=3.0,
    )
    pos2, vel2, mass2 = init_spiral_galaxy(
        n_each, rng,
        center=np.array([sep / 2, 0.0, 0.0]),
        bulk_vel=np.array([-2.0, -0.5, 0.0]),
        central_mass=500.0,
        disk_radius=3.0,
    )

    positions = np.concatenate([pos1, pos2], axis=0)
    velocities = np.concatenate([vel1, vel2], axis=0)
    masses = np.concatenate([mass1, mass2], axis=0)

    return positions, velocities, masses, total


def init_cluster(num_bodies, rng):
    """Random spherical cluster with low velocities for gravitational collapse."""
    n = num_bodies
    # Uniform in a sphere
    phi = np.arccos(1.0 - 2.0 * rng.uniform(size=n))
    theta = rng.uniform(0, 2 * np.pi, n)
    r = 5.0 * rng.uniform(size=n) ** (1.0 / 3.0)

    positions = np.zeros((n, 3), dtype=np.float32)
    positions[:, 0] = r * np.sin(phi) * np.cos(theta)
    positions[:, 1] = r * np.sin(phi) * np.sin(theta)
    positions[:, 2] = r * np.cos(phi)

    velocities = rng.normal(0, 0.3, (n, 3)).astype(np.float32)
    masses = np.ones(n, dtype=np.float32) * 2.0

    return positions, velocities, masses


# ── Galaxy Simulator Class ────────────────────────────────────────────────


class GalaxySimulator:
    def __init__(self, preset="spiral", num_bodies=10240, G=1.0, dt=0.001,
                 num_trail_steps=8, use_graph=True, benchmark=False):
        self.G = G
        self.dt = dt
        self.num_trail_steps = num_trail_steps
        self.use_graph = use_graph and wp.get_device().is_cuda
        self.trail_write_idx = 0
        self.preset = preset

        rng = np.random.default_rng(42)

        # Round num_bodies to multiple of TILE_SIZE (64)
        num_bodies = max((num_bodies // 64) * 64, 64)

        if preset == "spiral":
            pos_np, vel_np, mass_np = init_spiral_galaxy(num_bodies, rng)
        elif preset == "collision":
            pos_np, vel_np, mass_np, num_bodies = init_collision(num_bodies, rng)
        elif preset == "cluster":
            pos_np, vel_np, mass_np = init_cluster(num_bodies, rng)
        else:
            raise ValueError(f"Unknown preset: {preset}")

        self.num_bodies = len(pos_np)
        print(f"Simulating {self.num_bodies} bodies (preset={preset}, G={G}, dt={dt})")

        # Allocate Warp arrays
        self.positions = wp.array(pos_np, dtype=wp.vec3)
        self.velocities = wp.array(vel_np, dtype=wp.vec3)
        self.masses = wp.array(mass_np, dtype=float)
        self.forces = wp.zeros(self.num_bodies, dtype=wp.vec3)

        # Trail buffer: [num_trail_steps, num_bodies]
        self.trail_buf = wp.zeros((num_trail_steps, self.num_bodies), dtype=wp.vec3)

        # Benchmark: compare tile vs naive kernel
        if benchmark and wp.get_device().is_cuda:
            self._benchmark()

        # CUDA graph capture for the simulation loop
        self.graph = None
        if self.use_graph:
            # Warm-up launches for compilation
            self._sim_step()
            wp.synchronize_device()

            with wp.ScopedCapture() as capture:
                self._sim_step()
            self.graph = capture.graph
            print("CUDA graph captured successfully")

    def _sim_step(self):
        """One simulation substep: compute forces, then integrate."""
        wp.launch(
            compute_forces_tiled,
            dim=self.num_bodies,
            inputs=[self.positions, self.masses, self.forces, self.num_bodies, self.G],
            block_dim=64,
        )
        wp.launch(
            integrate,
            dim=self.num_bodies,
            inputs=[self.positions, self.velocities, self.forces, self.masses, self.dt],
        )

    def step(self, num_substeps=4):
        """Advance simulation by multiple substeps, then store trail."""
        for _ in range(num_substeps):
            if self.graph is not None:
                wp.capture_launch(self.graph)
            else:
                self._sim_step()

        # Store trail position
        slot = self.trail_write_idx % self.num_trail_steps
        wp.launch(
            store_trail,
            dim=self.num_bodies,
            inputs=[self.positions, self.trail_buf, slot],
        )
        self.trail_write_idx += 1

    def get_state(self):
        """Return positions, velocities, masses, and trails as NumPy arrays."""
        pos = self.positions.numpy()
        vel = self.velocities.numpy()
        masses = self.masses.numpy()
        trails = self.trail_buf.numpy()  # [num_trail_steps, num_bodies, 3]
        return pos, vel, masses, trails

    def _benchmark(self):
        """Time tile-based vs naive kernel."""
        print("\nBenchmarking force kernels...")

        # Warm up both kernels
        wp.launch(
            compute_forces_tiled,
            dim=self.num_bodies,
            inputs=[self.positions, self.masses, self.forces, self.num_bodies, self.G],
            block_dim=64,
        )
        wp.launch(
            compute_forces_naive,
            dim=self.num_bodies,
            inputs=[self.positions, self.masses, self.forces, self.num_bodies, self.G],
        )
        wp.synchronize_device()

        n_iters = 10

        # Time tile kernel
        t0 = time.perf_counter()
        for _ in range(n_iters):
            wp.launch(
                compute_forces_tiled,
                dim=self.num_bodies,
                inputs=[self.positions, self.masses, self.forces, self.num_bodies, self.G],
                block_dim=64,
            )
        wp.synchronize_device()
        tile_time = (time.perf_counter() - t0) / n_iters

        # Time naive kernel
        t0 = time.perf_counter()
        for _ in range(n_iters):
            wp.launch(
                compute_forces_naive,
                dim=self.num_bodies,
                inputs=[self.positions, self.masses, self.forces, self.num_bodies, self.G],
            )
        wp.synchronize_device()
        naive_time = (time.perf_counter() - t0) / n_iters

        print(f"  Tile kernel:  {tile_time*1000:.2f} ms/iter")
        print(f"  Naive kernel: {naive_time*1000:.2f} ms/iter")
        if tile_time > 0:
            print(f"  Speedup:      {naive_time/tile_time:.2f}x")
        print()


# ── Visualization ─────────────────────────────────────────────────────────


def render_frame(ax, pos, vel, masses, trails, trail_write_idx, num_trail_steps,
                 preset, frame_idx, view_range=None):
    """Render a single frame with velocity-colored particles and trails."""
    ax.clear()

    # 2D projection: x-y plane
    x, y = pos[:, 0], pos[:, 1]
    speed = np.sqrt(vel[:, 0] ** 2 + vel[:, 1] ** 2 + vel[:, 2] ** 2)

    # Normalize speed for colormap
    v_min, v_max = np.percentile(speed, [2, 98])
    if v_max - v_min < 1e-6:
        v_max = v_min + 1.0
    speed_norm = np.clip((speed - v_min) / (v_max - v_min), 0, 1)

    # Particle size proportional to mass (log scale), clamped
    log_mass = np.log1p(masses)
    size_min, size_max = 0.5, 12.0
    m_min, m_max = log_mass.min(), log_mass.max()
    if m_max - m_min < 1e-6:
        sizes = np.full_like(log_mass, 1.5)
    else:
        sizes = size_min + (size_max - size_min) * (log_mass - m_min) / (m_max - m_min)

    # Draw motion trails (fading alpha)
    num_valid = min(trail_write_idx, num_trail_steps)
    if num_valid > 1:
        # Subsample particles for trails to keep rendering fast
        n_trail_particles = min(len(x), 2000)
        trail_indices = np.linspace(0, len(x) - 1, n_trail_particles, dtype=int)

        for t in range(num_valid - 1):
            slot = (trail_write_idx - num_valid + t) % num_trail_steps
            alpha = 0.03 + 0.12 * (t / max(num_valid - 1, 1))
            tx = trails[slot, trail_indices, 0]
            ty = trails[slot, trail_indices, 1]
            ax.scatter(tx, ty, s=0.2, c="white", alpha=alpha, edgecolors="none", rasterized=True)

    # Draw particles
    ax.scatter(x, y, s=sizes, c=speed_norm, cmap="inferno", alpha=0.8,
               edgecolors="none", vmin=0, vmax=1, rasterized=True)

    # Styling
    ax.set_facecolor("black")
    ax.set_aspect("equal")

    if view_range is not None:
        ax.set_xlim(-view_range, view_range)
        ax.set_ylim(-view_range, view_range)
    else:
        # Auto range based on particle extent with padding
        margin = 1.5
        extent = max(np.percentile(np.abs(x), 99), np.percentile(np.abs(y), 99)) * margin
        extent = max(extent, 1.0)
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)

    ax.set_xticks([])
    ax.set_yticks([])

    preset_labels = {"spiral": "Spiral Galaxy", "collision": "Galaxy Collision", "cluster": "Random Cluster"}
    ax.set_title(f"{preset_labels.get(preset, preset)} — Frame {frame_idx}",
                 color="white", fontsize=12, pad=8)


def run_simulation(preset, num_frames, num_bodies, output_dir, dt, G,
                   save_frames=True, make_gif=True, benchmark=False):
    """Run the full simulation and produce output."""
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    sim = GalaxySimulator(
        preset=preset,
        num_bodies=num_bodies,
        G=G,
        dt=dt,
        num_trail_steps=8,
        use_graph=True,
        benchmark=benchmark,
    )

    # Determine a stable view range based on initial conditions
    pos0 = sim.positions.numpy()
    initial_extent = max(np.percentile(np.abs(pos0[:, 0]), 99),
                         np.percentile(np.abs(pos0[:, 1]), 99))
    if preset == "collision":
        view_range = initial_extent * 1.8
    elif preset == "cluster":
        view_range = None  # adaptive for cluster
    else:
        view_range = initial_extent * 1.3

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), facecolor="black")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02)

    frame_paths = []
    gif_name = f"galaxy_{preset}.gif"
    gif_path = os.path.join(output_dir, gif_name)

    # Key frames to save as high-res PNGs
    key_frame_indices = set()
    if num_frames > 0:
        # Save first, last, and several intermediate frames
        for frac in [0.0, 0.05, 0.15, 0.25, 0.5, 0.75, 1.0]:
            key_frame_indices.add(min(int(frac * (num_frames - 1)), num_frames - 1))

    print(f"Running {num_frames} frames...")
    t_start = time.perf_counter()

    for frame in range(num_frames):
        sim.step(num_substeps=4)

        pos, vel, masses, trails = sim.get_state()
        render_frame(ax, pos, vel, masses, trails,
                     sim.trail_write_idx, sim.num_trail_steps,
                     preset, frame, view_range=view_range)

        frame_path = os.path.join(output_dir, f"galaxy_frame_{frame:04d}.png")
        dpi = 150 if frame in key_frame_indices else 80
        fig.savefig(frame_path, dpi=dpi, facecolor="black")
        frame_paths.append(frame_path)

        if (frame + 1) % 50 == 0 or frame == num_frames - 1:
            elapsed = time.perf_counter() - t_start
            fps = (frame + 1) / elapsed
            print(f"  Frame {frame + 1}/{num_frames}  ({fps:.1f} frames/sec)")

    plt.close(fig)

    elapsed = time.perf_counter() - t_start
    print(f"\nSimulation complete: {num_frames} frames in {elapsed:.1f}s "
          f"({num_frames/elapsed:.1f} frames/sec)")

    # Compile GIF
    if make_gif and frame_paths:
        try:
            from PIL import Image  # noqa: PLC0415

            print(f"Creating GIF: {gif_path}")
            imgs = [Image.open(p) for p in frame_paths]
            imgs[0].save(
                gif_path,
                save_all=True,
                append_images=imgs[1:],
                duration=50,  # ms per frame
                loop=0,
            )
            print(f"Saved {gif_path}")
        except ImportError:
            print("Pillow not available, skipping GIF creation")

    # Clean up low-res frame PNGs, keep high-res key frames
    if save_frames:
        kept = []
        for i, path in enumerate(frame_paths):
            if i in key_frame_indices:
                kept.append(path)
            else:
                os.remove(path)
        print(f"Kept {len(kept)} high-res key frames")
    else:
        for path in frame_paths:
            os.remove(path)

    # Save a representative screenshot (mid-simulation looks most interesting)
    screenshot_path = os.path.join(output_dir, "mega_prompt_4.png")
    if kept:
        import shutil  # noqa: PLC0415
        # Pick the frame around 25-50% through the simulation
        mid_idx = max(0, len(kept) // 3)
        shutil.copy2(kept[mid_idx], screenshot_path)
        print(f"Screenshot saved: {screenshot_path}")

    return gif_path


# ── Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GPU N-body Galaxy Simulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", type=str, default=None, help="Warp device (cpu, cuda:0, etc.)")
    parser.add_argument("--preset", type=str, default="spiral",
                        choices=["spiral", "collision", "cluster"],
                        help="Initial condition preset")
    parser.add_argument("--num-frames", type=int, default=200, help="Number of animation frames")
    parser.add_argument("--num-bodies", type=int, default=10240,
                        help="Number of particles (rounded to multiple of 64)")
    parser.add_argument("--dt", type=float, default=0.001, help="Simulation timestep")
    parser.add_argument("--G", type=float, default=1.0, help="Gravitational constant")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (defaults to script directory)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run tile vs naive kernel benchmark")
    parser.add_argument("--no-gif", action="store_true", help="Skip GIF creation")

    args = parser.parse_known_args()[0]

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(__file__))

    with wp.ScopedDevice(args.device):
        run_simulation(
            preset=args.preset,
            num_frames=args.num_frames,
            num_bodies=args.num_bodies,
            output_dir=output_dir,
            dt=args.dt,
            G=args.G,
            save_frames=True,
            make_gif=not args.no_gif,
            benchmark=args.benchmark,
        )
