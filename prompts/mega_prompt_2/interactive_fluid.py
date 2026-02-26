"""Interactive GPU Particle Fluid with Interactive Forces.

A real-time 2D SPH fluid simulator rendered on GPU using Warp.
Click to attract or repel particles, spawn new particles, and
watch the fluid respond in real time.

Controls:
    Left click     Attract particles toward mouse
    Right click    Repel particles from mouse
    Space          Pause / resume
    R              Reset simulation
    +/=            Increase particle count (reset)
    -              Decrease particle count (reset)
    ESC            Exit

Usage:
    uv run --with "pyglet>=2.0" --with matplotlib python interactive_fluid.py
    uv run --with matplotlib python interactive_fluid.py --headless --num-frames 200
"""

import argparse
import math

import numpy as np

import warp as wp


# -- SPH Kernel Functions ---------------------------------------------------


@wp.func
def cubic_spline_w(r: float, h: float):
    """Cubic spline SPH kernel W(r, h) in 2D."""
    q = r / h
    sigma = 10.0 / (7.0 * 3.14159265 * h * h)
    w = float(0.0)
    if q <= 1.0:
        w = sigma * (1.0 - 1.5 * q * q + 0.75 * q * q * q)
    elif q <= 2.0:
        t = 2.0 - q
        w = sigma * 0.25 * t * t * t
    return w


@wp.func
def cubic_spline_grad_w(rij: wp.vec3, dist: float, h: float):
    """Gradient of the cubic spline SPH kernel in 2D."""
    q = dist / h
    sigma = 10.0 / (7.0 * 3.14159265 * h * h)
    grad_mag = float(0.0)
    if dist < 1.0e-8:
        return wp.vec3(0.0, 0.0, 0.0)
    if q <= 1.0:
        grad_mag = sigma / h * (-3.0 * q + 2.25 * q * q)
    elif q <= 2.0:
        t = 2.0 - q
        grad_mag = sigma / h * (-0.75 * t * t)
    return rij * (grad_mag / dist)


@wp.func
def eos_pressure(density: float, rest_density: float, k: float):
    """Weakly compressible equation of state."""
    return k * (density / rest_density - 1.0)


# -- Kernels ----------------------------------------------------------------


@wp.kernel
def initialize_dam_break(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    n: int,
    cols: int,
    spacing: float,
):
    """Initialize particles in a dam-break column on the left side."""
    tid = wp.tid()
    if tid >= n:
        return

    row = tid // cols
    col = tid % cols

    x = spacing * 0.5 + float(col) * spacing
    y = spacing * 0.5 + float(row) * spacing

    state = wp.rand_init(42, tid)
    jitter_x = wp.randf(state) * spacing * 0.1
    jitter_y = wp.randf(state) * spacing * 0.1

    positions[tid] = wp.vec3(x + jitter_x, y + jitter_y, 0.0)
    velocities[tid] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def compute_density(
    grid: wp.uint64,
    positions: wp.array(dtype=wp.vec3),
    densities: wp.array(dtype=float),
    n: int,
    h: float,
    particle_mass: float,
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if i >= n:
        return

    xi = positions[i]
    rho = float(0.0)

    neighbors = wp.hash_grid_query(grid, xi, h * 2.0)
    for index in neighbors:
        if index >= n:
            continue
        xj = positions[index]
        rij = xi - xj
        dist = wp.length(rij)
        rho += particle_mass * cubic_spline_w(dist, h)

    densities[i] = wp.max(rho, 1.0)


@wp.kernel
def compute_forces(
    grid: wp.uint64,
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    densities: wp.array(dtype=float),
    forces: wp.array(dtype=wp.vec3),
    n: int,
    h: float,
    particle_mass: float,
    rest_density: float,
    gas_constant: float,
    viscosity: float,
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if i >= n:
        return

    xi = positions[i]
    vi = velocities[i]
    rhoi = densities[i]
    pi = eos_pressure(rhoi, rest_density, gas_constant)

    f_pressure = wp.vec3(0.0, 0.0, 0.0)
    f_viscosity = wp.vec3(0.0, 0.0, 0.0)

    neighbors = wp.hash_grid_query(grid, xi, h * 2.0)
    for index in neighbors:
        if index == i:
            continue
        if index >= n:
            continue

        xj = positions[index]
        rij = xi - xj
        dist = wp.length(rij)

        if dist < 1.0e-8:
            continue

        rhoj = densities[index]
        pj = eos_pressure(rhoj, rest_density, gas_constant)
        vj = velocities[index]

        # Pressure force (symmetric)
        grad = cubic_spline_grad_w(rij, dist, h)
        f_pressure -= particle_mass * (pi / (rhoi * rhoi) + pj / (rhoj * rhoj)) * grad

        # Viscosity force
        f_viscosity += particle_mass * viscosity * (vj - vi) / rhoj * cubic_spline_w(dist, h)

    forces[i] = (f_pressure + f_viscosity) * rhoi


@wp.kernel
def apply_external_forces(
    forces: wp.array(dtype=wp.vec3),
    positions: wp.array(dtype=wp.vec3),
    densities: wp.array(dtype=float),
    n: int,
    gravity_x: float,
    gravity_y: float,
    mouse_x: float,
    mouse_y: float,
    mouse_strength: float,
    mouse_active: int,
):
    tid = wp.tid()
    if tid >= n:
        return

    f = forces[tid]
    rho = densities[tid]

    # Gravity
    f = f + wp.vec3(gravity_x, gravity_y, 0.0) * rho

    # Mouse interaction force
    if mouse_active != 0:
        pos = positions[tid]
        dx = pos[0] - mouse_x
        dy = pos[1] - mouse_y
        dist_sq = dx * dx + dy * dy + 0.001
        dist = wp.sqrt(dist_sq)
        radius = 0.15
        if dist < radius:
            strength = mouse_strength * (1.0 - dist / radius)
            f = f + wp.vec3(dx, dy, 0.0) * (strength / dist)

    forces[tid] = f


@wp.kernel
def integrate(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    forces: wp.array(dtype=wp.vec3),
    densities: wp.array(dtype=float),
    n: int,
    dt: float,
):
    """Symplectic Euler integration."""
    tid = wp.tid()
    if tid >= n:
        return

    rho = densities[tid]
    v = velocities[tid]
    x = positions[tid]

    # Update velocity
    v = v + dt * forces[tid] / rho
    # Damping
    v = v * 0.999

    # Update position
    x = x + dt * v

    velocities[tid] = v
    positions[tid] = x


@wp.kernel
def enforce_boundaries(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    n: int,
    domain_x: float,
    domain_y: float,
):
    tid = wp.tid()
    if tid >= n:
        return

    x = positions[tid]
    v = velocities[tid]
    eps = 0.002
    damping = -0.5

    # Left wall
    if x[0] < eps:
        x = wp.vec3(eps, x[1], 0.0)
        v = wp.vec3(v[0] * damping, v[1] * 0.99, 0.0)
    # Right wall
    if x[0] > domain_x - eps:
        x = wp.vec3(domain_x - eps, x[1], 0.0)
        v = wp.vec3(v[0] * damping, v[1] * 0.99, 0.0)
    # Floor
    if x[1] < eps:
        x = wp.vec3(x[0], eps, 0.0)
        v = wp.vec3(v[0] * 0.99, v[1] * damping, 0.0)
    # Ceiling
    if x[1] > domain_y - eps:
        x = wp.vec3(x[0], domain_y - eps, 0.0)
        v = wp.vec3(v[0] * 0.99, v[1] * damping, 0.0)

    # Clamp z
    x = wp.vec3(x[0], x[1], 0.0)
    v = wp.vec3(v[0], v[1], 0.0)

    positions[tid] = x
    velocities[tid] = v


@wp.kernel
def compute_colors(
    velocities: wp.array(dtype=wp.vec3),
    colors: wp.array(dtype=wp.vec3),
    n: int,
    max_speed: float,
):
    """Map velocity magnitude to color (blue=slow, red=fast)."""
    tid = wp.tid()
    if tid >= n:
        return

    v = velocities[tid]
    speed = wp.length(v)
    t = wp.clamp(speed / max_speed, 0.0, 1.0)

    # Blue -> Cyan -> Green -> Yellow -> Red
    r = float(0.0)
    g = float(0.0)
    b = float(0.0)

    if t < 0.25:
        s = t / 0.25
        r = 0.0
        g = s
        b = 1.0
    elif t < 0.5:
        s = (t - 0.25) / 0.25
        r = 0.0
        g = 1.0
        b = 1.0 - s
    elif t < 0.75:
        s = (t - 0.5) / 0.25
        r = s
        g = 1.0
        b = 0.0
    else:
        s = (t - 0.75) / 0.25
        r = 1.0
        g = 1.0 - s
        b = 0.0

    colors[tid] = wp.vec3(r, g, b)


# -- Simulation Class -------------------------------------------------------


class FluidSimulation:
    def __init__(self, num_particles=10000, domain_x=1.0, domain_y=1.0):
        self.domain_x = domain_x
        self.domain_y = domain_y
        self.num_particles = num_particles

        # SPH parameters
        self.h = 0.025
        self.rest_density = 1000.0
        self.gas_constant = 2000.0
        self.viscosity = 0.1
        self.gravity = (0.0, -9.81)
        self.dt = 0.00008
        spacing = self.h * 0.6
        self.particle_mass = self.rest_density * spacing * spacing

        # Mouse interaction state
        self.mouse_x = 0.0
        self.mouse_y = 0.0
        self.mouse_strength = 0.0
        self.mouse_active = 0

        # Allocate arrays
        self._allocate(num_particles)
        self._initialize()

    def _allocate(self, n):
        self.num_particles = n
        max_n = n + 1024  # extra headroom
        self.positions = wp.zeros(max_n, dtype=wp.vec3)
        self.velocities = wp.zeros(max_n, dtype=wp.vec3)
        self.densities = wp.zeros(max_n, dtype=float)
        self.forces = wp.zeros(max_n, dtype=wp.vec3)
        self.colors = wp.zeros(max_n, dtype=wp.vec3)

        grid_dim = max(int(self.domain_x / (self.h * 2.0)), 16)
        self.grid = wp.HashGrid(grid_dim, grid_dim, 1)

    def _initialize(self):
        n = self.num_particles
        spacing = self.h * 0.6
        # Compute columns to fill left 40% of domain; rows fill from bottom
        cols = max(int(self.domain_x * 0.4 / spacing), 1)
        rows = math.ceil(n / cols)
        # If the block would exceed domain height, widen the block
        if rows * spacing > self.domain_y * 0.95:
            rows = max(int(self.domain_y * 0.95 / spacing), 1)
            cols = math.ceil(n / rows)
        wp.launch(
            initialize_dam_break,
            dim=n,
            inputs=[self.positions, self.velocities, n, cols, spacing],
        )

    def reset(self, num_particles=None):
        if num_particles is not None and num_particles != self.num_particles:
            self._allocate(num_particles)
        else:
            self.velocities.zero_()
            self.forces.zero_()
            self.densities.zero_()
        self._initialize()

    def step(self):
        n = self.num_particles
        # Rebuild spatial hash
        self.grid.build(self.positions, self.h * 2.0)

        # Compute densities
        wp.launch(
            compute_density,
            dim=n,
            inputs=[self.grid.id, self.positions, self.densities, n, self.h, self.particle_mass],
        )

        # Compute pressure + viscosity forces
        wp.launch(
            compute_forces,
            dim=n,
            inputs=[
                self.grid.id,
                self.positions,
                self.velocities,
                self.densities,
                self.forces,
                n,
                self.h,
                self.particle_mass,
                self.rest_density,
                self.gas_constant,
                self.viscosity,
            ],
        )

        # Apply gravity and mouse forces
        wp.launch(
            apply_external_forces,
            dim=n,
            inputs=[
                self.forces,
                self.positions,
                self.densities,
                n,
                self.gravity[0],
                self.gravity[1],
                self.mouse_x,
                self.mouse_y,
                self.mouse_strength,
                self.mouse_active,
            ],
        )

        # Integrate
        wp.launch(
            integrate,
            dim=n,
            inputs=[self.positions, self.velocities, self.forces, self.densities, n, self.dt],
        )

        # Boundaries
        wp.launch(
            enforce_boundaries,
            dim=n,
            inputs=[self.positions, self.velocities, n, self.domain_x, self.domain_y],
        )

    def update_colors(self):
        n = self.num_particles
        wp.launch(compute_colors, dim=n, inputs=[self.velocities, self.colors, n, 3.0])


# -- Headless Renderer -------------------------------------------------------


def run_headless(num_particles=10000, num_frames=200, output="interactive_fluid.png"):
    import matplotlib.pyplot as plt  # noqa: PLC0415

    sim = FluidSimulation(num_particles=num_particles)

    print(f"Running {num_frames} frames with {num_particles} particles...")
    substeps = 20
    for frame in range(num_frames):
        for _ in range(substeps):
            sim.step()
        if (frame + 1) % 50 == 0:
            print(f"  Frame {frame + 1}/{num_frames}")

    sim.update_colors()

    # Get particle data
    pos_np = sim.positions.numpy()[: sim.num_particles]
    col_np = sim.colors.numpy()[: sim.num_particles]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, sim.domain_x)
    ax.set_ylim(0, sim.domain_y)
    ax.set_aspect("equal")
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    ax.scatter(
        pos_np[:, 0],
        pos_np[:, 1],
        c=col_np,
        s=3.0,
        alpha=0.85,
        edgecolors="none",
    )

    ax.set_title(
        f"SPH Fluid Simulation - {num_particles} particles, {num_frames} frames",
        color="white",
        fontsize=14,
    )
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved {output}")
    plt.close()


# -- Interactive Renderer ----------------------------------------------------


def run_interactive(num_particles=10000, width=900, height=900):
    import time  # noqa: PLC0415

    import pyglet  # noqa: PLC0415
    from pyglet.window import key, mouse  # noqa: PLC0415

    sim = FluidSimulation(num_particles=num_particles)

    window = pyglet.window.Window(
        width, height, caption="Interactive GPU Particle Fluid", vsync=False
    )
    key_handler = key.KeyStateHandler()
    window.push_handlers(key_handler)

    # State
    paused = [False]
    running = [True]
    mouse_pos = [0.0, 0.0]
    mouse_buttons = [False, False]  # left, right

    @window.event
    def on_close():
        running[0] = False

    @window.event
    def on_key_press(symbol, modifiers):
        if symbol == key.ESCAPE:
            running[0] = False
        elif symbol == key.SPACE:
            paused[0] = not paused[0]
        elif symbol == key.R:
            sim.reset()
        elif symbol in (key.PLUS, key.EQUAL):
            new_n = min(sim.num_particles + 2000, 30000)
            sim.reset(num_particles=new_n)
        elif symbol == key.MINUS:
            new_n = max(sim.num_particles - 2000, 2000)
            sim.reset(num_particles=new_n)

    @window.event
    def on_mouse_motion(x, y, dx, dy):
        mouse_pos[0] = x / float(width) * sim.domain_x
        mouse_pos[1] = y / float(height) * sim.domain_y

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        mouse_pos[0] = x / float(width) * sim.domain_x
        mouse_pos[1] = y / float(height) * sim.domain_y

    @window.event
    def on_mouse_press(x, y, button, modifiers):
        if button == mouse.LEFT:
            mouse_buttons[0] = True
        if button == mouse.RIGHT:
            mouse_buttons[1] = True

    @window.event
    def on_mouse_release(x, y, button, modifiers):
        if button == mouse.LEFT:
            mouse_buttons[0] = False
        if button == mouse.RIGHT:
            mouse_buttons[1] = False

    # Labels
    info_label = pyglet.text.Label(
        "",
        font_size=11,
        x=8,
        y=height - 18,
        color=(220, 220, 220, 200),
    )
    controls_label = pyglet.text.Label(
        "LMB: attract | RMB: repel | Space: pause | R: reset | +/-: particles | ESC: quit",
        font_size=9,
        x=8,
        y=6,
        color=(180, 180, 180, 160),
    )

    last_time = time.perf_counter()
    frame_count = 0
    fps_clock = time.perf_counter()
    fps_text = "FPS: --"

    substeps = 4

    while running[0]:
        window.dispatch_events()

        now = time.perf_counter()
        dt = now - last_time
        last_time = now

        # Update mouse force
        if mouse_buttons[0]:  # left = attract
            sim.mouse_x = mouse_pos[0]
            sim.mouse_y = mouse_pos[1]
            sim.mouse_strength = -50.0
            sim.mouse_active = 1
        elif mouse_buttons[1]:  # right = repel
            sim.mouse_x = mouse_pos[0]
            sim.mouse_y = mouse_pos[1]
            sim.mouse_strength = 80.0
            sim.mouse_active = 1
        else:
            sim.mouse_active = 0

        if not paused[0]:
            for _ in range(substeps):
                sim.step()

        sim.update_colors()

        # Read data from GPU
        pos_np = sim.positions.numpy()[: sim.num_particles]
        col_np = sim.colors.numpy()[: sim.num_particles]

        # Convert positions to pixel coords
        px = (pos_np[:, 0] / sim.domain_x * width).astype(np.float32)
        py = (pos_np[:, 1] / sim.domain_y * height).astype(np.float32)
        cr = (col_np[:, 0] * 255).astype(np.uint8)
        cg = (col_np[:, 1] * 255).astype(np.uint8)
        cb = (col_np[:, 2] * 255).astype(np.uint8)

        # Render
        window.switch_to()
        window.clear()

        # Rasterize particles into a pixel buffer and blit (pyglet 2.x compatible)
        framebuf = np.zeros((height, width, 3), dtype=np.uint8)
        framebuf[:, :] = (15, 15, 25)  # dark background

        # Convert to integer pixel coords and clip
        ix = np.clip(px.astype(np.int32), 0, width - 1)
        iy = np.clip(py.astype(np.int32), 0, height - 1)

        # Draw particles (later particles overwrite earlier — fine for visualization)
        framebuf[iy, ix, 0] = cr
        framebuf[iy, ix, 1] = cg
        framebuf[iy, ix, 2] = cb

        image = pyglet.image.ImageData(width, height, "RGB", framebuf.tobytes(), pitch=width * 3)
        image.blit(0, 0)

        # FPS
        frame_count += 1
        if now - fps_clock >= 1.0:
            fps_text = f"FPS: {frame_count}"
            frame_count = 0
            fps_clock = now

        state_str = " [PAUSED]" if paused[0] else ""
        info_label.text = f"{fps_text} | Particles: {sim.num_particles}{state_str}"
        info_label.draw()
        controls_label.draw()

        window.flip()

    window.close()


# -- Entry Point -------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactive GPU Particle Fluid Simulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--headless", action="store_true", help="Run headless and save PNG output.")
    parser.add_argument("--num-particles", type=int, default=10000, help="Number of particles.")
    parser.add_argument("--num-frames", type=int, default=200, help="Number of simulation frames (headless).")
    parser.add_argument("--width", type=int, default=900, help="Window width (interactive).")
    parser.add_argument("--height", type=int, default=900, help="Window height (interactive).")
    parser.add_argument("--output", type=str, default="interactive_fluid.png", help="Output PNG path (headless).")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        if args.headless:
            run_headless(
                num_particles=args.num_particles,
                num_frames=args.num_frames,
                output=args.output,
            )
        else:
            run_interactive(
                num_particles=args.num_particles,
                width=args.width,
                height=args.height,
            )
