"""Mini 2D Lattice Boltzmann CFD Solver.

A D2Q9 Lattice Boltzmann Method (LBM) solver for 2D incompressible flow around
obstacles. Produces von Karman vortex streets with real-time visualization of
vorticity and velocity fields.

Usage:
    uv run --with matplotlib --with Pillow python lbm_solver.py --num-steps 20000
    uv run --with matplotlib --with Pillow python lbm_solver.py --Re 400 --grid 1600x400
"""

import argparse
import os
import time

import numpy as np

import warp as wp

wp.config.enable_backward = False

# D2Q9 lattice constants
EX_CONST = wp.constant(wp.types.vector(9, int)(0, 1, 0, -1, 0, 1, -1, -1, 1))
EY_CONST = wp.constant(wp.types.vector(9, int)(0, 0, 1, 0, -1, 1, 1, -1, -1))
OPP_CONST = wp.constant(wp.types.vector(9, int)(0, 3, 4, 1, 2, 7, 8, 5, 6))

W0 = wp.constant(4.0 / 9.0)
W1 = wp.constant(1.0 / 9.0)
W5 = wp.constant(1.0 / 36.0)


@wp.func
def get_weight(q: int) -> float:
    if q == 0:
        return W0
    if q >= 1 and q <= 4:
        return W1
    return W5


@wp.func
def compute_feq(q: int, rho: float, ux: float, uy: float) -> float:
    ex = float(EX_CONST[q])
    ey = float(EY_CONST[q])
    w = get_weight(q)
    eu = ex * ux + ey * uy
    usq = ux * ux + uy * uy
    return w * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * usq)


# -- Kernels ------------------------------------------------------------------


@wp.kernel
def init_obstacle_kernel(
    solid: wp.array(dtype=int),
    cx: float,
    cy: float,
    radius: float,
    nx: int,
    ny: int,
):
    tid = wp.tid()
    j = tid / nx
    i = tid % nx
    dx = float(i) - cx
    dy = float(j) - cy
    if dx * dx + dy * dy <= radius * radius:
        solid[tid] = 1
    if j == 0 or j == ny - 1:
        solid[tid] = 1


@wp.kernel
def init_equilibrium_kernel(
    f: wp.array(dtype=float),
    solid: wp.array(dtype=int),
    u_in: float,
    nx: int,
    ny: int,
):
    tid = wp.tid()
    u = u_in
    if solid[tid] == 1:
        u = 0.0
    stride = ny * nx
    for q in range(9):
        f[q * stride + tid] = compute_feq(q, 1.0, u, 0.0)


@wp.kernel
def perturbation_kernel(
    f: wp.array(dtype=float),
    solid: wp.array(dtype=int),
    cx: float,
    cy: float,
    radius: float,
    u_in: float,
    nx: int,
    ny: int,
):
    """Apply small asymmetric perturbation near cylinder to trigger vortex shedding."""
    tid = wp.tid()
    j = tid / nx
    i = tid % nx
    if solid[tid] == 1:
        return
    dx = float(i) - cx
    dy = float(j) - cy
    dist = wp.sqrt(dx * dx + dy * dy)
    # Perturb in a ring around the cylinder
    if dist > radius and dist < radius * 3.0:
        # Small transverse velocity perturbation (asymmetric in y)
        uy_pert = u_in * 0.1 * wp.sin(2.0 * 3.14159265 * dy / radius)
        uy_pert = uy_pert * wp.exp(-(dist - radius * 1.5) * (dist - radius * 1.5) / (radius * radius))
        stride = ny * nx
        for q in range(9):
            f[q * stride + tid] = compute_feq(q, 1.0, u_in, uy_pert)


@wp.kernel
def collision_kernel(
    f: wp.array(dtype=float),
    f_post: wp.array(dtype=float),
    solid: wp.array(dtype=int),
    inv_tau: float,
    nx: int,
    ny: int,
):
    """BGK collision. Skip solid cells (just copy)."""
    tid = wp.tid()
    stride = ny * nx

    if solid[tid] == 1:
        for q in range(9):
            idx = q * stride + tid
            f_post[idx] = f[idx]
        return

    # Compute macroscopic at this cell
    r = float(0.0)
    vx = float(0.0)
    vy = float(0.0)
    for q in range(9):
        fq = f[q * stride + tid]
        r += fq
        vx += fq * float(EX_CONST[q])
        vy += fq * float(EY_CONST[q])
    if r > 1.0e-10:
        vx = vx / r
        vy = vy / r
    else:
        r = 1.0
        vx = 0.0
        vy = 0.0

    for q in range(9):
        idx = q * stride + tid
        feq = compute_feq(q, r, vx, vy)
        f_post[idx] = f[idx] - (f[idx] - feq) * inv_tau


@wp.kernel
def streaming_kernel(
    f_post: wp.array(dtype=float),
    f: wp.array(dtype=float),
    nx: int,
    ny: int,
):
    """Pull-based streaming with periodic wrapping (like np.roll).

    Wraps around at domain boundaries so that boundary conditions
    (inlet/outlet) can fix the values afterwards.
    """
    tid = wp.tid()
    j = tid / nx
    i = tid % nx
    stride = ny * nx

    for q in range(9):
        # Source with periodic wrapping
        src_i = (i - EX_CONST[q] + nx) % nx
        src_j = (j - EY_CONST[q] + ny) % ny
        src_tid = src_j * nx + src_i
        f[q * stride + tid] = f_post[q * stride + src_tid]


@wp.kernel
def bounceback_kernel(
    f: wp.array(dtype=float),
    f_post: wp.array(dtype=float),
    solid: wp.array(dtype=int),
    nx: int,
    ny: int,
):
    """Apply bounce-back at solid cells and fix fluid cells adjacent to solid.

    Step 1: At solid cells, swap directions (f[q] = f_streamed[opp(q)]).
    Step 2: At fluid cells whose streaming source was solid, use pre-streaming
    post-collision value in opposite direction from self.
    """
    tid = wp.tid()
    j = tid / nx
    i = tid % nx
    stride = ny * nx

    if solid[tid] == 1:
        # Solid cell: reverse all directions
        for q in range(9):
            opp_q = OPP_CONST[q]
            f[q * stride + tid] = f[opp_q * stride + tid]
        return

    # Fluid cell: fix directions whose source was solid
    for q in range(9):
        src_i = (i - EX_CONST[q] + nx) % nx
        src_j = (j - EY_CONST[q] + ny) % ny
        src_tid = src_j * nx + src_i
        if solid[src_tid] == 1:
            # Use pre-streaming post-collision value in opposite direction
            opp_q = OPP_CONST[q]
            f[q * stride + tid] = f_post[opp_q * stride + tid]


@wp.kernel
def inlet_zou_he_kernel(
    f: wp.array(dtype=float),
    u_in: float,
    solid: wp.array(dtype=int),
    nx: int,
    ny: int,
):
    j = wp.tid()
    if j >= ny:
        return
    tid = j * nx
    if solid[tid] == 1:
        return
    stride = ny * nx

    f0 = f[0 * stride + tid]
    f2 = f[2 * stride + tid]
    f3 = f[3 * stride + tid]
    f4 = f[4 * stride + tid]
    f6 = f[6 * stride + tid]
    f7 = f[7 * stride + tid]

    r = (f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / (1.0 - u_in)
    f[1 * stride + tid] = f3 + 2.0 / 3.0 * r * u_in
    f[5 * stride + tid] = f7 - 0.5 * (f2 - f4) + 1.0 / 6.0 * r * u_in
    f[8 * stride + tid] = f6 + 0.5 * (f2 - f4) + 1.0 / 6.0 * r * u_in


@wp.kernel
def outlet_kernel(
    f: wp.array(dtype=float),
    nx: int,
    ny: int,
):
    j = wp.tid()
    if j >= ny:
        return
    stride = ny * nx
    tid_out = j * nx + nx - 1
    tid_src = j * nx + nx - 2
    for q in range(9):
        f[q * stride + tid_out] = f[q * stride + tid_src]


@wp.kernel
def macroscopic_kernel(
    f: wp.array(dtype=float),
    rho: wp.array(dtype=float),
    ux: wp.array(dtype=float),
    uy: wp.array(dtype=float),
    solid: wp.array(dtype=int),
    nx: int,
    ny: int,
):
    tid = wp.tid()
    stride = ny * nx
    if solid[tid] == 1:
        rho[tid] = 1.0
        ux[tid] = 0.0
        uy[tid] = 0.0
        return
    r = float(0.0)
    vx = float(0.0)
    vy = float(0.0)
    for q in range(9):
        fq = f[q * stride + tid]
        r += fq
        vx += fq * float(EX_CONST[q])
        vy += fq * float(EY_CONST[q])
    if r > 1.0e-10:
        vx = vx / r
        vy = vy / r
    else:
        r = 1.0
        vx = 0.0
        vy = 0.0
    rho[tid] = r
    ux[tid] = vx
    uy[tid] = vy


@wp.kernel
def vorticity_kernel(
    ux: wp.array(dtype=float),
    uy: wp.array(dtype=float),
    vort: wp.array(dtype=float),
    nx: int,
    ny: int,
):
    tid = wp.tid()
    inner_nx = nx - 2
    jj = tid / inner_nx + 1
    ii = tid % inner_nx + 1
    if jj >= ny - 1:
        return
    duy_dx = (uy[jj * nx + ii + 1] - uy[jj * nx + ii - 1]) * 0.5
    dux_dy = (ux[(jj + 1) * nx + ii] - ux[(jj - 1) * nx + ii]) * 0.5
    vort[jj * nx + ii] = duy_dx - dux_dy


# -- Solver class --------------------------------------------------------------


class LBMSolver:
    def __init__(self, nx=800, ny=200, re=200.0, u_in=0.04, cylinder_d=40):
        self.nx = nx
        self.ny = ny
        self.re = re
        self.u_in = u_in
        self.cylinder_d = cylinder_d
        self.cylinder_r = cylinder_d / 2.0

        self.nu = u_in * cylinder_d / re
        self.tau = 3.0 * self.nu + 0.5
        self.inv_tau = 1.0 / self.tau

        self.cx = float(nx // 5)
        self.cy = float(ny // 2)

        total = nx * ny
        self.f = wp.zeros(9 * total, dtype=float)
        self.f_post = wp.zeros(9 * total, dtype=float)
        self.rho = wp.zeros(total, dtype=float)
        self.ux = wp.zeros(total, dtype=float)
        self.uy = wp.zeros(total, dtype=float)
        self.solid = wp.zeros(total, dtype=int)
        self.vorticity = wp.zeros(total, dtype=float)

        wp.launch(init_obstacle_kernel, dim=total,
                  inputs=[self.solid, self.cx, self.cy, self.cylinder_r, nx, ny])
        wp.launch(init_equilibrium_kernel, dim=total,
                  inputs=[self.f, self.solid, 0.0, nx, ny])

        # Apply asymmetric perturbation to trigger vortex shedding
        if cylinder_d > 0:
            wp.launch(perturbation_kernel, dim=total,
                      inputs=[self.f, self.solid, self.cx, self.cy,
                              self.cylinder_r, u_in, nx, ny])

        print(f"LBM Solver initialized:")
        print(f"  Grid:      {nx} x {ny}")
        print(f"  Re:        {re}")
        print(f"  u_in:      {u_in}")
        print(f"  nu:        {self.nu:.6f}")
        print(f"  tau:       {self.tau:.6f}")
        print(f"  Cylinder:  D={cylinder_d}, center=({self.cx:.0f}, {self.cy:.0f})")

    def step(self, current_u_in=None):
        nx, ny = self.nx, self.ny
        total = nx * ny
        u = current_u_in if current_u_in is not None else self.u_in

        # 1. Collision
        wp.launch(collision_kernel, dim=total,
                  inputs=[self.f, self.f_post, self.solid, self.inv_tau, nx, ny])

        # 2. Streaming (pull with periodic wrapping)
        wp.launch(streaming_kernel, dim=total,
                  inputs=[self.f_post, self.f, nx, ny])

        # 3. Bounce-back (solid cells + fluid cells adjacent to solid)
        wp.launch(bounceback_kernel, dim=total,
                  inputs=[self.f, self.f_post, self.solid, nx, ny])

        # 4. Inlet / outlet boundary conditions
        wp.launch(inlet_zou_he_kernel, dim=ny,
                  inputs=[self.f, u, self.solid, nx, ny])
        wp.launch(outlet_kernel, dim=ny,
                  inputs=[self.f, nx, ny])

        # 5. Macroscopic
        wp.launch(macroscopic_kernel, dim=total,
                  inputs=[self.f, self.rho, self.ux, self.uy, self.solid, nx, ny])

    def compute_vorticity(self):
        nx, ny = self.nx, self.ny
        self.vorticity.zero_()
        inner = (nx - 2) * (ny - 2)
        wp.launch(vorticity_kernel, dim=inner,
                  inputs=[self.ux, self.uy, self.vorticity, nx, ny])

    def get_vorticity_numpy(self):
        self.compute_vorticity()
        return self.vorticity.numpy().reshape(self.ny, self.nx)

    def get_velocity_magnitude_numpy(self):
        ux_np = self.ux.numpy().reshape(self.ny, self.nx)
        uy_np = self.uy.numpy().reshape(self.ny, self.nx)
        return np.sqrt(ux_np**2 + uy_np**2)

    def get_solid_numpy(self):
        return self.solid.numpy().reshape(self.ny, self.nx)


# -- Visualization -------------------------------------------------------------


def save_frame(solver, step, output_dir, save_vorticity_png=True, save_velocity_png=True):
    import matplotlib.pyplot as plt  # noqa: PLC0415

    solid_np = solver.get_solid_numpy()
    solid_mask = solid_np.astype(bool)

    if save_vorticity_png:
        vort_np = solver.get_vorticity_numpy()
        vort_masked = np.where(solid_mask, np.nan, vort_np)
        vmax = 0.02
        fig, ax = plt.subplots(figsize=(16, 4), dpi=100)
        im = ax.imshow(vort_masked, origin="lower", cmap="coolwarm",
                       vmin=-vmax, vmax=vmax, interpolation="bilinear", aspect="equal")
        ax.contour(solid_np, levels=[0.5], colors="black", linewidths=1.5)
        ax.set_title(f"Vorticity (step {step})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, label="vorticity", shrink=0.8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"lbm_vorticity_{step:05d}.png"),
                    dpi=100, bbox_inches="tight")
        plt.close(fig)

    if save_velocity_png:
        vel_np = solver.get_velocity_magnitude_numpy()
        vel_masked = np.where(solid_mask, np.nan, vel_np)
        fig, ax = plt.subplots(figsize=(16, 4), dpi=100)
        im = ax.imshow(vel_masked, origin="lower", cmap="viridis",
                       vmin=0.0, vmax=solver.u_in * 2.0,
                       interpolation="bilinear", aspect="equal")
        ax.contour(solid_np, levels=[0.5], colors="black", linewidths=1.5)
        ax.set_title(f"Velocity Magnitude (step {step})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, label="|u|", shrink=0.8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"lbm_velocity_{step:05d}.png"),
                    dpi=100, bbox_inches="tight")
        plt.close(fig)


def make_gif(output_dir, pattern="lbm_vorticity_", output_name="lbm_vorticity.gif"):
    from PIL import Image  # noqa: PLC0415

    frames = []
    files = sorted(f for f in os.listdir(output_dir)
                   if f.startswith(pattern) and f.endswith(".png"))
    if not files:
        print(f"No frames found for GIF ({pattern})")
        return
    for fname in files:
        img = Image.open(os.path.join(output_dir, fname))
        frames.append(img.copy())
        img.close()

    gif_path = os.path.join(output_dir, output_name)
    frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                   duration=80, loop=0)
    print(f"Saved GIF: {gif_path} ({len(frames)} frames)")


# -- Main ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Mini 2D Lattice Boltzmann CFD Solver",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", type=str, default=None, help="Warp device.")
    parser.add_argument("--num-steps", type=int, default=20000, help="Number of simulation steps.")
    parser.add_argument("--Re", type=float, default=200.0, help="Reynolds number.")
    parser.add_argument("--u-in", type=float, default=0.1, help="Inlet velocity.")
    parser.add_argument("--cylinder-d", type=int, default=40, help="Cylinder diameter in cells.")
    parser.add_argument("--grid", type=str, default="800x200", help="Grid size NxM.")
    parser.add_argument("--save-interval", type=int, default=100, help="Save frame every N steps.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory.")

    args = parser.parse_known_args()[0]
    nx, ny = (int(x) for x in args.grid.split("x"))

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)

    with wp.ScopedDevice(args.device):
        solver = LBMSolver(
            nx=nx, ny=ny, re=args.Re,
            u_in=args.u_in, cylinder_d=args.cylinder_d,
        )

        num_steps = args.num_steps
        save_interval = args.save_interval
        total_cells = nx * ny
        bench_interval = 1000
        ramp_steps = 2000

        print(f"\nRunning {num_steps} steps (ramp-up over {ramp_steps} steps)...")
        t_start = time.perf_counter()
        t_bench = t_start

        for step in range(1, num_steps + 1):
            if step < ramp_steps:
                current_u = args.u_in * float(step) / float(ramp_steps)
            else:
                current_u = args.u_in
            solver.step(current_u_in=current_u)

            if step % bench_interval == 0:
                wp.synchronize_device()
                t_now = time.perf_counter()
                elapsed = t_now - t_bench
                mlups = (total_cells * bench_interval) / elapsed / 1.0e6
                print(f"  Step {step:6d}/{num_steps} | {elapsed:.3f}s per {bench_interval} steps | {mlups:.1f} MLUPS")
                t_bench = t_now

            if step % save_interval == 0:
                save_vel = step % (save_interval * 10) == 0
                save_frame(solver, step, output_dir,
                           save_vorticity_png=True, save_velocity_png=save_vel)

        wp.synchronize_device()
        t_end = time.perf_counter()
        total_time = t_end - t_start
        avg_mlups = (total_cells * num_steps) / total_time / 1.0e6

        print(f"\nSimulation complete:")
        print(f"  Total time:    {total_time:.2f}s")
        print(f"  Avg MLUPS:     {avg_mlups:.1f}")
        print(f"  Grid:          {nx} x {ny}")
        print(f"  Re:            {args.Re}")

        save_frame(solver, num_steps, output_dir,
                   save_vorticity_png=True, save_velocity_png=True)

        print("\nCompiling GIF...")
        make_gif(output_dir)
        print("Done.")


if __name__ == "__main__":
    main()
