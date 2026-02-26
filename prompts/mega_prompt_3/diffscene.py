"""Differentiable Scene Optimizer: Image-to-3D Primitives.

Optimizes 3D sphere parameters (positions, radii, colors) to match a target
image using a differentiable soft rasterizer and automatic differentiation via
wp.Tape. Uses the Adam optimizer from warp.optim.

The renderer uses a soft, differentiable compositing approach: each sphere
contributes to each pixel based on a smooth opacity function of the signed
distance from the ray to the sphere center, enabling gradient flow through
all parameters.

Usage:
    uv run --with matplotlib --with Pillow python prompts/mega_prompt_3/diffscene.py
    uv run --with matplotlib --with Pillow python prompts/mega_prompt_3/diffscene.py --device cuda:0
    uv run --with matplotlib --with Pillow python prompts/mega_prompt_3/diffscene.py --steps 300 --num-spheres 30
"""

import argparse
import os

import numpy as np

import warp as wp
import warp.optim


# -- Differentiable soft renderer kernel ------------------------------------


@wp.func
def sigmoid(x: float):
    """Numerically stable sigmoid with clamping to avoid exp overflow."""
    cx = wp.clamp(x, -20.0, 20.0)
    return 1.0 / (1.0 + wp.exp(-cx))


@wp.kernel
def render_spheres_soft(
    width: int,
    height: int,
    num_spheres: int,
    sphere_positions: wp.array(dtype=wp.vec3),
    sphere_radii: wp.array(dtype=float),
    sphere_colors: wp.array(dtype=wp.vec3),
    cam_pos: wp.vec3,
    cam_fwd: wp.vec3,
    cam_right: wp.vec3,
    cam_up: wp.vec3,
    light_dir: wp.vec3,
    sharpness: float,
    rendered: wp.array(dtype=wp.vec3),
):
    """Soft differentiable sphere renderer using front-to-back alpha compositing.

    For each pixel, we shoot a ray and compute a smooth opacity contribution
    from each sphere based on the closest distance from the ray to the sphere
    surface. Spheres are composited in a fixed order (sorted by index) using
    the over operator: C_out = alpha * C_sphere + (1 - alpha) * C_accum.
    """
    tid = wp.tid()
    px = tid % width
    py = tid / width

    aspect = float(width) / float(height)
    u = (2.0 * (float(px) + 0.5) / float(width) - 1.0) * aspect
    v = 2.0 * (float(py) + 0.5) / float(height) - 1.0

    ray_o = cam_pos
    ray_d = wp.normalize(cam_fwd + cam_right * u + cam_up * v)

    # Front-to-back compositing
    accum_color = wp.vec3(0.0, 0.0, 0.0)
    transmittance = float(1.0)

    for i in range(num_spheres):
        center = sphere_positions[i]
        radius = wp.abs(sphere_radii[i]) + 0.05

        # Compute closest approach of ray to sphere center
        oc = ray_o - center
        b = wp.dot(oc, ray_d)
        # Squared distance from ray to sphere center
        c = wp.dot(oc, oc) - b * b
        # Signed distance from ray to sphere surface (negative = inside)
        dist_to_surface = wp.sqrt(wp.max(c, 0.0001)) - radius

        # Soft alpha: sigmoid of signed distance, scaled by sharpness
        alpha = sigmoid(-dist_to_surface * sharpness)

        # Compute approximate normal for lighting (project center onto ray)
        t_closest = -b
        t_hit = wp.max(t_closest, 0.1)
        hit_point = ray_o + ray_d * t_hit
        normal = wp.normalize(hit_point - center)

        # Diffuse shading
        ndotl = wp.max(wp.dot(normal, light_dir), 0.0)
        ambient = float(0.15)

        raw_c = sphere_colors[i]
        base_r = wp.clamp(raw_c[0], 0.0, 1.0)
        base_g = wp.clamp(raw_c[1], 0.0, 1.0)
        base_b = wp.clamp(raw_c[2], 0.0, 1.0)
        base_color = wp.vec3(base_r, base_g, base_b)

        shaded = base_color * (ambient + ndotl * 0.85)

        # Over operator
        contribution = alpha * transmittance
        accum_color = accum_color + shaded * contribution
        transmittance = transmittance * (1.0 - alpha)

    # Background: sky gradient
    sky_t = 0.5 * (ray_d[1] + 1.0)
    bg = wp.vec3(0.8, 0.85, 0.95) * (1.0 - sky_t) + wp.vec3(0.4, 0.55, 0.8) * sky_t
    accum_color = accum_color + bg * transmittance

    rendered[tid] = accum_color


# -- Loss kernel -------------------------------------------------------------


@wp.kernel
def compute_loss(
    rendered: wp.array(dtype=wp.vec3),
    target: wp.array(dtype=wp.vec3),
    loss: wp.array(dtype=float),
):
    tid = wp.tid()
    diff = rendered[tid] - target[tid]
    wp.atomic_add(loss, 0, wp.dot(diff, diff))


# -- Utility functions -------------------------------------------------------


def save_image(pixels_np, width, height, path):
    """Save a pixel array as PNG using Pillow."""
    from PIL import Image  # noqa: PLC0415

    img_data = pixels_np.reshape(height, width, 3)
    # Flip vertically (screen y-up to image y-down)
    img_data = img_data[::-1]
    # Replace NaN with 0 before clipping
    img_data = np.nan_to_num(img_data, nan=0.0)
    img_data = np.clip(img_data * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_data)
    img.save(path)


def make_camera(distance=5.0, elevation=0.3):
    """Create a fixed camera looking at the origin."""
    cam_pos = wp.vec3(0.0, elevation, distance)

    fwd_np = np.array([0.0 - 0.0, 0.0 - elevation, 0.0 - distance])
    fwd_np = fwd_np / np.linalg.norm(fwd_np)

    world_up = np.array([0.0, 1.0, 0.0])
    right_np = np.cross(fwd_np, world_up)
    right_np = right_np / np.linalg.norm(right_np)
    up_np = np.cross(right_np, fwd_np)

    cam_fwd = wp.vec3(float(fwd_np[0]), float(fwd_np[1]), float(fwd_np[2]))
    cam_right = wp.vec3(float(right_np[0]), float(right_np[1]), float(right_np[2]))
    cam_up = wp.vec3(float(up_np[0]), float(up_np[1]), float(up_np[2]))

    return cam_pos, cam_fwd, cam_right, cam_up


def create_target_scene():
    """Create a known scene of 5 spheres as the optimization target."""
    positions = np.array(
        [
            [-1.2, 0.0, -0.5],
            [1.0, 0.3, 0.2],
            [0.0, -0.5, -1.0],
            [-0.5, 0.6, 0.8],
            [0.8, -0.3, -0.8],
        ],
        dtype=np.float32,
    )
    radii = np.array([0.45, 0.35, 0.5, 0.3, 0.4], dtype=np.float32)
    colors = np.array(
        [
            [0.9, 0.2, 0.2],  # Red
            [0.2, 0.8, 0.3],  # Green
            [0.2, 0.3, 0.9],  # Blue
            [0.9, 0.8, 0.1],  # Yellow
            [0.8, 0.3, 0.8],  # Purple
        ],
        dtype=np.float32,
    )
    return positions, radii, colors


def render_target(positions, radii, colors, width, height, cam_pos, cam_fwd, cam_right, cam_up, light_dir, sharpness):
    """Render the target scene (no gradients needed)."""
    num_spheres = len(positions)
    pos_wp = wp.array(positions, dtype=wp.vec3)
    rad_wp = wp.array(radii, dtype=float)
    col_wp = wp.array(colors, dtype=wp.vec3)
    rendered = wp.zeros(width * height, dtype=wp.vec3)

    wp.launch(
        render_spheres_soft,
        dim=width * height,
        inputs=[
            width,
            height,
            num_spheres,
            pos_wp,
            rad_wp,
            col_wp,
            cam_pos,
            cam_fwd,
            cam_right,
            cam_up,
            light_dir,
            sharpness,
            rendered,
        ],
    )

    return rendered


def run(
    width=512,
    height=512,
    num_spheres=20,
    num_steps=500,
    lr=0.01,
    save_every=50,
    output_dir=None,
):
    from PIL import Image  # noqa: PLC0415

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    sharpness = 15.0

    # Camera and lighting
    cam_pos, cam_fwd, cam_right, cam_up = make_camera(distance=5.0, elevation=0.3)
    light_dir_np = np.array([0.5, 0.8, 0.6], dtype=np.float32)
    light_dir_np = light_dir_np / np.linalg.norm(light_dir_np)
    light_dir = wp.vec3(float(light_dir_np[0]), float(light_dir_np[1]), float(light_dir_np[2]))

    # Generate and save target image
    target_pos, target_rad, target_col = create_target_scene()
    target_rendered = render_target(
        target_pos, target_rad, target_col, width, height, cam_pos, cam_fwd, cam_right, cam_up, light_dir, sharpness
    )
    target_np = target_rendered.numpy()
    target_path = os.path.join(output_dir, "target.png")
    save_image(target_np, width, height, target_path)
    print(f"Saved target image: {target_path}")

    # Keep target as a non-grad array for the loss
    target_pixels = wp.array(target_np, dtype=wp.vec3)

    # Initialize random spheres for optimization
    rng = np.random.default_rng(42)
    init_positions = rng.uniform(-1.5, 1.5, size=(num_spheres, 3)).astype(np.float32)
    init_radii = rng.uniform(0.1, 0.5, size=num_spheres).astype(np.float32)
    init_colors = rng.uniform(0.1, 0.9, size=(num_spheres, 3)).astype(np.float32)

    # Create differentiable parameter arrays
    sphere_positions = wp.array(init_positions, dtype=wp.vec3, requires_grad=True)
    sphere_radii = wp.array(init_radii, dtype=float, requires_grad=True)
    sphere_colors = wp.array(init_colors, dtype=wp.vec3, requires_grad=True)

    # Rendered image buffer (needs grad for backprop through loss)
    rendered = wp.zeros(width * height, dtype=wp.vec3, requires_grad=True)
    loss = wp.zeros(1, dtype=float, requires_grad=True)

    # Adam optimizer
    optimizer = warp.optim.Adam(
        [sphere_positions, sphere_radii, sphere_colors],
        lr=lr,
    )

    # Collect progress frames for GIF
    progress_frames = []
    loss_history = []

    print(f"Optimizing {num_spheres} spheres over {num_steps} steps...")
    print(f"Target: 5 spheres, Image: {width}x{height}")
    print(f"Sharpness: {sharpness}, Learning rate: {lr}")
    print("-" * 50)

    for step in range(num_steps):
        # Zero buffers
        rendered.zero_()
        loss.zero_()

        # Forward pass with tape for autodiff
        tape = wp.Tape()
        with tape:
            wp.launch(
                render_spheres_soft,
                dim=width * height,
                inputs=[
                    width,
                    height,
                    num_spheres,
                    sphere_positions,
                    sphere_radii,
                    sphere_colors,
                    cam_pos,
                    cam_fwd,
                    cam_right,
                    cam_up,
                    light_dir,
                    sharpness,
                    rendered,
                ],
            )
            wp.launch(
                compute_loss,
                dim=width * height,
                inputs=[rendered, target_pixels, loss],
            )

        # Backward pass
        tape.backward(loss)

        # Optimizer step
        optimizer.step([
            tape.gradients[sphere_positions],
            tape.gradients[sphere_radii],
            tape.gradients[sphere_colors],
        ])

        # Zero gradients
        tape.zero()

        # Record loss
        loss_val = loss.numpy()[0]
        loss_history.append(loss_val)

        if step % save_every == 0 or step == num_steps - 1:
            print(f"Step {step:4d}/{num_steps}  Loss: {loss_val:.2f}")

            # Save progress image
            img_np = rendered.numpy()
            img_path = os.path.join(output_dir, f"optimized_step_{step:03d}.png")
            save_image(img_np, width, height, img_path)

            # Collect frame for GIF
            img_data = img_np.reshape(height, width, 3)[::-1]
            img_data = np.clip(img_data * 255, 0, 255).astype(np.uint8)
            progress_frames.append(Image.fromarray(img_data))

    # Save final step if not already saved
    if (num_steps - 1) % save_every != 0:
        img_np = rendered.numpy()
        img_path = os.path.join(output_dir, f"optimized_step_{num_steps - 1:03d}.png")
        save_image(img_np, width, height, img_path)
        img_data = img_np.reshape(height, width, 3)[::-1]
        img_data = np.clip(img_data * 255, 0, 255).astype(np.uint8)
        progress_frames.append(Image.fromarray(img_data))

    # Save optimization GIF
    if len(progress_frames) > 1:
        gif_path = os.path.join(output_dir, "optimization.gif")
        progress_frames[0].save(
            gif_path,
            save_all=True,
            append_images=progress_frames[1:],
            duration=300,
            loop=0,
        )
        print(f"Saved optimization GIF: {gif_path}")

    # Save loss plot
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(loss_history)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss (MSE)")
        ax.set_title("Optimization Convergence")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        plot_path = os.path.join(output_dir, "loss_curve.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved loss curve: {plot_path}")
    except ImportError:
        pass

    print("-" * 50)
    print(f"Initial loss: {loss_history[0]:.2f}")
    print(f"Final loss:   {loss_history[-1]:.2f}")
    if loss_history[-1] > 0:
        print(f"Reduction:    {loss_history[0] / loss_history[-1]:.1f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Differentiable Scene Optimizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", type=str, default=None, help="Warp device (cpu, cuda:0, etc.)")
    parser.add_argument("--width", type=int, default=512, help="Image width.")
    parser.add_argument("--height", type=int, default=512, help="Image height.")
    parser.add_argument("--num-spheres", type=int, default=20, help="Number of spheres to optimize.")
    parser.add_argument("--steps", type=int, default=500, help="Optimization steps.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for Adam optimizer.")
    parser.add_argument("--save-every", type=int, default=50, help="Save progress image every N steps.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        run(
            width=args.width,
            height=args.height,
            num_spheres=args.num_spheres,
            num_steps=args.steps,
            lr=args.lr,
            save_every=args.save_every,
        )
