"""Interactive GPU Ray Marcher with Camera Navigation.

A real-time SDF scene explorer rendered entirely on GPU using Warp.
Navigate a 3D scene of procedural objects with keyboard controls.

Controls:
    WASD        Move forward/back/left/right
    Q/E         Rotate camera left/right
    R/F         Move up/down
    Arrow keys  Rotate camera
    ESC         Exit

Usage:
    uv run --with "pyglet>=2.0" --with matplotlib python interactive_raymarch.py
    uv run --with matplotlib python interactive_raymarch.py --headless
"""

import argparse
import math

import numpy as np

import warp as wp


# ── SDF Primitives ──────────────────────────────────────────────────────────


@wp.func
def sdf_sphere(p: wp.vec3, center: wp.vec3, r: float):
    return wp.length(p - center) - r


@wp.func
def sdf_torus(p: wp.vec3, center: wp.vec3, major_r: float, minor_r: float):
    q = p - center
    xz = wp.sqrt(q[0] * q[0] + q[2] * q[2]) - major_r
    return wp.sqrt(xz * xz + q[1] * q[1]) - minor_r


@wp.func
def sdf_rounded_box(p: wp.vec3, center: wp.vec3, b: wp.vec3, r: float):
    q = p - center
    dx = wp.abs(q[0]) - b[0]
    dy = wp.abs(q[1]) - b[1]
    dz = wp.abs(q[2]) - b[2]
    outer = wp.length(wp.vec3(wp.max(dx, 0.0), wp.max(dy, 0.0), wp.max(dz, 0.0)))
    inner = wp.min(wp.max(dx, wp.max(dy, dz)), 0.0)
    return outer + inner - r


@wp.func
def sdf_plane(p: wp.vec3):
    return p[1]


@wp.func
def smin(a: float, b: float, k: float):
    h = wp.max(k - wp.abs(a - b), 0.0) / k
    return wp.min(a, b) - h * h * k * 0.25


# ── Scene Composition ───────────────────────────────────────────────────────


@wp.func
def sdf_scene(p: wp.vec3):
    d = sdf_plane(p)
    d = smin(d, sdf_sphere(p, wp.vec3(-2.0, 0.8, -1.0), 0.8), 0.3)
    d = smin(d, sdf_sphere(p, wp.vec3(0.0, 0.6, -2.0), 0.6), 0.2)
    d = wp.min(d, sdf_sphere(p, wp.vec3(2.5, 0.7, 0.5), 0.7))
    d = smin(d, sdf_torus(p, wp.vec3(1.0, 0.35, -1.0), 0.6, 0.15), 0.15)
    d = smin(d, sdf_rounded_box(p, wp.vec3(-1.0, 0.5, 1.5), wp.vec3(0.35, 0.35, 0.35), 0.08), 0.2)
    return d


# ── Shading Functions ───────────────────────────────────────────────────────


@wp.func
def scene_normal(p: wp.vec3):
    eps = 1.0e-4
    dx = sdf_scene(p + wp.vec3(eps, 0.0, 0.0)) - sdf_scene(p - wp.vec3(eps, 0.0, 0.0))
    dy = sdf_scene(p + wp.vec3(0.0, eps, 0.0)) - sdf_scene(p - wp.vec3(0.0, eps, 0.0))
    dz = sdf_scene(p + wp.vec3(0.0, 0.0, eps)) - sdf_scene(p - wp.vec3(0.0, 0.0, eps))
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def soft_shadow(ro: wp.vec3, rd: wp.vec3):
    t = float(0.02)
    res = float(1.0)
    for _ in range(64):
        d = sdf_scene(ro + rd * t)
        if d < 0.001:
            return 0.0
        res = wp.min(res, 8.0 * d / t)
        t = t + wp.clamp(d, 0.02, 0.2)
        if t > 20.0:
            return res
    return wp.clamp(res, 0.0, 1.0)


@wp.func
def ambient_occlusion(p: wp.vec3, n: wp.vec3):
    occ = float(0.0)
    scale = float(1.0)
    for i in range(5):
        t = 0.01 + 0.12 * float(i)
        d = sdf_scene(p + n * t)
        occ = occ + (t - d) * scale
        scale = scale * 0.95
    return wp.clamp(1.0 - 3.0 * occ, 0.0, 1.0)


@wp.func
def checkerboard(p: wp.vec3):
    fx = int(wp.floor(p[0]))
    fz = int(wp.floor(p[2]))
    if (fx + fz) % 2 == 0:
        return wp.vec3(0.85, 0.85, 0.85)
    return wp.vec3(0.35, 0.35, 0.35)


@wp.func
def get_material(p: wp.vec3):
    """Return (color, material_id) for the closest surface at point p."""
    d_min = sdf_plane(p)
    color = checkerboard(p)
    mat_id = 0

    d = sdf_sphere(p, wp.vec3(-2.0, 0.8, -1.0), 0.8)
    if d < d_min:
        d_min = d
        color = wp.vec3(0.95, 0.93, 0.88)
        mat_id = 1

    d = sdf_sphere(p, wp.vec3(0.0, 0.6, -2.0), 0.6)
    if d < d_min:
        d_min = d
        color = wp.vec3(0.7, 0.15, 0.15)
        mat_id = 2

    d = sdf_sphere(p, wp.vec3(2.5, 0.7, 0.5), 0.7)
    if d < d_min:
        d_min = d
        color = wp.vec3(1.0, 0.85, 0.2)
        mat_id = 3

    d = sdf_torus(p, wp.vec3(1.0, 0.35, -1.0), 0.6, 0.15)
    if d < d_min:
        d_min = d
        color = wp.vec3(0.2, 0.6, 0.9)
        mat_id = 4

    d = sdf_rounded_box(p, wp.vec3(-1.0, 0.5, 1.5), wp.vec3(0.35, 0.35, 0.35), 0.08)
    if d < d_min:
        d_min = d
        color = wp.vec3(0.4, 0.8, 0.3)
        mat_id = 5

    return color, mat_id


# ── Ray March Kernel ────────────────────────────────────────────────────────


@wp.kernel
def raymarch_kernel(
    cam_pos: wp.vec3,
    cam_rot: wp.quat,
    width: int,
    height: int,
    pixels: wp.array(dtype=wp.vec3),
):
    x, y = wp.tid()

    aspect = float(width) / float(height)
    sx = (2.0 * (float(x) + 0.5) / float(width) - 1.0) * aspect
    sy = 2.0 * (float(y) + 0.5) / float(height) - 1.0

    ro = cam_pos
    rd = wp.quat_rotate(cam_rot, wp.normalize(wp.vec3(sx, sy, -1.5)))

    t = float(0.0)
    for _ in range(128):
        d = sdf_scene(ro + rd * t)
        t = t + d

    idx = y * width + x

    if d < 0.01:
        p = ro + rd * t
        n = scene_normal(p)
        light_dir = wp.normalize(wp.vec3(0.6, 0.8, -0.4))

        color, mat_id = get_material(p)
        ao = ambient_occlusion(p, n)
        diffuse = wp.max(wp.dot(n, light_dir), 0.0)

        h = wp.normalize(light_dir - rd)
        spec_base = wp.clamp(wp.dot(n, h), 0.0, 1.0)
        spec_power = float(32.0)
        if mat_id == 1:
            spec_power = 128.0
        if mat_id == 2:
            spec_power = 8.0
        if mat_id == 4:
            spec_power = 96.0
        specular = spec_base ** spec_power

        shadow_val = soft_shadow(p + n * 0.01, light_dir)

        if mat_id == 3:
            result = color * 1.5
        else:
            result = color * (0.15 * ao + diffuse * shadow_val) + wp.vec3(1.0, 1.0, 1.0) * specular * shadow_val * 0.5

        pixels[idx] = wp.vec3(
            wp.clamp(result[0] ** 0.4545, 0.0, 1.0),
            wp.clamp(result[1] ** 0.4545, 0.0, 1.0),
            wp.clamp(result[2] ** 0.4545, 0.0, 1.0),
        )
    else:
        sky_t = 0.5 * (rd[1] + 1.0)
        sky = wp.vec3(1.0, 1.0, 1.0) * (1.0 - sky_t) + wp.vec3(0.5, 0.7, 1.0) * sky_t
        pixels[idx] = sky


# ── Camera ──────────────────────────────────────────────────────────────────


class Camera:
    def __init__(self, pos=(0.0, 2.0, 5.0), yaw=0.0, pitch=-0.3):
        self.pos = np.array(pos, dtype=np.float64)
        self.yaw = yaw
        self.pitch = pitch
        self.speed = 3.0
        self.rot_speed = 1.5

    def get_forward(self):
        cos_p = math.cos(self.pitch)
        return np.array([-math.sin(self.yaw) * cos_p, math.sin(self.pitch), -math.cos(self.yaw) * cos_p])

    def get_right(self):
        return np.array([math.cos(self.yaw), 0.0, -math.sin(self.yaw)])

    def get_rotation(self):
        yaw_q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), self.yaw)
        pitch_q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), self.pitch)
        # Quaternion composition: apply pitch in local frame, then yaw
        x1, y1, z1, w1 = float(yaw_q[0]), float(yaw_q[1]), float(yaw_q[2]), float(yaw_q[3])
        x2, y2, z2, w2 = float(pitch_q[0]), float(pitch_q[1]), float(pitch_q[2]), float(pitch_q[3])
        return wp.quat(
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        )

    def get_wp_pos(self):
        return wp.vec3(float(self.pos[0]), float(self.pos[1]), float(self.pos[2]))

    def update(self, key_handler, dt):
        from pyglet.window import key  # noqa: PLC0415

        if key_handler[key.LEFT] or key_handler[key.Q]:
            self.yaw += self.rot_speed * dt
        if key_handler[key.RIGHT] or key_handler[key.E]:
            self.yaw -= self.rot_speed * dt

        fwd = self.get_forward()
        right = self.get_right()
        move = np.zeros(3)
        if key_handler[key.W]:
            move += fwd
        if key_handler[key.S]:
            move -= fwd
        if key_handler[key.A]:
            move -= right
        if key_handler[key.D]:
            move += right
        if key_handler[key.R]:
            move[1] += 1.0
        if key_handler[key.F]:
            move[1] -= 1.0

        norm = np.linalg.norm(move)
        if norm > 1e-6:
            self.pos += (move / norm) * self.speed * dt


# ── Headless Renderer ───────────────────────────────────────────────────────


def run_headless(width=2048, height=1024, output="interactive_raymarch.png"):
    import matplotlib.pyplot as plt  # noqa: PLC0415

    camera = Camera()
    pixels = wp.zeros(width * height, dtype=wp.vec3)

    wp.launch(
        raymarch_kernel,
        dim=(width, height),
        inputs=[camera.get_wp_pos(), camera.get_rotation(), width, height, pixels],
    )

    pixels_np = pixels.numpy().reshape(height, width, 3)
    plt.figure(figsize=(16, 8))
    plt.imshow(pixels_np, origin="lower", interpolation="antialiased")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight", pad_inches=0)
    print(f"Saved {output} ({width}x{height})")
    plt.close()


# ── Interactive Renderer ────────────────────────────────────────────────────


def run_interactive(width=1024, height=512):
    import time  # noqa: PLC0415

    import pyglet  # noqa: PLC0415
    from pyglet.window import key  # noqa: PLC0415

    window = pyglet.window.Window(width, height, caption="Interactive GPU Ray Marcher", vsync=False)
    key_handler = key.KeyStateHandler()
    window.push_handlers(key_handler)

    camera = Camera()
    pixels = wp.zeros(width * height, dtype=wp.vec3)

    controls_label = pyglet.text.Label(
        "WASD: move | Q/E: rotate | R/F: up/down | ESC: quit",
        font_size=10,
        x=4,
        y=4,
        color=(200, 200, 200, 180),
    )
    fps_label = pyglet.text.Label(
        "",
        font_size=10,
        x=4,
        y=height - 16,
        color=(200, 200, 200, 180),
    )

    running = [True]

    @window.event
    def on_close():
        running[0] = False

    @window.event
    def on_key_press(symbol, modifiers):
        if symbol == key.ESCAPE:
            running[0] = False

    last_time = time.perf_counter()
    frame_count = 0
    fps_clock = time.perf_counter()
    fps_text = "FPS: --"

    # Manual event loop (matches how warp.render.OpenGLRenderer works internally)
    while running[0]:
        # Process all pending OS events — this updates the KeyStateHandler
        window.dispatch_events()

        # Compute delta time
        now = time.perf_counter()
        dt = now - last_time
        last_time = now

        # Update camera from continuous key states
        camera.update(key_handler, dt)

        # Launch ray march kernel with updated camera
        wp.launch(
            raymarch_kernel,
            dim=(width, height),
            inputs=[camera.get_wp_pos(), camera.get_rotation(), width, height, pixels],
        )

        # Read pixels from GPU (.numpy() implicitly synchronizes)
        pixels_np = pixels.numpy().reshape(height, width, 3)
        pixels_uint8 = (pixels_np * 255).astype(np.uint8)

        # Draw to window
        window.switch_to()
        window.clear()
        image = pyglet.image.ImageData(width, height, "RGB", pixels_uint8.tobytes(), pitch=width * 3)
        image.blit(0, 0)

        # FPS counter
        frame_count += 1
        if now - fps_clock >= 1.0:
            fps_text = f"FPS: {frame_count}"
            frame_count = 0
            fps_clock = now
        fps_label.text = fps_text
        fps_label.draw()
        controls_label.draw()

        window.flip()

    window.close()


# ── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactive GPU Ray Marcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--headless", action="store_true", help="Render a single frame to PNG.")
    parser.add_argument("--width", type=int, default=None, help="Image width in pixels.")
    parser.add_argument("--height", type=int, default=None, help="Image height in pixels.")
    parser.add_argument("--output", type=str, default="interactive_raymarch.png", help="Output PNG path (headless).")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        if args.headless:
            run_headless(args.width or 2048, args.height or 1024, args.output)
        else:
            run_interactive(args.width or 1024, args.height or 512)
