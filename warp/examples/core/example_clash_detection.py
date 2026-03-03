# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Clash Detection
#
# Demonstrates a complete clash detection pipeline using Warp:
#   1. Procedural generation of building/CAD-like geometry (beams, pipes, plates)
#   2. Broadphase AABB overlap detection via BVH queries
#   3. Narrowphase triangle-triangle intersection tests
#   4. Clearance (soft clash) detection via mesh point queries
#   5. Ray-cast rendering of the scene with clash highlighting
#   6. Image output via matplotlib
#
# Clash types detected:
#   - Hard clash: triangles of two objects physically intersect
#   - Soft clash: objects are within a specified clearance tolerance
#
###########################################################################

import math

import numpy as np

import warp as wp

# ---------------------------------------------------------------------------
# Warp helper functions
# ---------------------------------------------------------------------------


@wp.func
def cw_min(a: wp.vec3, b: wp.vec3):
    return wp.vec3(wp.min(a[0], b[0]), wp.min(a[1], b[1]), wp.min(a[2], b[2]))


@wp.func
def cw_max(a: wp.vec3, b: wp.vec3):
    return wp.vec3(wp.max(a[0], b[0]), wp.max(a[1], b[1]), wp.max(a[2], b[2]))


# ---------------------------------------------------------------------------
# Kernel: hard clash detection (triangle-triangle intersection)
# ---------------------------------------------------------------------------


@wp.kernel
def detect_hard_clash(
    mesh_a: wp.uint64,
    mesh_b: wp.uint64,
    num_faces_a: int,
    result: wp.array(dtype=int),
):
    """Test every face of mesh_a against the BVH of mesh_b for intersection."""
    tid = wp.tid()
    face = tid

    if face >= num_faces_a:
        return

    # Load triangle vertices from mesh_a
    v0 = wp.mesh_eval_position(mesh_a, face, 1.0, 0.0)
    v1 = wp.mesh_eval_position(mesh_a, face, 0.0, 1.0)
    v2 = wp.mesh_eval_position(mesh_a, face, 0.0, 0.0)

    # Compute AABB of this triangle
    lower = cw_min(cw_min(v0, v1), v2)
    upper = cw_max(cw_max(v0, v1), v2)

    # Query mesh_b's BVH for candidate faces
    query = wp.mesh_query_aabb(mesh_b, lower, upper)

    for f in query:
        u0 = wp.mesh_eval_position(mesh_b, f, 1.0, 0.0)
        u1 = wp.mesh_eval_position(mesh_b, f, 0.0, 1.0)
        u2 = wp.mesh_eval_position(mesh_b, f, 0.0, 0.0)

        if wp.intersect_tri_tri(v0, v1, v2, u0, u1, u2) > 0:
            # Record the intersecting face index (any non-zero signals a clash)
            wp.atomic_add(result, 0, 1)
            return


# ---------------------------------------------------------------------------
# Kernel: soft clash / clearance detection (closest-point distance)
# ---------------------------------------------------------------------------


@wp.kernel
def detect_soft_clash(
    mesh_a: wp.uint64,
    sample_points: wp.array(dtype=wp.vec3),
    clearance: float,
    min_dist_out: wp.array(dtype=float),
    violation_count: wp.array(dtype=int),
):
    """For each sample point, find the closest point on mesh_a.

    If the distance is less than *clearance*, count it as a soft clash.
    """
    tid = wp.tid()
    pt = sample_points[tid]

    query = wp.mesh_query_point_sign_normal(mesh_a, pt, clearance)
    if query.result:
        p = wp.mesh_eval_position(mesh_a, query.face, query.u, query.v)
        dist = wp.length(pt - p)

        # Track minimum distance (atomically)
        wp.atomic_min(min_dist_out, 0, dist)

        if dist < clearance:
            wp.atomic_add(violation_count, 0, 1)


# ---------------------------------------------------------------------------
# Kernel: ray-cast rendering with clash color coding
# ---------------------------------------------------------------------------


@wp.kernel
def render_scene(
    mesh_ids: wp.array(dtype=wp.uint64),
    mesh_colors: wp.array(dtype=wp.vec3),
    num_meshes: int,
    cam_pos: wp.vec3,
    cam_forward: wp.vec3,
    cam_right: wp.vec3,
    cam_up: wp.vec3,
    width: int,
    height: int,
    fov_scale: float,
    pixels: wp.array(dtype=wp.vec3),
):
    """Cast one ray per pixel and shade by the closest mesh hit."""
    tid = wp.tid()

    x = tid % width
    y = tid // width

    # Normalized device coordinates [-1, 1]
    aspect = float(width) / float(height)
    sx = (2.0 * float(x) / float(width) - 1.0) * aspect * fov_scale
    sy = (2.0 * float(y) / float(height) - 1.0) * fov_scale

    ro = cam_pos
    rd = wp.normalize(cam_forward + cam_right * sx + cam_up * sy)

    best_t = float(1.0e12)
    best_color = wp.vec3(0.18, 0.18, 0.22)  # background

    for m in range(num_meshes):
        mesh_id = mesh_ids[m]
        query = wp.mesh_query_ray(mesh_id, ro, rd, best_t)
        if query.result:
            if query.t < best_t:
                best_t = query.t

                # Simple directional lighting
                n = query.normal
                if query.sign < 0.0:
                    n = n * (-1.0)

                light_dir = wp.normalize(wp.vec3(0.3, 1.0, 0.5))
                ndotl = wp.max(wp.dot(n, light_dir), 0.0)

                base = mesh_colors[m]
                ambient = base * 0.25
                diffuse = base * ndotl * 0.75
                best_color = ambient + diffuse

    pixels[tid] = best_color


# ---------------------------------------------------------------------------
# Procedural geometry helpers (CPU / NumPy)
# ---------------------------------------------------------------------------


def make_box(center, half_extents):
    """Create a triangulated axis-aligned box.

    Returns (vertices, indices) as NumPy arrays.
    """
    cx, cy, cz = center
    hx, hy, hz = half_extents

    verts = np.array(
        [
            [cx - hx, cy - hy, cz - hz],
            [cx + hx, cy - hy, cz - hz],
            [cx + hx, cy + hy, cz - hz],
            [cx - hx, cy + hy, cz - hz],
            [cx - hx, cy - hy, cz + hz],
            [cx + hx, cy - hy, cz + hz],
            [cx + hx, cy + hy, cz + hz],
            [cx - hx, cy + hy, cz + hz],
        ],
        dtype=np.float32,
    )

    # 12 triangles (2 per face)
    tris = np.array(
        [
            # -Z face
            0,
            1,
            2,
            0,
            2,
            3,
            # +Z face
            4,
            6,
            5,
            4,
            7,
            6,
            # -Y face
            0,
            5,
            1,
            0,
            4,
            5,
            # +Y face
            3,
            2,
            6,
            3,
            6,
            7,
            # -X face
            0,
            3,
            7,
            0,
            7,
            4,
            # +X face
            1,
            5,
            6,
            1,
            6,
            2,
        ],
        dtype=np.int32,
    )
    return verts, tris


def make_cylinder(center, radius, height, segments=16):
    """Create a triangulated cylinder along the Y axis.

    Returns (vertices, indices) as NumPy arrays.
    """
    cx, cy, cz = center
    verts = []
    tris = []

    # Bottom and top center vertices
    bottom_center = len(verts)
    verts.append([cx, cy - height / 2, cz])
    top_center = bottom_center + 1
    verts.append([cx, cy + height / 2, cz])

    # Ring vertices
    bottom_start = len(verts)
    for i in range(segments):
        angle = 2.0 * math.pi * i / segments
        x = cx + radius * math.cos(angle)
        z = cz + radius * math.sin(angle)
        verts.append([x, cy - height / 2, z])

    top_start = len(verts)
    for i in range(segments):
        angle = 2.0 * math.pi * i / segments
        x = cx + radius * math.cos(angle)
        z = cz + radius * math.sin(angle)
        verts.append([x, cy + height / 2, z])

    # Bottom cap
    for i in range(segments):
        i_next = (i + 1) % segments
        tris.extend([bottom_center, bottom_start + i_next, bottom_start + i])

    # Top cap
    for i in range(segments):
        i_next = (i + 1) % segments
        tris.extend([top_center, top_start + i, top_start + i_next])

    # Side quads (two triangles each)
    for i in range(segments):
        i_next = (i + 1) % segments
        b0 = bottom_start + i
        b1 = bottom_start + i_next
        t0 = top_start + i
        t1 = top_start + i_next
        tris.extend([b0, b1, t1])
        tris.extend([b0, t1, t0])

    return np.array(verts, dtype=np.float32), np.array(tris, dtype=np.int32)


def make_sphere(center, radius, rings=12, segments=16):
    """Create a triangulated UV sphere.

    Returns (vertices, indices) as NumPy arrays.
    """
    cx, cy, cz = center
    verts = []
    tris = []

    # Top pole
    verts.append([cx, cy + radius, cz])

    # Intermediate rings
    for i in range(1, rings):
        phi = math.pi * i / rings
        for j in range(segments):
            theta = 2.0 * math.pi * j / segments
            x = cx + radius * math.sin(phi) * math.cos(theta)
            y = cy + radius * math.cos(phi)
            z = cz + radius * math.sin(phi) * math.sin(theta)
            verts.append([x, y, z])

    # Bottom pole
    bottom_idx = len(verts)
    verts.append([cx, cy - radius, cz])

    # Top cap triangles
    for j in range(segments):
        j_next = (j + 1) % segments
        tris.extend([0, 1 + j, 1 + j_next])

    # Middle quads
    for i in range(rings - 2):
        for j in range(segments):
            j_next = (j + 1) % segments
            row0 = 1 + i * segments
            row1 = 1 + (i + 1) * segments
            tris.extend([row0 + j, row1 + j, row1 + j_next])
            tris.extend([row0 + j, row1 + j_next, row0 + j_next])

    # Bottom cap triangles
    last_ring_start = 1 + (rings - 2) * segments
    for j in range(segments):
        j_next = (j + 1) % segments
        tris.extend([bottom_idx, last_ring_start + j_next, last_ring_start + j])

    return np.array(verts, dtype=np.float32), np.array(tris, dtype=np.int32)


def sample_surface_points(vertices, indices, num_samples, rng):
    """Uniformly sample points on the surface of a triangle mesh."""
    vertices = np.asarray(vertices, dtype=np.float64)
    tri_indices = indices.reshape(-1, 3)
    v0 = vertices[tri_indices[:, 0]]
    v1 = vertices[tri_indices[:, 1]]
    v2 = vertices[tri_indices[:, 2]]

    # Compute triangle areas for weighted sampling
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    probs = areas / areas.sum()

    # Choose triangles proportional to area
    chosen = rng.choice(len(tri_indices), size=num_samples, p=probs)

    # Random barycentric coordinates
    r1 = rng.random(num_samples)
    r2 = rng.random(num_samples)
    sqrt_r1 = np.sqrt(r1)
    u = 1.0 - sqrt_r1
    v = sqrt_r1 * (1.0 - r2)
    w = sqrt_r1 * r2

    points = u[:, None] * v0[chosen] + v[:, None] * v1[chosen] + w[:, None] * v2[chosen]
    return points.astype(np.float32)


# ---------------------------------------------------------------------------
# ClashDetector: main class
# ---------------------------------------------------------------------------


class ClashResult:
    """Container for the result of a single pairwise clash test."""

    def __init__(self, obj_a_name, obj_b_name):
        self.obj_a = obj_a_name
        self.obj_b = obj_b_name
        self.hard_clash_count = 0
        self.soft_clash_count = 0
        self.min_distance = float("inf")

    @property
    def has_hard_clash(self):
        return self.hard_clash_count > 0

    @property
    def has_soft_clash(self):
        return self.soft_clash_count > 0 and not self.has_hard_clash

    def __repr__(self):
        status = "HARD CLASH" if self.has_hard_clash else ("SOFT CLASH" if self.has_soft_clash else "CLEAR")
        return (
            f"ClashResult({self.obj_a} vs {self.obj_b}: {status}, "
            f"hard={self.hard_clash_count}, soft={self.soft_clash_count}, "
            f"min_dist={self.min_distance:.4f})"
        )


class ClashDetector:
    """GPU-accelerated clash detection for triangulated geometry.

    Args:
        clearance: Distance threshold for soft-clash detection.
        surface_samples: Number of surface sample points for clearance checks.
    """

    def __init__(self, clearance=0.15, surface_samples=2048):
        self.clearance = clearance
        self.surface_samples = surface_samples
        self.objects = {}  # name -> {verts, tris, mesh, color}
        self.results = []
        self.rng = np.random.default_rng(42)

    # -- Object registration --------------------------------------------------

    def add_object(self, name, vertices, indices, color=(0.5, 0.5, 0.5)):
        """Register a triangulated mesh for clash testing.

        Args:
            name: Unique identifier for this object.
            vertices: (N, 3) float32 vertex positions.
            indices: Flat int32 triangle index array (length divisible by 3).
            color: RGB display color in [0, 1].
        """
        verts_wp = wp.array(np.asarray(vertices, dtype=np.float32), dtype=wp.vec3)
        inds_wp = wp.array(np.asarray(indices, dtype=np.int32), dtype=int)
        mesh = wp.Mesh(points=verts_wp, indices=inds_wp)

        self.objects[name] = {
            "vertices": np.asarray(vertices, dtype=np.float32),
            "indices": np.asarray(indices, dtype=np.int32),
            "mesh": mesh,
            "color": color,
        }

    def add_box(self, name, center, half_extents, color=(0.5, 0.5, 0.5)):
        v, t = make_box(center, half_extents)
        self.add_object(name, v, t, color)

    def add_cylinder(self, name, center, radius, height, segments=16, color=(0.5, 0.5, 0.5)):
        v, t = make_cylinder(center, radius, height, segments)
        self.add_object(name, v, t, color)

    def add_sphere(self, name, center, radius, rings=12, segments=16, color=(0.5, 0.5, 0.5)):
        v, t = make_sphere(center, radius, rings, segments)
        self.add_object(name, v, t, color)

    # -- Detection -------------------------------------------------------------

    def run_detection(self):
        """Run pairwise clash detection on all registered objects.

        Returns:
            List of :class:`ClashResult` for each pair.
        """
        self.results = []
        names = list(self.objects.keys())

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                result = self._check_pair(names[i], names[j])
                self.results.append(result)

        return self.results

    def _check_pair(self, name_a, name_b):
        obj_a = self.objects[name_a]
        obj_b = self.objects[name_b]

        result = ClashResult(name_a, name_b)

        mesh_a = obj_a["mesh"]
        mesh_b = obj_b["mesh"]

        num_faces_a = len(obj_a["indices"]) // 3

        # --- Hard clash: triangle-triangle intersection ---
        hard_result = wp.zeros(1, dtype=int)
        wp.launch(
            kernel=detect_hard_clash,
            dim=num_faces_a,
            inputs=[mesh_a.id, mesh_b.id, num_faces_a, hard_result],
        )
        result.hard_clash_count = int(hard_result.numpy()[0])

        # --- Soft clash: clearance violation via surface sampling ---
        samples_b = sample_surface_points(obj_b["vertices"], obj_b["indices"], self.surface_samples, self.rng)
        sample_pts = wp.array(samples_b, dtype=wp.vec3)
        min_dist = wp.array([1e12], dtype=float)
        violation_count = wp.zeros(1, dtype=int)

        wp.launch(
            kernel=detect_soft_clash,
            dim=self.surface_samples,
            inputs=[mesh_a.id, sample_pts, self.clearance, min_dist, violation_count],
        )

        result.soft_clash_count = int(violation_count.numpy()[0])
        result.min_distance = float(min_dist.numpy()[0])

        return result

    # -- Visualization ---------------------------------------------------------

    def render_image(self, width=1024, height=768, cam_pos=None, cam_target=None):
        """Ray-cast the scene and return an (H, W, 3) float32 image.

        Clash-involved objects are recolored: red for hard clashes,
        yellow for soft clashes.

        Args:
            width: Image width in pixels.
            height: Image height in pixels.
            cam_pos: Camera position as (x, y, z). Auto-computed if ``None``.
            cam_target: Look-at target as (x, y, z). Auto-computed if ``None``.
        """
        # Determine clash colors
        clash_colors = self._compute_clash_colors()

        names = list(self.objects.keys())
        num_meshes = len(names)

        mesh_ids_np = np.array([self.objects[n]["mesh"].id for n in names], dtype=np.uint64)
        mesh_colors_np = np.array([clash_colors[n] for n in names], dtype=np.float32)

        mesh_ids = wp.array(mesh_ids_np, dtype=wp.uint64)
        mesh_colors = wp.array(mesh_colors_np, dtype=wp.vec3)

        # Auto-compute camera if needed
        if cam_pos is None or cam_target is None:
            all_verts = np.concatenate([self.objects[n]["vertices"] for n in names], axis=0)
            scene_center = all_verts.mean(axis=0)
            scene_extent = all_verts.max(axis=0) - all_verts.min(axis=0)
            max_extent = float(scene_extent.max())

            if cam_target is None:
                cam_target = tuple(scene_center.tolist())
            if cam_pos is None:
                cam_pos = (
                    scene_center[0] + max_extent * 1.2,
                    scene_center[1] + max_extent * 0.6,
                    scene_center[2] + max_extent * 1.2,
                )

        # Build camera basis
        forward = np.array(cam_target) - np.array(cam_pos)
        forward = forward / np.linalg.norm(forward)
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        fov_scale = 0.6  # ~60 degree FOV

        pixels = wp.zeros(width * height, dtype=wp.vec3)

        wp.launch(
            kernel=render_scene,
            dim=width * height,
            inputs=[
                mesh_ids,
                mesh_colors,
                num_meshes,
                wp.vec3(*cam_pos),
                wp.vec3(*forward.tolist()),
                wp.vec3(*right.tolist()),
                wp.vec3(*up.tolist()),
                width,
                height,
                fov_scale,
                pixels,
            ],
        )

        image = pixels.numpy().reshape((height, width, 3))
        # Flip vertically so Y-up in world maps to top of image
        image = image[::-1]
        return np.clip(image, 0.0, 1.0)

    def _compute_clash_colors(self):
        """Determine display color for each object based on clash results."""
        colors = {}
        for name, obj in self.objects.items():
            colors[name] = obj["color"]

        for r in self.results:
            if r.has_hard_clash:
                colors[r.obj_a] = (0.9, 0.15, 0.15)  # red
                colors[r.obj_b] = (0.9, 0.15, 0.15)
            elif r.has_soft_clash:
                if colors[r.obj_a] != (0.9, 0.15, 0.15):
                    colors[r.obj_a] = (0.95, 0.75, 0.1)  # yellow
                if colors[r.obj_b] != (0.9, 0.15, 0.15):
                    colors[r.obj_b] = (0.95, 0.75, 0.1)

        return colors

    def render_usd(self, stage_path="clash_detection.usd"):
        """Export the scene to a USD file with clash-colored meshes.

        Args:
            stage_path: Output file path.
        """
        import warp.render  # noqa: PLC0415

        renderer = warp.render.UsdRenderer(stage_path)
        clash_colors = self._compute_clash_colors()

        renderer.begin_frame(0.0)

        for name, obj in self.objects.items():
            renderer.render_mesh(
                name=name,
                points=obj["vertices"],
                indices=obj["indices"],
                colors=clash_colors[name],
            )

        renderer.end_frame()
        renderer.save()
        return renderer

    def print_report(self):
        """Print a human-readable clash report to stdout."""
        print("=" * 60)
        print("CLASH DETECTION REPORT")
        print("=" * 60)

        hard_count = sum(1 for r in self.results if r.has_hard_clash)
        soft_count = sum(1 for r in self.results if r.has_soft_clash)
        clear_count = len(self.results) - hard_count - soft_count

        print(f"Total pairs checked: {len(self.results)}")
        print(f"  Hard clashes:  {hard_count}")
        print(f"  Soft clashes:  {soft_count}")
        print(f"  Clear:         {clear_count}")
        print("-" * 60)

        for r in self.results:
            print(f"  {r}")

        print("=" * 60)


# ---------------------------------------------------------------------------
# Example scene setup
# ---------------------------------------------------------------------------


class Example:
    """Clash detection demo with a building-like scene.

    Creates beams, pipes, and plates that intentionally overlap to demonstrate
    hard and soft clash detection.
    """

    def __init__(self, stage_path="example_clash_detection.usd", width=1024, height=768):
        self.stage_path = stage_path
        self.width = width
        self.height = height

        self.detector = ClashDetector(clearance=0.15, surface_samples=2048)
        self._build_scene()

        self.image = None

    def _build_scene(self):
        """Construct a scene with known clashes."""
        d = self.detector

        # --- Structural beams (boxes) ---
        # Horizontal beam along X
        d.add_box(
            "beam_horizontal",
            center=(0.0, 0.0, 0.0),
            half_extents=(2.0, 0.15, 0.15),
            color=(0.4, 0.55, 0.7),
        )

        # Vertical column
        d.add_box(
            "column_a",
            center=(-1.5, 1.0, 0.0),
            half_extents=(0.15, 1.0, 0.15),
            color=(0.4, 0.55, 0.7),
        )

        # --- Pipe (cylinder) that INTERSECTS the horizontal beam (hard clash) ---
        d.add_cylinder(
            "pipe_a",
            center=(0.5, 0.0, 0.0),
            radius=0.25,
            height=1.5,
            segments=20,
            color=(0.6, 0.35, 0.2),
        )

        # --- Pipe running CLOSE to the beam but not touching (soft clash) ---
        d.add_cylinder(
            "pipe_b",
            center=(-0.5, 0.25, 0.0),
            radius=0.12,
            height=1.0,
            segments=20,
            color=(0.6, 0.35, 0.2),
        )

        # --- Plate that clips through the column (hard clash) ---
        d.add_box(
            "plate",
            center=(-1.5, 1.8, 0.0),
            half_extents=(0.6, 0.04, 0.6),
            color=(0.3, 0.65, 0.3),
        )

        # --- Equipment box well clear of everything ---
        d.add_box(
            "equipment",
            center=(3.0, 0.5, 2.0),
            half_extents=(0.4, 0.4, 0.4),
            color=(0.55, 0.55, 0.55),
        )

        # --- Spherical fitting on pipe_a (hard clash with beam) ---
        d.add_sphere(
            "fitting",
            center=(0.5, -0.4, 0.0),
            radius=0.2,
            rings=10,
            segments=16,
            color=(0.7, 0.5, 0.2),
        )

    def step(self):
        """Run clash detection."""
        with wp.ScopedTimer("clash_detection"):
            self.detector.run_detection()

    def render(self):
        """Render the scene to an image and optionally to USD."""
        with wp.ScopedTimer("render"):
            self.image = self.detector.render_image(
                width=self.width,
                height=self.height,
            )

        if self.stage_path:
            self.detector.render_usd(self.stage_path)

    def save_image(self, path="clash_detection.png"):
        """Save the rendered image to a PNG file via matplotlib."""
        if self.image is None:
            return

        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415

        fig, ax = plt.subplots(1, 1, figsize=(self.width / 100, self.height / 100), dpi=100)
        ax.imshow(self.image, origin="lower", interpolation="antialiased")
        ax.set_axis_off()
        fig.tight_layout(pad=0)
        fig.savefig(path, dpi=100, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        print(f"Saved clash detection image to {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Clash detection tool using Warp",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage-path",
        type=lambda x: None if x == "None" else str(x),
        default="example_clash_detection.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--width", type=int, default=1024, help="Render width in pixels.")
    parser.add_argument("--height", type=int, default=768, help="Render height in pixels.")
    parser.add_argument("--image-path", type=str, default="clash_detection.png", help="Output image path.")
    parser.add_argument("--clearance", type=float, default=0.15, help="Soft-clash clearance tolerance.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode, suppressing the opening of any graphical windows.",
    )

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, width=args.width, height=args.height)
        example.detector.clearance = args.clearance

        example.step()
        example.detector.print_report()
        example.render()
        example.save_image(args.image_path)

        if not args.headless and example.image is not None:
            try:
                import matplotlib.pyplot as plt

                plt.imshow(example.image, origin="lower", interpolation="antialiased")
                plt.title("Clash Detection Results")
                plt.axis("off")
                plt.show()
            except Exception:
                pass
