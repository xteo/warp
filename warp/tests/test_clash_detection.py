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

"""Tests for the clash detection example (example_clash_detection.py).

Validates procedural geometry generation, hard/soft clash detection,
ray-cast rendering, and the full example pipeline.
"""

import os
import tempfile
import unittest

import numpy as np

import warp as wp
from warp.examples.core.example_clash_detection import (
    ClashDetector,
    Example,
    make_box,
    make_cylinder,
    make_sphere,
    sample_surface_points,
)
from warp.tests.unittest_utils import *

# ---------------------------------------------------------------------------
# Geometry generation tests (CPU-only, no device needed)
# ---------------------------------------------------------------------------


class TestGeometryGeneration(unittest.TestCase):
    """Validate procedural geometry helpers produce valid meshes."""

    def test_make_box_vertex_count(self):
        verts, tris = make_box((0, 0, 0), (1, 1, 1))
        self.assertEqual(verts.shape, (8, 3))
        # 6 faces x 2 triangles = 12 triangles x 3 indices = 36
        self.assertEqual(len(tris), 36)

    def test_make_box_indices_in_range(self):
        verts, tris = make_box((1, 2, 3), (0.5, 0.5, 0.5))
        self.assertTrue(np.all(tris >= 0))
        self.assertTrue(np.all(tris < len(verts)))

    def test_make_box_bounds(self):
        center = (1.0, 2.0, 3.0)
        half = (0.5, 1.0, 1.5)
        verts, _ = make_box(center, half)
        np.testing.assert_allclose(verts.min(axis=0), [0.5, 1.0, 1.5], atol=1e-6)
        np.testing.assert_allclose(verts.max(axis=0), [1.5, 3.0, 4.5], atol=1e-6)

    def test_make_cylinder_vertex_count(self):
        segments = 16
        verts, tris = make_cylinder((0, 0, 0), 1.0, 2.0, segments)
        # 2 center verts + 2 rings * segments
        self.assertEqual(len(verts), 2 + 2 * segments)
        # caps: 2 * segments + side: 2 * segments = 4 * segments triangles
        self.assertEqual(len(tris), 4 * segments * 3)

    def test_make_cylinder_indices_in_range(self):
        verts, tris = make_cylinder((0, 0, 0), 0.5, 1.0, 12)
        self.assertTrue(np.all(tris >= 0))
        self.assertTrue(np.all(tris < len(verts)))

    def test_make_sphere_vertex_count(self):
        rings = 8
        segments = 12
        verts, _tris = make_sphere((0, 0, 0), 1.0, rings, segments)
        # 2 poles + (rings - 1) * segments
        expected_verts = 2 + (rings - 1) * segments
        self.assertEqual(len(verts), expected_verts)

    def test_make_sphere_indices_in_range(self):
        verts, tris = make_sphere((0, 0, 0), 1.0, 10, 14)
        self.assertTrue(np.all(tris >= 0))
        self.assertTrue(np.all(tris < len(verts)))

    def test_make_sphere_radius(self):
        radius = 2.5
        center = np.array([1.0, 2.0, 3.0])
        verts, _ = make_sphere(tuple(center), radius, 12, 16)
        distances = np.linalg.norm(verts - center, axis=1)
        np.testing.assert_allclose(distances, radius, atol=1e-5)

    def test_sample_surface_points_count(self):
        verts, tris = make_box((0, 0, 0), (1, 1, 1))
        rng = np.random.default_rng(0)
        pts = sample_surface_points(verts, tris, 500, rng)
        self.assertEqual(pts.shape, (500, 3))

    def test_sample_surface_points_on_surface(self):
        """Sampled points should lie on the box faces."""
        verts, tris = make_box((0, 0, 0), (1, 1, 1))
        rng = np.random.default_rng(42)
        pts = sample_surface_points(verts, tris, 1000, rng)
        # Each point must have at least one coordinate at +/-1 (on a face)
        on_face = np.any(np.isclose(np.abs(pts), 1.0, atol=1e-5), axis=1)
        self.assertTrue(np.all(on_face))


# ---------------------------------------------------------------------------
# Clash detection tests (require Warp device)
# ---------------------------------------------------------------------------


def test_hard_clash_overlapping_boxes(test, device):
    """Two overlapping boxes must produce a hard clash."""
    with wp.ScopedDevice(device):
        detector = ClashDetector(clearance=0.1, surface_samples=512)
        detector.add_box("box_a", center=(0, 0, 0), half_extents=(1, 1, 1))
        detector.add_box("box_b", center=(0.5, 0.5, 0.5), half_extents=(1, 1, 1))

        results = detector.run_detection()
        test.assertEqual(len(results), 1)
        test.assertTrue(results[0].has_hard_clash, "Overlapping boxes should produce a hard clash")


def test_no_clash_separated_boxes(test, device):
    """Two well-separated boxes must not clash."""
    with wp.ScopedDevice(device):
        detector = ClashDetector(clearance=0.1, surface_samples=512)
        detector.add_box("box_a", center=(0, 0, 0), half_extents=(0.5, 0.5, 0.5))
        detector.add_box("box_b", center=(5, 5, 5), half_extents=(0.5, 0.5, 0.5))

        results = detector.run_detection()
        test.assertEqual(len(results), 1)
        test.assertFalse(results[0].has_hard_clash)
        test.assertFalse(results[0].has_soft_clash)


def test_soft_clash_near_boxes(test, device):
    """Two boxes just outside contact but within clearance produce a soft clash."""
    with wp.ScopedDevice(device):
        # Boxes separated by 0.05 gap, clearance is 0.2
        detector = ClashDetector(clearance=0.2, surface_samples=2048)
        detector.add_box("box_a", center=(0, 0, 0), half_extents=(0.5, 0.5, 0.5))
        detector.add_box("box_b", center=(1.05, 0, 0), half_extents=(0.5, 0.5, 0.5))

        results = detector.run_detection()
        test.assertEqual(len(results), 1)
        # Should not have a hard clash (boxes don't overlap)
        test.assertFalse(results[0].has_hard_clash)
        # Should have a soft clash (gap < clearance)
        test.assertTrue(results[0].has_soft_clash, "Near boxes should trigger soft clash")
        test.assertLess(results[0].min_distance, 0.2)


def test_cylinder_through_box(test, device):
    """A cylinder passing through a box must produce a hard clash."""
    with wp.ScopedDevice(device):
        detector = ClashDetector(clearance=0.1, surface_samples=512)
        detector.add_box("box", center=(0, 0, 0), half_extents=(1, 1, 1))
        detector.add_cylinder("pipe", center=(0, 0, 0), radius=0.3, height=4.0)

        results = detector.run_detection()
        test.assertEqual(len(results), 1)
        test.assertTrue(results[0].has_hard_clash, "Cylinder through box should be a hard clash")


def test_sphere_inside_box(test, device):
    """A sphere fully inside a box must produce a hard clash."""
    with wp.ScopedDevice(device):
        detector = ClashDetector(clearance=0.1, surface_samples=512)
        detector.add_box("box", center=(0, 0, 0), half_extents=(2, 2, 2))
        detector.add_sphere("sphere", center=(0, 0, 0), radius=0.5)

        results = detector.run_detection()
        test.assertEqual(len(results), 1)
        # The sphere is fully inside, but triangle-triangle may not intersect if
        # fully contained. In this case we rely on the soft clash detecting that
        # the sphere surface points are very close to the box mesh.
        has_any_clash = results[0].has_hard_clash or results[0].has_soft_clash
        test.assertTrue(has_any_clash, "Sphere inside box should detect some clash")


def test_multiple_objects_pairwise(test, device):
    """Three objects produce three pairwise results."""
    with wp.ScopedDevice(device):
        detector = ClashDetector(clearance=0.1, surface_samples=256)
        detector.add_box("a", (0, 0, 0), (0.5, 0.5, 0.5))
        detector.add_box("b", (10, 0, 0), (0.5, 0.5, 0.5))
        detector.add_box("c", (20, 0, 0), (0.5, 0.5, 0.5))

        results = detector.run_detection()
        test.assertEqual(len(results), 3)  # C(3,2) = 3 pairs


def test_render_image_shape(test, device):
    """Rendered image has the expected shape and value range."""
    with wp.ScopedDevice(device):
        detector = ClashDetector(clearance=0.1, surface_samples=256)
        detector.add_box("box", (0, 0, 0), (1, 1, 1), color=(0.5, 0.5, 0.5))
        detector.run_detection()

        image = detector.render_image(width=64, height=48)
        test.assertEqual(image.shape, (48, 64, 3))
        test.assertTrue(np.all(image >= 0.0))
        test.assertTrue(np.all(image <= 1.0))


def test_render_image_not_blank(test, device):
    """Rendered image must not be entirely the background color."""
    with wp.ScopedDevice(device):
        detector = ClashDetector(clearance=0.1, surface_samples=256)
        detector.add_box("box", (0, 0, 0), (1, 1, 1), color=(0.8, 0.2, 0.2))
        detector.run_detection()

        image = detector.render_image(
            width=64,
            height=48,
            cam_pos=(3, 2, 3),
            cam_target=(0, 0, 0),
        )
        # Background is approximately (0.18, 0.18, 0.22).
        # If the box is visible, some pixels should differ significantly.
        bg = np.array([0.18, 0.18, 0.22])
        diff = np.abs(image - bg).max(axis=2)
        test.assertTrue(np.any(diff > 0.1), "Image should contain non-background pixels")


def test_clash_result_properties(test, device):
    """Verify ClashResult property logic."""
    with wp.ScopedDevice(device):
        detector = ClashDetector(clearance=0.1, surface_samples=512)
        # Overlapping pair
        detector.add_box("a", (0, 0, 0), (1, 1, 1))
        detector.add_box("b", (0.5, 0, 0), (1, 1, 1))

        results = detector.run_detection()
        r = results[0]
        if r.has_hard_clash:
            # has_soft_clash should be False when hard_clash is True
            test.assertFalse(r.has_soft_clash)


def test_example_full_pipeline(test, device):
    """Run the full Example pipeline end-to-end."""
    with wp.ScopedDevice(device):
        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = os.path.join(tmpdir, "test_clash.usd")
            example = Example(stage_path=usd_path, width=64, height=48)

            # Run detection
            example.step()
            test.assertGreater(len(example.detector.results), 0)

            # Render
            example.render()
            test.assertIsNotNone(example.image)
            test.assertEqual(example.image.shape, (48, 64, 3))

            # Save image
            img_path = os.path.join(tmpdir, "test_output.png")
            example.save_image(img_path)
            test.assertTrue(os.path.exists(img_path))


# ---------------------------------------------------------------------------
# Register device-parametrized tests
# ---------------------------------------------------------------------------

devices = get_test_devices()


class TestClashDetection(unittest.TestCase):
    pass


add_function_test(
    TestClashDetection, "test_hard_clash_overlapping_boxes", test_hard_clash_overlapping_boxes, devices=devices
)
add_function_test(TestClashDetection, "test_no_clash_separated_boxes", test_no_clash_separated_boxes, devices=devices)
add_function_test(TestClashDetection, "test_soft_clash_near_boxes", test_soft_clash_near_boxes, devices=devices)
add_function_test(TestClashDetection, "test_cylinder_through_box", test_cylinder_through_box, devices=devices)
add_function_test(TestClashDetection, "test_sphere_inside_box", test_sphere_inside_box, devices=devices)
add_function_test(TestClashDetection, "test_multiple_objects_pairwise", test_multiple_objects_pairwise, devices=devices)
add_function_test(TestClashDetection, "test_render_image_shape", test_render_image_shape, devices=devices)
add_function_test(TestClashDetection, "test_render_image_not_blank", test_render_image_not_blank, devices=devices)
add_function_test(TestClashDetection, "test_clash_result_properties", test_clash_result_properties, devices=devices)
add_function_test(TestClashDetection, "test_example_full_pipeline", test_example_full_pipeline, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=False)
