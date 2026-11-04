"""Synthetic math/provenance tests; no captured data or hosted inference."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

import observed_surface_completion as C


class SurfaceCompletionTests(unittest.TestCase):
    def setUp(self):
        self.K = np.array([[90., 0, 40], [0, 90., 35], [0, 0, 1.]])
        self.pose = (np.eye(3), np.zeros(3))

    def scene(self, shape=(70, 80)):
        yy, xx = np.indices(shape)
        depth = 1 / (.25 + (xx-40)*.0001 + (yy-35)*.0002)
        hole = np.zeros(shape, bool)
        hole[25:35, 30:42] = True
        observed = depth.copy()
        observed[hole] = np.nan
        domain = np.zeros(shape, bool)
        domain[5:-5, 5:-5] = True
        labels = domain.astype(int)
        return depth, observed, hole, domain, labels

    def test_harmonic_inverse_depth_recovers_planar_inverse_depth(self):
        truth, observed, hole, _, _ = self.scene()
        recovered = C.harmonic_inverse_depth(observed, hole)
        np.testing.assert_allclose(recovered, truth[hole], atol=1e-12)

    def test_accepted_hole_is_explicitly_inferred_and_observations_unchanged(self):
        truth, observed, hole, domain, labels = self.scene()
        original = observed.copy()
        completed, inferred, report = C.infer_interior_depth(observed, domain, labels)
        self.assertTrue(np.array_equal(inferred, hole))
        np.testing.assert_allclose(completed, truth, atol=1e-12)
        np.testing.assert_array_equal(observed, original)
        self.assertEqual(len(report['accepted_regions']), 1)
        self.assertEqual(report['accepted_regions'][0]['physical_confidence'], 'unknown')
        self.assertFalse(report['accepted_regions'][0]['depth_observed'])

    def test_gap_open_to_mask_boundary_is_never_filled(self):
        _, observed, hole, domain, labels = self.scene()
        observed[5:26, 34] = np.nan
        _, inferred, report = C.infer_interior_depth(observed, domain, labels)
        self.assertEqual(int(inferred.sum()), 0)
        self.assertIn('open_to_mask_or_image_boundary', report['rejected_regions'])

    def test_other_component_label_prevents_cross_part_filling(self):
        _, observed, _, domain, labels = self.scene()
        labels[:, 36:] = 2
        _, inferred, report = C.infer_interior_depth(observed, domain, labels)
        self.assertEqual(int(inferred.sum()), 0)
        self.assertIn('crosses_component_labels', report['rejected_regions'])

    def test_exclusion_mask_and_guard_remain_empty(self):
        mask = np.ones((70, 80), np.uint8)
        other = np.zeros_like(mask)
        other[10:60, 38:42] = 1
        domain, labels = C.component_domain(mask, [other])
        self.assertFalse(domain[10:60, 34:46].any())
        self.assertTrue((labels[10:60, 34:46] == 2).all())

    def test_large_enclosed_gap_rejected_without_rescaling_thresholds(self):
        depth = np.ones((200, 200)) * 4
        depth[50:150, 50:150] = np.nan
        _, inferred, report = C.infer_interior_depth(depth, np.ones(depth.shape, bool), np.ones(depth.shape, int))
        self.assertFalse(inferred.any())
        self.assertIn('gap_exceeds_frozen_size_bounds', report['rejected_regions'])

    def test_boundary_depth_step_rejected(self):
        _, observed, _, domain, labels = self.scene()
        observed[24, 30:42] = 6
        _, inferred, report = C.infer_interior_depth(observed, domain, labels)
        self.assertFalse(inferred.any())
        self.assertIn('boundary_depth_discontinuity', report['rejected_regions'])

    def test_nonfinite_or_negative_boundary_does_not_become_support(self):
        _, observed, _, domain, labels = self.scene()
        observed[24, 30] = np.inf
        observed[23, 30] = -4
        completed, inferred, _ = C.infer_interior_depth(observed, domain, labels)
        self.assertFalse(np.isinf(completed[inferred]).any())
        self.assertTrue((completed[inferred] > 0).all())

    def test_harmonic_solver_rejects_open_boundary(self):
        depth = np.full((12, 12), np.nan)
        hole = np.zeros((12, 12), bool)
        hole[5:7, 5:7] = True
        with self.assertRaisesRegex(ValueError, 'complete positive-depth boundary'):
            C.harmonic_inverse_depth(depth, hole)

    def test_mesh_only_has_inferred_faces_and_stays_in_component(self):
        _, observed, _, domain, labels = self.scene()
        completed, inferred, _ = C.infer_interior_depth(observed, domain, labels)
        image = np.zeros((*domain.shape, 3), np.uint8)
        image[:] = [12, 34, 56]
        frame = {'id': 'source', 'image_url': '/files/data/test/source.jpg'}
        mesh = C.inferred_mesh(completed, inferred, domain, labels, image, self.pose, self.K, frame)
        self.assertIsNotNone(mesh)
        self.assertEqual(mesh['observed_face_count'], 0)
        self.assertEqual(set(mesh['face_confidence']), {'unknown'})
        self.assertEqual(set(mesh['face_provenance']), {'inferred_harmonic_inverse_depth'})
        self.assertFalse(mesh['hidden_surface_completion'])
        pixels = np.array(mesh['vertex_source_pixels'])
        self.assertTrue(domain[pixels[:, 1], pixels[:, 0]].all())
        self.assertEqual(set(map(tuple, mesh['colors'])), {(56, 34, 12)})
        for triangle in mesh['triangles']:
            uv = pixels[triangle]
            self.assertTrue(inferred[uv[:, 1], uv[:, 0]].any())
        projected, _ = C.project(mesh['vertices'], self.pose, self.K)
        np.testing.assert_allclose(projected, pixels, atol=1e-12)

    def test_rasterization_is_perspective_correct_and_keeps_nearest_layer(self):
        pixels = np.array([[20., 20.], [55., 20.], [20., 55.]])
        depths = np.array([3., 4., 5.])
        vertices = C.unproject(pixels, depths, self.pose, self.K)
        mesh = {'vertices': vertices.tolist(), 'triangles': [[0, 1, 2]]}
        found = C.rasterize_depth(mesh, self.pose, self.K, (70, 80))
        bary = np.linalg.solve(np.vstack([pixels.T, np.ones(3)]), [30, 30, 1])
        self.assertAlmostEqual(found[30, 30], 1 / np.sum(bary / depths), places=12)
        farther = C.unproject(pixels, depths*2, self.pose, self.K)
        mesh['vertices'] += farther.tolist()
        mesh['triangles'] += [[3, 4, 5]]
        nearer = C.rasterize_depth(mesh, self.pose, self.K, (70, 80))
        np.testing.assert_array_equal(nearer, found)

    def test_invalid_vertex_rejected(self):
        mesh = {'vertices': [[0, 0, float('nan')], [1, 0, 4], [0, 1, 4]], 'triangles': [[0, 1, 2]]}
        with self.assertRaisesRegex(ValueError, 'finite XYZ'):
            C.rasterize_depth(mesh, self.pose, self.K, (70, 80))

    def test_nonidentity_projection_roundtrip(self):
        pose = (cv2.Rodrigues(np.array([.1, -.05, .02]))[0], np.array([.1, -.2, .3]))
        xyz = np.array([[.1, .2, 4], [-.5, .1, 5]])
        pixels, depth = C.project(xyz, pose, self.K)
        np.testing.assert_allclose(C.unproject(pixels, depth, pose, self.K), xyz, atol=1e-12)

    def test_completion_cannot_reuse_output_or_escape_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'inside project data'):
                C.complete(root, 'outside/source', 'data/output')
            (root/'data/existing').mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                C.complete(root, 'data/source', 'data/existing')


if __name__ == '__main__':
    unittest.main()
