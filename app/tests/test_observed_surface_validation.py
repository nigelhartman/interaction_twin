from pathlib import Path
import unittest

import cv2
import numpy as np

import observed_surface_validation as validation


class HeldoutSurfaceValidationTests(unittest.TestCase):
    def setUp(self):
        self.K = np.array([[200., 0, 160], [0, 200., 120], [0, 0, 1]])
        self.pose = (np.eye(3), np.zeros(3))

    def images(self):
        rng = np.random.default_rng(77)
        left = cv2.GaussianBlur(rng.integers(0, 256, (240, 320), np.uint8), (3, 3), .5)
        right = np.zeros_like(left)
        right[:, :-20] = left[:, 20:]
        def frame(image, baseline):
            return {'gray': image, 'pose': (np.eye(3), np.array([-baseline, 0, 0])),
                    'core': np.ones_like(image, bool), 'known_depth': np.full(image.shape, 4.), 'anchor_pixels': []}
        return frame(left, 0), frame(right, .4)

    def test_project_nonidentity_agrees_with_opencv(self):
        xyz = np.array([[.2, .3, 4.], [-.5, .2, 5.]])
        rvec, t = np.array([.1, -.2, .03]), np.array([.2, -.1, .5])
        xy, z = validation.project(xyz, (cv2.Rodrigues(rvec)[0], t), self.K)
        expected = cv2.projectPoints(xyz, rvec, t, self.K, None)[0][:, 0]
        np.testing.assert_allclose(xy, expected, atol=1e-12)
        self.assertTrue((z > 0).all())

    def test_planar_true_geometry_heldout_consistency(self):
        reference, heldout = self.images()
        point = [-.2, -.2, 4.]
        result = validation.check_point(point, reference, heldout, self.K)
        self.assertEqual(result['status'], 'photo_consistent', result)
        self.assertAlmostEqual(result['zncc'], 1.)
        self.assertTrue(result['physical_visibility_unknown'])
        self.assertEqual(result['projected_xy'], [130., 110.])

    def test_corrupt_heldout_image_reports_inconsistency_without_mutation(self):
        reference, heldout = self.images()
        heldout['gray'] = np.roll(heldout['gray'], 7, axis=1)
        point = np.array([-.2, -.2, 4.])
        before = point.copy()
        result = validation.check_point(point, reference, heldout, self.K)
        self.assertEqual(result['status'], 'photo_inconsistent', result)
        np.testing.assert_array_equal(point, before)

    def test_missing_patch_depth_is_skipped_not_visible(self):
        reference, heldout = self.images()
        heldout['known_depth'][:] = np.inf
        result = validation.check_point([-.2, -.2, 4.], reference, heldout, self.K)
        self.assertEqual(result['status'], 'skipped')
        self.assertEqual(result['reason'], 'target_unknown_known_patch_coverage')
        self.assertIsNone(result['zncc'])

    def test_known_nearer_surface_is_skipped(self):
        reference, heldout = self.images()
        heldout['known_depth'][:] = 3.
        result = validation.check_point([-.2, -.2, 4.], reference, heldout, self.K)
        self.assertEqual(result['reason'], 'target_occluded_by_represented_patch')

    def test_guard_does_not_score_sparse_anchor_neighbourhood(self):
        reference, heldout = self.images()
        heldout['anchor_pixels'] = [[130, 110]]
        result = validation.check_point([-.2, -.2, 4.], reference, heldout, self.K)
        self.assertEqual(result['reason'], 'near_sparse_pose_anchor_centre')

    def test_mask_bilinear_footprint_cannot_cross_hole(self):
        reference, heldout = self.images()
        heldout['core'][110, 130] = False
        result = validation.check_point([-.2, -.2, 4.], reference, heldout, self.K)
        self.assertEqual(result['reason'], 'target_outside_mask_core')

    def test_textureless_has_no_numeric_score(self):
        reference, heldout = self.images()
        reference['gray'][:] = 100
        result = validation.check_point([-.2, -.2, 4.], reference, heldout, self.K)
        self.assertEqual(result['reason'], 'insufficient_reference_or_target_texture')
        self.assertIsNone(result['zncc'])

    def test_behind_camera_is_skipped_with_json_safe_coordinates(self):
        import json
        reference, heldout = self.images()
        result = validation.check_point([0, 0, 0], reference, heldout, self.K)
        self.assertEqual(result['reason'], 'target_behind_camera_or_nonfinite')
        json.dumps(result, allow_nan=False)

    def test_backfacing_mesh_does_not_create_visibility(self):
        vertices = [[-1, -1, 4], [1, -1, 4], [1, 1, 4], [-1, 1, 4]]
        depth = validation.rasterize_known_patch(vertices, [[0, 1, 2], [0, 2, 3]], self.pose, self.K, (240, 320))
        self.assertFalse(np.isfinite(depth).any())
        depth = validation.rasterize_known_patch(vertices, [[0, 2, 1], [0, 3, 2]], self.pose, self.K, (240, 320))
        self.assertAlmostEqual(depth[120, 160], 4.)
        self.assertTrue(np.isinf(depth[20, 20]))

    def test_rasterized_depth_is_perspective_correct(self):
        # Pixel vertices (100,80),(200,80),(100,180); positive front-facing order.
        uv = np.array([[100., 80], [200, 80], [100, 180]])
        z = np.array([3., 5., 4.])
        xyz = np.c_[uv, np.ones(3)] @ np.linalg.inv(self.K).T * z[:, None]
        depth = validation.rasterize_known_patch(xyz, [[0, 2, 1]], self.pose, self.K, (240, 320))
        # (125,105): projected barycentric weights .5,.25,.25.
        self.assertAlmostEqual(depth[105, 125], 1 / (.5 / 3 + .25 / 5 + .25 / 4))

    def test_no_hole_filling_in_triangle_rasterization(self):
        vertices = [[-.5, -.5, 4], [.5, -.5, 4], [-.5, .5, 4]]
        depth = validation.rasterize_known_patch(vertices, [[0, 2, 1]], self.pose, self.K, (240, 320))
        self.assertTrue(np.isfinite(depth[100, 140]))
        self.assertTrue(np.isinf(depth[140, 180]))

    def test_path_guard_rejects_escape_and_remote(self):
        root = Path('/private/tmp/test-observed-validation')
        for path in ['https://evil.test/data/x', 'data/../../x', '/files/data/a?x=1', 'data/a\\b']:
            with self.assertRaises(ValueError):
                validation.local_path(root, path)


if __name__ == '__main__':
    unittest.main()
