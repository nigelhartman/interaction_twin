import unittest
from unittest.mock import patch

import cv2
import numpy as np

from alignment_fit import solve_pose


class AlignmentFitTests(unittest.TestCase):
    def setUp(self):
        random = np.random.default_rng(29)
        angle = random.uniform(-1.2, 1.2, 80)
        self.points = .02 * np.c_[.5*np.cos(angle), .5*np.sin(angle),
                                  random.uniform(-1.9, 1.9, 80)]
        self.K = np.array([[1030., 0, 479.5], [0, 1030., 269.5], [0, 0, 1]])
        self.rvec = np.array([.5, -.3, .15])
        self.R = cv2.Rodrigues(self.rvec)[0]
        self.t = np.array([-.025, -.008, .36])
        self.uv = cv2.projectPoints(self.points, self.rvec, self.t, self.K, None)[0][:, 0]

    def test_fixed_size_pose_recovery_from_cylinder_surface(self):
        result = solve_pose(self.points, self.uv, self.K)
        self.assertTrue(result['success'], result['reason'])
        np.testing.assert_allclose(result['rotation'], self.R, atol=1e-5)
        np.testing.assert_allclose(result['translation'], self.t, atol=1e-7)
        self.assertLess(np.max(result['residual_px']), 1e-5)
        self.assertFalse(result['scale_estimated'])
        self.assertGreater(np.min((self.points @ result['rotation'].T
                                   + result['translation'])[:, 2]), 0)

    def test_outliers_invalid_rows_and_deterministic_original_indices(self):
        random = np.random.default_rng(6)
        uv = self.uv + random.normal(0, .15, self.uv.shape)
        outliers = np.arange(0, 80, 4)
        uv[outliers] += random.choice([-1, 1], (len(outliers), 2))*80
        uv[1] = np.nan
        result = solve_pose(self.points, uv, self.K)
        repeated = solve_pose(self.points, uv, self.K)
        self.assertTrue(result['success'], result['reason'])
        self.assertTrue(np.isnan(result['residual_px'][1]))
        self.assertTrue(set(result['inlier_indices']).isdisjoint(set(outliers) | {1}))
        self.assertGreaterEqual(len(result['inlier_indices']), 58)
        self.assertLess(np.linalg.norm(result['translation'] - self.t), .001)
        np.testing.assert_allclose(result['rotation'], self.R, atol=.03)
        np.testing.assert_array_equal(result['inlier_indices'], repeated['inlier_indices'])
        np.testing.assert_allclose(result['translation'], repeated['translation'], atol=1e-12)

    def test_insufficient_and_collinear_evidence_is_unsupported(self):
        few = solve_pose(self.points[:5], self.uv[:5], self.K)
        self.assertFalse(few['success'])
        self.assertEqual(few['reason'], 'fewer_than_six_finite_correspondences')
        line = np.c_[np.arange(10.), np.zeros((10, 2))]
        degenerate = solve_pose(line, np.c_[np.arange(10.), np.arange(10.)], self.K)
        self.assertFalse(degenerate['success'])
        self.assertEqual(degenerate['reason'], 'degenerate_correspondence_geometry')

    def test_behind_camera_refinement_is_rejected(self):
        behind = np.array([[0.], [0.], [-1.]])
        consensus = np.arange(80, dtype=np.int32)[:, None]
        with patch('alignment_fit.cv2.solvePnPRansac',
                   return_value=(True, np.zeros((3, 1)), behind, consensus)), \
             patch('alignment_fit.cv2.solvePnPRefineLM',
                   return_value=(np.zeros((3, 1)), behind)):
            result = solve_pose(self.points, self.uv, self.K)
        self.assertFalse(result['success'])
        self.assertEqual(result['reason'], 'no_positive_depth_supported_refinement')

    def test_initial_pose_cannot_bypass_positive_depth_or_scale_contract(self):
        result = solve_pose(self.points, self.uv, self.K,
                            initial_R=np.eye(3), initial_t=[0, 0, -1])
        self.assertTrue(result['success'], result['reason'])
        self.assertGreater(np.min((self.points @ result['rotation'].T
                                   + result['translation'])[:, 2]), 0)
        with self.assertRaises(ValueError):
            solve_pose(self.points, self.uv, self.K, initial_R=np.eye(3))


if __name__ == '__main__':
    unittest.main()
