import importlib.util
import unittest
from pathlib import Path

import cv2
import numpy as np

spec = importlib.util.spec_from_file_location('generated_motion', Path(__file__).parents[1]/'scripts/stage4_track_generated.py')
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


class GeneratedMotionTests(unittest.TestCase):
    def test_loss_is_absorbing_in_each_direction_and_raw_evidence_survives(self):
        uv = np.full((7, 2, 2), 3., np.float32)
        visible = np.ones((7, 2), bool)
        visible[1, 0] = False
        visible[5, 1] = False
        masks = [np.ones((8, 8), bool) for _ in range(7)]
        masks[4][3, 3] = False
        result = M.support_masks(uv, visible, masks, 3)
        self.assertFalse(result['valid_no_reacquisition'][0, 0])
        self.assertTrue(result['valid_no_reacquisition'][2, 0])
        self.assertFalse(result['valid_no_reacquisition'][6, 1])
        self.assertTrue(result['instantaneous_support'][6, 1])
        np.testing.assert_array_equal(uv, np.full((7, 2, 2), 3., np.float32))

    def test_split_is_disjoint_and_invalid_reference_points_cannot_enter(self):
        valid = np.array([True, True, False, True, True, True, False, True])
        train, held = M.split_ids(valid)
        self.assertFalse(set(train) & set(held))
        self.assertEqual(set(train) | set(held), set(np.flatnonzero(valid)))
        np.testing.assert_array_equal(held, [0, 3])

    def test_outside_and_nan_points_are_never_indexed_as_valid_masks(self):
        uv = np.array([[[np.nan, 2], [-1, 2], [10, 2], [3, 3]]])
        result = M.support_masks(uv, np.ones((1, 4), bool), [np.ones((8, 8), bool)], 0)
        np.testing.assert_array_equal(result['valid_no_reacquisition'], [[False, False, False, True]])

    def test_generic_queries_are_deterministic_inside_eroded_mask(self):
        rng = np.random.default_rng(3)
        image = rng.integers(0, 255, (100, 80, 3), dtype=np.uint8)
        mask = np.zeros((100, 80), bool)
        mask[20:80, 15:65] = True
        a, b = M.query_points(image, mask), M.query_points(image, mask)
        np.testing.assert_array_equal(a, b)
        self.assertGreaterEqual(len(a), 24)
        interior = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        pixel = np.rint(a).astype(int)
        self.assertTrue(interior[pixel[:, 1], pixel[:, 0]].all())

    def test_heldout_corruption_and_bad_mask_cannot_be_hidden_by_good_training(self):
        self.assertEqual(M.pose_gate(30, 28, np.ones(12), .9, .95, 'plausible_visible_mask'), [])
        bad = M.pose_gate(30, 28, np.full(12, 20.), .9, .95, 'plausible_visible_mask')
        self.assertIn('heldout_error_above_3px_median_or_8px_p90', bad)
        self.assertIn('source_mask_not_plausible_in_model_audit', M.pose_gate(30, 28, np.ones(12), .9, .95, 'uncertain'))
        self.assertIn('raw_full_mesh_silhouette_gate_failed', M.pose_gate(30, 28, np.ones(12), .1, .2, 'plausible_visible_mask'))

    def test_disjoint_synthetic_pose_checks_do_not_fit_heldout_points(self):
        rng = np.random.default_rng(8)
        xyz = rng.uniform(-.3, .3, (48, 3))
        k = np.array([[420., 0, 270], [0, 420, 480], [0, 0, 1.]])
        r = cv2.Rodrigues(np.array([.1, -.2, .03]))[0]
        t = np.array([.2, .1, 2.4])
        uv = M.project(xyz, k, r, t)
        train, held = M.split_ids(np.ones(len(xyz), bool))
        fit = M.solve_pose(xyz[train], uv[train], k)
        self.assertTrue(fit['success'])
        errors = np.linalg.norm(M.project(xyz[held], k, fit['rotation'], fit['translation'])-uv[held], axis=1)
        self.assertLess(float(np.max(errors)), .001)


if __name__ == '__main__':
    unittest.main()
