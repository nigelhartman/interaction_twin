"""Independent camera-hypothesis software fixtures, not physical captures."""
import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import automatic_camera as A
import observed_geometry as G
from tests.test_observed_calibration import fixture


class AutomaticCameraTests(unittest.TestCase):
    def test_normalized_pixel_centres_map_without_recentering(self):
        normalized = np.array([[.9, 0, .45], [0, 1.6, .55], [0, 0, 1.]])
        k = A.pixel_intrinsics(normalized, 960, 540)
        np.testing.assert_allclose(k, [[864, 0, 431.5], [0, 864, 296.5], [0, 0, 1]])

    def test_resize_pixel_centre_equivalence(self):
        normalized = np.array([[.9543153, 0, .5], [0, 1.6965605, .5], [0, 0, 1.]])
        native = A.pixel_intrinsics(normalized, 960, 540)
        resized = A.pixel_intrinsics(normalized, 512, 288)
        q = np.array([[1., -.2, 3.], [-.4, 1.2, 4.]])
        a = (q @ native.T); a = a[:, :2] / a[:, 2:]
        b = (q @ resized.T); b = b[:, :2] / b[:, 2:]
        np.testing.assert_allclose((a + .5) * [512/960, 288/540] - .5, b, atol=1e-12)

    def test_invalid_K_rejected(self):
        for k in [np.eye(2), [[0, 0, 0], [0, 1, 0], [0, 0, 1]],
                [[1, .2, 0], [0, 1, 0], [0, 0, 1]], np.full((3, 3), np.nan)]:
            with self.assertRaises(ValueError):
                A.validate_K(k)
        k = np.eye(3); k[2, 2] += 1e-7
        with self.assertRaises(ValueError): A.validate_K(k)
        for width, height in [(0, 540), (960., 540), (True, 540), (960, -1)]:
            with self.assertRaises(ValueError): A.pixel_intrinsics(np.eye(3), width, height)

    def test_custom_K_residual_matches_true_projection(self):
        model, tracks, old_K = fixture(count=20)
        K = np.array([[710., 0., 301.5], [0., 730., 231.5], [0., 0., 1.]])
        xyz = np.array(list(model['points'].values()))
        for fi, pose in model['poses'].items():
            uv, _ = G.project(xyz, pose, K)
            for j, tid in enumerate(tracks):
                tracks[tid][fi] = uv[j]
        problem = A.FixedCameraProblem(model, tracks, 640, 480, set(model['poses']), K)
        residual, depth = problem.residual_and_depth(problem.initial)
        self.assertLess(np.max(np.abs(residual)), 1e-9)
        self.assertTrue((depth > 0).all())
        self.assertTrue(A.support_diagnostics(problem)['registered_camera_graph_connected'])

    def test_wrong_K_has_nonzero_original_residual_and_independent_initial_state(self):
        model, tracks, K = fixture(count=20)
        a = A.FixedCameraProblem(model, tracks, 640, 480, set(model['poses']), K)
        wrong = K.copy(); wrong[0, 0] *= 1.1
        b = A.FixedCameraProblem(model, tracks, 640, 480, set(model['poses']), wrong)
        np.testing.assert_array_equal(a.initial, b.initial)
        self.assertGreater(np.linalg.norm(b.residual_and_depth(b.initial)[0]), 1.)
        b.initial[0] += .2
        self.assertNotEqual(a.initial[0], b.initial[0])

    def test_reserved_tracks_cannot_enter_fit(self):
        model, tracks, K = fixture(count=20)
        tracks['p0000'][99] = [1., 2.]
        with self.assertRaises(ValueError):
            A.FixedCameraProblem(model, tracks, 640, 480, set(model['poses']), K)

    @staticmethod
    def rows():
        return [{'id': name, 'training': {'optimizer_success': True, 'finite': True,
            'positive_training_depth_fraction': 1., 'robust_train_cost': c,
            'mean_robust_train_cost': c / 100}} for name, c in [('a', 1.), ('b', 2.)]]

    def test_train_preference_never_promotes_camera(self):
        decision = A.select_training(self.rows())
        self.assertEqual(decision['training_preferred_hypothesis_id'], 'a')
        self.assertIsNone(decision['selected_production_camera_id'])
        for key in ['camera_calibrated', 'physical_shape_verified', 'physical_scale_known',
                'production_camera_changed', 'reserved_used_for_selection']:
            self.assertFalse(decision[key])

    def test_unconverged_lower_cost_not_selected(self):
        rows = self.rows(); rows[0]['training']['optimizer_success'] = False
        self.assertIsNone(A.select_training(rows)['training_preferred_hypothesis_id'])

    def test_similar_costs_remain_ambiguous(self):
        rows = self.rows(); rows[1]['training'].update(robust_train_cost=1.01, mean_robust_train_cost=.0101)
        self.assertIsNone(A.select_training(rows)['training_preferred_hypothesis_id'])

    def test_reserved_metric_changes_cannot_change_training_choice(self):
        rows = self.rows(); before = A.select_training(rows)
        rows[0]['reserved'] = {'median_px': 1e100}; rows[1]['reserved'] = {'median_px': 0.}
        self.assertEqual(before, A.select_training(rows))

    def test_source_loader_rejects_escape_symlink_and_changed_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); (root / 'data').mkdir()
            p = root / 'data/x.json'; p.write_text('{}')
            self.assertEqual(A.data_file(root, 'data/x.json', A.digest(p)), p)
            for name in ['../x.json', '/etc/passwd', 'data/../x.json']:
                with self.assertRaises(ValueError): A.data_file(root, name)
            (root / 'data/link.json').symlink_to(p)
            with self.assertRaises(ValueError): A.data_file(root, 'data/link.json')
            with self.assertRaises(ValueError): A.data_file(root, 'data/x.json', '0' * 64)

    def test_known_K_exact_fixture_converges_without_shape_claim(self):
        model, tracks, K = fixture(count=15, views=4)
        p = A.FixedCameraProblem(model, tracks, 640, 480, set(model['poses']), K)
        fit = p.fit_fixed(max_evaluations=10)
        self.assertTrue(fit['metrics']['optimizer_success'])
        self.assertLess(fit['metrics']['train_median_px'], 1e-8)
        self.assertEqual(fit['metrics']['heldout_observations_used'], 0)

    def test_derived_display_recomputes_finite_errors_without_moving_geometry(self):
        from scripts.export_automatic_camera_display import corrected_view
        view = {'cameras': [{'frame_id': 'f', 'R': np.eye(3).tolist(), 't': [0., 0., 0.],
                'K': [[100., 0., 50.], [0., 100., 40.], [0., 0., 1.]]}],
            'points': [{'id': 'p', 'xyz': [0., 0., 2.]}],
            'frames': [{'id': 'f', 'split': 'heldout', 'timestamp_s': 1.,
                'observations': [{'track_id': 'p', 'xy': [51., 40.], 'predicted_xy': [50., 40.], 'role': 'heldout'}]}],
            'heldout_evaluation': [{'frame_index': 0, 'passes': True}],
            'metrics': {'heldout_finite_error_median_px': 1234.}}
        original = copy.deepcopy(view)
        result, audit = corrected_view(view)
        self.assertEqual(view, original)
        self.assertEqual(result['points'][0]['xyz'], original['points'][0]['xyz'])
        self.assertEqual(result['cameras'], original['cameras'])
        self.assertEqual(result['metrics']['heldout_finite_error_median_px'], 1.)
        self.assertEqual(result['metrics']['heldout_scored_observations'], 1)
        self.assertFalse(audit['geometry_or_pose_or_K_changes'])

    def test_reserved_anchor_is_not_scored_in_display_metrics(self):
        from scripts.export_automatic_camera_display import corrected_view
        view = {'cameras': [{'frame_id': 'f', 'R': np.eye(3).tolist(), 't': [0., 0., 0.], 'K': np.eye(3).tolist()}],
            'points': [{'id': 'p', 'xyz': [0., 0., 2.]}],
            'frames': [{'id': 'f', 'split': 'heldout', 'timestamp_s': 1.,
                'observations': [{'track_id': 'p', 'xy': [1000., 1000.], 'role': 'pose_anchor'}]}],
            'heldout_evaluation': [{'frame_index': 0, 'passes': True}], 'metrics': {}}
        result, audit = corrected_view(view)
        self.assertEqual(result['metrics']['heldout_scored_observations'], 0)
        self.assertIsNone(result['metrics']['heldout_finite_error_median_px'])


if __name__ == '__main__':
    unittest.main()
