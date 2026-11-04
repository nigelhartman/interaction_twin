"""Mathematical fixtures only; none are genuine reconstruction assets."""
import copy
import unittest

import cv2
import numpy as np

import observed_calibration as C
import observed_geometry as G


def fixture(count=45, views=7, parallel=False):
    rng = np.random.default_rng(20261031)
    xyz = np.column_stack([rng.uniform(-1., 1., count), rng.uniform(-.7, .7, count), rng.uniform(3.5, 5.5, count)])
    K = G.camera_matrix(640, 480, 1.15)
    poses = {}
    tracks = {f'p{i:04d}': {} for i in range(count)}
    for i in range(views):
        R = np.eye(3) if parallel else cv2.Rodrigues(np.array([.06 * np.sin(i), -.05 * i, .03 * np.cos(i)]))[0]
        center = np.array([.22 * i, .08 * np.sin(i), .06 * np.cos(i)])
        poses[i] = (R, -R @ center)
        uv, _ = G.project(xyz, poses[i], K)
        for j, t in enumerate(tracks):
            tracks[t][i] = uv[j].copy()
    model = {'seed': {'a': 0, 'b': views - 1}, 'points': {t: xyz[i].copy() for i, t in enumerate(tracks)}, 'poses': poses}
    return model, tracks, K


class CalibrationTests(unittest.TestCase):
    def test_normalized_gauge_preserves_projections(self):
        model, tracks, K = fixture()
        points, poses = C.normalize_gauge(model['points'], model['poses'], 2, 6)
        for fi in poses:
            a, _ = G.project(np.array(list(model['points'].values())), model['poses'][fi], K)
            b, _ = G.project(np.array(list(points.values())), poses[fi], K)
            np.testing.assert_allclose(a, b, atol=1e-10)
        np.testing.assert_allclose(poses[2][0], np.eye(3))
        np.testing.assert_allclose(poses[2][1], 0.)
        self.assertAlmostEqual(np.linalg.norm(poses[6][1]), 1., places=12)

    def test_unit_baseline_has_no_radial_gauge(self):
        for angles in [[0., 0.], [1., .8], [-2., -1.3]]:
            direction = C.unit_direction(angles)
            self.assertAlmostEqual(np.linalg.norm(direction), 1., places=12)
            np.testing.assert_allclose(C.unit_direction(C.direction_angles(direction)), direction, atol=1e-12)

    def test_second_rotation_and_direction_are_not_fixed(self):
        model, tracks, K = fixture()
        problem = C.BundleProblem(model, tracks, 640, 480, set(model['poses']))
        x = problem.initial.copy()
        before, _ = problem.decode(x)
        x[:3] += [.04, .02, -.03]
        x[3:5] += [.1, -.1]
        after, _ = problem.decode(x)
        self.assertGreater(np.linalg.norm(before[problem.second][0] - after[problem.second][0]), .01)
        self.assertGreater(np.linalg.norm(before[problem.second][1] - after[problem.second][1]), .01)
        self.assertAlmostEqual(np.linalg.norm(after[problem.second][1]), 1., places=12)
        np.testing.assert_array_equal(after[problem.first][0], np.eye(3))

    def test_known_nonplanar_focal_recovered(self):
        model, tracks, K = fixture()
        problem = C.BundleProblem(model, tracks, 640, 480, set(model['poses']))
        result = problem.fit(K[0, 0] * .9, max_nfev=150)
        self.assertLess(abs(result['metrics']['focal_px'] / K[0, 0] - 1), .03)
        self.assertLess(result['metrics']['train_median_px'], .03)
        self.assertGreater(result['metrics']['positive_training_depth_fraction'], .99)

    def test_parallel_translation_admits_wrong_focal_same_projection(self):
        model, tracks, K = fixture(parallel=True)
        xyz = np.array(list(model['points'].values()))
        for factor in [.65, 1.6]:
            wrong = K.copy(); wrong[0, 0] *= factor; wrong[1, 1] *= factor
            altered = xyz.copy(); altered[:, 2] *= factor
            for fi, pose in model['poses'].items():
                t = pose[1].copy(); t[2] *= factor
                original, _ = G.project(xyz, pose, K)
                alternate, _ = G.project(altered, (pose[0], t), wrong)
                np.testing.assert_allclose(original, alternate, atol=1e-10)
        # Low projection error alone therefore cannot establish focal.

    def test_observability_rejects_flat_profiles_and_disagreement(self):
        def row(f, cost=1.):
            return {'metrics': {'optimizer_success': True, 'finite': True, 'at_focal_boundary': False,
                                'positive_training_depth_fraction': 1., 'focal_px': f,
                                'robust_train_cost': cost, 'mean_robust_train_cost': cost / 100}}
        attempts = [row(650.), row(1000.), row(1600.)]
        diagnostic = C.observability(attempts, copy.deepcopy(attempts), 0)
        self.assertFalse(diagnostic['supported'])
        self.assertIn('multistart_focal_solutions_disagree', diagnostic['reasons'])
        self.assertIn('alternative_focals_have_similar_training_cost', diagnostic['reasons'])

    def test_boundary_and_unconverged_are_not_calibrated(self):
        def row(f):
            return {'metrics': {'optimizer_success': True, 'finite': True, 'at_focal_boundary': False,
                                'positive_training_depth_fraction': 1., 'focal_px': f,
                                'robust_train_cost': 1., 'mean_robust_train_cost': .01}}
        attempts = [row(1000.) for _ in range(3)]
        attempts[1]['metrics']['optimizer_success'] = False
        attempts[2]['metrics']['at_focal_boundary'] = True
        diagnostic = C.observability(attempts, [row(f) for f in [650., 1000., 1600.]], 0)
        self.assertFalse(diagnostic['supported'])
        self.assertIn('not_all_predeclared_starts_converged', diagnostic['reasons'])
        self.assertIn('focal_solution_reaches_predeclared_boundary', diagnostic['reasons'])
        self.assertFalse(diagnostic['camera_calibrated'])

    def test_holdout_tracks_or_poses_rejected(self):
        model, tracks, _ = fixture()
        with self.assertRaises(ValueError):
            C.BundleProblem(model, tracks, 640, 480, set(range(6)))
        extra = copy.deepcopy(tracks); extra['p0000'][10] = [100., 100.]
        with self.assertRaises(ValueError):
            C.BundleProblem(model, extra, 640, 480, set(range(7)))

    def test_training_selection_uses_same_fixed_cost_only(self):
        rows = [{'metrics': {'finite': True, 'positive_training_depth_fraction': 1., 'robust_train_cost': c},
                 'heldout_error': h} for c, h in [(5., 0.), (2., 100.), (3., 1.)]]
        self.assertEqual(C.select_training_winner(rows), 1)
        rows[1]['metrics']['positive_training_depth_fraction'] = .5
        self.assertEqual(C.select_training_winner(rows), 2)

    def test_invalid_baseline_and_focal_fail_closed(self):
        model, tracks, _ = fixture()
        model['poses'][6] = copy.deepcopy(model['poses'][0])
        with self.assertRaises(ValueError):
            C.BundleProblem(model, tracks, 640, 480, set(range(7)))
        model, tracks, _ = fixture()
        problem = C.BundleProblem(model, tracks, 640, 480, set(range(7)))
        with self.assertRaises(ValueError):
            problem.fit(1.)


if __name__ == '__main__':
    unittest.main()
