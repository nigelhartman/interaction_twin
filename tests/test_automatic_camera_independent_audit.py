"""Numerical and leakage checks independent from automatic-camera fitting."""
import copy
import unittest

import cv2
import numpy as np

import automatic_camera_independent_audit as A
import automatic_camera as F
import consistency
import observed_calibration as C
import observed_geometry as G


class IndependentCameraTests(unittest.TestCase):
    def test_normalized_centres_project_to_native_integer_centres(self):
        rng = np.random.default_rng(17654)
        normalized = np.array([[.9, 0, .47], [0, 1.6, .52], [0, 0, 1.]])
        for width, height in [(960, 540), (540, 960), (641, 359)]:
            K = A.normalized_camera_to_pixels(normalized, width, height)
            pixels = rng.uniform([0, 0], [width - 1, height - 1], size=(1000, 2))
            uv = (pixels + .5) / [width, height]
            rays = np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(normalized).T
            depths = rng.uniform(.3, 8., len(uv))
            actual, _ = A.opencv_projection(rays * depths[:, None], np.eye(3), np.zeros(3), K)
            np.testing.assert_allclose(actual, pixels, atol=5e-12)
            np.testing.assert_allclose(K, consistency.pixel_intrinsics(normalized, width, height), atol=1e-12)

    def test_resize_roundtrip_retains_half_pixel_offset(self):
        normal = np.array([[.9543153047561646, 0, .5], [0, 1.6965605020523071, .5], [0, 0, 1.]])
        source = A.normalized_camera_to_pixels(normal, 960, 540)
        model = A.normalized_camera_to_pixels(normal, 512, 288)
        scale = 960 / 512
        source_from_model = np.array([[scale, 0, (scale - 1) / 2], [0, scale, (scale - 1) / 2], [0, 0, 1.]])
        np.testing.assert_allclose(source_from_model @ model, source, atol=1e-12)
        self.assertEqual(source[0, 2], 479.5)
        self.assertEqual(source[1, 2], 269.5)
        self.assertGreater(abs(source[0, 2] - G.camera_matrix(960, 540)[0, 2]), .49)

    def test_bad_camera_rejected(self):
        for dimensions in [(False, 2), (0, 2), (1.2, 2), (2, -1)]:
            with self.assertRaises(ValueError):
                A.normalized_camera_to_pixels(np.eye(3), *dimensions)
        for key, value in [((0, 0), -1), ((1, 1), np.inf), ((2, 2), 2)]:
            K = np.eye(3); K[key] = value
            with self.assertRaises(ValueError):
                A.normalized_camera_to_pixels(K, 640, 480)

    def test_opencv_projection_matches_engine_without_using_engine_residual(self):
        rng = np.random.default_rng(7)
        points = rng.uniform(-1, 1, size=(300, 3)); points[:, 2] += 5
        for rvec in [np.zeros(3), np.array([.12, -.07, .02])]:
            R = cv2.Rodrigues(rvec)[0]; t = np.array([.2, -.1, .7])
            K = np.array([[927., 0, 479.5], [0, 912., 269.5], [0, 0, 1.]])
            independent, z = A.opencv_projection(points, R, t, K)
            production, production_z = G.project(points, (R, t), K)
            np.testing.assert_allclose(independent, production, atol=1e-10)
            np.testing.assert_allclose(z, production_z, atol=1e-12)

    def test_projection_rejects_reflections_skew_and_invalid_homogeneous_scale(self):
        xyz = np.array([[0., 0., 2.]])
        K = G.camera_matrix(640, 480)
        with self.assertRaises(ValueError):
            A.opencv_projection(xyz, np.diag([-1., 1., 1.]), np.zeros(3), K)
        for entry, value in [((0, 1), .001), ((2, 2), 1.00000001)]:
            invalid = K.copy(); invalid[entry] = value
            with self.assertRaises(ValueError):
                A.opencv_projection(xyz, np.eye(3), np.zeros(3), invalid)

    def test_gauge_normalization_preserves_anisotropic_offcentre_camera(self):
        rng = np.random.default_rng(13)
        points = {str(i): rng.normal(size=3) + [0, 0, 8] for i in range(30)}
        poses = {i: (cv2.Rodrigues(np.array([.01*i, .04*i, -.02*i]))[0], np.array([.2*i, -.1*i, .03*i])) for i in range(5)}
        K = np.array([[837., 0, 471.2], [0, 915., 267.9], [0, 0, 1.]])
        newpoints, newposes = C.normalize_gauge(points, poses, 1, 4)
        for fi, (R, t) in poses.items():
            before, _ = A.opencv_projection(list(points.values()), R, t, K)
            after, _ = A.opencv_projection(list(newpoints.values()), *newposes[fi], K)
            np.testing.assert_allclose(before, after, atol=1e-10)
        self.assertAlmostEqual(np.linalg.norm(newposes[4][1]), 1., places=12)

    def test_fixed_full_K_bundle_residual_matches_independent_opencv(self):
        rng = np.random.default_rng(29)
        points = {str(i): rng.uniform([-1, -.5, 4], [1, .5, 7]) for i in range(24)}
        poses = {i: (cv2.Rodrigues(np.array([.03*i, -.02*i, .01*i]))[0], np.array([.2*i, 0, .03*i])) for i in range(4)}
        K = np.array([[831., 0, 318.5], [0, 802., 235.5], [0, 0, 1.]])
        tracks = {tid: {} for tid in points}
        for fi, pose in poses.items():
            for tid, point in points.items():
                tracks[tid][fi] = A.opencv_projection(point[None], *pose, K)[0][0]
        problem = F.FixedCameraProblem({'seed': {'a': 0, 'b': 3}, 'points': points, 'poses': poses},
                                       tracks, 640, 480, set(poses), K)
        for _ in range(6):
            state = problem.initial + rng.normal(0, .005, len(problem.initial))
            cameras, xyz = problem.decode(state)
            expected = []
            expected_depths = []
            for fi, tid, observed in problem.observations:
                projected, depth = A.opencv_projection(xyz[problem.point_index[tid]][None], *cameras[fi], K)
                expected.extend(projected[0] - observed)
                expected_depths.append(depth[0])
            actual, depths = problem.residual_and_depth(state)
            np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)
            np.testing.assert_allclose(depths, expected_depths, atol=1e-12, rtol=0)

    def test_raw_support_rejects_reserved_observations_or_poses(self):
        points = {'a': [0, 0, 4], 'b': [1, 0, 4]}
        tracks = {'a': {0: [0, 0], 1: [1, 0]}, 'b': {1: [2, 0], 3: [3, 0]}}
        with self.assertRaises(ValueError):
            A.raw_support(points, {0: None, 1: None}, tracks, {0, 1})
        with self.assertRaises(ValueError):
            A.raw_support(points, {0: None, 1: None, 2: None}, {'a': {0: [0, 0]}, 'b': {1: [0, 0]}}, {0, 1})

    def test_raw_support_exposes_unobserved_camera_and_two_view_points(self):
        result = A.raw_support({'a': [0, 0, 4], 'b': [1, 0, 4]}, {0: None, 1: None, 3: None},
                               {'a': {0: [0, 0], 1: [1, 0]}, 'b': {1: [2, 0]}}, {0, 1, 3})
        self.assertEqual(result['unobserved_cameras'], [3])
        self.assertEqual(result['camera_graph_components'], [[0, 1], [3]])
        self.assertEqual(result['point_view_count_histogram'], {1: 1, 2: 1})
        self.assertEqual(result['points_with_fewer_than_two_views'], 1)

    def test_viewer_audit_detects_forged_projection_and_reserved_role(self):
        point = np.array([.2, .1, 3.])
        K = G.camera_matrix(640, 480)
        image, _ = A.opencv_projection(point[None], np.eye(3), np.zeros(3), K)
        role = 'heldout' if G.score_track('a') else 'pose_anchor'
        viewer = {'points': [{'id': 'a', 'xyz': point.tolist()}],
            'cameras': [{'frame_id': 'f', 'R': np.eye(3).tolist(), 't': [0,0,0], 'K': K.tolist()}],
            'frames': [{'id': 'f', 'split': 'heldout', 'observations': [{'track_id': 'a', 'xy': image[0].tolist(),
                'predicted_xy': image[0].tolist(), 'error_px': 0., 'role': role}]}]}
        self.assertTrue(A.viewer_projection_audit(viewer)['passed'])
        unmatched = copy.deepcopy(viewer)
        unmatched['frames'][0]['observations'].append({'track_id': 'feature_0042', 'xy': [0, 0],
            'predicted_xy': None, 'error_px': None, 'role': 'unmatched_feature'})
        self.assertTrue(A.viewer_projection_audit(unmatched)['passed'])
        forged = copy.deepcopy(viewer); forged['frames'][0]['observations'][0]['predicted_xy'][0] += .1
        self.assertFalse(A.viewer_projection_audit(forged)['passed'])
        forged = copy.deepcopy(viewer); forged['frames'][0]['observations'][0]['role'] = 'made_up'
        self.assertFalse(A.viewer_projection_audit(forged)['passed'])


if __name__ == '__main__':
    unittest.main()
