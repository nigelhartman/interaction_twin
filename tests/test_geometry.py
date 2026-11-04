import unittest
import numpy as np
from geometry import intrinsics_from_hfov, lift_pixels, project_points, rigid_fit


class GeometryTests(unittest.TestCase):
    def test_depth_lift_round_trip_and_invalid_pixels(self):
        k = intrinsics_from_hfov(9, 7, 60)
        uv = np.array([[4, 3], [1, 1], [-1, 3], [9, 3], [np.nan, 1]])
        depth = np.full((7, 9), 2.0)
        xyz = lift_pixels(uv, depth, k)
        np.testing.assert_allclose(xyz[0], [0, 0, 2])
        np.testing.assert_allclose(project_points(xyz, k)[:2], uv[:2], atol=1e-10)
        self.assertTrue(np.isnan(xyz[2:]).all())
        depth[1, 1] = 0
        self.assertTrue(np.isnan(lift_pixels([[1, 1]], depth, k)).all())

    def test_rigid_fit_preserves_scale_and_recovers_known_motion(self):
        a = np.random.default_rng(7).normal(size=(30, 3))
        r = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        t = np.array([1, 2, 3])
        transform, errors = rigid_fit(a, a @ r.T + t)
        np.testing.assert_allclose(transform[:3, :3], r, atol=1e-12)
        np.testing.assert_allclose(transform[:3, 3], t, atol=1e-12)
        self.assertLess(errors.max(), 1e-12)
        _, scaled_errors = rigid_fit(a, 2*a)
        self.assertGreater(np.median(scaled_errors), .1)

    def test_rigid_fit_rejects_unobservable_collinear_points(self):
        p = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        with self.assertRaises(ValueError):
            rigid_fit(p, p)


if __name__ == '__main__':
    unittest.main()
