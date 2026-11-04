import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

import observed_stereo as stereo


class ObservedStereoTests(unittest.TestCase):
    def setUp(self):
        self.K = np.array([[200., 0, 160], [0, 200., 120], [0, 0, 1]])
        self.pose = (np.eye(3), np.zeros(3))

    def frame(self, image, baseline=0.):
        mask = np.zeros(image.shape[:2], np.uint8)
        mask[12:-12, 12:-12] = 1
        return {'pose': (np.eye(3), np.array([-baseline, 0., 0.])),
                'gray': image, 'image': cv2.cvtColor(image, cv2.COLOR_GRAY2BGR),
                'mask': mask > 0, 'core': stereo.mask_core(mask)}

    def plane_images(self):
        rng = np.random.default_rng(42)
        left = rng.integers(0, 256, (240, 320), dtype=np.uint8)
        left = cv2.GaussianBlur(left, (3, 3), .5)
        right = np.zeros_like(left)
        right[:, :-20] = left[:, 20:]
        return self.frame(left), self.frame(right, .4)

    def test_project_unproject_nonidentity_pose(self):
        R = cv2.Rodrigues(np.array([.1, -.2, .03]))[0]
        pose = (R, np.array([.3, -.1, .2]))
        xyz = np.array([[.1, .2, 4.], [-.5, .1, 5.]])
        uv, z = stereo.project(xyz, pose, self.K)
        np.testing.assert_allclose(stereo.unproject(uv, z, pose, self.K), xyz, atol=1e-12)

    def test_lr_consistency_rejects_wrong_correspondence(self):
        left = np.full((10, 40), 6., np.float32)
        right = np.full((10, 40), -6., np.float32)
        keep = stereo.left_right_consistency(left, right)
        self.assertTrue(keep[:, 6:].all())
        self.assertFalse(keep[:, :6].any())
        right[:] = -10
        self.assertFalse(stereo.left_right_consistency(left, right).any())

    def test_lr_rejects_invalid_and_negative_disparity(self):
        left = np.full((5, 40), -2., np.float32)
        right = np.full((5, 40), 2., np.float32)
        self.assertFalse(stereo.left_right_consistency(left, right).any())
        left[:] = np.nan
        self.assertFalse(stereo.left_right_consistency(left, right).any())

    def test_mask_at_rejects_nonfinite_and_outside(self):
        mask = np.zeros((20, 30), bool)
        mask[5:15, 5:25] = True
        self.assertEqual(stereo.mask_at(mask, [[10, 10], [1, 1], [30, 10], [np.nan, 4]]).tolist(),
                         [True, False, False, False])

    def test_patch_never_crosses_mask_hole(self):
        image = np.ones((20, 20), np.uint8) * 100
        mask = np.ones((20, 20), bool)
        mask[10, 10] = False
        self.assertIsNone(stereo.sample_patch(image, mask, [[9, 9], [10, 10]]))

    def test_zncc_rejects_untextured_and_handles_brightness(self):
        self.assertIsNone(stereo.zncc(np.ones(49), np.ones(49)))
        a = np.arange(49)
        self.assertAlmostEqual(stereo.zncc(a, a * 2 + 20), 1.)

    def test_true_multiview_plane_is_photometrically_supported(self):
        left, right = self.plane_images()
        xyz = stereo.unproject([[150, 100]], [4.], self.pose, self.K)[0]
        result = stereo.photometric_check(xyz, left, right, self.K, discriminate_depth=True)
        self.assertTrue(result['passes'], result)
        self.assertGreater(result['depth_discrimination_margin'], .025)
        self.assertGreaterEqual(result['alternative_count'], 2)

    def test_wrong_depth_rejected_in_additional_view(self):
        left, right = self.plane_images()
        xyz = stereo.unproject([[150, 100]], [6.], self.pose, self.K)[0]
        result = stereo.photometric_check(xyz, left, right, self.K, discriminate_depth=True)
        self.assertFalse(result['passes'], result)

    def test_masked_extra_view_does_not_support_geometry(self):
        left, right = self.plane_images()
        right['core'][:] = False
        xyz = stereo.unproject([[150, 100]], [4.], self.pose, self.K)[0]
        self.assertFalse(stereo.photometric_check(xyz, left, right, self.K, discriminate_depth=True)['passes'])

    def test_no_baseline_cannot_discriminate_depth(self):
        left, _ = self.plane_images()
        xyz = stereo.unproject([[150, 100]], [4.], self.pose, self.K)[0]
        result = stereo.photometric_check(xyz, left, left, self.K, discriminate_depth=True)
        self.assertFalse(result['passes'])
        self.assertEqual(result['reason'], 'insufficient_depth_discrimination')

    def test_rectification_triangulation_roundtrip_nonidentity_left(self):
        gray = np.zeros((240, 320), np.uint8)
        left = self.frame(gray)
        right = self.frame(gray, .4)
        # Move the object coordinate frame; rectified coordinates must still invert it.
        R = cv2.Rodrigues(np.array([.07, .11, -.03]))[0]
        t = np.array([.1, -.2, .3])
        left['pose'] = (R, t)
        right['pose'] = (R, t + np.array([-.4, 0, 0]))
        rect = stereo.rectify_pair(left, right, self.K)
        xyz = np.array([[.2, .1, 4.], [-.3, -.2, 4.5]])
        a, _ = stereo.project(xyz, left['pose'], self.K)
        b, _ = stereo.project(xyz, right['pose'], self.K)
        ra = cv2.undistortPoints(a[:, None], self.K, None, R=rect['R1'], P=rect['P1'])[:, 0]
        rb = cv2.undistortPoints(b[:, None], self.K, None, R=rect['R2'], P=rect['P2'])[:, 0]
        np.testing.assert_allclose(stereo.unrectify_pixels(ra, rect['R1'], rect['P1'], self.K), a, atol=1e-10)
        np.testing.assert_allclose(stereo.unrectify_pixels(rb, rect['R2'], rect['P2'], self.K), b, atol=1e-10)
        found = stereo.rectify_points_to_world(ra, ra[:, 0] - rb[:, 0], rect, left['pose'])
        np.testing.assert_allclose(found, xyz, atol=1e-10)

    def test_reverse_order_rejects_negative_disparity(self):
        gray = np.zeros((240, 320), np.uint8)
        with self.assertRaisesRegex(ValueError, 'positive horizontal'):
            stereo.rectify_pair(self.frame(gray, .4), self.frame(gray), self.K)

    def test_zbuffer_rejects_known_occluded_layer(self):
        xyz = np.array([[0., 0., 4.], [0., 0., 6.], [.1, 0., 4.]])
        self.assertEqual(stereo.zbuffer_visibility(xyz, self.pose, self.K, (240, 320)).tolist(), [True, False, True])

    def test_robust_depth_bounds_exclude_single_outlier(self):
        z = np.r_[np.linspace(7., 8., 200), 19.]
        points = np.c_[np.zeros(len(z)), np.zeros(len(z)), z]
        bounds = stereo.robust_depth_bounds(points, self.pose, self.K)
        self.assertLess(bounds['upper'], 9.)
        self.assertEqual(bounds['excluded_sparse_outlier_count'], 1)

    def test_sgbm_recovers_synthetic_plane_disparity(self):
        left, right = self.plane_images()
        dl, dr, valid, settings = stereo.stereo_disparities(left['gray'], right['gray'], np.full(100, 20.))
        roi = np.s_[40:200, 80:260]
        keep = stereo.left_right_consistency(dl, dr)[roi] & valid[roi]
        self.assertGreater(keep.mean(), .9)
        self.assertLess(np.median(np.abs(dl[roi][keep] - 20.)), .1)

    def test_visible_mesh_has_source_uv_and_no_hole_bridge(self):
        image = np.zeros((240, 320), np.uint8)
        left = self.frame(image)
        right = self.frame(image, .4)
        left.update(id='train_a', image_url='/files/data/test/frame.jpg', source_row={'image_sha256': 'a' * 64})
        pixels = [[140, 100], [142, 100], [140, 102], [142, 102], [160, 100]]
        xyz = stereo.unproject(pixels, np.full(len(pixels), 4.), self.pose, self.K)
        points = [{'xyz': p.tolist(), 'rgb': [1, 2, 3], 'reference_depth_arbitrary_units': 4.,
                   'rectified_reference_pixel': uv} for p, uv in zip(xyz, pixels)]
        mesh = stereo.visible_lattice_mesh(points, left, right, self.K)
        self.assertEqual(len(mesh['triangles']), 2)
        self.assertFalse(mesh['hole_filling'])
        self.assertFalse(mesh['watertight'])
        self.assertNotIn(4, np.array(mesh['triangles']).ravel())
        np.testing.assert_allclose(mesh['uv'][0], [140.5 / 320, 1 - 100.5 / 240])

    def test_visible_mesh_rejects_depth_discontinuity(self):
        image = np.zeros((240, 320), np.uint8)
        left, right = self.frame(image), self.frame(image, .4)
        pixels = [[140, 100], [142, 100], [140, 102]]
        depths = [4., 4., 5.]
        xyz = stereo.unproject(pixels, depths, self.pose, self.K)
        points = [{'xyz': p.tolist(), 'rgb': [1, 2, 3], 'reference_depth_arbitrary_units': z,
                   'rectified_reference_pixel': uv} for p, uv, z in zip(xyz, pixels, depths)]
        self.assertIsNone(stereo.visible_lattice_mesh(points, left, right, self.K))

    def test_source_texture_uv_maps_pixel_centres(self):
        uv = stereo.texture_uv([[0, 0], [319, 239]], 320, 240)
        np.testing.assert_allclose(uv, [[.5 / 320, 1 - .5 / 240], [319.5 / 320, 1 - 239.5 / 240]])


if __name__ == '__main__':
    unittest.main()
