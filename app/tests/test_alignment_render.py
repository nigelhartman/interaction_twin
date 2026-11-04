import unittest
import numpy as np

from alignment_render import ray_mesh_points, render_mesh
from geometry import project_points


class AlignmentRenderTests(unittest.TestCase):
    def setUp(self):
        self.K = np.array([[10., 0, 5], [0, 10, 5], [0, 0, 1]])
        self.R = np.eye(3)
        self.t = np.zeros(3)

    def camera_vertices(self, uv, z):
        return np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(self.K).T * np.asarray(z)[:, None]

    def test_perspective_depth_and_color_are_not_linear_screen_interpolation(self):
        uv = np.array([[2, 2], [8, 2], [2, 8]])
        vertices = self.camera_vertices(uv, [2, 4, 8])
        result = render_mesh(vertices, [[0, 1, 2]], self.K, self.R, self.t, 11, 11,
                             vertex_colors=np.eye(3) * 255)
        self.assertAlmostEqual(result['depth'][4, 4], 24 / 7, places=6)
        np.testing.assert_array_equal(result['rgb'][4, 4], [146, 73, 36])
        self.assertTrue(result['mask'][4, 4])
        self.assertTrue(np.isinf(result['depth'][10, 10]))
        self.assertEqual(result['face_index'][10, 10], -1)

    def test_z_buffer_is_order_independent_and_intersections_select_front(self):
        uv = np.array([[2, 2], [8, 2], [2, 8]])
        vertices = np.r_[self.camera_vertices(uv, [5] * 3), self.camera_vertices(uv, [2] * 3)]
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        a = render_mesh(vertices, faces, self.K, self.R, self.t, 11, 11)
        b = render_mesh(vertices, faces[::-1], self.K, self.R, self.t, 11, 11)
        np.testing.assert_array_equal(a['depth'], b['depth'])
        self.assertEqual(a['face_index'][4, 4], 1)
        hits, valid = ray_mesh_points(np.array([[4., 4.], [10., 10.], [np.nan, 3.]]),
                                     vertices, faces, self.K, self.R, self.t)
        np.testing.assert_array_equal(valid, [True, False, False])
        np.testing.assert_allclose(hits[0], [-.2, -.2, 2])
        self.assertTrue(np.isnan(hits[1:]).all())

    def test_ray_hits_return_canonical_coordinates_with_fixed_scale_and_rotation(self):
        vertices = np.array([[-1., -1, 0], [1, -1, 0], [0, 1, 0]])
        faces = np.array([[0, 1, 2]])
        rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        canonical = np.array([[.1, -.2, 0], [-.2, .3, 0]])
        translation = np.array([.2, -.1, 5])
        camera = 2 * canonical @ rotation.T + translation
        uv = project_points(camera, self.K)
        hits, valid = ray_mesh_points(uv, vertices, faces, self.K, rotation, translation, scale=2)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(hits, canonical, atol=1e-12)
        first = render_mesh(vertices, faces, self.K, rotation, translation, 11, 11, scale=2)
        second = render_mesh(vertices, faces, self.K, rotation, translation + [0, 0, 3], 11, 11, scale=2)
        self.assertGreater(first['mask'].sum(), second['mask'].sum())
        np.testing.assert_array_equal(vertices, [[-1, -1, 0], [1, -1, 0], [0, 1, 0]])

    def test_near_plane_clipping_and_negative_depth(self):
        vertices = np.array([[-.4, -.4, -.2], [.4, -.4, 1], [0, .4, 1]])
        result = render_mesh(vertices, [[0, 1, 2]], self.K, self.R, self.t, 11, 11)
        self.assertTrue(result['mask'].any())
        self.assertTrue((result['depth'][result['mask']] > 0).all())
        behind = render_mesh(vertices - [0, 0, 3], [[0, 1, 2]], self.K, self.R, self.t, 11, 11)
        self.assertFalse(behind['mask'].any())


if __name__ == '__main__':
    unittest.main()
