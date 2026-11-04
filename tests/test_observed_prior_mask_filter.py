"""Mask filtering must remove faces, never invent or silently move geometry."""
import copy
import unittest

import numpy as np

import observed_prior_mask_filter as F


class PriorMaskFilterTests(unittest.TestCase):
    def test_unique_camera_iou_selection(self):
        target = np.zeros((20, 30), bool)
        target[3:17, 4:26] = True
        wrong = np.zeros_like(target)
        wrong[2:6, 2:6] = True
        selected, report = F.choose_camera([('wrong', wrong), ('target', target)], target)
        self.assertEqual(selected, 'target')
        self.assertEqual(report['status'], 'unique_native_camera_selected')

    def test_ambiguous_duplicate_cameras_fail_closed(self):
        target = np.ones((12, 12), bool)
        selected, report = F.choose_camera([('a', target), ('b', target)], target)
        self.assertIsNone(selected)
        self.assertEqual(report['status'], 'ambiguous_native_camera_instances')

    def test_tiny_camera_overlap_fails_closed(self):
        target = np.ones((12, 12), bool)
        tiny = np.zeros_like(target)
        tiny[:2, :2] = True
        selected, _ = F.choose_camera([('a', tiny)], target)
        self.assertIsNone(selected)

    def test_all_hands_are_unioned_and_guarded(self):
        camera = np.ones((30, 40), bool)
        a, b = np.zeros_like(camera), np.zeros_like(camera)
        a[8:10, 8:10] = True
        b[20:22, 28:30] = True
        domain, union, guard = F.refined_domain(camera, [a, b])
        self.assertEqual(int(union.sum()), 8)
        self.assertFalse(domain[6:12, 6:12].any())
        self.assertFalse(domain[18:24, 26:32].any())
        self.assertTrue((guard[a|b]).all())

    def test_mask_dimensions_cannot_be_resized_implicitly(self):
        with self.assertRaisesRegex(ValueError, 'exact source dimensions'):
            F.refined_domain(np.ones((12, 12)), [np.ones((6, 6))])

    def test_edge_crossing_mask_gap_rejected_even_if_vertices_inside(self):
        mask = np.ones((30, 30), bool)
        mask[5:8, 12:15] = False
        triangle = [[5, 6], [24, 6], [5, 24]]
        self.assertTrue(F.strict_mask_membership(mask, triangle))
        keep, reason = F.triangle_allowed(mask, triangle)
        self.assertFalse(keep)
        self.assertEqual(reason, 'edge_or_centroid_outside_refined_domain')

    def test_interior_mask_hole_rejected_even_when_edges_and_centroid_inside(self):
        mask = np.ones((35, 35), bool)
        mask[9, 9] = False
        triangle = [[4, 4], [30, 4], [4, 30]]
        keep, reason = F.triangle_allowed(mask, triangle)
        self.assertFalse(keep)
        self.assertEqual(reason, 'triangle_interior_outside_refined_domain')

    def test_subpixel_samples_require_both_neighbouring_pixels(self):
        mask = np.ones((10, 10), bool)
        mask[4, 5] = False
        self.assertFalse(F.strict_mask_membership(mask, [[4.2, 4]]))
        self.assertTrue(F.strict_mask_membership(mask, [[3.2, 4]]))
        self.assertFalse(F.strict_mask_membership(mask, [[float('nan'), 2]]))

    def test_filter_remaps_without_changing_original_coordinates_or_attributes(self):
        surface = {'kind':'model_prior_source_visible_patch',
                   'vertices':[[2.,2.,4.],[4.,2.,4.],[2.,4.,4.],[8.,8.,4.]],
                   'vertex_source_pixels':[[2,2],[4,2],[2,4],[8,8]],
                   'triangles':[[0,1,2],[1,2,3]],
                   'colors':[[1,2,3],[4,5,6],[7,8,9],[10,11,12]],
                   'uv':[[.2,.2],[.4,.2],[.2,.4],[.8,.8]],
                   'vertex_provenance':['model_prior']*4,
                   'face_provenance':['model_prior_aligned_inverse_depth']*2,
                   'face_confidence':['unknown']*2}
        original = copy.deepcopy(surface)
        mask = np.ones((12,12),bool)
        mask[7:,7:] = False
        result, report = F.filter_surface(surface, mask)
        self.assertEqual(surface, original)
        self.assertEqual(result['triangles'], [[0,1,2]])
        self.assertEqual(result['original_vertex_ids'], [0,1,2])
        self.assertEqual(result['original_face_ids'], [0])
        self.assertEqual(result['vertices'], surface['vertices'][:3])
        self.assertEqual(result['colors'], surface['colors'][:3])
        self.assertEqual(report['dropped_triangles'], 1)
        self.assertEqual(report['new_vertices_or_faces_created'], 0)
        self.assertFalse(result['clean_segmentation_established'])

    def test_all_rejected_faces_do_not_produce_substitute_mesh(self):
        surface = {'kind':'model_prior_source_visible_patch','vertices':[[0,0,4],[1,0,4],[0,1,4]],
                   'vertex_source_pixels':[[0,0],[1,0],[0,1]],'triangles':[[0,1,2]]}
        result, report = F.filter_surface(surface, np.zeros((10,10),bool))
        self.assertIsNone(result)
        self.assertEqual(report['retained_vertices'], 0)
        self.assertEqual(report['retained_triangles'], 0)


if __name__=='__main__':
    unittest.main()
