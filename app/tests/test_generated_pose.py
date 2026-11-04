import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

spec = importlib.util.spec_from_file_location('generated_pose', Path(__file__).parents[1] / 'scripts/stage4_generated_pose.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class GeneratedPoseTests(unittest.TestCase):
    def test_pixel_centre_resize_preserves_projection(self):
        k = module.intrinsics(960, 540, 60)
        resized = module.resized_intrinsics(k, (960, 540), (256, 144))
        xyz = np.array([[.3, -.2, 2.], [-.5, .4, 3.]])
        original = module.project(xyz, np.eye(3), np.zeros(3), k)
        actual = module.project(xyz, np.eye(3), np.zeros(3), resized)
        np.testing.assert_allclose(actual, (original + .5) * (256 / 960) - .5)

    def test_portrait_resize_retains_distinct_xy_pixel_mapping(self):
        k = module.intrinsics(540, 960, 60)
        resized = module.resized_intrinsics(k, (540, 960), (256, 455))
        xyz = np.array([[.3, .8, 2.], [-.5, -.6, 3.]])
        original = module.project(xyz, np.eye(3), np.zeros(3), k)
        actual = module.project(xyz, np.eye(3), np.zeros(3), resized)
        np.testing.assert_allclose(actual, (original + .5) * np.array([256/540, 455/960]) - .5)

    def test_nested_selection_resolves_case_by_matching_source_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / 'case'
            selection = case / 'inputs' / 'track' / 'selection.json'
            selection.parent.mkdir(parents=True)
            selection.write_text('{}')
            (case / 'manifest.json').write_text(json.dumps({'source': {'sha256': 'correct'}}))
            (selection.parent / 'manifest.json').write_text(json.dumps({'source': {'sha256': 'wrong'}}))
            self.assertEqual(module.find_case_root(selection, 'correct'), case.resolve())
            with self.assertRaisesRegex(ValueError, 'No ancestor case'):
                module.find_case_root(selection, 'absent')

    def test_mask_provenance_does_not_label_model_seeds_as_manual(self):
        proposed = module.mask_seed_provenance({'mask_provenance': {
            'prompt_origin': 'model_generated_prompt', 'manual_seed_boxes_used': False,
            'original_seed_origin': 'codex_cli_hosted_OpenAI_model_proposal'}})
        self.assertFalse(proposed['manual_seed_boxes_used'])
        self.assertFalse(proposed['sam_masks_inherit_assistant_prompts'])
        manual = module.mask_seed_provenance({'mask_provenance': {'prompt_origin': 'assistant_visual_annotation'}})
        self.assertTrue(manual['manual_seed_boxes_used'])
        unknown = module.mask_seed_provenance({})
        self.assertIsNone(unknown['manual_seed_boxes_used'])

    def test_generated_pose_gate_never_confuses_silhouette_with_truth(self):
        gate = module.usable_pose_gate(.999, independent_pose_support=False, ambiguous=False)
        self.assertFalse(gate['accepted'])
        self.assertIn('no_independent_depth_or_multiview_pose_support', gate['reasons'])
        self.assertFalse(gate['reference_silhouette_is_independent_pose_truth'])

    def test_rotation_search_is_deterministic_and_contains_proper_rotations(self):
        a, b = module.generic_orientations(16), module.generic_orientations(16)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (40, 3, 3))
        np.testing.assert_allclose(np.linalg.det(a), 1., atol=1e-12)

    def test_triangle_projection_retains_full_geometry_outside_observed_region(self):
        vertices = np.array([[-1., -1, 0], [1, -1, 0], [0, 1, 0]])
        k = np.array([[20., 0, 20], [0, 20, 20], [0, 0, 1.]])
        mask = module.silhouette(vertices, np.array([[0, 1, 2]]), np.eye(3), np.array([0., 0, 2]), k, (41, 41))
        self.assertTrue(mask[20, 20])
        self.assertTrue(mask[10, 10])
        behind = module.silhouette(vertices, np.array([[0, 1, 2]]), np.eye(3), np.array([0., 0, -2]), k, (41, 41))
        self.assertFalse(behind.any())

    def test_bounding_box_initializer_places_synthetic_mesh_in_frame(self):
        vertices = np.array([[-.5, -.3, -.2], [.5, -.3, -.2], [.5, .3, -.2], [-.5, .3, -.2],
                             [-.5, -.3, .2], [.5, -.3, .2], [.5, .3, .2], [-.5, .3, .2]])
        r = Rotation.from_euler('xyz', [20, 30, 15], degrees=True).as_matrix()
        k = module.intrinsics(200, 120, 60)
        target = np.zeros((120, 200), bool)
        target[35:85, 65:145] = True
        t = module.initial_translation(vertices, r, k, target)
        uv = module.project(vertices, r, t, k)
        self.assertIsNotNone(uv)
        np.testing.assert_allclose((uv.max(0) + uv.min(0)) / 2, [104.5, 59.5], atol=1.)

    def test_asymmetric_vertex_appearance_disambiguates_identical_silhouettes(self):
        vertices = np.array([[-1., -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]])
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        colors = np.array([[255., 20, 20], [30, 30, 40], [10, 250, 20], [20, 40, 250]])
        k = np.array([[50., 0, 40], [0, 50, 40], [0, 0, 1.]])
        t = np.array([0., 0, 2])
        upright = module.render_mesh(vertices, faces, k, np.eye(3), t, 81, 81, vertex_colors=colors)
        inverted = module.render_mesh(vertices, faces, k, np.diag([-1., -1, 1]), t, 81, 81, vertex_colors=colors)
        np.testing.assert_array_equal(upright['mask'], inverted['mask'])
        correct = module.appearance_similarity(upright['rgb'], upright['rgb'], upright['mask'])
        wrong = module.appearance_similarity(upright['rgb'], inverted['rgb'], upright['mask'])
        self.assertAlmostEqual(correct['correlation'], 1., places=5)
        self.assertGreater(correct['correlation'], wrong['correlation'] + .5)
        self.assertGreater(module.reference_rank_score(1., correct, .2), module.reference_rank_score(1., wrong, .2))
        # Appearance is still reference fitting, not independent pose validation.
        self.assertFalse(module.usable_pose_gate(1., False, False)['accepted'])

    def test_flat_appearance_does_not_fabricate_orientation_evidence(self):
        image = np.full((64, 64, 3), 120, np.uint8)
        appearance = module.appearance_similarity(image, image, np.ones((64, 64), bool))
        self.assertFalse(appearance['available'])
        self.assertIsNone(appearance['correlation'])


if __name__ == '__main__':
    unittest.main()
