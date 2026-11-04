"""Analytic relative-gauge observations; no source recordings or learned calls."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

import assembly_scale_feasibility as A


def rows():
    scale, offset = .35, np.array([.2, -.1, .08])
    relative = Rotation.from_rotvec([.12, -.06, .08]).as_matrix()
    result = []
    for i in range(8):
        ra = Rotation.from_rotvec([i * .045, i * -.02, i * .03]).as_matrix()
        ta = np.array([.2 * np.sin(i), .1 * np.cos(i), 3. + i * .12])
        rb = ra @ relative
        tb = (ta + ra @ offset) / scale
        result.append({'sample_index': i + 1, 'timestamp_s': float(i), 'reserved': i % 3 == 2,
                       'a_reference': i == 0, 'b_reference': i == 1,
                       'a_R': ra, 'a_t': ta, 'b_R': rb, 'b_t': tb})
    return result, scale, offset, relative


def documents():
    rs, *_ = rows()
    scope = {'scope_reviewed': True, 'scope_safe': True, 'single_component_visible': True,
             'component_boundary_plausible': True, 'other_components_inside_mask': False}
    result = []
    for role in ['a', 'b']:
        frames = []
        for row in rs:
            reference = row[role + '_reference']
            frames.append({'sample_index': row['sample_index'], 'timestamp_s': row['timestamp_s'],
                           'source_image_sha256': str(row['sample_index']) * 64, 'width': 40, 'height': 30,
                           'evaluation_timestamp_reserved_before_reference_selection': row['reserved'],
                           'status': A.REFERENCE if reference else A.SUPPORTED, 'reference': reference,
                           'scope_assessment': dict(scope), 'candidate_rotation': row[role + '_R'].tolist(),
                           'candidate_translation': row[role + '_t'].tolist(), 'mask_sha256': role * 64})
        result.append({'format': 'short-interval-motion', 'version': 1, 'status': 'complete_with_gaps',
                       'coordinate_convention': A.CONVENTION, 'manual_geometry_or_poses': False,
                       'source_video_sha256': 'a' * 64, 'interaction_sha256': 'b' * 64,
                       'component_id': role, 'episodes': [{'id': 'episode-01',
                       'intrinsics': {'matrix': np.eye(3).tolist(), 'width': 40, 'height': 30}, 'frames': frames}]})
    return result


class AssemblyScaleTests(unittest.TestCase):
    def test_exact_excited_pair_recovers_scalar_and_offset(self):
        rs, scale, offset, rotation = rows()
        result = A.fit_compatibility(rs)
        self.assertEqual(result['status'], 'conditionally_compatible_scale_hypothesis')
        self.assertAlmostEqual(result['diagnostic_fit']['scale_b_in_a_native_units'], scale, places=12)
        np.testing.assert_allclose(result['diagnostic_fit']['offset_b_origin_in_a_native_units'], offset, atol=1e-12)
        matrix = np.array(result['common_frame_transform_a_from_b'])
        np.testing.assert_allclose(matrix[:3, :3], scale * rotation, atol=1e-12)
        self.assertFalse(result['physical_assembly_established'])

    def test_reserved_observations_never_change_fitted_parameters(self):
        rs, *_ = rows()
        old = A.fit_compatibility(rs)
        for row in rs:
            if row['reserved']:
                row['b_t'] = row['b_t'] + [1., .3, 0.]
        changed = A.fit_compatibility(rs)
        self.assertEqual(old['diagnostic_fit'], changed['diagnostic_fit'] | {'fit_status': 'accepted_compatibility_only'})
        self.assertIn('reserved_relative_translation_inconsistent', changed['reasons'])
        self.assertIsNone(changed['common_frame_transform_a_from_b'])

    def test_static_camera_pose_does_not_establish_scale(self):
        rs, *_ = rows()
        for row in rs:
            row['a_R'], row['b_R'] = np.eye(3), np.eye(3)
            row['a_t'], row['b_t'] = np.array([0., 0., 3.]), np.array([.1, 0., 4.])
        result = A.fit_compatibility(rs)
        self.assertIn('scale_unobservable', result['reasons'])
        self.assertIsNone(result['common_frame_transform_a_from_b'])

    def test_relative_motion_is_rejected_instead_of_becoming_attachment(self):
        rs, *_ = rows()
        rs[4]['b_R'] = rs[4]['b_R'] @ Rotation.from_euler('x', 25, degrees=True).as_matrix()
        result = A.fit_compatibility(rs)
        self.assertIn('training_relative_rotation_inconsistent', result['reasons'])
        self.assertIsNone(result['common_frame_transform_a_from_b'])

    def test_common_camera_rotation_cancels(self):
        rs, *_ = rows()
        old = A.fit_compatibility(rs)
        q = Rotation.from_rotvec([.1, -.04, .12]).as_matrix()
        for r in rs:
            r['a_R'], r['b_R'] = q @ r['a_R'], q @ r['b_R']
            r['a_t'], r['b_t'] = q @ r['a_t'], q @ r['b_t']
        new = A.fit_compatibility(rs)
        self.assertEqual(new['status'], old['status'])
        np.testing.assert_allclose(new['common_frame_transform_a_from_b'], old['common_frame_transform_a_from_b'], atol=1e-12)

    def test_native_origin_change_does_not_change_relative_scale(self):
        rs, scale, *_ = rows()
        for r in rs:
            r['a_t'] = r['a_t'] + r['a_R'] @ np.array([.07, -.03, .05])
            r['b_t'] = r['b_t'] + r['b_R'] @ np.array([-.02, .08, .03])
        result = A.fit_compatibility(rs)
        self.assertEqual(result['status'], 'conditionally_compatible_scale_hypothesis')
        self.assertAlmostEqual(result['diagnostic_fit']['scale_b_in_a_native_units'], scale, places=12)

    def test_negative_and_unstable_scale_fail_closed(self):
        rs, *_ = rows()
        for r in rs:
            r['b_t'] = np.array([0., 0., 20.]) - r['a_R'] @ (r['a_R'].T @ r['a_t'])
            r['b_R'] = r['a_R']
        self.assertNotEqual(A.fit_compatibility(rs)['status'], 'conditionally_compatible_scale_hypothesis')
        u = np.arange(18, dtype=float).reshape(6, 3)
        s, _, _ = A.solve_train_scale(u, -2 * u + 100.)
        self.assertAlmostEqual(s, -.5)

    def test_unverified_references_do_not_replace_reserved_check(self):
        rs, *_ = rows()
        for r in rs:
            r['reserved'] = False
        result = A.fit_compatibility(rs)
        self.assertIn('insufficient_pre_reserved_timestamps', result['reasons'])
        self.assertGreater(result['training_reference_conditioned_count'], 0)

    def test_invalid_rotation_nonfinite_and_bounds_rejected(self):
        rs, *_ = rows()
        rs[0]['a_R'] = np.eye(3) * 2
        with self.assertRaises(ValueError): A.fit_compatibility(rs)
        rs, *_ = rows(); rs[0]['a_t'][2] = -1
        with self.assertRaises(ValueError): A.fit_compatibility(rs)
        with self.assertRaises(ValueError): A.fit_compatibility(rs * 20)

    def test_duplicate_source_timestamps_do_not_inflate_support(self):
        rs, *_ = rows()
        rs[2]['timestamp_s'] = rs[1]['timestamp_s']
        with self.assertRaisesRegex(ValueError, 'Duplicate'): A.fit_compatibility(rs)

    def test_exact_join_scope_and_reference_identity_remain_separate(self):
        first, second = documents()
        group = A.prepare_episode_pairs(first, second)[0]
        self.assertEqual(len(group['rows']), 8)
        self.assertEqual(A.fit_compatibility(group['rows'])['training_reference_conditioned_count'], 2)
        second['episodes'][0]['frames'][0]['scope_assessment']['other_components_inside_mask'] = True
        self.assertEqual(len(A.prepare_episode_pairs(first, second)[0]['rows']), 7)
        second['episodes'][0]['frames'][1]['source_image_sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'identical'): A.prepare_episode_pairs(first, second)

    def test_camera_and_source_mismatch_do_not_create_assembly(self):
        first, second = documents()
        second['episodes'][0]['intrinsics']['matrix'][0][0] = 2.
        with self.assertRaisesRegex(ValueError, 'shared source camera'): A.prepare_episode_pairs(first, second)
        first, second = documents(); second['source_video_sha256'] = 'different'
        with self.assertRaisesRegex(ValueError, 'one exact source'): A.prepare_episode_pairs(first, second)

    def test_independent_episodes_are_never_pooled(self):
        first, second = documents()
        for doc in [first, second]:
            original = doc['episodes'][0]
            later = copy.deepcopy(original); later['id'] = 'episode-02'; later['frames'] = later['frames'][4:]
            original['frames'] = original['frames'][:4]; doc['episodes'].append(later)
        groups = A.prepare_episode_pairs(first, second)
        self.assertEqual(len(groups), 2)
        self.assertTrue(all(len(group['rows']) == 4 for group in groups))

    def test_saved_paths_cannot_escape_or_follow_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); (root / 'data').mkdir(); (root / 'data/file.json').write_text('{}')
            self.assertEqual(A.data_file(root, '/files/data/file.json'), root / 'data/file.json')
            for name in ['../outside', '/absolute', 'data/../outside', '.secrets/meta.env']:
                with self.assertRaises(ValueError): A.data_file(root, name)
            (root / 'data/link.json').symlink_to(root / 'data/file.json')
            with self.assertRaises(ValueError): A.data_file(root, 'data/link.json')


if __name__ == '__main__':
    unittest.main()
