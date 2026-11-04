"""Mesh-anchor kinematics/math contracts; no paid models, GPU or geometry edits."""
import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

import sam3d_mechanisms as M
import upgrade_bundles as U


class MechanismMathTests(unittest.TestCase):
    @staticmethod
    def cup_points():
        angle = np.linspace(0, 2 * np.pi, 160, endpoint=False)
        rim = np.c_[np.cos(angle), .55 * np.sin(angle), np.full(len(angle), .15)]
        xx, yy = np.meshgrid(np.linspace(-.95, .95, 30), np.linspace(-.5, .5, 20))
        solid = np.c_[xx.ravel(), yy.ravel(), np.full(xx.size, -.15)]
        return np.concatenate([rim, solid])

    def test_cavity_prior_requires_asymmetric_extreme_faces_and_detects_the_actual_opening_side(self):
        vertices = self.cup_points()
        original = vertices.copy(); frame = M.cavity_rim_frame(vertices, np.array([1., 0., 0.]))
        self.assertIsNotNone(frame); self.assertGreater(float(frame['normal'] @ [0, 0, 1]), .99)
        np.testing.assert_array_equal(vertices, original)
        symmetric = np.concatenate([vertices, vertices * [1, 1, -1]])
        self.assertIsNone(M.cavity_rim_frame(symmetric, np.array([1., 0., 0.])))

    def test_cavity_mating_uses_opposed_openings_aligned_footprints_and_real_rim_vertices(self):
        vertices = self.cup_points(); mesh = SimpleNamespace(vertices=vertices)
        matrix = np.eye(4); matrix[2, 3] = 4.
        poses = {i: {'status': 'estimated', 'timestamp_s': i / 3, 'camera_from_native_mesh': matrix.tolist()} for i in range(6)}
        hint = {'source_sample_index': 0, 'timestamp_s': 0., 'endpoints_px': [[0., 0.], [200., 0.]], 'uncertainty_px': 1000.}
        region = {'pca_axis': np.array([1., 0., 0.]), 'used_hints': [hint], 'vertex_ids': [],
                  'points': np.empty((0, 3)), 'candidate_mean_hint_distance_px': []}
        p_region, c_region = copy.deepcopy(region), copy.deepcopy(region)
        model = {'axis': np.array([1., 0., 0.]), 'rows': {i: {'q': 0.} for i in poses}}
        fitted = M.cavity_mating_model(model, mesh, mesh, poses, poses, list(poses), p_region, c_region, 1., 1., np.eye(3))
        prior = fitted['cavity_mating_prior']; self.assertEqual(prior['status'], 'estimated_geometry_prior')
        pn, cn = np.array(prior['parent']['normal']), np.array(prior['child']['normal'])
        self.assertAlmostEqual(float(pn @ fitted['zero'] @ cn), -1.)
        h, c = p_region['points'][fitted['pi']], c_region['points'][fitted['ci']]
        transformed = vertices @ fitted['zero'].T + h - fitted['zero'] @ c
        np.testing.assert_allclose(np.mean(transformed[:, :2], axis=0), np.mean(vertices[:, :2], axis=0), atol=.02)
        self.assertIn(p_region['vertex_ids'][fitted['pi']], list(range(160)))
        self.assertIn(c_region['vertex_ids'][fitted['ci']], list(range(160)))

    def test_native_scale_is_applied_once_and_mesh_anchors_coincide_at_every_intermediate_angle(self):
        parent_scale, child_scale = .42, .17
        parent_anchor = np.array([.2, -.3, .1]); child_anchor = np.array([-.4, .1, .2])
        axis = np.array([1., 2., -.5]); axis /= np.linalg.norm(axis)
        zero = Rotation.from_euler('xyz', [.2, -.1, .3]).as_matrix()
        a, b = M.joint_frames(parent_scale, child_scale, parent_anchor, child_anchor, zero)
        for time in np.linspace(0, 1, 101):
            parent = np.eye(4); parent[:3, :3] = parent_scale * Rotation.from_euler('xy', [time, time * .3]).as_matrix()
            parent[:3, 3] = [time, -.4, 2.]
            child = M.forward(parent, a, axis, time * 2.4 - .5, b)
            p = parent @ np.r_[parent_anchor / parent_scale, 1.]
            c = child @ np.r_[child_anchor / child_scale, 1.]
            np.testing.assert_allclose(p, c, atol=1e-12)
            np.testing.assert_allclose(np.linalg.svd(child[:3, :3], compute_uv=False), [child_scale] * 3, atol=1e-12)
            self.assertGreater(np.linalg.det(child[:3, :3]), 0)

    def test_revolute_relative_motion_and_discrete_mesh_anchor_pair_are_recovered_without_floating_pivots(self):
        axis = np.array([1., 0., 0.]); angles = np.linspace(0, 1.8, 20)
        zero = Rotation.from_euler('y', .3).as_matrix()
        rotations = Rotation.from_rotvec(angles[:, None] * axis).as_matrix() @ zero
        parent = np.array([[.2, .3, .1], [.5, .5, .5], [0., 0., 0.]])
        child = np.array([[-.2, .1, .2], [.4, .5, .1], [.1, .1, .1]])
        target = parent[0] - np.einsum('tij,j->ti', rotations, child[0])
        pi, ci, residuals = M.choose_anchors(parent, child, rotations, target, 1.)
        self.assertEqual((pi, ci), (0, 0)); np.testing.assert_allclose(residuals, 0, atol=1e-12)
        region_p = {'pca_axis': axis}; region_c = {'pca_axis': zero.T @ axis}
        result = M.choose_axis(rotations, zero, region_p, region_c)
        self.assertAlmostEqual(abs(float(result['axis'] @ axis)), 1.)
        np.testing.assert_allclose(result['rotation_residual_rad'], 0, atol=1e-12)

    def test_twist_curve_unwraps_without_discontinuous_full_turns(self):
        axis = np.array([0., 0., 1.]); angles = np.linspace(2.8, 4., 20)
        matrices = Rotation.from_rotvec(angles[:, None] * axis).as_matrix()
        recovered = M.twist_angles(matrices, np.eye(3), axis)
        np.testing.assert_allclose(np.diff(recovered), np.diff(angles), atol=1e-12)

    def test_noisy_free_pose_rotation_cannot_choose_an_axis_across_the_mesh_hinge_seam(self):
        seam = np.array([1., 0., 0.]); wrong = np.array([0., 1., 0.])
        rotations = Rotation.from_rotvec(np.linspace(0, 1., 10)[:, None] * wrong).as_matrix()
        result = M.choose_axis(rotations, np.eye(3), {'pca_axis': seam}, {'pca_axis': seam})
        self.assertGreater(abs(float(result['axis'] @ seam)), .85)

    def test_surface_candidates_remain_actual_vertices_near_the_cited_hinge_seam(self):
        line = np.c_[np.linspace(-1, 1, 40), np.zeros(40), np.zeros(40)]
        unrelated = np.c_[np.linspace(-1, 1, 40), np.ones(40), np.zeros(40)]
        vertices = np.concatenate([line, unrelated]); before = vertices.copy()
        matrix = np.eye(4); matrix[:3, :3] *= .4; matrix[2, 3] = 3.
        k = np.array([[300, 0, 160], [0, 300, 120], [0, 0, 1.]])
        poses = {1: {'status': 'estimated', 'timestamp_s': 0., 'camera_from_native_mesh': matrix.tolist()}}
        hints = [{'source_sample_index': 1, 'timestamp_s': 0., 'endpoints_px': [[120, 120], [200, 120]], 'uncertainty_px': 3.}]
        result = M.surface_candidates(vertices, poses, hints, k)
        self.assertTrue(all(i < 40 for i in result['vertex_ids']))
        np.testing.assert_array_equal(result['points'], vertices[result['vertex_ids']]); np.testing.assert_array_equal(vertices, before)
        self.assertGreater(abs(float(result['pca_axis'][0])), .99)
        hints[0]['timestamp_s'] = .1
        with self.assertRaisesRegex(ValueError, 'PTS'):
            M.surface_candidates(vertices, poses, hints, k)

    def test_malformed_scale_axis_and_2d_seam_fail_closed(self):
        with self.assertRaises(ValueError):
            M.axis_rotation([2, 0, 0], .2)
        bad = np.eye(4); bad[0, 0] = 2
        with self.assertRaises(ValueError):
            M.rotation_pose(bad)
        with self.assertRaises(ValueError):
            M.line_distance(np.ones((5, 2)), [[1, 1], [1, 1]])


class MechanismSourceBindingTests(unittest.TestCase):
    def fixture(self, root, joint_type='prismatic'):
        folder = root / 'data/input'; folder.mkdir(parents=True)
        for name in ['sam3d_mechanisms.py', 'sam3d_common_scene.py', 'native_appearance.py']:
            (root / name).write_text('INERT_SOURCE_BOUND_UNIT_TEST_FIXTURE_' + name)
        asset = folder / 'asset.glb'; asset.write_bytes(b'INERT_UNSUPPORTED_ATTACHMENT_FIXTURE_NOT_NATIVE_GEOMETRY')
        pose = {'source_sample_index': 1, 'timestamp_s': 0., 'status': 'estimated', 'camera_from_native_mesh': np.eye(4).tolist()}
        scene = {'format': M.S.FORMAT, 'accepted_motion_poses': 0, 'source': {'sha256': 'a' * 64},
                 'original_program': {'sha256': 'b' * 64}, 'limitations': [],
                 'components': [{'id': id, 'asset': M.S.ref(asset, root), 'poses': [copy.deepcopy(pose)]} for id in ['parent', 'child']]}
        proposal = {'format': 'sam3d-mechanism-proposals', 'version': 1, 'source_sha256': 'a' * 64,
                    'interaction_sha256': 'b' * 64, 'physical_accuracy_verified': False,
                    'joints': [{'id': 'j1', 'parent_id': 'parent', 'child_id': 'child', 'joint_type': joint_type, 'attachment': 'persistent'}]}
        scene_path, proposal_path = folder / 'scene.json', folder / 'proposals.json'
        U.write(scene_path, scene); U.write(proposal_path, proposal)
        return scene_path, proposal_path

    def test_unsupported_declared_attachment_hides_child_and_exact_cache_never_refits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); scene, proposal = self.fixture(root)
            original = U.sha(scene)
            result = M.build(scene, proposal, root / 'data/output', root=root)
            self.assertEqual(result['unresolved_joints'], 1)
            built = U.read(root / result['manifest']['path'])
            child = next(c for c in built['components'] if c['id'] == 'child')
            self.assertIsNone(child['poses'][0]['camera_from_native_mesh']); self.assertEqual(child['poses'][0]['status'], 'unknown')
            self.assertFalse(built['mechanics']['free_floating_child_fallback'])
            self.assertEqual(U.sha(scene), original)
            with patch.object(M, 'fit_revolute', side_effect=AssertionError('cache attempted fitting')):
                repeated = M.build(scene, proposal, root / 'data/output', root=root)
            self.assertTrue(repeated['reused_verified_cache'])
            self.assertEqual(repeated['manifest'], result['manifest'])
            (root / result['manifest']['path']).write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Cached mechanism output changed'):
                M.build(scene, proposal, root / 'data/output', root=root)

    def test_failed_revolute_fit_preserves_declared_topology_without_free_child_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); scene, proposal = self.fixture(root, 'revolute')
            with patch.object(M, 'fit_revolute', side_effect=ValueError('source evidence unavailable')):
                result = M.build(scene, proposal, root / 'data/output', root=root)
            built = U.read(root / result['manifest']['path'])
            pending = built['mechanics']['unresolved_children'][0]
            self.assertEqual(pending['diagnostic'], 'source evidence unavailable')
            self.assertEqual(pending['parent_id'], 'parent'); self.assertEqual(pending['child_id'], 'child')
            self.assertIsNone(next(c for c in built['components'] if c['id'] == 'child')['poses'][0]['camera_from_native_mesh'])

    def test_verified_cache_refuses_changed_source_or_code_instead_of_silent_refit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); scene, proposal = self.fixture(root)
            M.build(scene, proposal, root / 'data/output', root=root)
            (root / 'sam3d_mechanisms.py').write_text('DIFFERENT_IMPLEMENTATION')
            with self.assertRaisesRegex(ValueError, 'inputs or implementation changed'):
                M.build(scene, proposal, root / 'data/output', root=root)

    def test_proposals_cannot_name_another_source_or_program_and_original_files_stay_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); folder = root / 'data/input'; folder.mkdir(parents=True)
            asset = folder / 'asset.glb'; asset.write_bytes(b'INERT_SHA_CONTRACT_NOT_A_NATIVE_MESH')
            scene = {'format': M.S.FORMAT, 'accepted_motion_poses': 0, 'source': {'sha256': 'a' * 64},
                     'original_program': {'sha256': 'b' * 64}, 'components': [{'asset': M.S.ref(asset, root)}]}
            proposal = {'format': 'sam3d-mechanism-proposals', 'version': 1, 'source_sha256': 'a' * 64,
                        'interaction_sha256': 'b' * 64, 'joints': [], 'physical_accuracy_verified': False}
            scene_path, proposal_path = folder / 'scene.json', folder / 'proposals.json'
            U.write(scene_path, scene); U.write(proposal_path, proposal)
            before = U.sha(scene_path); M.scene_inputs(scene_path, proposal_path, root)
            self.assertEqual(U.sha(scene_path), before)
            for field in ['source_sha256', 'interaction_sha256']:
                changed = dict(proposal, **{field: 'c' * 64}); U.write(proposal_path, changed)
                with self.assertRaisesRegex(ValueError, 'Source/program'):
                    M.scene_inputs(scene_path, proposal_path, root)


if __name__ == '__main__':
    unittest.main()
