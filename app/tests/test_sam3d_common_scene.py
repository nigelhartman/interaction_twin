"""Automatic estimated-scene math and cache boundaries; no inference or network."""
import copy
import itertools
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

import sam3d_common_scene as S
import upgrade_bundles as U


class SceneMathTests(unittest.TestCase):
    def test_background_similarity_recovers_matching_3d_identities_without_reflection(self):
        rng = np.random.default_rng(4); points = rng.normal(size=(40, 3))
        rotation = Rotation.from_euler('xyz', [.2, -.1, .3]).as_matrix(); translation = np.array([.1, -.4, 2.])
        scale, recovered, offset = S.fit_similarity(points, 1.3 * points @ rotation.T + translation)
        self.assertAlmostEqual(scale, 1.3, places=12)
        np.testing.assert_allclose(recovered, rotation, atol=1e-12); np.testing.assert_allclose(offset, translation, atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(recovered), 1.)
        with self.assertRaisesRegex(ValueError, 'collinear'):
            S.fit_similarity(np.zeros((3, 3)), np.zeros((3, 3)))

    def test_large_mask_keeps_full_convex_contour_instead_of_area_based_subsampling(self):
        mask = np.zeros((960, 540), np.uint8); cv2.rectangle(mask, (20, 40), (500, 900), 1, -1)
        shape = S.mask_shape(mask > 0)
        self.assertGreater(shape['area'], 300000); self.assertGreaterEqual(len(shape['contour']), 4)
        np.testing.assert_allclose(shape['extent'], [472, 844], atol=3)

    def test_mask_fit_retains_native_origin_and_uniform_scale_exactly_once(self):
        vertices = np.array(list(itertools.product([-.5, .5], repeat=3))) + [1., 2., 3.]
        native = vertices.copy(); rotation = Rotation.from_euler('xy', [12, -18], degrees=True).as_matrix()
        k = np.array([[300, 0, 160], [0, 300, 120], [0, 0, 1.]])
        mask = np.zeros((240, 320), np.uint8); cv2.ellipse(mask, (190, 130), (35, 31), 0, 0, 360, 1, -1)
        result = S.estimate_pose(vertices, rotation, .4, k, mask, 2., search=False)
        matrix = result['matrix']; singular = np.linalg.svd(matrix[:3, :3], compute_uv=False)
        np.testing.assert_allclose(singular, [.4] * 3, atol=1e-12); np.testing.assert_array_equal(vertices, native)
        camera = vertices @ matrix[:3, :3].T + matrix[:3, 3]; uvw = camera @ k.T
        np.testing.assert_allclose((uvw[:, :2] / uvw[:, 2:]).mean(0), [190, 130], atol=2)
        self.assertTrue(np.all(camera[:, 2] > 0))

    def pose(self, time, sample, status='estimated', reason='no_usable_source_mask', degrees=0):
        matrix = np.eye(4); matrix[:3, :3] = .2 * Rotation.from_euler('y', degrees, degrees=True).as_matrix(); matrix[0, 3] = time
        return {'timestamp_s': time, 'source_sample_index': sample, 'status': status, 'reason': reason,
                'camera_from_native_mesh': matrix.tolist() if status == 'estimated' else None}

    def test_missing_pose_interpolation_uses_both_anchors_without_extrapolation_or_acceptance(self):
        poses = [self.pose(0, 1), self.pose(.3, 2, 'unknown'), self.pose(.6, 3, degrees=30)]
        before = copy.deepcopy(poses); result = S.interpolate_short_missing(poses)
        self.assertEqual(result[1]['status'], 'estimated'); self.assertEqual(result[1]['interpolation_neighbors'], [1, 3])
        self.assertFalse(result[1]['accepted_motion_evidence']); self.assertEqual(poses, before)
        matrix = np.asarray(result[1]['camera_from_native_mesh']); self.assertAlmostEqual(matrix[0, 3], .3)
        np.testing.assert_allclose(np.linalg.svd(matrix[:3, :3], compute_uv=False), [.2] * 3)
        for rows in [[self.pose(0, 1, 'unknown'), self.pose(.3, 2)],
                     [self.pose(0, 1), self.pose(.6, 2, 'unknown'), self.pose(1.5, 3)],
                     [self.pose(0, 1), self.pose(.3, 2, 'unknown', 'abrupt_mask_area_change_possible_occlusion'), self.pose(.6, 3)]]:
            self.assertEqual(S.interpolate_short_missing(rows), rows)

    def test_sparse_background_descriptors_do_not_crash_knn_pair_unpacking(self):
        feature = mock.Mock(); feature.detectAndCompute.return_value = ([mock.Mock(pt=(1, 1))], np.zeros((1, 32), np.uint8))
        matcher = mock.Mock(); matcher.knnMatch.return_value = [[mock.Mock(distance=0)]]
        image = np.zeros((16, 16, 3), np.uint8); points = np.ones_like(image, dtype=float); allowed = np.ones((16, 16), bool)
        with mock.patch.object(S.cv2, 'ORB_create', return_value=feature), mock.patch.object(S.cv2, 'BFMatcher', return_value=matcher):
            result = S.background_scale(image, points, allowed, image, points, allowed)
        self.assertEqual(result['status'], 'unknown_background_gauge'); self.assertEqual(result['scale'], 1.)

    def test_withheld_or_geometrically_rejected_masks_are_distinct_from_interpolatable_absence(self):
        self.assertEqual(S.unavailable_mask_reason(None), 'no_usable_source_mask')
        for row, withheld in [({'sample_index': 2, 'zero_placeholder_for_missing_or_withheld': True}, []),
                              ({'sample_index': 2, 'status': 'withheld_due_to_occlusion_or_ambiguity'}, []),
                              ({'sample_index': 2, 'status': 'candidate_visible'}, [2])]:
            reason = S.unavailable_mask_reason(row, withheld)
            self.assertNotEqual(reason, 'no_usable_source_mask')
            poses = [self.pose(0, 1), self.pose(.3, 2, 'unknown', reason), self.pose(.6, 3)]
            self.assertEqual(S.interpolate_short_missing(poses), poses)


class SceneCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup); self.root = Path(temporary.name).resolve()
        for name in S.DEPENDENCIES:
            shutil.copyfile(S.ROOT / name, self.root / name)
        self.case = self.root / 'data/case'; self.case.mkdir(parents=True)
        self.program = self.case / 'interaction.json'; U.write(self.program, {'source': {'sha256': 'a' * 64, 'time_domain': U.TIME_DOMAIN}})
        self.model = self.root / 'data/model.json'; self.model.write_text('{"software_contract_fixture":true}\n')
        self.mask = self.case / 'mask.png'; self.mask.write_bytes(b'SOFTWARE CONTRACT PIXEL HASH FIXTURE')
        self.out = self.root / 'data/sam3d-scene-fixture'; self.out.mkdir()
        dependencies = [S.source_ref(self.root / n, self.root) for n in S.DEPENDENCIES]
        inputs = [S.ref(p, self.root) for p in [self.program, self.model, self.mask]]
        self.scene = {'format': S.FORMAT, 'version': 1, 'original_program': S.ref(self.program, self.root),
                      'model_metadata_paths': [self.model.relative_to(self.root).as_posix()], 'inputs': inputs,
                      'dependencies': dependencies, 'input_signature': U.digest(U.encoded({'inputs': inputs, 'dependencies': dependencies})),
                      'source': {'sha256': 'a' * 64}, 'physical_accuracy_verified': False, 'simulation_ready': False,
                      'components': [{'poses': [{'status': 'estimated'}, {'status': 'unknown'}]}]}
        U.write(self.out / 'scene.json', self.scene)
        U.write(self.out / 'build.json', {'format': 'sam3d-common-scene-build', 'manifest': S.ref(self.out / 'scene.json', self.root)})

    def build(self):
        return S.build(self.program, self.out, metadata_paths=[self.model], root=self.root)

    def test_cached_scene_checks_all_frozen_bytes_before_reuse_without_loading_or_fitting_models(self):
        with mock.patch.object(S, 'load_model', side_effect=AssertionError('No repeated native decode')):
            result = self.build()
        self.assertEqual(result['status'], 'reused_verified_estimated_scene'); self.assertEqual(result['estimated_pose_count'], 1)
        self.mask.write_bytes(b'changed mask')
        with self.assertRaisesRegex(ValueError, 'frozen source/model/mask'):
            self.build()

    def test_changed_output_or_builder_never_overwrites_previous_scene(self):
        original = (self.out / 'scene.json').read_bytes(); self.scene['components'][0]['poses'][0]['status'] = 'unknown'
        U.write(self.out / 'scene.json', self.scene)
        with self.assertRaisesRegex(ValueError, 'output changed'):
            self.build()
        (self.out / 'scene.json').write_bytes(original)
        (self.root / S.DEPENDENCIES[0]).write_text('changed implementation')
        with self.assertRaisesRegex(ValueError, 'implementation changed'):
            self.build()
        self.assertEqual((self.out / 'scene.json').read_bytes(), original)

    def test_local_registry_preserves_other_sources_and_refuses_changed_manifest(self):
        (self.root / 'showcase').mkdir()
        result = self.build(); published = S.publish_registry(result, root=self.root)
        self.assertEqual(published['scenes'], 1)
        self.assertEqual(S.publish_registry(result, root=self.root)['scenes'], 1)
        (self.out / 'scene.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'changed'):
            S.publish_registry(result, root=self.root)


class ActualSavedSceneTests(unittest.TestCase):
    def test_real_four_part_scene_preserves_native_hashes_and_keeps_every_pose_estimated_or_unknown(self):
        path = S.ROOT / 'data/sam3d-scene-jabra-20261102-v6/scene.json'
        if not path.exists():
            self.skipTest('Actual saved scene is optional outside the local capture workspace')
        scene = U.read(path)
        self.assertEqual({c['id'] for c in scene['components']}, {'case_base', 'case_lid', 'earbud_a', 'earbud_b'})
        self.assertEqual(scene['accepted_motion_poses'], 0); self.assertFalse(scene['manual_geometry_or_poses_used'])
        self.assertFalse(scene['common_frame']['camera_motion_removed']); self.assertFalse(scene['physical_accuracy_verified'])
        for component in scene['components']:
            self.assertEqual(S.ref(component['asset']['path'], S.ROOT), component['asset'])
            scale = component['native_uniform_scale']
            for pose in component['poses']:
                self.assertIn(pose['status'], ['estimated', 'unknown'])
                if pose['status'] == 'unknown':
                    self.assertIsNone(pose['camera_from_native_mesh'])
                else:
                    matrix = np.asarray(pose['camera_from_native_mesh'])
                    self.assertTrue(np.isfinite(matrix).all()); self.assertFalse(pose['accepted_motion_evidence'])
                    np.testing.assert_allclose(np.linalg.svd(matrix[:3, :3], compute_uv=False), [scale] * 3, atol=1e-9)


if __name__ == '__main__':
    unittest.main()
