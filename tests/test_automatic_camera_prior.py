"""Bounded offline camera adapter tests with fake weights and fake predictions.

No torch import, actual model call, checkpoint download or network operation.
"""
from contextlib import redirect_stdout
from pathlib import Path
import copy
import io
import json
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import automatic_camera_prior as P
import observed_geometry as G


NORMALIZED = np.array([[.8, 0, .5], [0, 1.2, .5], [0, 0, 1.]])


class CameraPriorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        images = self.root / 'data/images'; images.mkdir(parents=True)
        weights = self.root / 'models/moge-2-vits-normal'; weights.mkdir(parents=True)
        self.checkpoint = weights / 'model.pt'; self.checkpoint.write_bytes(b'FAKE TEST WEIGHTS; NEVER LOADED')
        self.provenance = weights / 'provenance.json'
        self.provenance.write_text(json.dumps({'checkpoint_sha256': P.sha(self.checkpoint), 'repo': 'fake-fixture'}))
        self.frames = []
        for i in range(6):
            image = np.random.default_rng(i).integers(0, 255, size=(40, 60, 3), dtype=np.uint8)
            path = images / f'{i}.png'; cv2.imwrite(str(path), image)
            self.frames.append({'id': f'frame_{i}', 'timestamp_s': float(i),
                'split': 'heldout' if i in [1, 3] else 'train', 'image_url': f'/files/data/images/{i}.png',
                'image_sha256': P.sha(path), 'image': image})
        self.output = self.root / 'data/result'; self.output.mkdir()
        self.provider = P.CameraPrior(self.root)

    @staticmethod
    def predictions(rows, directory, normalized=None):
        result = []
        for i, frame in enumerate(rows):
            h, w = frame['image'].shape[:2]
            P.CameraPrior._call_started(directory, frame, (w, h))
            prediction = {'frame_id': frame['id'], 'input_size': [w, h],
                          'normalized_K': (NORMALIZED if normalized is None else normalized[i]).tolist()}
            P.write_once(directory / f'prediction-{i+1}.json', prediction)
            result.append(prediction)
        return result

    def run_fake(self, fake=None, frames=None):
        if fake is None:
            fake = lambda provider, rows, directory, checkpoint: self.predictions(rows, directory)
        with patch.object(P.CameraPrior, '_predict', fake), redirect_stdout(io.StringIO()):
            return self.provider.estimate(self.frames if frames is None else frames, self.output)

    def report(self):
        return json.loads((self.output / 'camera-prior/camera-prior.json').read_text())

    def test_describe_is_a_deep_copy_and_declares_uncertainty(self):
        for description in [P.describe(), self.provider.describe()]:
            description['maximum_train_images'] = 999
            self.assertEqual(P.POLICY['maximum_train_images'], 3)
        self.assertEqual(self.provider.describe()['minimum_train_images'], 2)
        self.assertFalse(P.describe()['camera_calibrated'])
        self.assertFalse(P.describe()['physical_scale_known'])
        self.assertFalse(P.describe()['consistency_is_accuracy'])

    def test_selection_is_deterministic_train_only_and_never_result_selected(self):
        chosen = P.select_training(list(reversed(self.frames)))
        self.assertEqual([row['id'] for row in chosen], ['frame_0', 'frame_4', 'frame_5'])
        self.assertTrue(all(row['split'] == 'train' for row in chosen))
        for frames in [[], [self.frames[0]], [self.frames[1], self.frames[3]]]:
            with self.assertRaises(ValueError):
                P.select_training(frames)

    def test_duplicate_identity_or_source_and_invalid_pts_are_rejected(self):
        for key in ['id', 'image_url']:
            frames = [dict(self.frames[0]), dict(self.frames[2])]; frames[1][key] = frames[0][key]
            with self.assertRaises(ValueError):
                P.select_training(frames)
        for value in [float('nan'), float('inf'), -1., True, '1.0']:
            frames = [dict(self.frames[0]), dict(self.frames[2])]; frames[0]['timestamp_s'] = value
            with self.assertRaises(ValueError):
                P.select_training(frames)
        frames = [dict(self.frames[0]), dict(self.frames[2])]; frames[1]['timestamp_s'] = frames[0]['timestamp_s']
        with self.assertRaisesRegex(ValueError, 'presentation times'):
            P.select_training(frames)

    def test_native_mapping_uses_full_frame_half_pixel_convention(self):
        K = P.native_intrinsics(NORMALIZED, 600, 400)
        np.testing.assert_allclose(K, [[480., 0, 299.5], [0, 480., 199.5], [0, 0, 1.]])
        for width, height in [(0, 4), (3.5, 4), (True, 4), (4, None), (4, float('inf'))]:
            with self.assertRaises(ValueError):
                P.native_intrinsics(NORMALIZED, width, height)

    def test_mapper_rejects_shape_skew_bottom_row_focal_bounds_and_pp(self):
        invalid = [np.eye(2), np.full((3, 3), np.nan)]
        for coordinate, value in [((2,2), 1.00000001), ((0,1), .01), ((0,0), .001), ((1,1), 10.), ((0,2), 2.)]:
            matrix = NORMALIZED.copy(); matrix[coordinate] = value; invalid.append(matrix)
        for matrix in invalid:
            with self.assertRaises(ValueError):
                P.native_intrinsics(matrix, 600, 400)

    def test_combine_requires_two_to_three_valid_individual_cameras(self):
        K = P.native_intrinsics(NORMALIZED, 600, 400)
        for cameras in [[K], [K]*4, np.zeros((2, 2, 2)), np.zeros((3,3)), [K, np.full((3,3), np.inf)]]:
            with self.assertRaises(ValueError):
                P.combine(cameras, 600, 400)
        negative = K.copy(); negative[0,0] = -480.
        positive = K.copy(); positive[0,0] = 1440.
        # A positive median must not conceal an invalid individual camera.
        with self.assertRaises(ValueError):
            P.combine([negative, positive], 600, 400)
        median, metrics = P.combine([K, K, K], 600, 400)
        np.testing.assert_array_equal(median, K)
        self.assertEqual(metrics['relative_focal_spread'], 0)
        self.assertFalse(metrics['agreement_measures_physical_accuracy'])

    def test_cross_frame_disagreement_rejected_without_accuracy_claim(self):
        K = P.native_intrinsics(NORMALIZED, 600, 400)
        other = K.copy(); other[0,0] *= 1.4
        with self.assertRaises(ValueError):
            P.combine([K, other], 600, 400)
        other = K.copy(); other[0,2] += 20
        with self.assertRaises(ValueError):
            P.combine([K, other], 600, 400)

    def test_success_is_unvalidated_with_exact_source_and_call_receipts(self):
        K, metadata = self.run_fake()
        report = self.report()
        self.assertEqual(report['status'], 'unvalidated_learned_camera_assumption')
        self.assertEqual(report['actual_model_calls'], 3)
        self.assertEqual(report['completed_model_calls'], 3)
        self.assertEqual(report['new_api_calls'], 0)
        self.assertTrue(report['input_integrity']['unchanged'])
        self.assertFalse(report['camera_calibrated'])
        self.assertFalse(report['physical_accuracy_verified'])
        self.assertFalse(report['physical_scale_known'])
        self.assertEqual(metadata['input_views'], report['input_views'])
        self.assertTrue(all(row['split'] == 'train' for row in report['input_views']))
        np.testing.assert_array_equal(K, P.native_intrinsics(NORMALIZED, 60, 40))
        protocol = json.loads((self.output / 'camera-prior/protocol.json').read_text())
        self.assertTrue(protocol['frozen_before_inference'])
        self.assertEqual(protocol['implementation_sha256'], P.LOADED_SOURCE_SHA256)
        self.assertEqual(P.sha(self.output / 'camera-prior/implementation.py'), P.LOADED_SOURCE_SHA256)
        provenance_snapshot = self.output / 'camera-prior/model-provenance.json'
        self.assertEqual(provenance_snapshot.read_bytes(), self.provenance.read_bytes())
        self.assertEqual(P.sha(provenance_snapshot), protocol['model_provenance_sha256'])
        self.assertEqual(json.loads(provenance_snapshot.read_text())['checkpoint_sha256'], protocol['checkpoint_sha256'])
        self.assertNotIn(str(self.root), provenance_snapshot.read_text())

    def test_only_declared_provenance_bytes_are_snapshotted_before_predict(self):
        adjacent = self.provenance.with_name('unrelated-settings.json')
        adjacent.write_text('{"private_fixture_value":"do not copy this unrelated file"}')
        def fake(provider, rows, directory, checkpoint):
            protocol = json.loads((directory / 'protocol.json').read_text())
            self.assertEqual((directory / 'model-provenance.json').read_bytes(), self.provenance.read_bytes())
            self.assertEqual(P.sha(directory / 'model-provenance.json'), protocol['model_provenance_sha256'])
            self.assertFalse((directory / 'unrelated-settings.json').exists())
            return self.predictions(rows, directory)
        self.run_fake(fake)

    def test_two_train_frames_are_supported_without_repeating_one_image(self):
        _, _ = self.run_fake(frames=[self.frames[0], self.frames[1], self.frames[2]])
        self.assertEqual(self.report()['actual_model_calls'], 2)

    def test_source_file_hash_or_decoded_pixels_must_match_before_any_call(self):
        for key, value in [('image_sha256', '0'*64), ('image', np.zeros((40, 60, 3), dtype=np.uint8))]:
            with self.subTest(key=key):
                # Each attempt has its own fresh preserved output directory.
                self.output = self.root / f'data/precheck-{key}'; self.output.mkdir()
                frames = [dict(frame) for frame in self.frames]; frames[0][key] = value
                with self.assertRaises(ValueError):
                    self.run_fake(frames=frames)
                self.assertEqual(self.report()['actual_model_calls'], 0)
                self.assertIsNone(self.report()['K'])

    def test_mixed_native_resolution_and_wrong_image_dtype_rejected(self):
        frames = [dict(frame) for frame in self.frames]
        frames[0]['image'] = np.zeros((32, 60, 3), np.uint8)
        with self.assertRaises(ValueError):
            self.run_fake(frames=frames)
        self.assertEqual(self.report()['actual_model_calls'], 0)
        self.output = self.root / 'data/bad-dtype'; self.output.mkdir()
        frames[0]['image'] = np.zeros((40, 60, 3), float)
        with self.assertRaises(ValueError):
            self.run_fake(frames=frames)

    def test_partial_inference_failure_counts_attempted_and_completed_separately(self):
        def fake(provider, rows, directory, checkpoint):
            self.predictions(rows[:1], directory)
            provider._call_started(directory, rows[1], (60, 40))
            raise RuntimeError('Fake second invocation failed')
        with self.assertRaisesRegex(RuntimeError, 'Fake second'):
            self.run_fake(fake)
        report = self.report()
        self.assertEqual(report['status'], 'unsupported')
        self.assertEqual(report['actual_model_calls'], 2)
        self.assertEqual(report['completed_model_calls'], 1)
        self.assertTrue(report['input_integrity']['unchanged'])

    def test_predictions_without_start_receipts_cannot_claim_success(self):
        def fake(provider, rows, directory, checkpoint):
            return [{'frame_id': row['id'], 'input_size': [60,40], 'normalized_K': NORMALIZED.tolist()} for row in rows]
        with self.assertRaisesRegex(ValueError, 'receipts'):
            self.run_fake(fake)
        self.assertEqual(self.report()['actual_model_calls'], 0)
        self.assertIsNone(self.report()['K'])

    def test_changed_source_checkpoint_or_provenance_during_prediction_rejected(self):
        cases = [('source', self.root/'data/images/0.png'), ('checkpoint', self.checkpoint), ('provenance', self.provenance)]
        for name, path in cases:
            original = path.read_bytes()
            self.output = self.root / f'data/mutation-{name}'; self.output.mkdir()
            def fake(provider, rows, directory, checkpoint):
                result = self.predictions(rows, directory)
                path.write_bytes(original + b'changed')
                return result
            try:
                with self.assertRaises(ValueError):
                    self.run_fake(fake)
                report = self.report()
                self.assertEqual(report['status'], 'unsupported')
                self.assertIsNone(report['K'])
                self.assertEqual(report['actual_model_calls'], 3)
                self.assertFalse(report['input_integrity']['unchanged'])
            finally:
                path.write_bytes(original)

    def test_source_mutation_also_detected_when_prediction_raises(self):
        path = self.root / 'data/images/0.png'
        def fake(provider, rows, directory, checkpoint):
            provider._call_started(directory, rows[0], (60,40))
            path.write_bytes(b'changed during failed inference')
            raise RuntimeError('Fake model failed too')
        with self.assertRaisesRegex(ValueError, 'pinned source'):
            self.run_fake(fake)
        self.assertEqual(self.report()['actual_model_calls'], 1)
        self.assertFalse(self.report()['input_integrity']['unchanged'])

    def test_implementation_change_detected_without_modifying_real_source(self):
        original_sha = P.sha
        changed = [False]
        def altered_hash(path):
            if Path(path).resolve() == Path(P.__file__).resolve() and changed[0]:
                return '0' * 64
            return original_sha(path)
        def fake(provider, rows, directory, checkpoint):
            result = self.predictions(rows, directory); changed[0] = True
            return result
        with patch.object(P, 'sha', altered_hash), self.assertRaises(ValueError):
            self.run_fake(fake)
        self.assertIsNone(self.report()['K'])
        self.assertFalse(self.report()['input_integrity']['unchanged'])

    def test_inconsistent_predictions_preserve_completed_receipts_and_fail(self):
        def fake(provider, rows, directory, checkpoint):
            values = [NORMALIZED.copy() for row in rows]; values[-1][0,0] *= 1.4
            return self.predictions(rows, directory, values)
        with self.assertRaisesRegex(ValueError, 'disagree'):
            self.run_fake(fake)
        self.assertEqual(self.report()['actual_model_calls'], 3)
        self.assertEqual(self.report()['completed_model_calls'], 3)
        self.assertTrue(self.report()['input_integrity']['unchanged'])

    def test_prepared_checkpoint_corruption_never_reaches_predictor(self):
        self.checkpoint.write_bytes(b'bad weight bytes')
        with patch.object(P.CameraPrior, '_predict', side_effect=AssertionError('Must never call predictor')):
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'checkpoint changed'):
                self.provider.estimate(self.frames, self.output)
        self.assertEqual(self.report()['actual_model_calls'], 0)

    def test_source_symlink_and_parent_symlink_rejected(self):
        image = self.root / 'data/images/0.png'
        (self.root / 'data/symlink.png').symlink_to(image)
        (self.root / 'data/shortcut').symlink_to(image.parent, target_is_directory=True)
        for url in ['/files/data/symlink.png', '/files/data/shortcut/0.png', '/files/data/../models/a', '/files/data/.hidden.png', '/files/data/a?b.png']:
            with self.assertRaises(ValueError):
                P.source_file(self.root, url)

    def test_checkpoint_symlink_or_ancestor_symlink_rejected(self):
        original = self.checkpoint.read_bytes(); real = self.root / 'real.pt'; real.write_bytes(original)
        self.checkpoint.unlink(); self.checkpoint.symlink_to(real)
        with self.assertRaises(ValueError):
            self.run_fake()
        self.assertEqual(self.report()['actual_model_calls'], 0)
        self.checkpoint.unlink(); self.checkpoint.write_bytes(original)
        self.output = self.root / 'data/second-check'; self.output.mkdir()
        model_dir = self.checkpoint.parent
        actual = self.root / 'models/real-moge'; model_dir.rename(actual)
        model_dir.symlink_to(actual, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.run_fake()
        self.assertEqual(self.report()['actual_model_calls'], 0)

    def test_output_symlink_or_existing_attempt_cannot_overwrite_files(self):
        self.run_fake()
        before = (self.output / 'camera-prior/camera-prior.json').read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_fake()
        self.assertEqual((self.output / 'camera-prior/camera-prior.json').read_bytes(), before)
        alias = self.root / 'data/alias'; alias.symlink_to(self.output, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.provider.estimate(self.frames, alias)

    def test_call_bound_rejects_a_fourth_started_invocation(self):
        directory = self.output / 'calls'; directory.mkdir()
        with redirect_stdout(io.StringIO()):
            for frame in P.select_training(self.frames):
                P.CameraPrior._call_started(directory, frame, (60,40))
            with self.assertRaisesRegex(ValueError, 'bound'):
                P.CameraPrior._call_started(directory, self.frames[4], (60,40))

    def test_reserved_frame_cannot_create_an_inference_start_receipt(self):
        directory = self.output / 'reserved-call'; directory.mkdir()
        with self.assertRaisesRegex(ValueError, 'TRAIN'):
            P.CameraPrior._call_started(directory, self.frames[1], (60,40))
        self.assertEqual(list(directory.iterdir()), [])

    def test_readonly_training_images_work_without_mutation(self):
        for frame in self.frames:
            frame['image'].setflags(write=False)
        self.run_fake()
        self.assertTrue(self.report()['input_integrity']['unchanged'])

    def test_real_adapter_contract_is_accepted_by_geometry_engine_with_fake_predictions(self):
        fake = lambda provider, rows, directory, checkpoint: self.predictions(rows, directory)
        with patch.object(P.CameraPrior, '_predict', fake), redirect_stdout(io.StringIO()):
            K, assumption = G.learned_camera_assumption(self.provider, self.frames, self.output, self.root,
                P.describe(), {'path': 'data/fixture.mp4', 'sha256': '0'*64})
        np.testing.assert_array_equal(K, P.native_intrinsics(NORMALIZED, 60, 40))
        self.assertTrue(assumption['geometry_reconstructed_with_this_assumption'])
        self.assertFalse(assumption['camera_calibrated'])
        self.assertEqual(assumption['report']['sha256'], P.sha(self.output / 'camera-prior/camera-prior.json'))
        self.assertTrue((self.output / 'camera-assumption.json').is_file())

    def test_prediction_receipt_mismatch_is_preserved_and_rejected(self):
        def fake(provider, rows, directory, checkpoint):
            result = self.predictions(rows, directory)
            result[0]['normalized_K'][0][0] += .01
            return result
        with self.assertRaisesRegex(ValueError, 'receipt differs'):
            self.run_fake(fake)
        self.assertEqual(self.report()['actual_model_calls'], 3)
        self.assertEqual(self.report()['completed_model_calls'], 3)
        self.assertIsNone(self.report()['K'])

    def test_changed_output_ancestor_never_writes_into_symlink_target(self):
        outside = self.root / 'outside-output'; outside.mkdir()
        moved = self.root / 'data/moved-attempt'
        def fake(provider, rows, directory, checkpoint):
            result = self.predictions(rows, directory)
            self.output.rename(moved)
            self.output.symlink_to(outside, target_is_directory=True)
            return result
        with self.assertRaisesRegex(ValueError, 'output ancestry changed'):
            self.run_fake(fake)
        self.assertEqual(list(outside.iterdir()), [])
        self.assertEqual(len(list((moved / 'camera-prior').glob('call-*-start.json'))), 3)


if __name__ == '__main__':
    unittest.main()
