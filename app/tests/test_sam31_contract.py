"""SAM 3.1 adapter contracts using explicit fixtures, never model inference."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from scripts import sam31_segment as adapter


class Sam31ContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.rgb = np.arange(20 * 12 * 3, dtype=np.uint8).reshape(20, 12, 3)
        self.mask = np.zeros((20, 12), np.uint8)
        self.mask[4:14, 2:9] = 255
        Image.fromarray(self.rgb).save(self.root / 'image.png')
        Image.fromarray(self.mask).save(self.root / 'mask.png')
        self.manifest = {'schema_version': '0.1', 'object_id': 'component_a',
            'image': 'image.png', 'image_sha256': adapter.sha256(self.root / 'image.png'),
            'mask': 'mask.png', 'mask_sha256': adapter.sha256(self.root / 'mask.png'),
            'source': {'video_sha256': 'a' * 64, 'sample_index': 14, 'timestamp_s': 13.001144,
                       'time_domain': adapter.TIME_DOMAIN},
            'mask_provenance': {'prompt_origin': 'model_generated_prompt',
                                'manual_seed_boxes_used': False, 'provider': 'synthetic contract fixture'}}
        self.path = self.root / 'input.json'
        self.save()

    def tearDown(self):
        self.temporary.cleanup()

    def save(self):
        self.path.write_text(json.dumps(self.manifest))

    def prediction(self):
        return {'out_obj_ids': np.array([7], dtype=np.int64), 'out_probs': np.array([.8]),
                'out_boxes_xywh': np.array([[2/12, 4/20, 7/12, 10/20]]),
                'out_binary_masks': (self.mask > 0)[None]}

    def test_pts_rgb_and_mask_provenance_are_preserved_without_mask_prompt(self):
        record, rgb, mask = adapter.read_input(self.path)
        np.testing.assert_array_equal(rgb, self.rgb)
        np.testing.assert_array_equal(mask, self.mask > 0)
        self.assertEqual((record['width'], record['height']), (12, 20))
        self.assertEqual(record['manifest']['source']['timestamp_s'], 13.001144)
        self.assertFalse(record['reference_mask_passed_to_model'])
        self.assertFalse(record['manifest']['mask_provenance']['manual_seed_boxes_used'])

    def test_reference_mask_is_optional_for_real_text_only_segmentation(self):
        for key in ['mask', 'mask_sha256', 'mask_provenance']:
            del self.manifest[key]
        self.save()
        record, _, mask = adapter.read_input(self.path)
        self.assertIsNone(mask)
        self.assertNotIn('reference_mask', record['files'])

    def test_bad_pts_hash_soft_and_resized_masks_fail_before_inference(self):
        for bad in [True, -1, float('nan'), '13.0']:
            self.manifest['source']['timestamp_s'] = bad
            self.save()
            with self.assertRaisesRegex(ValueError, 'PTS'):
                adapter.read_input(self.path)
        self.manifest['source']['timestamp_s'] = 13.
        self.manifest['image_sha256'] = 'b' * 64
        self.save()
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            adapter.read_input(self.path)
        self.manifest['image_sha256'] = adapter.sha256(self.root / 'image.png')
        for bad in [np.ones((20, 11), np.uint8), np.full((20, 12), 127, np.uint8)]:
            Image.fromarray(bad).save(self.root / 'mask.png')
            self.manifest['mask_sha256'] = adapter.sha256(self.root / 'mask.png')
            self.save()
            with self.assertRaisesRegex(ValueError, 'full-frame'):
                adapter.read_input(self.path)

    def test_exif_rotation_is_rejected_instead_of_silently_remapping_pixels(self):
        exif = Image.Exif()
        exif[274] = 6
        Image.fromarray(self.rgb).save(self.root / 'rotated.jpg', exif=exif)
        self.manifest.update(image='rotated.jpg', image_sha256=adapter.sha256(self.root / 'rotated.jpg'))
        self.save()
        with self.assertRaisesRegex(ValueError, 'Normalize image orientation'):
            adapter.read_input(self.path)

    def test_local_checkpoint_requires_exact_version_revision_and_hash(self):
        checkpoint = self.root / 'sam3.1_multiplex.pt'
        checkpoint.write_bytes(b'Explicit synthetic inventory fixture, not executable or real weights')
        manifest = {'model_repository': 'facebook/sam3.1', 'model_revision': 'a' * 40,
                    'checkpoint_file': checkpoint.name, 'checkpoint_sha256': adapter.sha256(checkpoint)}
        path = self.root / 'checkpoint.json'
        path.write_text(json.dumps(manifest))
        record, errors = adapter.inspect_checkpoint(path)
        self.assertEqual(errors, [])
        self.assertEqual(record['sha256'], manifest['checkpoint_sha256'])
        manifest['model_repository'] = 'facebook/sam3'
        manifest['checkpoint_sha256'] = 'b' * 64
        path.write_text(json.dumps(manifest))
        _, errors = adapter.inspect_checkpoint(path)
        self.assertTrue(any('not SAM 3' in text for text in errors))
        self.assertTrue(any('SHA256' in text for text in errors))

    def test_checkpoint_partial_wrong_shape_or_empty_model_never_passes(self):
        expected = {'detector.weight': np.ones((2, 3)), 'tracker.bias': np.ones(2)}
        coverage = adapter.verify_checkpoint_coverage(expected, {'model': expected})
        self.assertTrue(coverage['strict_key_and_shape_coverage'])
        for invalid in [{}, {'detector.weight': np.ones((2, 3))},
                        {**expected, 'random.extra': np.ones(1)},
                        {**expected, 'tracker.bias': np.ones(3)}]:
            with self.assertRaisesRegex(RuntimeError, 'no inference or fallback'):
                adapter.verify_checkpoint_coverage(expected, invalid)

    def test_actual_output_boundary_preserves_masks_instances_and_source_pixels(self):
        out = self.root / 'output'
        result = adapter.export_prediction(self.prediction(), self.rgb, out, self.mask > 0)
        self.assertEqual(result['instance_count'], 1)
        self.assertEqual(result['instances'][0]['model_instance_id'], 7)
        self.assertEqual(result['instances'][0]['iou_with_earlier_reference_mask_not_accuracy'], 1.)
        self.assertFalse(result['handcrafted_mask_fallback'])
        np.testing.assert_array_equal(np.asarray(Image.open(out / 'instance-000001-mask.png')), self.mask)
        rgba = np.asarray(Image.open(out / 'instance-000001-rgba.png'))
        np.testing.assert_array_equal(rgba[:, :, :3], self.rgb)
        np.testing.assert_array_equal(rgba[:, :, 3], self.mask)

    def test_empty_model_detection_is_legitimate_and_missing_or_soft_masks_are_not(self):
        empty = {'out_obj_ids': np.zeros(0, np.int64), 'out_probs': np.zeros(0),
                 'out_boxes_xywh': np.zeros((0, 4)), 'out_binary_masks': np.zeros((0, 20, 12), bool)}
        record = adapter.export_prediction(empty, self.rgb, self.root / 'empty')
        self.assertEqual(record['instance_count'], 0)
        self.assertEqual(list((self.root / 'empty').glob('instance-*')), [])
        for invalid in [{}, {**self.prediction(), 'out_binary_masks': np.ones((1, 20, 12), np.float32)},
                        {**self.prediction(), 'out_binary_masks': np.ones((1, 12, 20), bool)}]:
            with self.assertRaises(RuntimeError):
                adapter.export_prediction(invalid, self.rgb, self.root / 'invalid')
        self.assertFalse((self.root / 'invalid').exists())

    def test_missing_runtime_stays_preflight_without_results_or_upstream_imports(self):
        args = argparse.Namespace(input_manifest=self.path, repo=self.root / 'absent-repo',
            expected_code_commit=adapter.PINNED_CODE_COMMIT, checkpoint_manifest=None,
            prompt='earbud', prompt_origin='synthetic model-proposed fixture')
        with mock.patch.object(adapter, 'inspect_runtime', return_value=({'cuda_available': False}, ['CUDA missing'])):
            report, rgb, _ = adapter.preflight(args)
        self.assertEqual(report['status'], 'prerequisites_missing')
        self.assertFalse(report['learned_inference_executed'])
        self.assertFalse(report['network_or_credential_access_attempted'])
        self.assertNotIn('outputs', report)
        self.assertIsNotNone(rgb)


if __name__ == '__main__':
    unittest.main()
