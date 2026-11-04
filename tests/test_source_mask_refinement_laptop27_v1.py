from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import meta_sam_api as S
import source_mask_refinement_laptop27_v1 as M


class LaptopOpeningMaskRefresh(unittest.TestCase):
    def test_actual_adapter_accepts_model_origin_and_exact_two_boxes_offline(self):
        with tempfile.TemporaryDirectory(prefix='test-source27-', dir=M.ROOT / 'data') as name:
            output = (Path(name) / 'prepared').relative_to(M.ROOT).as_posix()
            with patch('meta_api.MetaClient.request', side_effect=AssertionError('No network in offline preflight')):
                M.prepare(output=output)
                plan, _ = M.verified_plan(output=output)
            self.assertEqual(plan['adapter_preflight']['status'], 'real_adapter_payload_captured_without_network')
            self.assertEqual(plan['adapter_preflight']['new_model_requests'], 0)
            self.assertEqual(plan['maximum_new_requests'], 1)
            self.assertEqual(plan['maximum_estimated_reservation_usd'], '0.002500')
            self.assertEqual(plan['source_review']['source']['sample_index'], 27)
            self.assertEqual(plan['source_review']['source']['timestamp_s'], 18.070422)
            self.assertEqual([o['id'] for o in plan['source_review']['objects']], [0, 1])
            self.assertEqual([r['track_id'] for r in plan['object_registry']], ['laptop_base', 'laptop_lid'])
            self.assertNotIn('manual', M.ORIGIN)
            self.assertIn('model', M.ORIGIN)
            with self.assertRaises(S.SamAPIError):
                S._model_box_prompt(plan['source_review']['objects'], width=722, height=960, frame_count=1,
                                    prompt_origin='model_manual_origin_fixture', media='image')

    def test_uncertain_once_only_attempt_cannot_implicitly_repeat(self):
        client = Mock()
        client.sam_callback.return_value.side_effect = TimeoutError('synthetic fixture only')
        with tempfile.TemporaryDirectory(prefix='test-source27-', dir=M.ROOT / 'data') as name:
            output = (Path(name) / 'prepared').relative_to(M.ROOT).as_posix()
            prepared = M.prepare(output=output)
            with self.assertRaises(TimeoutError):
                M.run(output=output, authorized_plan_sha256=prepared['sha256'], allow_paid_api=True, client=client)
            folder = M.ROOT / output
            self.assertTrue((folder / 'attempt/failure.json').exists())
            with self.assertRaisesRegex(ValueError, 'One mask-refresh attempt'):
                M.run(output=output, authorized_plan_sha256=prepared['sha256'], allow_paid_api=True, client=client)
        client.sam_callback.assert_called_once()
        client.sam_callback.return_value.assert_called_once()


if __name__ == '__main__':
    unittest.main()
