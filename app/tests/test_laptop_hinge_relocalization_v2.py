import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import laptop_hinge_relocalization_v2 as L


class RGBMaskHingeRelocalization(unittest.TestCase):
    def test_actual_frozen_packet_is_source_only_three_rgb_six_exact_masks(self):
        with patch('meta_api.MetaClient.request', side_effect=AssertionError('No inference during packet verification')):
            doc, extras, folder = L.verified_plan()
        images = doc['context']['images']
        self.assertEqual(len(images), 9)
        self.assertEqual([i['evidence_id'] for i in images if i['role'] == 'source_rgb'],
                         ['source_000027', 'source_000030', 'source_000044'])
        self.assertEqual(len([i for i in images if i['role'] == 'sam_binary_mask']), 6)
        self.assertEqual({i['component_id'] for i in images if i['role'] == 'sam_binary_mask'}, {'laptop_base', 'laptop_lid'})
        self.assertTrue(doc['context']['source_only'])
        self.assertFalse(doc['context']['manual_coordinates_supplied'])
        self.assertEqual(doc['maximum_new_requests'], 1)
        self.assertLess(doc['wire_provenance']['payload_bytes'], 32 * 1024 * 1024)
        self.assertEqual(doc['contract']['max_output_tokens'], 6000)

    def test_uncertain_transport_is_once_only_and_never_resubmitted(self):
        client = Mock()
        client.json_result.side_effect = TimeoutError('fake uncertain transport')
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve()
            folder = root / 'data/relocalization'
            folder.mkdir(parents=True)
            (folder / 'plan.json').write_text('{}')
            doc = {'context': {'images': []}}
            extras = {'prompt': 'fixture', 'provider_schema': {}}
            authorization = L.M.sha(folder / 'plan.json')
            with patch.object(L, 'verified_plan', return_value=(doc, extras, folder)):
                with self.assertRaises(TimeoutError):
                    L.run(root=root, authorized_plan_sha256=authorization, allow_paid_api=True, client=client)
                failure = json.loads((folder / 'attempt/failure.json').read_text())
                self.assertEqual(failure['status'], 'preserved_no_automatic_retry')
                with self.assertRaisesRegex(ValueError, 'Once-only'):
                    L.run(root=root, authorized_plan_sha256=authorization, allow_paid_api=True, client=client)
        self.assertEqual(client.json_result.call_count, 1)


if __name__ == '__main__':
    unittest.main()
