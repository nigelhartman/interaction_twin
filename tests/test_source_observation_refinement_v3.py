import copy
from unittest.mock import patch
import unittest

import source_observation_refinement_v2 as V2
import source_observation_refinement_v3 as R


class ExplicitAccountedSourceFollowup(unittest.TestCase):
    def test_new_payload_keeps_eight_source_images_and_larger_total_output_bound(self):
        with patch('meta_api.MetaClient.request', side_effect=AssertionError('offline only')):
            doc = R.build_plan()
            old, _, _ = V2.verified_plan()
        self.assertEqual(doc['context']['images'], old['context']['images'])
        self.assertEqual(doc['max_output_tokens'], 16000)
        self.assertEqual(doc['maximum_estimated_reservation_usd'], '1.378720')
        self.assertNotEqual(doc['wire_provenance']['payload_sha256'], old['wire_provenance']['payload_sha256'])
        self.assertEqual(doc['maximum_new_requests'], 1)
        self.assertFalse(doc['automatic_retries'])
        self.assertEqual(doc['explicit_accounted_followup']['request_id'], R.PREDECESSOR_REQUEST_ID)
        self.assertEqual(doc['explicit_accounted_followup']['reported_reasoning_tokens'], 6438)

    def test_unknown_charge_or_different_predecessor_never_admits_followup(self):
        original_read = R.read
        for field, value in [('status', 'uncertain'), ('request_id', 'other'), ('automatic_retries', True)]:
            def altered(path, field=field, value=value):
                doc = original_read(path)
                if str(path).endswith('/request/meta-request.json'):
                    doc = copy.deepcopy(doc)
                    doc[field] = value
                return doc
            with self.subTest(field=field), patch.object(R, 'read', side_effect=altered), self.assertRaises(ValueError):
                R.accounted_predecessor()


if __name__ == '__main__':
    unittest.main()
