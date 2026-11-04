from copy import deepcopy
import unittest
from component_seed_review import derive


class SeedReviewTests(unittest.TestCase):
    def setUp(self):
        self.config = {'tracks': [{'id': 'part', 'prompts': [
            {'sample_index': 1, 'box_xyxy': [1, 2, 3, 4]},
            {'sample_index': 2, 'box_xyxy': [5, 6, 7, 8]}]}]}
        self.audit = {'status': 'complete_candidate_visual_audit', 'source_track_index_sha256': 'abc',
            'result': {'observations': [{'track_id': 'part', 'sample_index': 1, 'verdict': 'reject_wrong_or_lost_mask', 'reason': 'background'},
                                        {'track_id': 'part', 'sample_index': 2, 'verdict': 'plausible_visible_mask', 'reason': 'part'}]}}

    def test_omits_bad_seed_preserves_good_coordinates_and_original(self):
        before = deepcopy(self.config)
        result, trace = derive(self.config, self.audit, 'abc')
        self.assertEqual(result['tracks'][0]['prompts'], [before['tracks'][0]['prompts'][1]])
        self.assertEqual(self.config, before)
        self.assertEqual([x['operation'] for x in trace], ['omit_seed', 'retain_identical_model_seed'])

    def test_does_not_keep_unchecked_or_uncertain_seeds(self):
        self.audit['result']['observations'] = self.audit['result']['observations'][:1]
        result, _ = derive(self.config, self.audit, 'abc')
        self.assertEqual(result['tracks'], [])

    def test_wrong_source_and_duplicate_audit_fail(self):
        with self.assertRaises(ValueError):
            derive(self.config, self.audit, 'different')
        self.audit['result']['observations'].append(self.audit['result']['observations'][0])
        with self.assertRaises(ValueError):
            derive(self.config, self.audit, 'abc')
