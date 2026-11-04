import copy
import importlib.util
import unittest
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location('discovery', Path(__file__).resolve().parents[1]/'scripts/stage2_discover.py')
D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D)


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{'index': i, 'timestamp_s': t, 'width': 480, 'height': 854} for i, t in [(1, .032), (2, 1.033), (3, 2.034)]]
        self.result = {'summary': 'Generic discovery', 'tracks': [
            {'id': 'rigid_part', 'label': 'Part candidate', 'description': 'Visible part', 'first_sample': 1, 'last_sample': 3,
             'seeds': [{'sample_index': 2, 'timestamp_s': 1.033, 'box_xyxy_1000': [100, 200, 500, 600], 'reason': 'Clear'}],
             'uncertainties': ['Occlusion']}],
            'actions': [{'description': 'Movement', 'participant_track_ids': ['rigid_part'], 'evidence_samples': [2, 3], 'uncertainties': ['Not contact proof']}],
            'uncertainties': []}

    def test_valid_proposals_become_explicit_model_prompts_and_preserve_pts(self):
        self.assertTrue(D.validate_discovery(self.result, self.rows))
        cfg = D.tracking_config(self.result, self.rows, 'test-case', 'abc')
        seed = cfg['tracks'][0]['prompts'][0]
        self.assertEqual(seed['box_xyxy'], [48, 170.8, 240, 512.4])
        self.assertEqual(seed['source_timestamp_s'], 1.033)
        self.assertFalse(cfg['manual_seed_boxes_used'])
        self.assertIn('hosted_OpenAI', cfg['seed_origin'])

    def test_invented_samples_times_boxes_ids_are_rejected(self):
        for kind in ['sample', 'timestamp', 'nan', 'bounds', 'collapsed', 'duplicate', 'range', 'action_sample', 'action_track']:
            value = copy.deepcopy(self.result)
            seed = value['tracks'][0]['seeds'][0]
            if kind == 'sample': seed['sample_index'] = 99
            if kind == 'timestamp': seed['timestamp_s'] = 1.2
            if kind == 'nan': seed['box_xyxy_1000'][0] = float('nan')
            if kind == 'bounds': seed['box_xyxy_1000'][0] = -1
            if kind == 'collapsed': seed['box_xyxy_1000'][2] = 100
            if kind == 'duplicate': value['tracks'].append(copy.deepcopy(value['tracks'][0]))
            if kind == 'range': value['tracks'][0]['first_sample'] = 3
            if kind == 'action_sample': value['actions'][0]['evidence_samples'] = [0]
            if kind == 'action_track': value['actions'][0]['participant_track_ids'] = ['imagined']
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                D.validate_discovery(value, self.rows)

    def test_index_without_source_continuity_is_rejected(self):
        for kind in ['index', 'time']:
            rows = copy.deepcopy(self.rows)
            if kind == 'index': rows[1]['index'] = 17
            else: rows[1]['timestamp_s'] = .001
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                D.validate_discovery(self.result, rows)

    def test_gate_withholds_empty_and_large_drift_without_claiming_accuracy(self):
        self.assertIn('empty_prediction', D.mask_gate(np.zeros((10, 10), bool), 10, 10, 10))
        full = np.ones((10, 10), bool)
        self.assertIn('covers_most_of_frame', D.mask_gate(full, 10, 10, 10))
        self.assertIn('area_above_four_times_seed', D.mask_gate(full, 10, 10, 10))
        modest = np.zeros((10, 10), bool)
        modest[2:6, 2:6] = True
        self.assertEqual(D.mask_gate(modest, 16, 10, 10), [])

    def test_generic_prompt_has_no_case_category_hints(self):
        prompt = D.prompt_text(self.rows).lower()
        for category in ['battery', 'camera', 'coin', 'pocket', 'earbud', 'jabra']:
            self.assertNotIn(category, prompt)
        self.assertIn('do not call any tools', prompt)
        self.assertIn('0..1000', prompt)

    def test_mask_audit_must_cover_only_and_all_actual_observations(self):
        evidence = [{'track_id': 'part', 'sample_index': 7}, {'track_id': 'part', 'sample_index': 9}]
        value = {'summary': 'Candidate audit', 'observations': [
            {'track_id': 'part', 'sample_index': i, 'verdict': 'uncertain', 'reason': 'Blurred'} for i in [7, 9]], 'limitations': []}
        self.assertTrue(D.validate_audit(value, evidence))
        for kind in ['invented', 'duplicate', 'omitted']:
            changed = copy.deepcopy(value)
            if kind == 'invented': changed['observations'][0]['sample_index'] = 8
            if kind == 'duplicate': changed['observations'].append(copy.deepcopy(changed['observations'][0]))
            if kind == 'omitted': changed['observations'].pop()
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                D.validate_audit(changed, evidence)


if __name__ == '__main__':
    unittest.main()
