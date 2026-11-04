import importlib.util
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('occlusion_evidence', ROOT / 'scripts/stage4_occlusion_evidence.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class OcclusionEvidenceTests(unittest.TestCase):
    def test_only_removes_pixels_and_preserves_inputs(self):
        raw = np.ones((5, 5), dtype=bool)
        hand = np.zeros((5, 5), dtype=bool)
        hand[0] = True
        other = np.zeros((5, 5), dtype=bool)
        other[:, 0] = True
        visible, stats = mod.derive_visible_support(raw, [hand], [other])
        self.assertEqual(int(raw.sum()), 25)
        self.assertEqual(int(visible.sum()), 16)
        self.assertEqual(stats['new_pixels_added'], 0)
        self.assertEqual(stats['removed_overlap_pixels'], 9)

    def test_coordinate_mismatch_is_not_resized_silently(self):
        with self.assertRaisesRegex(ValueError, 'shape mismatch'):
            mod.derive_visible_support(np.ones((5, 5)), [np.ones((6, 5))], [])

    def test_occluder_seed_must_be_supplied_at_exact_pts(self):
        rows = [{'index': i, 'timestamp_s': i * .2, 'width': 540, 'height': 960} for i in range(1, 4)]
        result = {'tracks': [{'id': 'foreground', 'first_sample': 1, 'last_sample': 3,
                    'seeds': [{'sample_index': 2, 'timestamp_s': .4, 'box_xyxy_1000': [0, 0, 100, 100]}]}], 'actions': []}
        mod.validate_occluder_proposal(result, rows, rows)
        result['tracks'][0]['seeds'][0]['timestamp_s'] = .5
        with self.assertRaisesRegex(ValueError, 'timestamp mismatch'):
            mod.validate_occluder_proposal(result, rows, rows)

    def test_event_cannot_invent_intermediate_or_out_of_window_evidence(self):
        rows = [{'index': 1}, {'index': 3}]
        event = {'id': 'event', 'first_sample': 1, 'last_sample': 3, 'evidence_samples': [1, 2]}
        with self.assertRaisesRegex(ValueError, 'outside supplied'):
            mod.validate_events({'events': [event]}, rows)

    def test_audit_cannot_omit_or_duplicate_pairs(self):
        evidence = [{'track_id': 'a', 'sample_index': 1}, {'track_id': 'a', 'sample_index': 2}]
        with self.assertRaisesRegex(ValueError, 'omitted or duplicated'):
            mod.validate_pairs({'observations': [evidence[0], evidence[0]]}, evidence)

    def test_prompt_edit_breaks_model_only_lineage(self):
        rows = [{'index': 1, 'timestamp_s': 0., 'width': 540, 'height': 960}]
        record = {'result': {'tracks': [{'id': 'foreground', 'label': 'foreground', 'description': 'test',
            'first_sample': 1, 'last_sample': 1, 'uncertainties': [],
            'seeds': [{'sample_index': 1, 'box_xyxy_1000': [0, 0, 100, 100], 'reason': 'model'}]}]}}
        cfg = mod.tracking_config(record['result'], rows, 'test', 'hash')
        cfg['scope'] = 'automatic_foreground_occluders_only'
        mod.verify_model_config(cfg, record, rows, 'test', 'hash')
        cfg['tracks'][0]['prompts'][0]['box_xyxy'][0] = 3
        with self.assertRaisesRegex(ValueError, 'differs from exact saved model'):
            mod.verify_model_config(cfg, record, rows, 'test', 'hash')


if __name__ == '__main__':
    unittest.main()
