import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

SCRIPTS = Path(__file__).parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('track_selection', SCRIPTS / 'stage3_select_track_views.py')
selection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selection)


class TrackViewSelectionTests(unittest.TestCase):
    def fixture(self, case):
        def write(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
        Image.fromarray(np.full((120, 80, 3), 140, np.uint8)).save(case / 'image.png')
        seg = case / 'segmentation' / 'run'
        seg.mkdir(parents=True)
        mask = np.zeros((120, 80), np.uint8)
        mask[20:60, 15:45] = 255
        Image.fromarray(mask).save(seg / 'mask.png')
        discovery = case / 'reasoning' / 'discovery' / 'discovery.json'
        write(discovery, {'provider': 'codex_cli', 'status': 'fixture'})
        write(case / 'manifest.json', {'source': {'sha256': 'video-hash', 'file': 'video.mp4'}})
        write(case / 'frames' / 'index.json', {'frames': [{'index': 1, 'timestamp_s': 2.25,
                                                        'width': 80, 'height': 120, 'file': 'image.png'}]})
        write(seg / 'metadata.json', {'status': 'complete', 'source_sha256': 'video-hash',
                                     'model': 'sam-fixture', 'device': 'cpu-fixture',
                                     'checkpoint_sha256': 'checkpoint-fixture',
                                     'seed_origin': 'codex_cli_hosted_OpenAI_model_proposal'})
        write(seg / 'annotations.json', {'seed_origin': 'codex_cli_hosted_OpenAI_model_proposal',
                                         'manual_seed_boxes_used': False,
                                         'discovery_record_sha256': selection.sha256(discovery),
                                         'tracks': [{'id': 'part', 'label': 'Part', 'identity_status': 'unverified',
                                                     'prompts': [{'sample_index': 1, 'box_xyxy': [15, 20, 45, 60]}]}]})
        write(seg / 'tracks.json', {'tracks': [{'track_id': 'part', 'frames': [{'sample_index': 1,
              'timestamp_s': 2.25, 'status': 'candidate_visible', 'raw_mask_file': 'mask.png'}]}]})
        write(seg / 'frames.json', {'frames': [{'sample_index': 1, 'tracks': [{'track_id': 'part',
                                                    'status': 'candidate_geometric_gate_only'}]}]})
        return seg

    def test_portrait_pixels_and_model_provenance_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp)
            self.fixture(case)
            candidates, _, _, automatic = selection.collect_track(case, 'run', 'part')
            self.assertTrue(automatic)
            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0]['metrics']['bbox_xyxy'], [15, 20, 45, 60])
            self.assertEqual(candidates[0]['timestamp_s'], 2.25)
            self.assertEqual(candidates[0]['mask_provenance']['prompt_origin'], 'model_generated_prompt')

    def test_automatic_seed_claim_requires_matching_discovery_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp)
            self.fixture(case)
            (case / 'reasoning' / 'discovery' / 'discovery.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'hash-matched'):
                selection.collect_track(case, 'run', 'part')

    def test_foreign_source_and_shifted_timestamp_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp)
            seg = self.fixture(case)
            meta = selection.read(seg / 'metadata.json')
            meta['source_sha256'] = 'wrong-video'
            (seg / 'metadata.json').write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, 'identical source'):
                selection.collect_track(case, 'run', 'part')
            meta['source_sha256'] = 'video-hash'
            (seg / 'metadata.json').write_text(json.dumps(meta))
            tracks = selection.read(seg / 'tracks.json')
            tracks['tracks'][0]['frames'][0]['timestamp_s'] += .1
            (seg / 'tracks.json').write_text(json.dumps(tracks))
            with self.assertRaisesRegex(ValueError, 'PTS mismatch'):
                selection.collect_track(case, 'run', 'part')

    def test_uncertain_audit_excluded_and_changed_audited_pixels_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp)
            seg = self.fixture(case)
            audit_dir = seg / 'audit'
            audit_dir.mkdir()
            audit = {'status': 'complete_candidate_visual_audit', 'source_sha256': 'video-hash',
                     'source_track_index_sha256': selection.sha256(seg / 'tracks.json'),
                     'result': {'observations': [{'sample_index': 1, 'track_id': 'part', 'verdict': 'uncertain'}]}}
            (audit_dir / 'audit.json').write_text(json.dumps(audit))
            evidence = [{'sample_index': 1, 'track_id': 'part', 'timestamp_s': 2.25,
                         'source_frame_sha256': selection.sha256(case / 'image.png'),
                         'raw_mask_sha256': selection.sha256(seg / 'mask.png')}]
            (audit_dir / 'evidence.json').write_text(json.dumps(evidence))
            self.assertEqual(selection.collect_track(case, 'run', 'part', 'audit')[0], [])
            audit['result']['observations'][0]['verdict'] = 'plausible_visible_mask'
            (audit_dir / 'audit.json').write_text(json.dumps(audit))
            self.assertEqual(len(selection.collect_track(case, 'run', 'part', 'audit')[0]), 1)
            Image.new('RGB', (80, 120), 'red').save(case / 'image.png')
            with self.assertRaisesRegex(ValueError, 'pixels or PTS changed'):
                selection.collect_track(case, 'run', 'part', 'audit')


if __name__ == '__main__':
    unittest.main()
