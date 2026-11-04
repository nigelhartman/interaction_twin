import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import part_ledger as L


class PartLedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.case = self.root/'data/case'
        self.seg = self.case/'segmentation/flow-components-2'
        self.seg.mkdir(parents=True)
        self.ids = ['base', 'lid', 'piece_a', 'piece_b']
        self.put(self.seg/'tracks.json', {'tracks': [{'track_id': tid, 'frames': [
            {'sample_index': 1, 'timestamp_s': 0., 'status': 'candidate_visible'}]} for tid in self.ids]})
        self.put(self.seg/'frames.json', {'frames': []})
        self.program = {'source': {'sha256': 'a'*64, 'original_sha256': 'b'*64},
            'tracklets': [{'id': tid, 'label': tid} for tid in self.ids],
            'relationships': [{'id': 'lid_hinge', 'subject_track_id': 'lid', 'object_track_id': 'base'}],
            'assets': [], 'inputs': [self.inp('mask_tracks', self.seg/'tracks.json', 'actual_sam_masks'),
                self.inp('mask_quality', self.seg/'frames.json', 'geometric_mask_gates')]}
        for i, tid in enumerate(self.ids[:3]):
            folder = self.root/'data/models'/tid
            folder.mkdir(parents=True)
            for name, raw in [('asset.glb', b'original mesh '+tid.encode()), ('rgb.png', b'original source RGB'), ('mask.png', b'original SAM mask')]:
                (folder/name).write_bytes(raw)
            model = {'object_id': tid, 'backend': 'Meta SAM 3D Objects', 'sam3d_output': True,
                'learned_inference_executed': True, 'manual_geometry_fallback': False, 'manual_pose_used': False,
                'source': {'video_sha256': 'a'*64, 'timestamp_s': 0., 'sample_index': 1,
                    'image_path': 'rgb.png', 'image_sha256': L.sha(folder/'rgb.png'),
                    'mask_path': 'mask.png', 'mask_sha256': L.sha(folder/'mask.png')},
                'artifact_files': [{'file': 'asset.glb', 'sha256': L.sha(folder/'asset.glb'), 'bytes': (folder/'asset.glb').stat().st_size}]}
            self.put(folder/'metadata.json', model)
            self.program['inputs'].append(self.inp('model_metadata_'+str(i+1), folder/'metadata.json', 'learned_asset_metadata'))
            self.program['assets'].append({'id': 'asset_'+str(i+1), 'track_id': tid,
                'path': (folder/'asset.glb').relative_to(self.root).as_posix(), 'sha256': L.sha(folder/'asset.glb'), 'backend': 'Meta SAM 3D Objects'})
        self.program_path = self.case/'compiled/interaction.json'
        self.put(self.program_path, self.program)
        self.validator = patch('stage5.validate_interaction')
        self.validator.start()
        self.addCleanup(self.validator.stop)

    def put(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value)+'\n')

    def inp(self, identifier, path, kind):
        return {'id': identifier, 'path': path.relative_to(self.root).as_posix(), 'sha256': L.sha(path), 'kind': kind}

    def test_four_ids_remain_distinct_and_missing_geometry_stays_pending(self):
        result = L.build(self.program_path, root=self.root)
        value = L.read(self.root/result['path'])
        self.assertEqual(value['summary']['components'], 4)
        self.assertEqual(value['summary']['native_generated'], 3)
        self.assertEqual(value['parts'][3]['id'], 'piece_b')
        self.assertIsNone(value['parts'][3]['native'])
        self.assertEqual(value['parts'][3]['geometry_status'], 'pending')
        self.assertIn('lid_hinge', value['parts'][1]['relationship_ids'])
        self.assertEqual(len(L.metadata_paths(result['path'], program_path=self.program_path, root=self.root)), 3)

    def test_later_segmentation_uses_exact_compiled_attempt(self):
        result = L.document(self.program_path, root=self.root)
        self.assertIn('/flow-components-2/', result['segmentation']['tracks']['path'])
        self.put(self.seg/'frames.json', {'frames': ['changed']})
        with self.assertRaisesRegex(ValueError, 'changed'):
            L.document(self.program_path, root=self.root)

    def test_cross_attempt_quality_and_uncompiled_native_are_rejected(self):
        other = self.case/'segmentation/flow-components-1/frames.json'
        self.put(other, {'frames': []})
        bad = copy.deepcopy(self.program)
        bad['inputs'][1] = self.inp('mask_quality', other, 'geometric_mask_gates')
        self.put(self.program_path, bad)
        with self.assertRaisesRegex(ValueError, 'different attempts'):
            L.document(self.program_path, root=self.root)
        bad = copy.deepcopy(self.program); bad['assets'].pop()
        self.put(self.program_path, bad)
        with self.assertRaisesRegex(ValueError, 'exact compiled'):
            L.document(self.program_path, root=self.root)

    def test_changed_native_or_foreign_source_cannot_be_adopted(self):
        path = self.root/self.program['inputs'][2]['path']
        model = L.read(path); model['source']['video_sha256'] = 'c'*64
        self.put(path, model)
        bad = copy.deepcopy(self.program)
        bad['inputs'][2]['sha256'] = L.sha(path);self.put(self.program_path, bad)
        with self.assertRaisesRegex(ValueError, 'source-matched'):
            L.document(self.program_path, root=self.root)

    def test_immutable_ledger_verifies_content_and_rejects_claim_promotion(self):
        result = L.build(self.program_path, root=self.root)
        value = L.read(self.root/result['path'])
        value['simulation_ready'] = True
        with self.assertRaisesRegex(ValueError, 'content'):
            L.validate(value, root=self.root)
        self.assertEqual(result, L.build(self.program_path, root=self.root))
        (self.root/self.program['assets'][0]['path']).write_bytes(b'changed geometry')
        with self.assertRaisesRegex(ValueError, 'changed'):
            L.metadata_paths(result['path'], root=self.root)

    def test_another_program_or_selection_ids_are_rejected(self):
        result = L.build(self.program_path, root=self.root)
        other = self.case/'other/interaction.json';self.put(other, self.program)
        with self.assertRaisesRegex(ValueError, 'different program'):
            L.metadata_paths(result['path'], program_path=other, root=self.root)
        selection = self.case/'selection.json'
        self.put(selection, {'decisions': [{'track_id': 'another_part'}]})
        with self.assertRaisesRegex(ValueError, 'IDs disagree'):
            L.build(self.program_path, selection_receipt=selection, root=self.root)

    def test_symlink_cannot_replace_source_or_result(self):
        target = self.seg/'tracks.json';target.rename(self.seg/'original.json');target.symlink_to(self.seg/'original.json')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            L.document(self.program_path, root=self.root)


if __name__ == '__main__':
    unittest.main()
