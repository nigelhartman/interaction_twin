"""Offline library boundaries: no models, network, inputs or paid requests."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('library_builder', ROOT / 'scripts/build_robotics_library.py')
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)


class LibraryBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.model = self.root / 'data/model'
        self.model.mkdir(parents=True)
        for name, data in [('asset.glb', b'preserved test bytes'), ('source-rgb.png', b'rgb'), ('source-mask.png', b'mask')]:
            (self.model / name).write_bytes(data)
        self.asset = {'id': 'asset1', 'path': 'data/model/asset.glb', 'sha256': B.digest(self.model / 'asset.glb'),
                      'track_id': 'part', 'backend': 'TripoSR'}
        self.doc = {'source': {'sha256': 'a' * 64, 'duration_s': 4}}
        self.metadata = {'learned_inference_executed': True, 'geometry_reconstructed_by_model': True,
                         'manual_geometry_fallback': False, 'manual_pose_used': False, 'metric_scale_verified': False,
                         'object_id': 'part', 'backend': 'TripoSR',
                         'source': {'video_sha256': 'a' * 64, 'image_path': 'source-rgb.png', 'mask_path': 'source-mask.png',
                                    'image_sha256': B.digest(self.model / 'source-rgb.png'),
                                    'mask_sha256': B.digest(self.model / 'source-mask.png'),
                                    'sample_index': 2, 'timestamp_s': 1.0, 'time_domain': 'source_video_presentation_seconds',
                                    'mask_provenance': {'manual_seed_boxes_used': False, 'prompt_origin': 'model_generated_prompt',
                                                        'model': 'sam-3.1'}},
                         'artifact_files': [{'file': 'asset.glb', 'sha256': self.asset['sha256']}]}

    def write_metadata(self):
        (self.model / 'metadata.json').write_text(json.dumps(self.metadata))

    def test_genuine_record_keeps_exact_native_bytes_and_source_anchors(self):
        self.write_metadata()
        result = B.genuine_asset(self.root, self.asset, self.doc)
        self.assertEqual(result['sha256'], self.asset['sha256'])
        self.assertEqual(result['anchor']['timestamp_s'], 1.0)
        self.assertFalse(result['physical_accuracy_verified'])
        self.assertFalse(result['source_video_pose_established_by_this_library'])

    def test_atomic_index_failure_preserves_previous_complete_bytes(self):
        target = self.root / 'library-index.json'
        target.write_bytes(b'{"old":"complete"}')
        with patch.object(B.os, 'replace', side_effect=OSError('fixture interrupted publication')):
            with self.assertRaises(OSError): B.atomic_write(target, b'{"new":"complete"}')
        self.assertEqual(target.read_bytes(), b'{"old":"complete"}')
        self.assertEqual(list(self.root.glob('.library-index.json-*.tmp')), [])

    def test_assisted_masks_fixture_or_manual_geometry_are_excluded(self):
        for key in ('manual_geometry_fallback', 'manual_pose_used', 'fixture'):
            with self.subTest(key=key):
                old = copy.deepcopy(self.metadata)
                self.metadata[key] = True
                self.write_metadata()
                with self.assertRaises(ValueError): B.genuine_asset(self.root, self.asset, self.doc)
                self.metadata = old
        self.metadata['source']['mask_provenance']['manual_seed_boxes_used'] = True
        self.write_metadata()
        with self.assertRaises(ValueError): B.genuine_asset(self.root, self.asset, self.doc)

    def test_wrong_video_or_tampered_glb_fails_instead_of_substituting(self):
        self.metadata['source']['video_sha256'] = 'b' * 64
        self.write_metadata()
        with self.assertRaises(ValueError): B.genuine_asset(self.root, self.asset, self.doc)
        self.metadata['source']['video_sha256'] = 'a' * 64
        self.write_metadata()
        (self.model / 'asset.glb').write_bytes(b'different bytes')
        with self.assertRaises(ValueError): B.genuine_asset(self.root, self.asset, self.doc)

    def test_anchor_symlink_or_outside_tree_is_refused(self):
        target = self.root / 'outside.png'
        target.write_bytes(b'mask')
        (self.model / 'source-mask.png').unlink()
        (self.model / 'source-mask.png').symlink_to(target)
        self.write_metadata()
        with self.assertRaises(ValueError): B.genuine_asset(self.root, self.asset, self.doc)
        with self.assertRaises(ValueError): B.local(self.root, 'data/../outside.png')

    def test_missing_input_hash_or_invalid_source_time_is_not_an_automatic_anchor(self):
        for change in ({'image_sha256': None}, {'timestamp_s': 100}, {'time_domain': 'relative_clip_time'}):
            original = copy.deepcopy(self.metadata)
            self.metadata['source'].update(change)
            self.write_metadata()
            with self.assertRaises(ValueError): B.genuine_asset(self.root, self.asset, self.doc)
            self.metadata = original

    def test_native_sam_anchor_accepts_only_exact_import_pair_with_inventory_hashes(self):
        self.asset['backend'] = 'Meta SAM 3D Objects'
        self.metadata['backend'] = self.asset['backend']
        self.metadata['sam3d_output'] = True
        for role, name in [('image', 'image-source.png'), ('mask', 'mask-source.png')]:
            nested = self.model / 'raw/inputs' / name
            nested.parent.mkdir(parents=True, exist_ok=True)
            nested.write_bytes((self.model / ('source-rgb.png' if role == 'image' else 'source-mask.png')).read_bytes())
            relative = 'raw/inputs/' + name
            self.metadata['source'][role + '_path'] = relative
            self.metadata['artifact_files'].append({'file': relative, 'sha256': B.digest(nested), 'bytes': nested.stat().st_size})
        self.write_metadata()
        result = B.genuine_asset(self.root, self.asset, self.doc)
        self.assertTrue(result['anchor']['image']['path'].endswith('/raw/inputs/image-source.png'))
        self.metadata['artifact_files'][-1]['sha256'] = 'b' * 64
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, 'inventory differs'): B.genuine_asset(self.root, self.asset, self.doc)
        self.metadata['source']['image_path'] = 'raw/inputs/other.png'
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, 'exact native SAM input'): B.genuine_asset(self.root, self.asset, self.doc)

    def test_empty_catalog_makes_no_synthetic_rows_or_model_calls(self):
        (self.root / 'workspace').mkdir()
        (self.root / 'workspace/cases-index.json').write_text('{"cases":[]}')
        (self.root / 'twin_catalog.py').write_text('# inert fixture')
        with patch.object(B, 'TwinCatalog') as catalog:
            catalog.return_value.entries.return_value = []
            result = B.build(self.root)
        self.assertEqual(result['records'], [])
        self.assertEqual(result['summary']['native_asset_drafts'], 0)
        self.assertFalse(result['physical_accuracy_verified'])
        manifest = json.loads((self.root / 'showcase/library-index.json').read_text())
        script = (self.root / 'showcase/library-index.js').read_text()
        self.assertEqual(manifest, json.loads(script.removeprefix('window.INTERACTION_LIBRARY_INDEX=').removesuffix(';\n')))
        self.assertEqual(manifest['catalog']['sha256'], B.digest(self.root / 'showcase/library-catalog.json'))
        self.assertEqual(manifest['catalog']['path'], 'library-catalog-' + manifest['catalog']['sha256'] + '.json')
        immutable = self.root / 'showcase' / manifest['catalog']['path']
        self.assertEqual(immutable.read_bytes(), (self.root / 'showcase/library-catalog.json').read_bytes())
        # A new generation leaves the old immutable reference usable, while
        # fetchable JSON and legacy JS publish the same new manifest.
        (self.root / 'twin_catalog.py').write_text('# updated inert fixture')
        with patch.object(B, 'TwinCatalog') as catalog:
            catalog.return_value.entries.return_value = []
            B.build(self.root)
        later = json.loads((self.root / 'showcase/library-index.json').read_text())
        self.assertNotEqual(later['catalog']['path'], manifest['catalog']['path'])
        self.assertEqual(B.digest(immutable), manifest['catalog']['sha256'])

    def program_record_fixture(self, assets=None):
        from tests.test_stage5 import fixture
        doc = fixture()
        doc['reasoning']['provider'] = 'meta_api'
        doc['assets'] = copy.deepcopy(assets or [])
        doc['motion'] = []
        doc['source']['sha256'] = 'a' * 64
        doc['source']['original_sha256'] = 'a' * 64
        for frame in doc['evidence']:
            path = self.root / 'data' / (frame['id'] + '.jpg')
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b'jpeg fixture')
            frame['frame_path'] = path.relative_to(self.root).as_posix()
            frame['frame_sha256'] = B.digest(path)
        path = self.root / 'data/interaction.json'
        path.write_text(json.dumps(doc))
        row = {'id': 'uploaded-20261102--procedure', 'interaction_path': 'data/interaction.json'}
        sha = B.digest(path)
        index = {row['id']: {'original_sha256': sha}}
        return doc, row, sha, index

    def test_muse_program_without_geometry_is_included_as_semantic_only(self):
        doc, row, sha, index = self.program_record_fixture()
        with patch.object(B, 'offline_queue', return_value=None), patch.object(B, 'cached_insights', return_value=[]):
            value = B.build_record(self.root, row, doc, sha, index)
        self.assertEqual(value['counts']['assets'], 0)
        self.assertEqual(value['search_index']['readiness'], ['semantic_only'])
        self.assertIsNone(value['scene_case']['scene_ref'])
        self.assertEqual(value['thumbnail']['sha256'], doc['evidence'][0]['frame_sha256'])
        self.assertTrue(all(c['asset'] is None for c in value['components']))

    def test_unaligned_native_assets_need_no_aggregate_scene_but_keep_real_asset_hashes(self):
        self.asset['backend'] = 'Meta SAM 3D Objects'
        self.asset['physical_scale_known'] = False
        self.asset['geometry_status'] = 'unreviewed_model_candidate'
        self.metadata['backend'] = self.asset['backend']
        self.metadata['sam3d_output'] = True
        self.write_metadata()
        doc, row, sha, index = self.program_record_fixture([self.asset])
        with patch.object(B, 'offline_queue', return_value=None), patch.object(B, 'cached_insights', return_value=[]):
            value = B.build_record(self.root, row, doc, sha, index)
        self.assertIsNone(value['scene_case']['scene_ref'])
        self.assertEqual(value['counts']['assets'], 1)
        self.assertEqual(value['components'][0]['asset']['sha256'], self.asset['sha256'])
        self.assertEqual(value['search_index']['readiness'], ['geometry_draft'])
        self.assertEqual(value['search_index']['native_motion_status'], 'no_native_motion')
        self.assertFalse(value['source_video_alignment_added'])
        self.assertFalse(value['simulation_ready'])
        # This exemption cannot hide an existing pose or legacy mesh scene.
        doc['motion'] = [{'track_id': 'part', 'status': 'supported_conditional_motion_candidate',
                          'camera_from_native_mesh': [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 2], [0, 0, 0, 1]]}]
        with self.assertRaisesRegex(ValueError, 'unaligned native SAM 3D'):
            B.build_record(self.root, row, doc, sha, index)
        self.asset['backend'] = 'TripoSR'
        self.metadata['backend'] = 'TripoSR'
        self.write_metadata()
        doc, row, sha, index = self.program_record_fixture([self.asset])
        with self.assertRaisesRegex(ValueError, 'unaligned native SAM 3D'):
            B.build_record(self.root, row, doc, sha, index)

    def test_stale_optional_queue_is_skipped_after_actual_queue_verifier_rejects_it(self):
        p = self.root / 'data/component-asset-queues/bad'
        p.mkdir(parents=True)
        (p / 'queue.json').write_text('{"same_track_but_wrong_input":"not accepted"}')
        with patch('component_asset_queue.inspect', side_effect=ValueError('Queue job binding differs')) as verifier:
            result = B.offline_queue(self.root, 'case', 'a' * 64, self.doc['source'], [])
        verifier.assert_called_once()
        self.assertIsNone(result)

    def test_prepared_initial_inputs_require_actual_verifier_and_exact_video_and_component(self):
        p = self.root / 'data/initial-sam3d-handoffs/example'
        p.mkdir(parents=True)
        (p / 'handoff.json').write_text('{"component_id":"part"}')
        source = {'video_sha256': 'a' * 64, 'timestamp_s': 1.0}
        (p / 'source-binding.json').write_text(json.dumps({'source': source}))
        verified = {'status': 'verified_initial_reconstruction_prepared', 'component_id': 'part', 'source': source}
        with patch('initial_sam3d_handoff.inspect', return_value=verified) as verifier:
            rows = B.initial_inputs(self.root, self.doc['source'], [{'id': 'part'}])
        verifier.assert_called_once()
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]['baseline_required'])
        self.assertTrue(rows[0]['no_inference'])
        with patch('initial_sam3d_handoff.inspect', side_effect=ValueError('Actual source changed')) as verifier:
            self.assertEqual(B.initial_inputs(self.root, self.doc['source'], [{'id': 'part'}]), [])
        verifier.assert_called_once()
        with patch('initial_sam3d_handoff.inspect') as verifier:
            self.assertEqual(B.initial_inputs(self.root, {'sha256': 'b' * 64}, [{'id': 'part'}]), [])
            self.assertEqual(B.initial_inputs(self.root, self.doc['source'], [{'id': 'other'}]), [])
        verifier.assert_not_called()

    def test_current_generated_catalog_is_three_families_with_no_claimed_simulation(self):
        data = json.loads((ROOT / 'showcase/library-catalog.json').read_text())
        self.assertEqual(data['summary']['recordings'], len({r['source']['original_sha256'] for r in data['records']}))
        self.assertEqual(data['summary']['native_asset_drafts'], len({c['asset']['sha256'] for r in data['records'] for c in r['components'] if c['asset']}))
        for r in data['records']:
            self.assertFalse(r['simulation_ready'])
            self.assertFalse(r['source_video_alignment_added'])
            self.assertFalse(r['preferred_model_changed'])
            self.assertEqual(B.digest(ROOT / r['interaction']['path']), r['interaction']['sha256'])


if __name__ == '__main__': unittest.main()
