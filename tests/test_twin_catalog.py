"""Catalog confinement, exact source-time semantics and handoff contracts."""
import copy
import importlib.util
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import stage5
from twin_catalog import TwinCatalog, MISSING_PHYSICS

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('catalog_test_procedure_fixture', ROOT/'tests/test_stage5.py')
fixtures = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixtures)


class TwinCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.doc = fixtures.fixture()
        self.doc['motion'] = []
        self.doc['links'] = {'scene_viewer': '', 'dense_viewer': ''}
        def file(relative, value):
            path = self.root / relative; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
            return stage5.sha(path)
        self.file = file
        source_hash = file('data/case/source/video.mp4', b'fake-video-only-for-contract-test')
        self.doc['source'].update(path='data/case/source/video.mp4', sha256=source_hash, original_sha256=source_hash)
        index = []
        for frame in self.doc['evidence']:
            relative = f"data/case/frames/{frame['source_index']}.jpg"
            frame['frame_path'] = relative
            frame['frame_sha256'] = file(relative, b'\xff\xd8contract-jpeg\xff\xd9')
            index.append({'index': frame['source_index'], 'timestamp_s': frame['timestamp_s'], 'file': relative.removeprefix('data/case/')})
        self.doc['inputs'] = [
            {'id': 'semantic_proposal', 'path': 'data/case/proposal.json', 'sha256': file('data/case/proposal.json', {}), 'kind': 'model_semantic_proposal'},
            {'id': 'source_manifest', 'path': 'data/case/manifest.json', 'sha256': file('data/case/manifest.json', {'source': {'sha256': source_hash, 'video': {'duration_s': 3.}}}), 'kind': 'source_manifest'},
            {'id': 'source_frame_index', 'path': 'data/case/frames/index.json', 'sha256': file('data/case/frames/index.json', {'frames': index}), 'kind': 'source_frame_index'},
        ]
        self.doc['assets'] = [{'id': 'asset_1', 'track_id': 'part', 'path': 'data/case/asset.glb',
                               'sha256': file('data/case/asset.glb', b'glTFcontract-test'), 'backend': 'Test fixture',
                               'geometry_status': 'unreviewed_model_candidate', 'physical_scale_known': False}]
        self.id = 'case--procedure'
        self.publish()
        self.catalog = TwinCatalog(self.root)

    def publish(self):
        digest = self.file('data/case/procedure/interaction.json', self.doc)
        self.row = {'id': self.id, 'title': 'Generic test component', 'description': 'Open a part beside a holder.',
                    'interaction_path': 'data/case/procedure/interaction.json', 'interaction': self.doc,
                    'original_sha256': digest, 'labels': {'assets': 1, 'steps': 1}}
        self.file('workspace/cases.json', {'cases': [self.row]})

    def test_search_limits_and_labels_do_not_claim_validated_physics(self):
        result = self.catalog.search('generic holder')
        self.assertEqual(result['twins'][0]['twin_id'], self.id)
        self.assertFalse(result['twins'][0]['simulation_ready'])
        self.assertEqual(self.catalog.search('absent')['twins'], [])
        for limit in [0, 26, True, 1.5]:
            with self.subTest(limit=limit), self.assertRaises(ValueError): self.catalog.search('', limit)

    def test_agent_query_uses_semantic_facets_and_groups_source_variants(self):
        current = copy.deepcopy(self.row)
        self.row['interaction']['reasoning']['provider'] = 'meta_api'
        current['interaction']['reasoning']['provider'] = 'meta_api'
        current['id'] = 'new-upload-20261102--procedure'
        current['interaction']['assets'] = []
        current['labels']['assets'] = 0
        self.file('workspace/cases.json', {'cases': [self.row, current]})
        with patch('task_intelligence.run_analysis') as paid:
            result = self.catalog.library_search(parts=['part'], actions=['visible_opening'],
                                                  relations=['component_of'], readiness=['semantic_only'])
            paid.assert_not_called()
        self.assertEqual(result['total'], 1)
        self.assertEqual(result['records'][0]['id'], current['id'])
        self.assertEqual(result['records'][0]['search_index']['selected_backends'], [])
        self.assertTrue(result['metadata_only'])
        self.assertTrue(result['artifact_inspection_required'])
        self.assertFalse(result['policy']['simulation_ready'])
        variants = self.catalog.search(group_by='variant', sort='id', limit=1)
        self.assertEqual(variants['total'], 2)
        self.assertTrue(variants['has_more'])
        self.assertEqual(self.catalog.library_facets()['query_contract'], result['query_contract'])

    def test_mcp_metadata_task_class_filter_uses_only_declared_typed_projection(self):
        self.row['interaction']['reasoning']['provider']='meta_api'
        self.row['task_classes']=['hinged_assemblies']
        self.file('workspace/cases.json',{'cases':[self.row]})
        found=self.catalog.search(task_classes=['hinged_assemblies'],automatic_only=True)
        self.assertEqual(found['twins'][0]['task_classes'],['hinged_assemblies'])
        self.assertEqual(self.catalog.search(task_classes=['sliding_mechanisms'])['total'],0)
        self.assertEqual(self.catalog.library_facets()['facets']['task_classes'][0]['count'],1)

    def test_invalid_discovery_request_is_rejected_before_native_artifact_scan(self):
        with patch('sam3d_review_catalog.discover_reviews') as scan:
            with self.assertRaises(ValueError): self.catalog.library_search(parts=[''])
            scan.assert_not_called()

    def candidate_record(self):
        record = {'component_id': 'part', 'source_sha256': self.doc['source']['sha256'],
                  'backend': 'Meta SAM 3D Objects', 'source_sample_index': 1, 'source_timestamp_s': 0,
                  'selected': False, 'physical_accuracy_verified': False}
        for field, name, value in [('native_glb', 'candidate.glb', b'glTFnew-candidate'),
                                   ('source_image', 'image.png', b'png-source'), ('source_mask', 'mask.png', b'png-mask'),
                                   ('review_manifest', 'review.json', b'{}')]:
            path = 'data/native/' + name
            digest = self.file(path, value)
            record[field] = {'path': path, 'sha256': digest, 'bytes': len(value)}
        return record

    def test_default_sam_candidate_is_retrievable_without_changing_legacy_assets(self):
        candidate = self.candidate_record()
        original = copy.deepcopy(self.doc)
        with patch('sam3d_review_catalog.reviews_for_record', return_value=[candidate]):
            descriptor = self.catalog.generated_candidate(self.id, 'part')
            self.assertEqual(self.catalog.generated_candidate_artifact(self.id, 'part', 'asset'), b'glTFnew-candidate')
            self.assertEqual(self.catalog.generated_candidate_artifact(self.id, 'part', 'source-mask'), b'png-mask')
            self.assertFalse(descriptor['selected_in_saved_program'])
            self.assertFalse(descriptor['source_video_alignment_verified'])
            self.assertEqual(descriptor['source_sha256'], self.doc['source']['sha256'])
            self.assertEqual(descriptor['artifacts']['asset']['sha256'], candidate['native_glb']['sha256'])
            self.assertEqual(self.catalog.asset(self.id, 'part', binary=True), b'glTFcontract-test')
            with self.assertRaises(ValueError): self.catalog.generated_candidate(self.id, 'missing')
            with self.assertRaises(ValueError): self.catalog.generated_candidate_artifact(self.id, 'part', '../file')
        self.assertEqual(self.doc, original)

    def test_native_resource_rejects_changed_bytes_wrong_source_or_promotion(self):
        candidate = self.candidate_record()
        with patch('sam3d_review_catalog.reviews_for_record', return_value=[candidate]):
            self.file(candidate['native_glb']['path'], b'changed-native-bytes')
            with self.assertRaisesRegex(ValueError, 'changed during read'):
                self.catalog.generated_candidate_artifact(self.id, 'part', 'asset')
        for change in [{'source_sha256': 'f' * 64}, {'selected': True}, {'physical_accuracy_verified': True}]:
            with self.subTest(change=change), patch('sam3d_review_catalog.reviews_for_record', return_value=[dict(candidate, **change)]):
                with self.assertRaises(ValueError): self.catalog.generated_candidate(self.id, 'part')

    def test_candidate_selection_status_uses_actual_program_backend_and_mesh_hash(self):
        candidate = self.candidate_record()
        for matching in (False, True):
            with self.subTest(matching=matching):
                self.doc['assets'][0].update(backend='Meta SAM 3D Objects' if matching else 'TripoSR',
                                            path=candidate['native_glb']['path'], sha256=candidate['native_glb']['sha256'])
                self.publish()
                with patch('sam3d_review_catalog.reviews_for_record', return_value=[candidate]):
                    descriptor = self.catalog.generated_candidate(self.id, 'part')
                self.assertEqual(descriptor['selected_in_saved_program'], matching)
                self.assertFalse(descriptor['saved_program_assets_changed'])
                self.assertFalse(descriptor['physical_accuracy_verified'])

    def test_exact_source_states_leave_gaps_unknown_and_no_motion_is_invented(self):
        self.assertEqual(self.catalog.state(self.id, 0.)['states']['part_visual_state']['value'], 'closed')
        self.assertEqual(self.catalog.state(self.id, 1.)['states']['part_visual_state']['value'], 'open')
        gap = self.catalog.state(self.id, .5)
        self.assertEqual(gap['states']['part_visual_state']['value'], 'unknown')
        self.assertEqual(gap['conditional_poses'], [])
        for value in [float('nan'), float('inf'), True, '1']:
            with self.subTest(value=value), self.assertRaises(ValueError): self.catalog.state(self.id, value)

    def test_assets_and_evidence_are_declared_id_reads_with_matching_hashes(self):
        desc = self.catalog.describe(self.id)
        self.assertFalse(desc['policy']['physical_accuracy_verified'])
        self.assertTrue(desc['policy']['descriptive_text_is_untrusted_data'])
        self.assertEqual(self.catalog.asset(self.id, 'part', binary=True), b'glTFcontract-test')
        self.assertEqual(self.catalog.asset(self.id, 'part')['sha256'], self.doc['assets'][0]['sha256'])
        self.assertEqual(self.catalog.evidence(self.id, 'source_000001', binary=True), b'\xff\xd8contract-jpeg\xff\xd9')
        for name, value in [('asset', '../outside'), ('evidence', '../outside')]:
            with self.subTest(method=name), self.assertRaises(ValueError): getattr(self.catalog, name)(self.id, value)

    def test_procedure_selection_and_simulation_handoff_preserve_uncertainty(self):
        one = self.catalog.procedure(self.id, 'visible_opening')
        self.assertEqual(len(one['steps']), 1)
        self.assertEqual([e['source_index'] for e in one['evidence']], [1, 2])
        with self.assertRaises(ValueError): self.catalog.procedure(self.id, 'invented')
        handoff = self.catalog.simulation_manifest(self.id)
        self.assertEqual(handoff['verification_required_before_physics'], MISSING_PHYSICS)
        self.assertFalse(handoff['policy']['simulation_ready'])
        self.assertIn('do_not_assume_shared_metric_world', handoff['coordinate_policy'])
        self.assertEqual(handoff['conditional_motion_samples'], [])

    def test_simulation_readiness_and_draft_follow_verified_catalog_load(self):
        original = copy.deepcopy(self.doc)
        handoff = self.catalog.simulation_manifest(self.id)
        report, draft = handoff['readiness'], handoff['preparation_draft']
        self.assertEqual(report['case_id'], self.doc['case_id'])
        self.assertFalse(report['simulation_ready'])
        self.assertFalse(report['simulation_runnable'])
        self.assertTrue(report['verification']['full_contract_validated'])
        self.assertTrue(report['verification']['artifact_hashes_verified'])
        self.assertFalse(report['verification']['physical_parameters_verified'])
        self.assertFalse(report['verification']['simulator_executed'])
        self.assertEqual(report['verification']['validated_program_sha256'], handoff['program_sha256'])
        self.assertEqual(draft['program_sha256'], handoff['program_sha256'])
        self.assertEqual(draft['program_uri'], handoff['program_uri'])
        self.assertEqual(draft['candidate_relationships'], handoff['candidate_relationships'])
        self.assertEqual(draft['demonstrated_steps'], handoff['demonstration_steps'])
        self.assertFalse(draft['simulation_runnable'])
        self.assertFalse(draft['physical_execution_authorized'])
        self.assertIsNone(draft['robot'])
        self.assertIsNone(draft['components'][0]['meters_per_native_unit'])
        self.assertIsNone(draft['components'][0]['collision_meshes'])
        self.assertIn('contact', report['blocking_requirement_ids'])
        self.assertIn('seating', report['blocking_requirement_ids'])
        self.assertEqual(self.doc, original)

    def test_invalid_artifacts_never_reach_readiness_or_draft_generation(self):
        self.file('data/case/source/video.mp4', b'changed-video')
        with patch('twin_catalog.audit_simulation_readiness') as audit, patch('twin_catalog.build_simulation_draft') as draft:
            with self.assertRaisesRegex(ValueError, 'changed evidence'):
                self.catalog.simulation_manifest(self.id)
            audit.assert_not_called()
            draft.assert_not_called()

    def test_physical_ready_claim_is_rejected_before_simulator_manifest(self):
        from jsonschema import ValidationError
        self.doc['runtime_policy']['simulation_ready'] = True
        self.publish()
        with patch('twin_catalog.audit_simulation_readiness') as audit:
            with self.assertRaises(ValidationError):
                self.catalog.simulation_manifest(self.id)
            audit.assert_not_called()

    def brief_record(self, key='current', *, mode='analysis', output=12000, images=None):
        images = [{'evidence_id': 'source_000001', 'sha256': self.doc['evidence'][0]['frame_sha256']}] if images is None else images
        return {'request_key': key, 'contract': {'mode': mode, 'question': None,
                    'interaction_sha256': self.row['original_sha256'], 'source_sha256': self.doc['source']['sha256'],
                    'max_output_tokens': output, 'images': images},
                'provider': 'meta_api', 'model': 'muse-spark-1.3',
                'result': {'title': key, 'claims': [{'title': 'Visible candidate', 'evidence_ids': ['source_000002']}],
                           'robotics_handoff': {'sequence_kind': 'hypothetical_storyboard_not_robot_commands'}},
                'citations': [{'frame_url': 'https://must-not-use.invalid', 'evidence_id': 'invented'}],
                'usage': {'private_test_accounting': 7}, 'budget': {'private_test_accounting': True},
                'estimated_cost_usd': 'SENSITIVE_TEST_COST', 'request_receipt': 'SENSITIVE_TEST_RECEIPT'}

    def test_cached_task_brief_prefers_current_contract_and_exposes_only_safe_data(self):
        canonical = self.brief_record()
        historic = self.brief_record('historical')
        plans = {'request_key': 'current', 'contract': {'images': canonical['contract']['images']}}
        with patch('task_intelligence.cached_analyses', return_value=[{'record': canonical}, {'record': historic}]), \
             patch('task_intelligence.plan_analysis', return_value=plans) as planner, \
             patch('task_intelligence.run_analysis') as paid, patch('meta_api.MetaClient') as client, \
             patch('meta_api.load_key') as key, patch('meta_budget.BudgetLedger') as ledger:
            result = self.catalog.describe(self.id)
            handoff = self.catalog.simulation_manifest(self.id)
            paid.assert_not_called(); client.assert_not_called(); key.assert_not_called(); ledger.assert_not_called()
        brief = result['task_brief']
        self.assertEqual(brief, handoff['task_brief'])
        self.assertEqual(brief['result']['title'], 'current')
        self.assertEqual(brief['selection'], 'current_exact_contract')
        self.assertEqual(brief['interaction_sha256'], self.row['original_sha256'])
        self.assertEqual(brief['source_sha256'], self.doc['source']['sha256'])
        self.assertFalse(brief['simulation_ready']); self.assertFalse(brief['physical_accuracy_verified'])
        self.assertFalse(brief['new_api_call'])
        self.assertEqual(brief['citations'], [{'evidence_id': 'source_000002', 'source_index': 2, 'timestamp_s': 1.,
                         'frame_sha256': self.doc['evidence'][1]['frame_sha256'],
                         'uri': f'twin://{self.id}/evidence/source_000002'}])
        encoded = json.dumps(brief)
        for private in ['request_key', 'request_receipt', 'budget', 'usage', 'estimated_cost_usd',
                        'SENSITIVE_TEST', 'must-not-use.invalid', 'frame_path']:
            self.assertNotIn(private, encoded)
        planner.assert_called_with(self.row['interaction_path'], root=self.root.resolve(), mode='analysis', question=None,
                                   max_images=12, max_output_tokens=12000)
        brief['result']['title'] = 'Changed client-side copy'
        self.assertEqual(canonical['result']['title'], 'current')

    def test_cached_task_brief_falls_back_only_to_matching_verified_historical_settings(self):
        earlier, latest = self.brief_record('earlier'), self.brief_record('latest')
        wrong_output, wrong_images = self.brief_record('wrong-output', output=8000), self.brief_record('wrong-images', images=[])
        ask = self.brief_record('ask-result', mode='ask')
        other_source = self.brief_record('other-source'); other_source['contract']['source_sha256'] = 'f' * 64
        other_program = self.brief_record('other-program'); other_program['contract']['interaction_sha256'] = 'e' * 64
        rows = [{'record': row} for row in [earlier, latest, wrong_output, wrong_images, ask, other_source, other_program]]
        planned = {'request_key': 'not-cached', 'contract': {'images': latest['contract']['images']}}
        with patch('task_intelligence.cached_analyses', return_value=rows), patch('task_intelligence.plan_analysis', return_value=planned):
            brief = self.catalog.describe(self.id)['task_brief']
        self.assertEqual(brief['result']['title'], 'latest')
        self.assertEqual(brief['selection'], 'verified_historical_contract')

    def test_absent_or_incompatible_cached_brief_is_explicit_and_does_not_generate(self):
        with patch('task_intelligence.cached_analyses', return_value=[]), \
             patch('task_intelligence.plan_analysis') as planner, patch('task_intelligence.run_analysis') as paid:
            brief = self.catalog.describe(self.id)['task_brief']
            self.assertEqual(brief['status'], 'not_available')
            planner.assert_not_called(); paid.assert_not_called()
        wrong = self.brief_record('8k-only', output=8000)
        with patch('task_intelligence.cached_analyses', return_value=[{'record': wrong}]), \
             patch('task_intelligence.plan_analysis', return_value={'request_key': 'current', 'contract': {'images': wrong['contract']['images']}}), \
             patch('task_intelligence.run_analysis') as paid:
            self.assertEqual(self.catalog.simulation_manifest(self.id)['task_brief']['status'], 'not_available')
            paid.assert_not_called()

    def test_task_brief_cache_read_occurs_only_after_catalog_hash_validation(self):
        self.file('data/case/source/video.mp4', b'changed-video')
        with patch('task_intelligence.cached_analyses') as cache:
            for method in [self.catalog.describe, self.catalog.simulation_manifest]:
                with self.assertRaisesRegex(ValueError, 'changed evidence'):
                    method(self.id)
            cache.assert_not_called()

    def test_recovered_task_brief_discloses_whole_item_omissions_without_private_trace(self):
        record = self.brief_record('recovered')
        record['recovery'] = {'recovery_policy': 'whole_invalid_item_omissions_only_v1',
                              'removed_count': 3, 'all_model_items_retained': False,
                              'original_response_sha256': 'd' * 64,
                              'original_attempt': 'PRIVATE_TEST_ORIGINAL_ATTEMPT',
                              'trace_path': 'PRIVATE_TEST_TRACE', 'original_request_key': 'PRIVATE_TEST_REQUEST'}
        planned = {'request_key': 'new-code-contract', 'contract': {'images': record['contract']['images']}}
        with patch('task_intelligence.cached_analyses', return_value=[{'record': record}]), \
             patch('task_intelligence.plan_analysis', return_value=planned):
            brief = self.catalog.describe(self.id)['task_brief']
        self.assertEqual(brief['recovery']['removed_count'], 3)
        self.assertFalse(brief['recovery']['all_model_items_retained'])
        self.assertEqual(brief['recovery']['additional_api_calls'], 0)
        self.assertEqual(brief['recovery']['original_response_sha256'], 'd' * 64)
        self.assertNotIn('PRIVATE_TEST', json.dumps(brief))
        self.assertEqual(brief['result'], record['result'])

    def test_unknown_ids_and_arbitrary_paths_are_rejected(self):
        for value in ['unknown', '../.secrets/meta.env', '/etc/passwd']:
            with self.subTest(value=value), self.assertRaises(ValueError): self.catalog.describe(value)
        for path in ['stage5.py', 'data/../../outside', '.secrets/meta.env']:
            with self.subTest(path=path), self.assertRaises(ValueError): self.catalog.data_path(path)

    def test_source_change_and_program_change_fail_integrity(self):
        self.file('data/case/source/video.mp4', b'changed-video')
        with self.assertRaisesRegex(ValueError, 'changed evidence'): self.catalog.describe(self.id)
        self.file('data/case/source/video.mp4', b'fake-video-only-for-contract-test')
        changed = copy.deepcopy(self.doc); changed['summary'] = 'Changed after catalog build'
        self.file('data/case/procedure/interaction.json', changed)
        with self.assertRaisesRegex(ValueError, 'changed since catalog'): self.catalog.describe(self.id)

    def test_tampered_input_or_data_symlink_never_reads_outside_data(self):
        (self.root/'.secrets').mkdir(); (self.root/'.secrets/fake.env').write_text('fake-test-no-secret')
        self.doc['inputs'][0]['path'] = '.secrets/fake.env'
        self.publish()
        with self.assertRaisesRegex(ValueError, 'Only declared data'): self.catalog.load(self.id)
        self.doc['inputs'][0]['path'] = 'data/link.json'
        (self.root/'data/link.json').symlink_to(self.root/'.secrets/fake.env')
        self.publish()
        with self.assertRaisesRegex(ValueError, 'escapes data'): self.catalog.load(self.id)

    def test_mislabeled_code_input_cannot_expand_known_code_allowlist(self):
        self.file('unlisted.py', b'not-a-code-input')
        self.doc['inputs'].append({'id': 'rogue', 'kind': 'deterministic_compiler_code', 'path': 'unlisted.py', 'sha256': '0'*64})
        self.publish()
        with self.assertRaisesRegex(ValueError, 'Only declared data'): self.catalog.load(self.id)

    def test_known_code_name_cannot_symlink_to_hidden_test_file(self):
        (self.root/'.secrets').mkdir(); (self.root/'.secrets/fake.json').write_text('fake fixture, not a credential')
        (self.root/'stage5.py').symlink_to(self.root/'.secrets/fake.json')
        self.doc['inputs'].append({'id': 'code', 'kind': 'deterministic_compiler_code', 'path': 'stage5.py', 'sha256': '0'*64})
        self.publish()
        with self.assertRaisesRegex(ValueError, 'Compiler input'): self.catalog.load(self.id)

    def test_hidden_or_symlinked_data_root_is_rejected(self):
        self.file('data/.hidden/frame.jpg', b'fake')
        with self.assertRaisesRegex(ValueError, 'Hidden'): self.catalog.data_path('data/.hidden/frame.jpg')
        (self.root/'data').rename(self.root/'saved-data')
        (self.root/'data').symlink_to(self.root/'saved-data', target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'Data root'): self.catalog.load(self.id)


if __name__ == '__main__':
    unittest.main()
