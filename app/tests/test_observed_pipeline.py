"""Gated workflow boundaries and immutable unsupported-stage contracts."""
import json
import io
import fcntl
from pathlib import Path
import shutil
import types
import threading
import unittest
from unittest.mock import patch

import observed_jobs as jobs
import observed_pipeline as pipeline
from tests.test_observed_jobs import JobFixture, ROOT


class WorkflowTests(JobFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        for name in ['observed_geometry.py', 'observed_stereo.py', 'observed_surface_validation.py',
                     'observed_surface_completion.py', 'observed_depth_completion.py',
                     'observed_surface_export.py', 'observed_completion_export.py', 'observed_pipeline.py']:
            shutil.copyfile(ROOT / name, self.root / name)
        shutil.copyfile(ROOT / 'scripts/run_observed_workflow.py', self.root / 'scripts/run_observed_workflow.py')
        self.extended = dict(self.body, workflow='gated-local-v1', feature_provider='sift', completion='none')

    def test_explicit_workflow_only_and_options_are_fixed_named_providers(self):
        request = jobs.canonical_request(self.extended, root=self.root)
        self.assertEqual(request['workflow'], 'gated-local-v1')
        self.assertEqual(request['feature_provider'], 'sift')
        for change in [dict(feature_provider='shell command'), dict(completion='unknown'),
                       dict(output='data/output'), dict(workflow='unknown'),
                       dict(feature_provider='reuse_verified', reuse_sparse_id='../file'),
                       dict(reuse_sparse_id='sparse-cache-' + '0' * 16)]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                jobs.canonical_request(self.extended | change, root=self.root)
        with self.assertRaises(ValueError):
            jobs.canonical_request(dict(self.body, feature_provider='cotracker'), root=self.root)

    def test_dense_gate_needs_four_train_views_forty_points_and_observed_provenance(self):
        row = {'metrics': {'recovered_points': 40, 'registered_train_views': 4},
               'status': 'sparse_multiview_candidate', 'prior_geometry_or_depth_used': False}
        self.assertTrue(pipeline.sparse_gate(row)['eligible'])
        for changed in [dict(row, metrics={'recovered_points': 39, 'registered_train_views': 4}),
                        dict(row, metrics={'recovered_points': 40, 'registered_train_views': 3}),
                        dict(row, prior_geometry_or_depth_used=True),
                        dict(row, status='linear_or_degenerate_sparse_candidate')]:
            self.assertFalse(pipeline.sparse_gate(changed)['eligible'])

    def test_no_sparse_geometry_blocks_downstream_preserves_real_failure_and_stage_receipts(self):
        request = jobs.canonical_request(self.extended, root=self.root)
        workflow = pipeline.Workflow(self.root, request, self.root / pipeline.BASE / 'observed-aaaaaaaaaaaaaaaaaaaaaaaa-workflow')
        folder = self.root / pipeline.BASE / 'observed-aaaaaaaaaaaaaaaaaaaaaaaa-sparse'
        folder.mkdir()
        viewer = {'id': folder.name, 'status': 'insufficient_correspondence', 'prior_geometry_or_depth_used': False,
                  'metrics': {'selected_views': 3, 'training_views': 2, 'heldout_views': 1, 'feature_tracks': 0,
                              'recovered_points': 0, 'registered_train_views': 0, 'heldout_views_passing': 0}}
        (folder / 'viewer.json').write_text(json.dumps(viewer))
        original = (folder / 'viewer.json').read_bytes()
        with patch.object(workflow, 'sparse', return_value=(folder, viewer)), \
             patch('observed_stereo.run', side_effect=AssertionError('No dense processing allowed')):
            result = workflow.run()
        self.assertEqual(result['status'], 'unsupported')
        self.assertEqual([r['status'] for r in result['stages']], ['completed', 'blocked', 'blocked', 'blocked', 'blocked'])
        self.assertEqual((folder / 'viewer.json').read_bytes(), original)
        self.assertEqual(result['exports'], {})
        self.assertEqual(result['new_api_calls'], 0)
        self.assertTrue(any(r['code'] == 'no_stable_tracks' for r in result['capture_diagnostics']))
        for stage in pipeline.STAGES:
            directory = workflow.output / 'stages' / stage
            self.assertTrue((directory / 'start.json').is_file())
            self.assertTrue((directory / 'result.json').is_file())
            start = json.loads((directory / 'start.json').read_text())
            expected_prior = {name: pipeline.sha(workflow.output / 'stages' / name / 'result.json')
                              for name in pipeline.STAGES[:pipeline.STAGES.index(stage)]}
            self.assertEqual(start['upstream_results_sha256'], expected_prior)
            self.assertEqual(start['workflow_contract_sha256'], pipeline.sha(workflow.output / 'contract.json'))
        self.assertEqual(result['stages'][0]['artifact_sha256'][folder.relative_to(self.root).as_posix() + '/viewer.json'],
                         pipeline.sha(folder / 'viewer.json'))
        before = (workflow.output / 'stages/sparse/result.json').read_bytes()
        with self.assertRaises(FileExistsError):
            workflow.stage('sparse', status='completed')
        self.assertEqual((workflow.output / 'stages/sparse/result.json').read_bytes(), before)

    def test_stage_error_recorded_once_no_retry_and_no_fake_result(self):
        request = jobs.canonical_request(self.extended, root=self.root)
        workflow = pipeline.Workflow(self.root, request, self.root / pipeline.BASE / 'observed-bbbbbbbbbbbbbbbbbbbbbbbb-workflow')
        with patch.object(workflow, 'sparse', side_effect=ValueError('Synthetic source mismatch')) as called:
            result = workflow.run()
        called.assert_called_once()
        self.assertEqual(result['status'], 'failed')
        self.assertIsNone(result['result_url'])
        self.assertEqual(result['stages'][0]['status'], 'failed')
        self.assertTrue((workflow.output / 'stages/sparse/failure.json').is_file())

    def test_local_job_uses_distinct_workflow_cli_and_preserves_old_sparse_semantics(self):
        old = jobs.ObservedRunner(jobs.canonical_request(self.body, root=self.root), root=self.root)
        new = jobs.ObservedRunner(jobs.canonical_request(self.extended, root=self.root), root=self.root)
        self.assertTrue(old.output.name.endswith('-sparse'))
        self.assertTrue(new.output.name.endswith('-workflow'))
        self.assertEqual(new.state['policy']['maximum_runtime_s'], 900)
        self.assertEqual(old.state['policy']['maximum_runtime_s'], 180)
        self.assertFalse(old.state['policy']['model_inference'])
        self.assertFalse(new.state['policy']['model_inference'])
        self.assertEqual(new.state['exports'], {})
        source = self.root / new.state['job_manager_source']['path']
        self.assertEqual(source.read_bytes(), jobs.LOADED_SOURCE)
        self.assertEqual(pipeline.sha(source), new.state['job_manager_source']['sha256'])

    def test_workflow_child_uses_fixed_argument_vector_and_immutable_canonical_request(self):
        runner = jobs.ObservedRunner(jobs.canonical_request(self.extended, root=self.root), root=self.root)
        def spawn(arguments):
            self.assertEqual(arguments, [str(self.root / '.venv-sam/bin/python'),
                str(self.root / 'scripts/run_observed_workflow.py'), '--request',
                (runner.directory / 'workflow-request.json').relative_to(self.root).as_posix(),
                '--output', runner.output.relative_to(self.root).as_posix()])
            self.assertEqual(json.loads((runner.directory / 'workflow-request.json').read_text()), runner.request)
            runner.output.mkdir()
            (runner.output / 'workflow.json').write_text(json.dumps({'id': runner.output.name,
                'new_api_calls': 0, 'production_assets_changed': False, 'status': 'unsupported',
                'result_url': None, 'capture_diagnostics': [], 'stages': [], 'exports': {}}))
            class Process:
                stdout = io.StringIO('{"phase":"workflow_complete","status":"unsupported"}\n')
                def wait(self): return 0
                def poll(self): return 0
            return Process()
        with patch.object(runner, '_launch', side_effect=spawn) as launch:
            runner.execute()
        launch.assert_called_once()
        self.assertEqual(runner.state['status'], 'completed')
        self.assertEqual(runner.state['workflow_status'], 'unsupported')
        self.assertIsNone(runner.state['result_url'])
        before = (runner.directory / 'workflow-request.json').read_bytes()
        with self.assertRaises(FileExistsError):
            pipeline.write_once(runner.directory / 'workflow-request.json', {'changed': True})
        self.assertEqual((runner.directory / 'workflow-request.json').read_bytes(), before)

    def test_standalone_job_respects_existing_cross_process_model_worker_lock(self):
        folder = self.root / 'data/flow-runs'
        folder.mkdir()
        with (folder / '.model-worker.lock').open('a+') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            runner = jobs.ObservedRunner(jobs.canonical_request(self.extended, root=self.root), root=self.root)
            with patch.object(runner, '_launch') as launch:
                runner.execute()
            launch.assert_not_called()
        self.assertEqual(runner.state['status'], 'failed')
        self.assertEqual(runner.state['failure_type'], 'BlockingIOError')
        self.assertFalse(runner.state['automatic_retry'])

    def test_nested_follow_on_is_bounded_registered_serial_and_cancellation_stops_later_children(self):
        program = json.loads(self.source.read_text())
        program['tracklets'].append(dict(program['tracklets'][0], id='second'))
        for frame in program['evidence']:
            frame['observations'].append(dict(frame['observations'][0], track_id='second'))
        self.source.write_text(json.dumps(program))
        catalog_path = self.root / 'workspace/cases.json'
        catalog = json.loads(catalog_path.read_text())
        catalog['cases'][0]['original_sha256'] = pipeline.sha(self.source)
        catalog_path.write_text(json.dumps(catalog))
        cancelled, started, stopped = threading.Event(), threading.Event(), threading.Event()
        saved = []
        def callback(state):
            saved.append(state)
            if state['status'] == 'running':
                started.set()
        def execute(runner):
            self.assertTrue(runner.parent_worker_lock_held)
            runner.state.update(status='running'); runner.save()
            self.assertTrue(runner.cancelled.wait(2))
            runner.state.update(status='cancelled'); runner.save(); stopped.set()
        result = {}
        with patch.object(jobs.ObservedRunner, 'execute', execute):
            worker = threading.Thread(target=lambda: result.update(jobs.run_follow_on_components(
                self.body['case_id'], root=self.root, cancel_event=cancelled, on_state=callback,
                parent_worker_lock_held=True)))
            worker.start(); self.assertTrue(started.wait(2)); cancelled.set(); worker.join(3)
        self.assertTrue(stopped.is_set())
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(result['components']), 1)
        self.assertEqual(result['unstarted_component_ids'], ['second'])
        self.assertEqual(result['components'][0]['status'], 'cancelled')
        self.assertTrue(result['cancelled'])
        self.assertTrue(saved[-1]['cancellation_requested'])
        self.assertEqual(saved[0]['completion'], 'smooth_gaps')
        self.assertEqual(saved[0]['feature_provider'], 'cotracker')
        with self.assertRaises(ValueError):
            jobs.run_follow_on_components(self.body['case_id'], root=self.root, cancel_event=threading.Event())
        with self.assertRaises(ValueError):
            jobs.run_follow_on_components(self.body['case_id'], ['part', 'second', 'part'], root=self.root,
                cancel_event=threading.Event(), parent_worker_lock_held=True)
        with self.assertRaises(ValueError):
            jobs.canonical_request(dict(self.extended, parent_worker_lock_held=True), root=self.root)
        parent = types.SimpleNamespace(state={'status': 'running', 'observed_child_job_id': result['components'][0]['id']})
        self.assertEqual(jobs.active_identifier(parent), result['components'][0]['id'])
        parent.state['status'] = 'cancelled'
        self.assertIsNone(jobs.active_identifier(parent))

    def make_sparse_cache(self):
        folder = self.root / pipeline.CACHE_ROOTS[0] / 'fixture-sparse-cache'
        folder.mkdir(parents=True)
        shutil.copyfile(self.root / 'observed_geometry.py', folder / 'implementation.py')
        document = pipeline.read(self.source)
        frames = []
        for index, frame in enumerate(document['evidence']):
            observation = frame['observations'][0]
            frames.append({'id': frame['id'], 'timestamp_s': frame['timestamp_s'],
                'image_url': '/files/' + frame['frame_path'], 'image_sha256': frame['frame_sha256'],
                'mask_url': '/files/' + observation['mask_path'], 'mask_sha256': observation['mask_sha256'],
                'split': 'heldout' if index == 2 else 'train'})
        protocol = dict(program_path=self.source.relative_to(self.root).as_posix(), program_sha256=pipeline.sha(self.source),
            implementation_sha256=pipeline.sha(folder / 'implementation.py'), track_id='part', input_views=frames,
            manual_coordinates=False, generated_meshes_read=False, depth_maps_read=False, old_poses_read=False,
            source=document['source'])
        (folder / 'protocol.json').write_text(json.dumps(protocol))
        (folder / 'viewer.json').write_text(json.dumps({'id': folder.name,
            'input_protocol_sha256': pipeline.sha(folder / 'protocol.json'), 'prior_geometry_or_depth_used': False,
            'status': 'insufficient_correspondence', 'metrics': {'recovered_points': 0}}))
        (folder / 'geometry.json').write_text('{"points":{},"tracks":{},"training_poses":{}}')
        return folder

    def test_explicit_cache_reuse_bound_to_publication_and_rechecks_saved_geometry_and_source_pixels(self):
        folder = self.make_sparse_cache()
        cache = pipeline.sparse_caches(self.root)[0]
        body = dict(self.extended, feature_provider='reuse_verified', reuse_sparse_id=cache['id'])
        request = jobs.canonical_request(body, root=self.root)
        workflow = pipeline.Workflow(self.root, request, self.root / pipeline.BASE / 'observed-cccccccccccccccccccccccc-workflow')
        actual, view = workflow.sparse()
        self.assertEqual(actual, folder)
        self.assertEqual(view['metrics']['recovered_points'], 0)
        (folder / 'geometry.json').write_text('corrupted after admission')
        with self.assertRaises(ValueError):
            workflow.sparse()

    def test_unavailable_mps_is_unsupported_without_running_completion_or_inventing_capture_failure(self):
        request = jobs.canonical_request(dict(self.extended, completion='moge'), root=self.root)
        workflow = pipeline.Workflow(self.root, request, self.root / pipeline.BASE / 'observed-dddddddddddddddddddddddd-workflow')
        unavailable = types.SimpleNamespace(backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)))
        with patch.dict('sys.modules', {'torch': unavailable}):
            row = workflow.stage('completion', lambda: workflow.completion(self.root / 'data/not-read', 'moge', []))
        self.assertEqual(row['status'], 'unsupported')
        self.assertIn('no MoGe inference', row['reason'])
        self.assertFalse(workflow.location('completion').exists())
        self.assertEqual(pipeline.capture_diagnostics(None)[0]['code'], 'reconstruction_not_run')

    def test_zero_eligible_dense_test_checks_block_inference_but_keep_observed_export(self):
        request = jobs.canonical_request(dict(self.extended, completion='moge'), root=self.root)
        workflow = pipeline.Workflow(self.root, request, self.root / pipeline.BASE / 'observed-eeeeeeeeeeeeeeeeeeeeeeee-workflow')
        sparse_folder = workflow.location('sparse'); sparse_folder.mkdir()
        sparse = {'status': 'sparse_multiview_candidate', 'prior_geometry_or_depth_used': False,
            'metrics': {'recovered_points': 40, 'registered_train_views': 4}}
        (sparse_folder / 'viewer.json').write_text(json.dumps(sparse))
        def dense(source, output, **options):
            output.mkdir()
            row = {'status': 'synthetic_contract_surface', 'metrics': {}, 'surface': {'source_frame_id': 'frame'}}
            (output / 'viewer-display.json').write_text(json.dumps(row))
            return row
        def validate(source, output, **options):
            output.mkdir()
            report = {'metrics': {'eligible_photo_checks': 0, 'conditional_photo_agreement_fraction': None}}
            (output / 'report.json').write_text(json.dumps(report))
            return report
        exported = {'glb_url': '/files/data/synthetic-contract-export.glb', 'glb_sha256': '0' * 64}
        with patch.object(workflow, 'sparse', return_value=(sparse_folder, sparse)), \
             patch('observed_stereo.run', side_effect=dense), patch('observed_stereo.derive_display_review'), \
             patch('observed_surface_validation.run', side_effect=validate), \
             patch.object(workflow, 'completion', side_effect=AssertionError('No completion may run')) as completion, \
             patch('observed_surface_export.export_run', return_value=exported), \
             patch.object(workflow, 'version_record', return_value='/files/data/synthetic-contract-version.json'):
            result = workflow.run()
        completion.assert_not_called()
        self.assertEqual(result['status'], 'completed_with_unsupported_stages')
        self.assertEqual(result['stages'][3]['status'], 'blocked')
        self.assertEqual(result['exports']['observed'], exported)
        self.assertNotIn('inferred', result['exports'])
        self.assertTrue(any(row['code'] == 'no_eligible_dense_test_checks' for row in result['capture_diagnostics']))

    def test_cotracker_provenance_rejects_modified_predictor_and_train_test_scope_leakage(self):
        folder = self.root / 'data/provider-contract-fixture'
        folder.mkdir()
        for name in ['correspondence-implementation.py', 'cotracker-predictor.py']:
            (folder / name).write_text('# Synthetic contract fixture; never executes inference')
        model = {'repo': 'fixture-only', 'checkpoint_sha256': '0' * 64}
        (folder / 'cotracker-model-provenance.json').write_text(json.dumps(model))
        provider = {'protocol': {'feature_provider': 'cotracker3_mask_core_v1'},
            'provider_implementation_sha256': pipeline.sha(folder / 'correspondence-implementation.py'),
            'predictor_implementation_sha256': pipeline.sha(folder / 'cotracker-predictor.py'),
            'checkpoint_sha256': model['checkpoint_sha256'], 'model_provenance': model,
            'maximum_model_calls': 4, 'training_source_frame_ids': ['train'], 'heldout_source_frame_ids': ['test'],
            'manual_queries': False, 'heldout_images_used_for_geometry': False}
        tracking = {'status': 'complete', 'actual_model_calls': 2, 'manual_queries': False,
            'learned_3d_or_depth_calls': 0, 'new_api_calls': 0, 'calls': [
                {'scope': 'train_geometry_forward', 'source_frame_ids': ['train']},
                {'scope': 'heldout_isolated_ref_test_forward', 'source_frame_ids': ['train', 'test']}]}
        audit = {'provider_implementation_sha256': provider['provider_implementation_sha256'],
            'checkpoint_sha256': provider['checkpoint_sha256'], 'actual_model_calls': 2,
            'training_model_calls': 1, 'heldout_model_calls': 1, 'geometry_used_only_training_images': True,
            'old_poses_or_generated_geometry_read': False}
        protocol = {'correspondence_provider': provider, 'input_views': [{'id': 'train', 'split': 'train'}, {'id': 'test', 'split': 'heldout'}]}
        def save():
            (folder / 'tracking.json').write_text(json.dumps(tracking))
            audit['tracking_json_sha256'] = pipeline.sha(folder / 'tracking.json')
            (folder / 'correspondence-audit.json').write_text(json.dumps(audit))
        save(); pipeline.verify_correspondence_provider(folder, protocol)
        tracking['calls'][0]['source_frame_ids'] = ['test']; save()
        with self.assertRaises(ValueError):
            pipeline.verify_correspondence_provider(folder, protocol)
        tracking['calls'][0]['source_frame_ids'] = ['train']; save()
        (folder / 'cotracker-predictor.py').write_text('modified')
        with self.assertRaises(ValueError):
            pipeline.verify_correspondence_provider(folder, protocol)


if __name__ == '__main__':
    unittest.main()
