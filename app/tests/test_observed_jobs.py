"""Bounded job/API contracts. Child-process fixtures perform no inference."""
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

from PIL import Image

import observed_jobs as jobs
import stage5
from tests.test_stage5 import fixture

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('serve_observed_jobs_tests', ROOT / 'scripts/serve_workspace.py')
server_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server_module)
CASE = 'fixture-case--procedure-v1'


class JobFixture:
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        folder = self.root / 'data/case'
        folder.mkdir(parents=True)
        (self.root / 'workspace').mkdir()
        (self.root / '.venv-sam/bin').mkdir(parents=True)
        (self.root / '.venv-sam/bin/python').write_text('Synthetic runtime path; never executed')
        (self.root / 'scripts').mkdir()
        (self.root / 'scripts/test_observed_geometry.py').write_text('# Synthetic test CLI; never executed')
        (self.root / 'observed_geometry.py').write_text('# Synthetic test engine; never executed')
        doc = fixture()
        (folder / 'source.mp4').write_bytes(b'Synthetic video fixture; not decoded')
        doc['source'].update(path='data/case/source.mp4', sha256=stage5.sha(folder / 'source.mp4'), original_sha256=stage5.sha(folder / 'source.mp4'))
        for frame in doc['evidence']:
            image = folder / (frame['id'] + '.png')
            mask = folder / (frame['id'] + '-mask.png')
            Image.new('RGB', (64, 64), '#334455').save(image)
            Image.new('L', (64, 64), 255).save(mask)
            frame.update(frame_path=image.relative_to(self.root).as_posix(), frame_sha256=stage5.sha(image),
                observations=[{'track_id': 'part', 'mask_path': mask.relative_to(self.root).as_posix(),
                    'mask_sha256': stage5.sha(mask), 'gate_status': 'candidate_geometric_gate_only', 'audit_status': 'not_model_audited'}])
        self.source = folder / 'interaction.json'
        self.source.write_text(json.dumps(doc))
        self.body = {'case_id': CASE, 'track_id': 'part'}
        (self.root / 'workspace/cases.json').write_text(json.dumps({'cases': [{'id': CASE, 'title': 'Synthetic contract fixture',
            'interaction_path': self.source.relative_to(self.root).as_posix(), 'original_sha256': stage5.sha(self.source)}]}))

    def tearDown(self):
        self.temporary.cleanup()

    def runner(self):
        return jobs.ObservedRunner(jobs.canonical_request(self.body, root=self.root), root=self.root)


class JobContracts(JobFixture, unittest.TestCase):
    def test_only_registered_case_and_existing_track_accepted_no_options_or_paths(self):
        self.assertEqual(jobs.canonical_request(self.body, root=self.root)['source_observations'], 3)
        for body in [dict(self.body, output='data/elsewhere'), dict(self.body, model='anything'),
                     dict(self.body, gpu=True), dict(self.body, max_views=1000),
                     dict(self.body, case_id='../escape'), dict(self.body, case_id='unregistered'),
                     dict(self.body, track_id='../escape'), dict(self.body, track_id='unknown')]:
            with self.subTest(body=body), self.assertRaises(ValueError):
                jobs.canonical_request(body, root=self.root)
        catalog = jobs.catalog(root=self.root)
        self.assertTrue(catalog['runtime_available'])
        self.assertTrue(catalog['cases'][0]['components'][0]['eligible'])
        self.assertFalse(catalog['policy']['model_inference'])

    def test_completed_insufficient_geometry_is_preserved_with_fixed_arguments_no_retry(self):
        runner = self.runner()
        def spawn(arguments):
            self.assertEqual(arguments[0], str(self.root / '.venv-sam/bin/python'))
            self.assertEqual(arguments[1], str(self.root / 'scripts/test_observed_geometry.py'))
            self.assertEqual(arguments[-5:], ['--max-views', '35', '--max-ba-evaluations', '70', '--no-sensitivity'])
            runner.output.mkdir()
            (runner.output / 'viewer.json').write_text(json.dumps({'id': runner.output.name,
                'new_api_calls': 0, 'prior_geometry_or_depth_used': False, 'status': 'insufficient_correspondence',
                'reason': 'Synthetic no-correspondence result', 'metrics': {'recovered_points': 0}}))
            class Process:
                stdout = io.StringIO('{"phase":"features","selected_views":3}\n')
                def wait(self): return 0
                def poll(self): return 0
            return Process()
        with patch.object(runner, '_launch', side_effect=spawn) as launcher:
            runner.execute()
        launcher.assert_called_once()
        self.assertEqual(runner.state['status'], 'completed')
        self.assertEqual(runner.state['result_status'], 'insufficient_correspondence')
        self.assertTrue(runner.state['viewer_url'].endswith('/viewer.json'))
        self.assertEqual(runner.state['metrics']['recovered_points'], 0)
        self.assertFalse(runner.state['automatic_retry'])
        self.assertTrue((runner.output / 'viewer.json').is_file())

    def test_cancel_before_execution_does_not_spawn_and_new_attempt_preserves_old_record(self):
        runner = self.runner(); runner.cancel()
        with patch.object(runner, '_launch') as spawn:
            runner.execute()
        spawn.assert_not_called()
        self.assertEqual(runner.state['status'], 'cancelled')
        original = (runner.directory / 'job.json').read_bytes()
        next_attempt = self.runner()
        self.assertNotEqual(runner.id, next_attempt.id)
        self.assertEqual((runner.directory / 'job.json').read_bytes(), original)

    def test_active_cancel_terminates_only_fixed_worker_and_preserves_partial_output(self):
        runner = self.runner()
        started, stopped = threading.Event(), threading.Event()
        class Process:
            def __init__(self): self.stdout = self.lines(); self.code = None
            def lines(self):
                started.set(); yield '{"phase":"tracks"}\n'
                stopped.wait(5)
            def poll(self): return self.code
            def terminate(self): self.code = -15; stopped.set()
            def kill(self): self.code = -9; stopped.set()
            def wait(self): stopped.wait(5); return self.code
        def spawn(arguments):
            runner.output.mkdir(); (runner.output / 'protocol.json').write_text('{"synthetic_partial":true}')
            return Process()
        with patch.object(runner, '_launch', side_effect=spawn) as launch:
            worker = threading.Thread(target=runner.execute)
            worker.start(); self.assertTrue(started.wait(2)); runner.cancel(); worker.join(3)
        self.assertFalse(worker.is_alive()); launch.assert_called_once()
        self.assertEqual(runner.state['status'], 'cancelled')
        self.assertTrue((runner.output / 'protocol.json').exists())
        self.assertTrue(runner.state['partial_outputs_preserved'])

    def test_failed_worker_keeps_protocol_and_does_not_fallback_or_retry(self):
        runner = self.runner()
        runner.output.mkdir(); (runner.output / 'protocol.json').write_text('{"synthetic_partial":true}')
        class Process:
            stdout = io.StringIO('Synthetic test process failure\n')
            def wait(self): return 1
            def poll(self): return 1
        with patch.object(runner, '_launch', return_value=Process()) as launch:
            runner.execute()
        launch.assert_called_once()
        self.assertEqual(runner.state['status'], 'failed')
        self.assertIsNone(runner.state['result_url'])
        self.assertTrue((runner.output / 'protocol.json').exists())

    def test_source_change_and_symlink_storage_rejected_without_child_execution(self):
        runner = self.runner(); self.source.write_text(self.source.read_text() + ' ')
        with patch.object(runner, '_launch') as launch:
            runner.execute()
        launch.assert_not_called(); self.assertEqual(runner.state['status'], 'failed')
        with self.assertRaises(ValueError):
            jobs.record('../escape', root=self.root)

    def test_symlinked_job_storage_cannot_write_outside_data(self):
        with tempfile.TemporaryDirectory() as outside:
            base = self.root / jobs.BASE
            base.symlink_to(outside)
            with self.assertRaises(ValueError):
                self.runner()
            self.assertEqual(list(Path(outside).iterdir()), [])

    def test_runtime_limit_stops_worker_without_automatic_retry(self):
        runner = self.runner()
        stopped = threading.Event()
        class Process:
            def __init__(self): self.stdout = self.lines(); self.code = None
            def lines(self):
                yield '{"phase":"features"}\n'
                stopped.wait(2)
            def poll(self): return self.code
            def terminate(self): self.code = -15; stopped.set()
            def kill(self): self.code = -9; stopped.set()
            def wait(self): stopped.wait(2); return self.code
        with patch.object(runner, '_launch', return_value=Process()) as launch, patch.object(jobs, 'MAX_SECONDS', .02):
            runner.execute()
        launch.assert_called_once()
        self.assertEqual(runner.state['status'], 'failed')
        self.assertEqual(runner.state['phase'], 'time_limit')
        self.assertFalse(runner.state['automatic_retry'])

    def test_restart_read_shows_interrupted_without_mutation_or_retry(self):
        runner = self.runner(); runner.state['status'] = 'running'; runner.save()
        before = (runner.directory / 'job.json').read_bytes()
        stale = jobs.record(runner.id, root=self.root, active_job_id=None)
        self.assertEqual(stale['status'], 'interrupted')
        self.assertEqual(stale['saved_status'], 'running')
        self.assertFalse(stale['automatic_retry'])
        self.assertEqual((runner.directory / 'job.json').read_bytes(), before)
        self.assertEqual(jobs.record(runner.id, root=self.root, active_job_id=runner.id)['status'], 'running')


class HTTPContracts(JobFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.root_patch = patch.object(server_module, 'ROOT', self.root); self.root_patch.start()
        self.server = server_module.WorkspaceServer(('127.0.0.1', 0), server_module.Handler)
        from tests.owner_http_fixture import OwnerHTTPSession
        self.owner_http=OwnerHTTPSession(self.server);self.addCleanup(self.owner_http.close)
        self.worker = threading.Thread(target=self.server.serve_forever, daemon=True); self.worker.start()
        self.base = 'http://127.0.0.1:' + str(self.server.server_port)

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.worker.join(2)
        self.root_patch.stop(); super().tearDown()

    def request(self, path, body=None, headers=None):
        request = urllib.request.Request(self.base + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers=headers or {'Content-Type': 'application/json', 'Origin': self.base})
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.load(response)

    def test_origin_host_json_and_registered_id_guards_cover_new_routes(self):
        for headers, body, code in [
            ({'Content-Type': 'application/json', 'Origin': 'https://attacker.example'}, self.body, 403),
            ({'Content-Type': 'text/plain'}, self.body, 400),
            ({'Content-Type': 'application/json', 'Host': 'attacker.example'}, self.body, 403),
            ({'Content-Type': 'application/json'}, dict(self.body, input_path='/private/file'), 400)]:
            with self.subTest(code=code), self.assertRaises(urllib.error.HTTPError) as caught:
                self.request('/api/observed/jobs', body, headers)
            self.assertEqual(caught.exception.code, code)
        self.assertEqual(self.request('/api/observed/catalog')[0], 200)
        self.assertEqual(self.request('/api/observed/jobs')[1]['jobs'], [])

    def test_shared_worker_conflict_start_poll_and_cancel(self):
        class Busy:
            id = 'other-worker'; state = {'status': 'running'}
            def cancel(self): pass
        self.server.active = Busy()
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request('/api/observed/jobs', self.body)
        self.assertEqual(caught.exception.code, 409)
        self.server.active = None
        with patch.object(jobs.ObservedRunner, 'execute', return_value=None):
            code, created = self.request('/api/observed/jobs', self.body)
        self.assertEqual(code, 202)
        self.assertEqual(self.request(created['status_url'])[1]['status'], 'queued')
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request('/api/observed/jobs', self.body)
        self.assertEqual(caught.exception.code, 409)
        self.assertEqual(self.request(created['status_url'] + '/cancel', {})[0], 202)
        state = self.request(created['status_url'])[1]
        self.assertTrue(state['cancellation_requested'])
        self.assertEqual(state['status'], 'queued')  # Holds the shared slot until worker exits.

    def test_explicit_gated_workflow_uses_same_http_worker_and_cancel_guards(self):
        for name in ['observed_pipeline.py', 'scripts/run_observed_workflow.py']:
            shutil.copyfile(ROOT / name, self.root / name)
        body = dict(self.body, workflow='gated-local-v1', feature_provider='sift', completion='none')
        with patch.object(jobs.ObservedRunner, 'execute', return_value=None):
            code, created = self.request('/api/observed/jobs', body)
        self.assertEqual(code, 202)
        state = self.request(created['status_url'])[1]
        self.assertEqual(state['workflow'], 'gated-local-v1')
        self.assertEqual(state['policy']['maximum_runtime_s'], 900)
        self.assertFalse(state['policy']['model_inference'])
        self.assertTrue(state['output_path'].endswith('-workflow'))
        self.assertEqual(state['stages'], [])
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request(created['status_url'] + '/cancel', {},
                {'Content-Type': 'application/json', 'Origin': 'https://attacker.example'})
        self.assertEqual(caught.exception.code, 403)
        self.assertFalse(self.request(created['status_url'])[1].get('cancellation_requested', False))
        self.assertEqual(self.request(created['status_url'] + '/cancel', {})[0], 202)
        self.assertTrue(self.request(created['status_url'])[1]['cancellation_requested'])


if __name__ == '__main__':
    unittest.main()
