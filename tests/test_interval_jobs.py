"""Registered boundary, source identity and concurrency; no model inference."""
import fcntl
import io
import json
from pathlib import Path
import threading
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch

import interval_jobs as jobs
from tests.test_observed_jobs import JobFixture, server_module


class MotionFixture(JobFixture):
    def setUp(self):
        super().setUp()
        folder = self.source.parent
        asset, baseline = folder / 'asset.glb', folder / 'baseline.json'
        asset.write_bytes(b'Explicit synthetic boundary fixture; never parsed as geometry')
        baseline.write_text('{"synthetic_contract_fixture":true}')
        self.doc = json.loads(self.source.read_text())
        self.doc['assets'] = [{'id': 'asset_1', 'track_id': 'part', 'path': asset.relative_to(self.root).as_posix(),
                              'sha256': jobs.sha(asset), 'backend': 'TripoSR',
                              'geometry_status': 'unreviewed_model_candidate', 'physical_scale_known': False}]
        row = next(r for r in self.doc['inputs'] if r['id'] == 'motion_1')
        row.update(path=baseline.relative_to(self.root).as_posix(), sha256=jobs.sha(baseline))
        self.source.write_text(json.dumps(self.doc))
        index = self.root / 'workspace/cases.json'
        value = json.loads(index.read_text()); value['cases'][0]['original_sha256'] = jobs.sha(self.source)
        index.write_text(json.dumps(value))
        for relative in ['interval_motion.py', 'scripts/stage4_interval_motion.py', 'interval_motion_search.py', 'motion_lab.py', 'models/cotracker3/scaled_offline.pth']:
            p = self.root / relative; p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('Synthetic boundary code/checkpoint path; never executed')

    def runner(self):
        return jobs.MotionRunner(jobs.canonical_request(self.body, root=self.root), root=self.root)

    def fake_spawn(self, runner, arguments, **changes):
        self.assertEqual(arguments[0], str(self.root / '.venv-sam/bin/python'))
        self.assertEqual(arguments[1], str(self.root / 'scripts/stage4_interval_motion.py'))
        self.assertEqual(arguments[2:6], ['--interaction', 'data/case/interaction.json', '--track', 'part'])
        # A separate descriptor must not be able to take the upload worker lock.
        with (self.root / 'data/flow-runs/.model-worker.lock').open('a+') as lock:
            with self.assertRaises(BlockingIOError):
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        runner.output.mkdir()
        result = {'format': 'short-interval-motion', 'version': 1, 'track_id': 'part',
                  'interaction_sha256': runner.request['interaction_sha256'],
                  'source_video_sha256': runner.request['source_video_sha256'],
                  'mesh_sha256': runner.request['asset_sha256'], 'new_api_calls': 0,
                  'physical_pose_accuracy_measured': False, 'usable_physical_pose_recovered': False,
                  'status': 'explicit_synthetic_test_no_usable_motion', 'episodes': []}
        result.update(changes)
        (runner.output / 'interval-motion.json').write_text(json.dumps(result))
        class Process:
            stdout = io.StringIO('{"phase":"synthetic_boundary_check"}\n')
            def wait(self): return 0
            def poll(self): return 0
        return Process()


class MotionJobTests(MotionFixture, unittest.TestCase):
    def test_browser_inputs_cannot_choose_paths_flags_commands_or_unregistered_objects(self):
        request = jobs.canonical_request(self.body, root=self.root)
        self.assertEqual(request['source_observations'], 3)
        self.assertEqual(request['baseline_supported_samples'], 1)
        for change in [{'command': 'anything'}, {'out': 'elsewhere'}, {'gpu': True}, {'audit': '/private/audit'},
                       {'track_id': '../escape'}, {'case_id': '../escape'}, {'track_id': 'unknown'}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                jobs.canonical_request(self.body | change, root=self.root)
        self.assertTrue(jobs.catalog(root=self.root)['runtime_available'])

    def test_published_hash_and_native_asset_hash_must_match(self):
        asset = self.root / self.doc['assets'][0]['path']; asset.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed'):
            jobs.canonical_request(self.body, root=self.root)

    def test_catalog_uses_one_attested_snapshot_but_admission_still_fully_validates(self):
        with patch.object(jobs.revisions, 'published_source', side_effect=AssertionError('catalog must not repeatedly parse embedded scene payloads')):
            display = jobs.catalog(root=self.root)
        self.assertTrue(display['cases'][0]['components'][0]['eligible'])
        with patch.object(jobs.revisions, 'published_source', side_effect=ValueError('full admission validation')):
            with self.assertRaisesRegex(ValueError, 'full admission validation'):
                jobs.canonical_request(self.body, root=self.root)
        self.source.write_text('{}')
        self.assertEqual(jobs.catalog(root=self.root)['cases'], [])

    def test_registered_cli_runs_under_shared_lock_and_normalizes_actual_result_only(self):
        runner = self.runner()
        with patch.object(runner, '_launch', side_effect=lambda args: self.fake_spawn(runner, args)), \
             patch('motion_lab.LOADED_SOURCE_SHA256', jobs.sha(self.root / 'motion_lab.py')), \
             patch('motion_lab.normalize', return_value={'synthetic_boundary_fixture': True}) as normalize:
            runner.execute()
        self.assertEqual(runner.state['status'], 'completed')
        normalize.assert_called_once_with(self.root, runner.output / 'interval-motion.json')
        self.assertEqual(runner.state['viewer_data_sha256'], jobs.sha(runner.output / 'viewer.json'))
        self.assertEqual(runner.state['policy']['new_api_calls'], 0)
        self.assertFalse(runner.state['policy']['physical_accuracy_verified'])

    def test_midrun_changes_and_stale_loaded_adapter_do_not_publish_a_viewer(self):
        for change in ['during_child', 'stale_adapter', 'during_normalize']:
            runner = self.runner()
            def spawn(args):
                process = self.fake_spawn(runner, args)
                if change == 'during_child':
                    (self.root / 'interval_motion.py').write_text('changed during child')
                return process
            def normalize(*args):
                if change == 'during_normalize':
                    (self.root / 'motion_lab.py').write_text('changed during normalization')
                return {'synthetic_boundary_fixture': True}
            digest = '0' * 64 if change == 'stale_adapter' else jobs.sha(self.root / 'motion_lab.py')
            with patch.object(runner, '_launch', side_effect=spawn), \
                 patch('motion_lab.LOADED_SOURCE_SHA256', digest), patch('motion_lab.normalize', side_effect=normalize):
                runner.execute()
            self.assertEqual(runner.state['status'], 'failed')
            self.assertTrue((runner.output / 'interval-motion.json').is_file())
            self.assertFalse((runner.output / 'viewer.json').exists())
            self.assertIsNone(runner.state['result_url'])

    def test_changed_implementation_or_cancellation_prevents_child_launch(self):
        for cancel in [False, True]:
            runner = self.runner()
            if cancel:
                runner.cancel()
            else:
                (self.root / 'interval_motion.py').write_text('changed after construction')
            with patch.object(runner, '_launch', side_effect=AssertionError('must not launch')):
                runner.execute()
            self.assertEqual(runner.state['status'], 'cancelled' if cancel else 'failed')

    def test_bad_result_source_or_physical_claim_is_preserved_but_never_published(self):
        for changes in [{'interaction_sha256': '0' * 64}, {'mesh_sha256': '0' * 64},
                        {'source_video_sha256': '0' * 64}, {'new_api_calls': 1},
                        {'physical_pose_accuracy_measured': True}, {'usable_physical_pose_recovered': True}]:
            runner = self.runner()
            with patch.object(runner, '_launch', side_effect=lambda args: self.fake_spawn(runner, args, **changes)):
                runner.execute()
            self.assertEqual(runner.state['status'], 'failed')
            self.assertTrue((runner.output / 'interval-motion.json').is_file())
            self.assertIsNone(runner.state['result_url'])

    def test_held_worker_lock_prevents_launch_in_another_operation(self):
        runner = self.runner()
        folder = self.root / 'data/flow-runs'; folder.mkdir()
        with (folder / '.model-worker.lock').open('a+') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch.object(runner, '_launch', side_effect=AssertionError('must not overlap')):
                runner.execute()
        self.assertEqual(runner.state['status'], 'failed')

    def test_get_marks_stale_worker_interrupted_without_mutation_or_retry(self):
        runner = self.runner(); runner.state['status'] = 'running'; runner.save()
        raw = (runner.directory / 'job.json').read_bytes()
        read = jobs.record(runner.id, root=self.root)
        self.assertEqual(read['status'], 'interrupted')
        self.assertFalse(read['automatic_retry'])
        self.assertEqual((runner.directory / 'job.json').read_bytes(), raw)
        self.assertEqual(jobs.record(runner.id, root=self.root, active_job_id=runner.id)['status'], 'running')
        for bad in ['../escape', 'interval-invalid', 'observed-' + 'a' * 24]:
            with self.assertRaises(ValueError): jobs.record(bad, root=self.root)


class MotionHTTPContracts(MotionFixture, unittest.TestCase):
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

    def test_real_http_origin_host_and_identifier_guards(self):
        for headers, body, expected in [
            ({'Content-Type': 'application/json', 'Origin': 'https://attacker.example'}, self.body, 403),
            ({'Content-Type': 'text/plain'}, self.body, 400),
            ({'Content-Type': 'application/json', 'Host': 'attacker.example'}, self.body, 403),
            ({'Content-Type': 'application/json'}, self.body | {'out': '/elsewhere'}, 400)]:
            with self.subTest(expected=expected), self.assertRaises(urllib.error.HTTPError) as caught:
                self.request('/api/motion/jobs', body, headers)
            self.assertEqual(caught.exception.code, expected)
        self.assertEqual(self.request('/api/motion/catalog')[0], 200)
        self.assertEqual(self.request('/api/motion/jobs')[1]['jobs'], [])

    def test_real_http_conflict_poll_and_cancellation_keep_shared_slot(self):
        class OtherWorker:
            id = 'other-worker'; state = {'status': 'running'}
        self.server.active = OtherWorker()
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request('/api/motion/jobs', self.body)
        self.assertEqual(caught.exception.code, 409)
        self.server.active = None
        with patch.object(jobs.MotionRunner, 'execute', return_value=None):
            status, created = self.request('/api/motion/jobs', self.body)
        self.assertEqual(status, 202)
        self.assertEqual(self.request(created['status_url'])[1]['status'], 'queued')
        for path, body, headers, expected in [
            ('/api/motion/jobs', self.body, None, 409),
            (created['status_url'] + '/cancel', {'unexpected': True}, None, 400),
            (created['status_url'] + '/cancel', {}, {'Content-Type': 'application/json', 'Origin': 'https://attacker.example'}, 403)]:
            with self.subTest(path=path), self.assertRaises(urllib.error.HTTPError) as caught:
                self.request(path, body, headers)
            self.assertEqual(caught.exception.code, expected)
        self.assertFalse(self.request(created['status_url'])[1].get('cancellation_requested', False))
        self.assertEqual(self.request(created['status_url'] + '/cancel', {})[0], 202)
        result = self.request(created['status_url'])[1]
        self.assertTrue(result['cancellation_requested'])
        self.assertEqual(result['status'], 'queued')
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request('/api/motion/jobs', self.body)
        self.assertEqual(caught.exception.code, 409)


if __name__ == '__main__': unittest.main()
