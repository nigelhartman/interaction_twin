import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pipeline as P

spec = importlib.util.spec_from_file_location('serve_workspace', ROOT / 'scripts/serve_workspace.py')
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


class PipelineContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / 'data')
        self.directory = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_profile_bounds_and_explicit_provider(self):
        for name in ['local_low', 'local_smoke', 'verified_evidence', 'cuda_preflight', 'meta_api_preflight']:
            profile = P.load_profile(name)
            self.assertEqual(profile['id'], name)
        with self.assertRaises(ValueError):
            P.load_profile('../secret')
        self.assertIn('hosted OpenAI', P.load_profile('local_low')['disclosure'])
        self.assertEqual(P.load_profile('local_low')['reconstruct_limit'], 3)
        self.assertEqual(P.load_profile('local_smoke')['reconstruct_limit'], 3)

    def selection_fixture(self, labels):
        specs, records, observations = [], [], []
        for index, (tid, label, score) in enumerate(labels):
            selected = self.directory / tid / 'selection.json'
            P.write(selected, {'selected': [{'score': score}]})
            specs.append({'id': tid, 'label': label, 'identity_status': 'unverified_after_occlusion',
                          'candidate_entity_ids': [tid + '_candidate']})
            records.append({'track_id': tid, 'selection': str(selected.relative_to(self.directory))})
            observations.append({'track_id': tid, 'frames': []})
        return records, {'tracks': specs}, {'tracks': observations}

    def test_selection_prioritizes_diversity_without_merging_uncertain_same_category(self):
        records, specs, tracks = self.selection_fixture([
            ('earbud_a', 'left earbud', .99), ('earbud_b', 'second right earbud', .98),
            ('lid', 'container lid', .8), ('base', 'container base', .7)])
        selected, receipt = P.choose_components(self.directory, records, specs, tracks, 3)
        self.assertEqual([row['track_id'] for row in selected], ['earbud_a', 'lid', 'base'])
        deferred = next(row for row in receipt['decisions'] if row['track_id'] == 'earbud_b')
        self.assertEqual(deferred['status'], 'skipped_budget')
        self.assertFalse(deferred['physical_identity_merged'])
        self.assertEqual(deferred['candidate_entity_ids'], ['earbud_b_candidate'])
        selected, receipt = P.choose_components(self.directory, records[:2], specs, tracks, 3)
        self.assertEqual([row['track_id'] for row in selected], ['earbud_a', 'earbud_b'])
        self.assertFalse(receipt['physical_identities_merged'])

    def test_four_native_parts_fit_the_default_budget_without_merging_identical_labels(self):
        records, specs, tracks = self.selection_fixture([
            ('earbud_a', 'left earbud', .99), ('earbud_b', 'second earbud', .98),
            ('lid', 'container lid', .8), ('base', 'container base', .7)])
        selected, receipt = P.choose_components(self.directory, records, specs, tracks, 4)
        self.assertEqual({r['track_id'] for r in selected}, {'earbud_a','earbud_b','lid','base'})
        self.assertFalse(receipt['physical_identities_merged'])
        self.assertEqual(P.load_profile('meta_sam3d_default')['reconstruct_limit'], 4)
        for invalid in [True, 4.5, 0, 9]:
            with self.assertRaises(ValueError):
                P.choose_components(self.directory, records, specs, tracks, invalid)

    def test_real_temporal_spatial_tracklet_labels_do_not_consume_whole_component_budget(self):
        # Regression from a fresh generic model discovery, not category hints.
        records, specs, tracks = self.selection_fixture([
            ('container_body', 'container body', .864965),
            ('container_lid', 'movable lid', .907710),
            ('left_piece_initial', 'initial image-left small piece', .936520),
            ('right_piece_initial', 'initial image-right small piece', .916349),
            ('loose_piece_a', 'first loose small piece', .927883),
            ('loose_piece_b', 'second loose small piece', .903588),
            ('left_piece_later', 'later image-left recessed piece', .900316),
            ('right_piece_later', 'later image-right recessed piece', .918679),
        ])
        selected, receipt = P.choose_components(self.directory, records, specs, tracks, 3)
        self.assertEqual([row['track_id'] for row in selected], ['left_piece_initial', 'container_lid', 'container_body'])
        pieces = [row for row in receipt['decisions'] if 'piece' in row['track_id']]
        self.assertEqual({row['category_key'] for row in pieces}, {'piece'})
        self.assertTrue(all(not row['physical_identity_merged'] for row in pieces))
        self.assertTrue(all(row['status'] == 'skipped_budget' for row in pieces if row['track_id'] != 'left_piece_initial'))
        self.assertNotEqual(P.component_category('upper housing'), P.component_category('lower housing'))

    def test_selection_suppresses_observed_duplicates_but_retains_original_tracklets(self):
        from PIL import Image
        records, specs, tracks = self.selection_fixture([
            ('earbud_a', 'left earbud', .95), ('earbud_b', 'second earbud', .9),
            ('lid', 'container lid', .85)])
        segmentation = self.directory / 'segmentation/test'
        segmentation.mkdir(parents=True)
        for observation in tracks['tracks'][:2]:
            for sample in [1, 2, 3, 4, 5]:
                mask = segmentation / f"{observation['track_id']}-{sample}.png"
                image = Image.new('L', (20, 20))
                image.paste(255, (5, 5, 15, 15)); image.save(mask)
                observation['frames'].append({'sample_index': sample, 'timestamp_s': sample * .5,
                                             'status': 'candidate_visible', 'raw_mask_file': mask.name})
        original = json.dumps(tracks)
        selected, receipt = P.choose_components(self.directory, records, specs, tracks, 3, segmentation=segmentation)
        self.assertEqual([row['track_id'] for row in selected], ['earbud_a', 'lid'])
        duplicate = next(row for row in receipt['decisions'] if row['track_id'] == 'earbud_b')
        self.assertEqual(duplicate['status'], 'skipped_redundant_observation')
        self.assertEqual(len(duplicate['redundant_with']['checks']), 3)
        self.assertEqual(json.dumps(tracks), original)
        self.assertFalse(duplicate['physical_identity_merged'])

    def test_selection_keeps_distinct_same_category_masks_and_reports_no_view(self):
        from PIL import Image
        records, specs, tracks = self.selection_fixture([('a', 'earbud', .8), ('b', 'earbud', .8)])
        segmentation = self.directory / 'segmentation/test'
        segmentation.mkdir(parents=True)
        for index, observation in enumerate(tracks['tracks']):
            for sample in [1, 2]:
                mask = segmentation / f"{observation['track_id']}-{sample}.png"
                image = Image.new('L', (30, 20)); image.paste(255, (index*15, 0, index*15+10, 10)); image.save(mask)
                observation['frames'].append({'sample_index': sample, 'timestamp_s': sample,
                                             'status': 'candidate_visible', 'raw_mask_file': mask.name})
        records.append({'track_id': 'no_view', 'status': 'resolution_gate_failed'})
        specs['tracks'].append({'id': 'no_view', 'label': 'container'})
        selected, receipt = P.choose_components(self.directory, records, specs, tracks, 3, segmentation=segmentation)
        self.assertEqual([row['track_id'] for row in selected], ['a', 'b'])
        self.assertEqual(receipt['decisions'][-1]['status'], 'skipped_no_eligible_view')

    def test_component_failure_preserves_success_logs_and_partial_failure_cache(self):
        runner = P.Runner('component-isolation', run_root=self.directory / 'runs')
        runner.profile = P.load_profile('local_smoke')
        dependency = self.directory / 'dependency.txt'; dependency.write_text('first')
        calls = []
        def action(component, directory):
            calls.append(component['track_id']); directory.mkdir()
            (directory / 'raw.txt').write_text(component['track_id'])
            with (runner.directory / 'pipeline.log').open('a') as log: log.write(component['track_id'] + ' real attempt\n')
            if component['track_id'] == 'broken': raise RuntimeError('model failed')
            return {'model': P.rel(directory / 'raw.txt')}
        def batch(attempt):
            return runner.component_batch('model', attempt, self.directory,
                                          [{'track_id': 'good'}, {'track_id': 'broken'}, {'track_id': 'later'}], action)
        result = runner.phase('reconstruct', [dependency], batch, optional=True)
        self.assertEqual([row['track_id'] for row in result['components'] if 'output' in row], ['good', 'later'])
        self.assertEqual(result['failures'][0]['track_id'], 'broken')
        self.assertTrue((self.directory / 'flow-model-1-2/raw.txt').is_file())
        self.assertIn('broken real attempt', (P.ROOT / result['failures'][0]['log']).read_text())
        self.assertTrue(result['has_gaps'])
        self.assertEqual(runner.state['phases'][0]['status'], 'complete_with_gaps')
        runner.phase('reconstruct', [dependency], batch, optional=True)
        self.assertEqual(calls, ['good', 'broken', 'later'])
        self.assertEqual(runner.state['phases'][0]['status'], 'reused_verified_with_gaps')
        # Tampering with either a failed artifact or success invalidates the complete batch receipt.
        (self.directory / 'flow-model-1-2/raw.txt').write_text('changed')
        runner.phase('reconstruct', [dependency], batch, optional=True)
        self.assertEqual(calls, ['good', 'broken', 'later'] * 2)
        self.assertEqual(runner.state['phases'][0]['attempt'], 2)

    def test_component_cancellation_propagates_and_never_attempts_sibling(self):
        runner = P.Runner('component-cancel', run_root=self.directory / 'runs')
        calls = []
        def action(component, directory):
            calls.append(component['track_id']); directory.mkdir()
            (directory / 'partial.txt').write_text('partial model artifact')
            runner.cancel()
            raise P.Cancelled('stop')
        with self.assertRaises(P.Cancelled):
            runner.component_batch('model', 1, self.directory,
                                   [{'track_id': 'first'}, {'track_id': 'never'}], action)
        self.assertEqual(calls, ['first'])
        report = P.read(self.directory / 'flow-model-receipts-1/components.json')
        self.assertEqual(report['components'][0]['status'], 'cancelled')
        self.assertNotIn('output', report['components'][0])
        self.assertTrue((self.directory / 'flow-model-1-1/partial.txt').is_file())

    def test_infer_keeps_native_models_after_sibling_reconstruction_and_pose_failures(self):
        """Exercise the actual phase closures, with model subprocesses replaced by fixtures."""
        runner = P.Runner('infer-isolation', run_root=self.directory / 'runs')
        runner.profile = P.load_profile('local_smoke')
        runner.state['provenance'] = {}
        case = self.directory / 'case'; case.mkdir()
        config = case / 'annotations.json'; P.write(config, {'tracks': []})
        components = [{'track_id': tid, 'selection': P.rel(case / (tid + '-selection.json'))}
                      for tid in ['fails_reconstruction', 'fails_pose', 'works']]
        for component in components:
            P.write(P.ROOT / component['selection'], {'object_id': component['track_id']})
        called = []
        actual_results = {}
        def phase(name, dependencies, action, **kwargs):
            if name == 'prepare': return {'case': P.rel(case)}
            if name == 'discover': return {'annotations': P.rel(config), 'discovery': P.rel(case / 'discovery.json')}
            if name == 'segment': return {'directory': P.rel(case / 'segmentation/test'), 'run': 'test'}
            if name == 'audit': return {'directory': P.rel(case / 'audit')}
            if name == 'select': return {'selections': [row['selection'] for row in components], 'components': components}
            if name in {'reconstruct', 'pose', 'motion'}:
                result, _ = action(1); actual_results[name] = result; return result
            return {}
        def command(args, timeout):
            script = Path(args[1]).name
            args = [str(item) for item in args[2:]]
            flag = '--output' if '--output' in args else '--out'
            directory = Path(args[args.index(flag)+1])
            if script == 'stage4_generated_observer.py':
                directory.write_text('diagnostic observer'); return
            directory.mkdir()
            input_flag = '--selection' if script == 'stage3_reconstruct_local.py' else '--manifest' if script == 'stage4_generated_pose.py' else '--reference'
            source = P.read(Path(args[args.index(input_flag)+1])); tid = source['object_id']
            called.append((script, tid))
            if script == 'stage3_reconstruct_local.py':
                if tid == 'fails_reconstruction':
                    (directory / 'partial.txt').write_text('unusable attempt retained')
                    raise RuntimeError('reconstruction failed')
                (directory / 'asset.glb').write_text('unit fixture learned asset')
                P.write(directory / 'metadata.json', {'object_id': tid, 'status': 'candidate_generated_pending_visual_review',
                                                     'outputs': {'glb_path': 'asset.glb'}})
            elif script == 'stage4_generated_pose.py':
                if tid == 'fails_pose': raise RuntimeError('pose failed')
                P.write(directory / 'experiment.json', {'object_id': tid, 'status': 'rejected_as_usable_pose_single_view_candidate_only',
                                                       'candidate': {'rotation': 'fixture'}, 'usable_pose_gate': {'accepted': False}})
            else:
                P.write(directory / 'experiment.json', {'object_id': tid, 'status': 'complete_conditional_tracking_experiment',
                                                       'supported_nonreference_frames': 2})
        with patch.object(runner, 'phase', side_effect=phase), patch.object(runner, 'command', side_effect=command):
            runner._infer(case / 'input.mp4', {'sha256': 'fixture'})
        self.assertEqual(len(runner.state['artifacts']['models']), 2)
        self.assertEqual(len(runner.state['artifacts']['motion']), 1)
        self.assertEqual(actual_results['reconstruct']['failures'][0]['track_id'], 'fails_reconstruction')
        self.assertEqual(actual_results['pose']['failures'][0]['track_id'], 'fails_pose')
        conditional = next(row['output'] for row in actual_results['pose']['components'] if row['track_id'] == 'works')
        self.assertFalse(conditional['physical_pose_accepted'])
        self.assertTrue(conditional['native_asset_retained'])
        self.assertNotIn(('stage4_track_generated.py', 'fails_pose'), called)
        failed_pose_asset = next(path for path in runner.state['artifacts']['models'] if P.read(P.ROOT / path)['object_id'] == 'fails_pose')
        self.assertTrue((P.ROOT / failed_pose_asset).with_name('asset.glb').is_file())

    def test_receipts_detect_changed_output_and_symlink_escape(self):
        path = self.directory / 'artifact.json'
        path.write_text('original')
        receipt = P.file_receipts([path])
        self.assertTrue(P.verify_receipts(receipt))
        path.write_text('modified')
        self.assertFalse(P.verify_receipts(receipt))
        (self.directory / 'escape').symlink_to('/etc/hosts')
        with self.assertRaises(ValueError):
            P.file_receipts([self.directory])

    def test_cache_reuses_only_unchanged_deps_and_outputs(self):
        dependency = self.directory / 'dependency.txt'
        dependency.write_text('one')
        runner = P.Runner('test-cache', run_root=self.directory / 'runs')
        runner.profile = P.load_profile('verified_evidence')
        count = []
        def action(attempt):
            count.append(attempt)
            out = self.directory / f'output-{attempt}.json'
            out.write_text(str(attempt))
            return {'attempt': attempt}, [out]
        runner.phase('prepare', [dependency], action)
        runner.phase('prepare', [dependency], action)
        self.assertEqual(count, [1])
        self.assertEqual(runner.state['phases'][0]['status'], 'reused_verified')
        dependency.write_text('two')
        runner.phase('prepare', [dependency], action)
        self.assertEqual(count, [1, 2])
        (self.directory / 'output-2.json').write_text('tampered')
        runner.phase('prepare', [dependency], action)
        self.assertEqual(count, [1, 2, 3])

    def test_failed_optional_phase_does_not_fabricate_result(self):
        dep = self.directory / 'in.txt'
        dep.write_text('data')
        runner = P.Runner('test-failure', run_root=self.directory / 'runs')
        runner.profile = P.load_profile('verified_evidence')
        def fail(attempt):
            raise RuntimeError('real model failed')
        result = runner.phase('reconstruct', [dep], fail, optional=True)
        self.assertEqual(result, {'unavailable': 'real model failed'})
        self.assertEqual(runner.state['phases'][0]['status'], 'failed')
        self.assertNotIn('models', result)

    def test_missing_optional_model_dependency_stays_explicit_gap(self):
        runner = P.Runner('test-missing-model', run_root=self.directory / 'runs')
        runner.profile = P.load_profile('verified_evidence')
        result = runner.phase('reconstruct', [self.directory / 'missing-weights.pt'], lambda attempt: self.fail('Should not fabricate model output'), optional=True)
        self.assertIn('Dependency unavailable', result['unavailable'])
        self.assertEqual(runner.state['phase'], 'reconstruct')
        self.assertEqual(runner.state['phases'][0]['status'], 'failed')

    def test_command_timeout_terminates_subprocess_group(self):
        runner = P.Runner('test-timeout', run_root=self.directory / 'runs')
        start = time.monotonic()
        with self.assertRaises(TimeoutError):
            runner.command([sys.executable, '-c', 'import time;time.sleep(60)'], .2)
        self.assertLess(time.monotonic() - start, 5)
        self.assertIsNotNone(runner.process.poll())

    def test_cancel_before_command_runs_nothing(self):
        runner = P.Runner('test-cancel', run_root=self.directory / 'runs')
        runner.cancel()
        with self.assertRaises(P.Cancelled):
            runner.command([sys.executable, '-c', 'raise RuntimeError()'], 2)
        self.assertIsNone(runner.process)

    def test_active_command_cancelled_and_reaped(self):
        runner = P.Runner('test-active-cancel', run_root=self.directory / 'runs')
        errors = []
        def execute():
            try:
                runner.command([sys.executable, '-c', 'import time;time.sleep(60)'], 10)
            except Exception as exc:
                errors.append(exc)
        worker = threading.Thread(target=execute)
        worker.start()
        deadline = time.monotonic() + 2
        while runner.process is None and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertIsNotNone(runner.process)
        runner.cancel()
        worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        self.assertIsInstance(errors[0], P.Cancelled)
        self.assertIsNotNone(runner.process.poll())

    def test_timeout_stops_child_heartbeat_too(self):
        heartbeat = self.directory / 'heartbeat.txt'
        child = 'import pathlib,time;p=pathlib.Path(' + repr(str(heartbeat)) + ');\nwhile True:\n p.write_text(str(time.time()));time.sleep(.03)'
        parent = 'import subprocess,sys,time;subprocess.Popen([sys.executable,"-c",' + repr(child) + ']);time.sleep(60)'
        runner = P.Runner('test-child-timeout', run_root=self.directory / 'runs')
        with self.assertRaises(TimeoutError):
            runner.command([sys.executable, '-c', parent], .4)
        before = heartbeat.read_text()
        time.sleep(.12)
        self.assertEqual(heartbeat.read_text(), before)

    def test_source_duration_and_frame_budget(self):
        path = self.directory / 'clip.mp4'
        path.write_bytes(b'not actually decoded in this unit fixture')
        response = subprocess.CompletedProcess([], 0, json.dumps({'format': {'duration': '10000'}, 'streams': [{'codec_type': 'video'}]}), '')
        with patch('pipeline.subprocess.run', return_value=response):
            with self.assertRaisesRegex(ValueError, 'budget'):
                P.validate_source(path, P.load_profile('local_smoke'))

    def test_static_paths_and_origin(self):
        for path in ['/files/../pipeline.py', '/files/data/../../pipeline.py', '/files/.venv/secrets', '/workspace/../pipeline.py', '/showcase/../pipeline.py', '/showcase/%2e%2e/pipeline.py', '/files/data/%2e%2e/%2e%2e/pipeline.py']:
            with self.assertRaises(ValueError, msg=path):
                S.static_path(path)
        self.assertTrue(S.valid_origin('http://127.0.0.1:8765', '127.0.0.1:8765'))
        self.assertFalse(S.valid_origin('https://attacker.example', '127.0.0.1:8765'))
        self.assertFalse(S.valid_origin('null', '127.0.0.1:8765'))
        with self.assertRaises(ValueError):
            S.upload_name('../clip.mp4')
        with self.assertRaises(ValueError):
            S.upload_name('script.py')

    def test_showcase_static_paths_are_confined_without_changing_workspace(self):
        self.assertEqual(S.static_path('/'),ROOT/'showcase/index.html')
        self.assertEqual(S.static_path('/showcase'),ROOT/'showcase/index.html')
        self.assertEqual(S.static_path('/showcase/'),ROOT/'showcase/index.html')
        self.assertEqual(S.static_path('/showcase/app.js'),ROOT/'showcase/app.js')
        self.assertEqual(S.static_path('/workspace'),ROOT/'workspace/index.html')
        self.assertEqual(S.static_path('/workspace/index.html'),ROOT/'workspace/index.html')
        root=self.directory/'routes'
        (root/'showcase').mkdir(parents=True)
        (root/'showcase/escape').symlink_to('/etc')
        with self.assertRaises(ValueError):
            S.static_path('/showcase/escape/hosts',root=root)

    def test_showcase_entry_redirect_keeps_relative_assets_on_showcase_route(self):
        for path in ['/', '/showcase', '/showcase/']:
            with self.subTest(path=path):
                handler=S.Handler.__new__(S.Handler)
                handler.path=path
                handler.headers={'Host':'127.0.0.1:8765'}
                codes=[];headers={}
                handler.send_response=codes.append
                handler.send_header=lambda key,value:headers.update({key:value})
                handler.end_headers=lambda:None
                with patch.object(handler,'serve_file',side_effect=AssertionError('Entry must redirect before serving relative assets')):
                    handler._read(False)
                self.assertEqual(codes,[302])
                self.assertEqual(headers['Location'],'/showcase/index.html')
                self.assertEqual(headers['Content-Length'],'0')

    def test_cancelled_video_range_never_sends_second_response(self):
        path = self.directory / 'range-test.mp4'
        path.write_bytes(bytes(range(256)) * 8)
        for disconnect in [BrokenPipeError, ConnectionResetError]:
            with self.subTest(disconnect=disconnect.__name__):
                class DisconnectedWriter:
                    def __init__(self):
                        self.attempts = []
                    def write(self, data):
                        self.attempts.append(bytes(data))
                        raise disconnect('browser cancelled old media range')
                handler = S.Handler.__new__(S.Handler)
                handler.path = P.public_url(path)
                from types import SimpleNamespace
                handler.server=SimpleNamespace(admin_auth=SimpleNamespace(authenticated=lambda _:True))
                handler.headers = {'Host': '127.0.0.1:8765', 'Range': 'bytes=5-128'}
                codes = []
                handler.send_response = codes.append
                handler.send_header = lambda *args: None
                handler.end_headers = lambda: None
                handler.wfile = DisconnectedWriter()
                with patch.object(handler, 'json', side_effect=AssertionError('Must not send a second response after client disconnect')):
                    handler._read(False)
                self.assertEqual(codes, [206])
                self.assertEqual(handler.wfile.attempts, [path.read_bytes()[5:129]])

    def test_preflight_does_not_call_model_or_network(self):
        runner = P.Runner('test-preflight', run_root=self.directory / 'runs')
        with patch.object(runner, 'command', side_effect=AssertionError('no subprocess expected')):
            result = runner.execute('cuda_preflight')
        self.assertEqual(result['status'], 'complete')
        self.assertFalse(P.read(runner.directory / 'preflight.json')['cluster_contacted'])
        snapshot = result['provenance']['orchestrator_source']
        self.assertEqual(P.sha(ROOT / snapshot['path']), P.LOADED_SOURCE_SHA256)

    def test_cli_interrupt_marks_run_cancelled(self):
        runner = P.Runner('test-interrupt', run_root=self.directory / 'runs')
        source = self.directory / 'fake-video.mp4'
        source.write_bytes(b'fixture')
        with patch('pipeline.validate_source', return_value={'path': str(source), 'sha256': P.sha(source)}), \
             patch.object(runner, '_infer', side_effect=KeyboardInterrupt):
            result = runner.execute('local_smoke', input_path=source)
        self.assertEqual(result['status'], 'cancelled')


class LocalHTTPContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = S.WorkspaceServer(('127.0.0.1', 0), S.Handler)
        from tests.owner_http_fixture import OwnerHTTPSession
        cls.owner_http=OwnerHTTPSession(cls.server)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)
        cls.owner_http.close()

    def test_health_and_config(self):
        with urllib.request.urlopen(self.url + '/health') as response:
            self.assertEqual(json.load(response)['status'], 'ready')
        with urllib.request.urlopen(self.url + '/api/config') as response:
            self.assertEqual(len(json.load(response)['profiles']), len(list((ROOT / 'profiles').glob('*.json'))))

    def test_real_owner_cookie_login_rotation_and_logout_protect_private_config(self):
        # A separate browser-shaped cookie jar starts anonymously; it bypasses
        # only the legacy fixture's request-header convenience wrapper.
        import http.cookiejar
        opener=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        with opener.open(self.url+'/api/admin/auth') as response:
            status=json.load(response);before=response.headers.get('Set-Cookie')
        self.assertFalse(status['authenticated']);self.assertIn('HttpOnly',before);self.assertIn('SameSite=Strict',before)
        with self.assertRaises(urllib.error.HTTPError) as denied:opener.open(self.url+'/api/config')
        self.assertEqual(denied.exception.code,401)
        request=urllib.request.Request(self.url+'/api/admin/login',data=json.dumps({'password':'Synthetic HTTP test owner password'}).encode(),
            headers={'Origin':self.url,'X-CSRF-Token':status['csrf_token'],'Content-Type':'application/json'})
        with opener.open(request) as response:
            owner=json.load(response);self.assertNotEqual(response.headers.get('Set-Cookie'),before)
        self.assertTrue(owner['authenticated']);self.assertNotEqual(owner['csrf_token'],status['csrf_token'])
        with opener.open(self.url+'/api/config') as response:self.assertEqual(response.status,200)
        request=urllib.request.Request(self.url+'/api/admin/logout',data=b'{}',headers={'Origin':self.url,
            'X-CSRF-Token':owner['csrf_token'],'Content-Type':'application/json'})
        with opener.open(request) as response:self.assertIn('Max-Age=0',response.headers.get('Set-Cookie'))
        with self.assertRaises(urllib.error.HTTPError) as denied:opener.open(self.url+'/api/config')
        self.assertEqual(denied.exception.code,401)

    def test_showcase_and_workspace_are_both_served_over_local_http(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'data') as temporary:
            root=Path(temporary)
            for folder in ['showcase','workspace']:(root/folder).mkdir()
            (root/'showcase/index.html').write_text('<title>Showcase route fixture</title>')
            (root/'showcase/style.css').write_text('body{color:navy}')
            (root/'workspace/index.html').write_text('<title>Workspace route fixture</title>')
            actual=S.static_path
            with patch.object(S,'static_path',side_effect=lambda route:actual(route,root=root)):
                with urllib.request.urlopen(self.url+'/') as response:
                    self.assertTrue(response.geturl().endswith('/showcase/index.html'))
                    self.assertEqual(response.read(),(root/'showcase/index.html').read_bytes())
                for relative in ['showcase/style.css','workspace/index.html']:
                    with urllib.request.urlopen(self.url+'/'+relative) as response:
                        self.assertEqual(response.read(),(root/relative).read_bytes())
                with self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(self.url+'/showcase/%2e%2e/pipeline.py')
                self.assertEqual(error.exception.code,404)

    def test_api_starts_real_preflight_and_reports_completion(self):
        request = urllib.request.Request(self.url + '/api/run', data=b'{"profile":"cuda_preflight"}', headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request) as response:
            result = json.load(response)
            self.assertEqual(response.status, 202)
        try:
            for _ in range(40):
                with urllib.request.urlopen(self.url + result['status_url']) as response:
                    state = json.load(response)
                if state['status'] == 'complete':
                    break
                time.sleep(.025)
            self.assertEqual(state['status'], 'complete')
            self.assertFalse(state['provenance']['cluster_contacted'])
            self.assertIn('preflight', state['artifacts'])
        finally:
            shutil.rmtree(ROOT / 'data/flow-runs' / result['id'])

    def test_cross_origin_write_and_missing_consent_rejected(self):
        for headers, data, expected in [({'Origin': 'https://attacker.example', 'Content-Type': 'application/json'}, {}, 403),
                                        ({'Content-Type': 'application/json'}, {'profile': 'local_low', 'case_id': 'anything'}, 400)]:
            request = urllib.request.Request(self.url + '/api/run', data=json.dumps(data).encode(), headers=headers)
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(request)
            self.assertEqual(error.exception.code, expected)

    def test_bounded_upload_and_single_active_worker(self):
        request = urllib.request.Request(self.url + '/api/upload', data=b'local-test-video-header',
                                         headers={'X-Filename': 'unit-test.mp4', 'Content-Type': 'application/octet-stream'})
        with urllib.request.urlopen(request) as response:
            result = json.load(response)
            self.assertEqual(response.status, 201)
        path = P.confined(ROOT / result['input_path'], ROOT / 'data/uploads')
        self.assertEqual(path.read_bytes(), b'local-test-video-header')
        path.unlink()
        path.parent.rmdir()
        class Busy:
            id = 'unit-busy'
            state = {'status': 'running'}
        previous = self.server.active
        self.server.active = Busy()
        try:
            request = urllib.request.Request(self.url + '/api/run', data=b'{"profile":"cuda_preflight"}', headers={'Content-Type': 'application/json'})
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(request)
            self.assertEqual(error.exception.code, 409)
        finally:
            self.server.active = previous

    def test_video_byte_range_and_traversal(self):
        path = ROOT / 'data/wallet-stage6-v1/procedure-v1/interaction.json'
        if not path.is_file():
            self.skipTest('Actual wallet artifact not available')
        relative = str(path.relative_to(ROOT))
        request = urllib.request.Request(self.url + '/files/' + relative, headers={'Range': 'bytes=0-9'})
        with urllib.request.urlopen(request) as response:
            self.assertEqual(response.status, 206)
            self.assertEqual(response.read(), path.read_bytes()[:10])
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(self.url + '/files/data/%2e%2e/pipeline.py')
        self.assertEqual(error.exception.code, 404)


if __name__ == '__main__':
    unittest.main()
