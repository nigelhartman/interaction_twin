"""Offline queue/admission/security tests; no SSH, GPU or learned models run."""
import base64
import copy
import json
import shlex
import threading
import tempfile
from pathlib import Path
import unittest
from unittest import mock

import sam3d_cluster_bridge as B
import upgrade_bundles as U
import tests.test_initial_sam3d_handoff as HandoffFixtures
import tests.test_initial_sam3d_import as NativeFixtures


class ClusterBridgeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = HandoffFixtures.InitialHandoffTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.prepare()
        self.root = self.fixture.root
        (self.root / 'config').mkdir()
        self.handoff = self.root / self.fixture.output / 'handoff.json'
        self.destination = self.root / 'data/new-upload/native'
        self.config = {'version': 1, 'enabled': True, 'ssh_control_path': '/private/tmp/task-test/cm-%C',
                       'maximum_parallel_workers': 2, 'poll_seconds': 1, 'wait_seconds': 0}

    def enable(self):
        U.write(self.root / B.CONFIG_PATH, self.config)

    def run_bridge(self, transport=None, destination=None, **kwargs):
        return B.reconstruct(self.handoff, destination or self.destination, root=self.root,
                             transport_factory=lambda config: transport, **kwargs)

    def test_no_worker_preserves_source_and_returns_real_queue_without_model(self):
        before = U.sha(self.handoff)
        result = self.run_bridge()
        self.assertEqual(result['status'], 'awaiting_gpu_worker')
        self.assertFalse(result['cluster_contacted'])
        self.assertFalse(result['remote_job_submitted'])
        self.assertNotIn('model', result)
        self.assertEqual(U.sha(self.handoff), before)
        self.assertFalse(self.destination.exists())

    def test_missing_ssh_resumes_same_identity_across_new_phase_destination(self):
        self.enable()
        unavailable = mock.Mock()
        unavailable.contacted = False
        unavailable.request.side_effect = B.WorkerUnavailable('no connection')
        one = self.run_bridge(unavailable)
        two = self.run_bridge(unavailable, self.root / 'data/new-phase/native')
        self.assertEqual(one['job_id'], two['job_id'])
        self.assertEqual(one['receipt'], two['receipt'])
        self.assertFalse(two['cluster_contacted'])
        self.assertNotIn('model', two)

    def test_running_dispatch_is_source_bound_and_resumable_without_new_job(self):
        self.enable()
        worker = mock.Mock()
        worker.request.side_effect = [{'status': 'ready', 'parent_allocation_modified': False}, {'status': 'running'}]
        one = self.run_bridge(worker)
        self.assertEqual(one['status'], 'awaiting_gpu_worker')
        self.assertTrue(one['remote_job_submitted'])
        request = worker.request.call_args_list[1].args[0]
        self.assertEqual(request['handoff_sha256'], U.sha(self.handoff))
        self.assertEqual({row['path'] for row in request['files']}, B.H.FILES | {'handoff.json'})
        for row in request['files']:
            raw = base64.b64decode(row['base64'])
            self.assertEqual(U.digest(raw), row['sha256'])
        worker.request.side_effect = [{'status': 'ready', 'parent_allocation_modified': False}, {'status': 'running'}]
        two = self.run_bridge(worker, self.root / 'data/another-destination/native')
        self.assertEqual(one['job_id'], two['job_id'])

    def test_failed_attempt_is_terminal_even_under_new_destination(self):
        self.enable()
        worker = mock.Mock()
        worker.request.side_effect = [{'status': 'ready', 'parent_allocation_modified': False}, {'status': 'failed_no_retry'}]
        one = self.run_bridge(worker)
        self.assertEqual(one['status'], 'failed_no_retry')
        worker.request.reset_mock()
        two = self.run_bridge(worker, self.root / 'data/next-phase/native')
        worker.request.assert_not_called()
        self.assertEqual(one['receipt'], two['receipt'])
        self.assertFalse(two['cluster_contacted'])

    def test_source_changed_before_dispatch_is_rejected_without_worker_contact(self):
        self.enable()
        self.fixture.video.write_bytes(b'changed source')
        worker = mock.Mock()
        with self.assertRaisesRegex(ValueError, 'source video'):
            self.run_bridge(worker)
        worker.request.assert_not_called()

    def test_cancelled_local_wait_never_cancels_or_contacts_worker(self):
        self.enable()
        event = threading.Event()
        event.set()
        worker = mock.Mock()
        result = self.run_bridge(worker, cancel_event=event)
        self.assertEqual(result['status'], 'awaiting_gpu_worker')
        worker.request.assert_not_called()

    def test_only_two_global_worker_slots_can_be_claimed(self):
        directory = self.root / '.state/workers'
        with B.worker_slot(directory, 2) as first, B.worker_slot(directory, 2) as second, B.worker_slot(directory, 2) as third:
            self.assertTrue(first)
            self.assertTrue(second)
            self.assertFalse(third)
        with B.worker_slot(directory, 2) as available:
            self.assertTrue(available)

    def test_configuration_cannot_add_credentials_hosts_or_resource_overrides(self):
        for key, value in [('token', 'do not use'), ('ssh_host', 'elsewhere'), ('maximum_parallel_workers', 3),
                           ('ssh_control_path', '/Users/account/.ssh/key'), ('wait_seconds', 99999)]:
            bad = dict(self.config, **{key: value})
            U.write(self.root / B.CONFIG_PATH, bad)
            with self.subTest(key=key), self.assertRaises(ValueError):
                B.configuration(self.root)

    def test_ssh_master_missing_never_attempts_a_new_connection(self):
        transport = B.SSHTransport(self.config)
        with mock.patch.object(B.subprocess, 'run', return_value=mock.Mock(returncode=255)) as run:
            with self.assertRaises(B.WorkerUnavailable):
                transport.request({'action': 'probe'})
        self.assertEqual(run.call_count, 1)
        args = run.call_args.args[0]
        self.assertIn('ProxyCommand=false', args)
        self.assertIn('BatchMode=yes', args)
        self.assertEqual(args[-3:], ['-O', 'check', 'KISSKI'])

    def test_return_transfer_rejects_unexpected_inventory_before_any_write(self):
        worker = mock.Mock()
        worker.request.return_value = {'status': 'native_output_ready', 'files': [{'path': '../outside', 'bytes': 1, 'sha256': 'a'*64}]}
        directory = self.root / 'data/return-contract'
        directory.mkdir()
        with self.assertRaisesRegex(ValueError, 'inventory'):
            B.fetch(worker, directory, 'upload-test')
        self.assertFalse((directory / 'returned').exists())

    def test_helper_failure_is_distinguished_from_missing_control_socket_without_raw_output(self):
        transport = B.SSHTransport(self.config)
        responses = [mock.Mock(returncode=0), mock.Mock(returncode=1, stdout=b'',
                    stderr=b'SyntaxError: future feature annotations is not defined\nDO_NOT_EXPOSE_RAW_OUTPUT')]
        with mock.patch.object(B.subprocess, 'run', side_effect=responses):
            with self.assertRaises(B.WorkerUnavailable) as caught:
                transport.request({'action': 'probe'})
        self.assertEqual(caught.exception.diagnostic['phase'], 'remote_helper_response')
        self.assertEqual(caught.exception.diagnostic['classification'], 'remote_python_syntax_error')
        self.assertNotIn('DO_NOT_EXPOSE', json.dumps(caught.exception.diagnostic))
        self.assertTrue(transport.contacted)

    def test_master_check_timeout_remains_noninteractive_resumable(self):
        transport = B.SSHTransport(self.config)
        with mock.patch.object(B.subprocess, 'run', side_effect=B.subprocess.TimeoutExpired('ssh', 12)):
            with self.assertRaises(B.WorkerUnavailable) as caught:
                transport.request({'action': 'probe'})
        self.assertEqual(caught.exception.diagnostic['phase'], 'master_check')
        self.assertEqual(caught.exception.diagnostic['classification'], 'timeout')
        self.assertFalse(transport.contacted)

    def test_running_master_but_unavailable_allocation_has_its_own_diagnostic(self):
        transport = B.SSHTransport(self.config)
        responses = [mock.Mock(returncode=0), mock.Mock(returncode=0, stdout=b'{"status":"worker_unavailable"}\n', stderr=b'')]
        with mock.patch.object(B.subprocess, 'run', side_effect=responses):
            with self.assertRaises(B.WorkerUnavailable) as caught:
                transport.request({'action': 'probe'})
        self.assertEqual(caught.exception.diagnostic['phase'], 'allocation_probe')

    def test_helper_uses_only_frozen_runtime_python_without_system_interpreter_fallback(self):
        transport = B.SSHTransport(self.config)
        responses = [mock.Mock(returncode=0), mock.Mock(returncode=0,
                     stdout=b'{"status":"ready","parent_allocation_modified":false}\n', stderr=b'')]
        with mock.patch.object(B.subprocess, 'run', side_effect=responses) as run:
            self.assertEqual(transport.request({'action': 'probe'})['status'], 'ready')
        invocation = run.call_args_list[1].args[0]
        remote = shlex.split(invocation[-1])
        self.assertEqual(remote[:3], [B.REMOTE_PYTHON, '-I', '-c'])
        self.assertEqual(remote[0], '/mnt/vast-kisski/home/nigel.hartman01/u29467/.local/state/ai-training/sam3d-20261101-v1/runtime/environment/bin/python')
        self.assertEqual(remote[3].encode(), transport.helper)
        self.assertNotIn('python3', remote[:3])
        self.assertEqual(run.call_count, 2)


class NativeQueueImportTests(unittest.TestCase):
    """Use the existing tiny native-contract fixture, not generated production data."""
    def setUp(self):
        self.native = NativeFixtures.InitialNativeImportTests()
        self.native.setUp()
        self.addCleanup(self.native.doCleanups)
        self.root = self.native.root
        (self.root / 'config').mkdir()
        self.handoff = self.native.handoff
        self.identifier = 'upload-' + U.sha(self.handoff)[:40]
        self.directory = self.root / 'data/sam3d-upload-jobs' / U.sha(self.handoff)
        self.directory.mkdir(parents=True)
        self.raw = {}
        for path in self.native.output.rglob('*'):
            if path.is_file():
                self.raw['native/' + path.relative_to(self.native.output).as_posix()] = path.read_bytes()
        # Execution receipts are fabricated software-contract fixtures only.
        spec = {'job_id': self.identifier, 'maximum_model_commands': 1, 'automatic_retry': False,
                'resources': {'existing_allocation': '16889584', 'nodes': 1, 'tasks': 1, 'gpus': 1,
                              'cpu_threads': 16, 'memory_gib': 100, 'time_seconds': 1800},
                'manifest': {'sha256': U.digest(b'{}\n')}}
        plan = {'job_id': self.identifier, 'handoff': {'sha256': U.sha(self.handoff)}, 'maximum_model_commands': 1, 'automatic_retry': False}
        self.raw['evidence/offline-manifest.json'] = b'{}\n'
        self.raw['evidence/plan.json'] = U.encoded(plan)
        self.raw['evidence/command-spec.json'] = U.encoded(spec)
        prepared = {'status': 'prepared_offline_preflight_passed_not_submitted',
                    'command_spec': {'sha256': U.digest(self.raw['evidence/command-spec.json'])},
                    'plan_sha256': U.digest(self.raw['evidence/plan.json'])}
        launch = {'format': 'sam3d-upload-bridge-launch', 'job_id': self.identifier, 'handoff_sha256': U.sha(self.handoff),
                  'helper_sha256': U.sha(B.REMOTE_HELPER), 'maximum_model_commands': 1, 'automatic_retry': False,
                  'parent_allocation_modified': False,
                  'argv': ['srun', '--jobid=16889584', '--job-name=object-reconstruction', '--exact', '--nodes=1', '--ntasks=1',
                           '--gres=gpu:A100:1', '--cpus-per-task=16', '--mem=100G', '--time=00:30:00', '--immediate=60',
                           'bash', str(B.R.TASK / 'run_component_job_v1.sh'), self.identifier, B.R.BASE, B.R.READY, U.sha(self.handoff)]}
        execution = {'format': 'sam3d-component-offline-execution', 'job_id': self.identifier,
                     'status': 'native_output_ready_for_independent_import_review', 'model_commands_attempted': 1,
                     'model_inference_executed': True, 'parent_allocation_modified': False, 'automatic_retry': False,
                     'executor_sha256': B.R.SCRIPTS['execute_component_job_v1.py'], 'ready_sha256': B.R.READY,
                     'spec_sha256': U.digest(self.raw['evidence/command-spec.json']),
                     'output_metadata_sha256': U.digest(self.raw['native/metadata.json']), 'commands': [{'returncode': 0}, {'returncode': 0}]}
        offline = {'status': 'adapter_completed_pending_output_review', 'adapter_completed': True, 'input_integrity_unchanged': True,
                   'wrapper_sha256': B.R.SCRIPTS['offline_run.py'], 'manifest_sha256': spec['manifest']['sha256']}
        values = {'launch.json': launch, 'supervisor-result.json': {'job_id': self.identifier, 'returncode': 0, 'automatic_retry': False},
                  'execution.json': execution, 'prepare.json': prepared, 'offline-run.json': offline,
                  'offline-preflight.json': {'manifest_sha256': spec['manifest']['sha256'], 'status': 'local_manifest_valid_runtime_not_tested'}}
        self.raw.update({'evidence/' + name: U.encoded(value) for name, value in values.items()})

    def worker(self):
        def response(request):
            if request['action'] == 'inventory':
                return {'status': 'native_output_ready', 'files': [{'path': name, 'sha256': U.digest(raw), 'bytes': len(raw)} for name, raw in self.raw.items()]}
            if request['action'] == 'chunk':
                return {'path': request['path'], 'offset': request['offset'],
                        'base64': base64.b64encode(self.raw[request['path']][request['offset']:request['offset']+4*2**20]).decode()}
            return {'status': 'ready', 'parent_allocation_modified': False} if request['action'] == 'probe' else {'status': 'native_output_ready'}
        return mock.Mock(request=mock.Mock(side_effect=response))

    def test_complete_verified_transfer_can_import_unchanged_initial_candidate(self):
        B.fetch(self.worker(), self.directory, self.identifier)
        B.verify_execution(self.directory, self.identifier, U.sha(self.handoff))
        report = B.N.import_output(self.handoff, self.directory / 'returned/native', self.directory / 'candidate', root=self.root)
        self.assertFalse(report['old_poses_read'])
        self.assertFalse(report['model_selection_changed'])
        self.assertEqual((self.directory / 'candidate/asset.glb').read_bytes(), self.raw['native/object.glb'])

    def test_rebound_execution_runtime_or_multiple_attempts_rejected(self):
        B.fetch(self.worker(), self.directory, self.identifier)
        path = self.directory / 'returned/evidence/execution.json'
        original = U.read(path)
        for key, value in [('ready_sha256', 'a'*64), ('model_commands_attempted', 2), ('parent_allocation_modified', True)]:
            U.write(path, dict(original, **{key: value}))
            with self.subTest(key=key), self.assertRaises(ValueError):
                B.verify_execution(self.directory, self.identifier, U.sha(self.handoff))
        U.write(path, original)

    def test_bad_chunk_hash_is_retained_partial_not_imported(self):
        worker = self.worker()
        real = worker.request.side_effect
        def corrupt(request):
            response = real(request)
            if request['action'] == 'chunk':
                response['base64'] = base64.b64encode(b'x' * len(base64.b64decode(response['base64']))).decode()
            return response
        worker.request.side_effect = corrupt
        with self.assertRaisesRegex(ValueError, 'native bytes'):
            B.fetch(worker, self.directory, self.identifier)
        self.assertFalse((self.directory / 'candidate').exists())

    def test_full_reconstruct_import_then_cached_call_does_not_contact_or_regenerate(self):
        config = {'version': 1, 'enabled': True, 'ssh_control_path': '/private/tmp/task-test/cm-%C',
                  'maximum_parallel_workers': 2, 'poll_seconds': 1, 'wait_seconds': 0}
        U.write(self.root / B.CONFIG_PATH, config)
        worker = self.worker()
        one = B.reconstruct(self.handoff, self.root / 'data/phase-one/native', root=self.root, transport_factory=lambda _: worker)
        self.assertEqual(one['status'], 'native_candidate_ready')
        self.assertTrue(one['remote_job_submitted'])
        self.assertTrue((self.root / one['native_asset']).is_file())
        worker.request.reset_mock()
        two = B.reconstruct(self.handoff, self.root / 'data/phase-two/native', root=self.root, transport_factory=lambda _: worker)
        worker.request.assert_not_called()
        self.assertEqual(one['model'], two['model'])
        self.assertFalse(two['cluster_contacted'])
        self.assertFalse(two['source_video_pose_validated'])


class RemoteAdmissionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.parent = {'JobId': '16889584', 'JobState': 'RUNNING', 'NodeList': 'ggpu182',
                       'EndTime': 'preserved', 'AllocTRES': 'cpu=32,mem=220G,gres/gpu=2'}
        self.patch = mock.patch.object(B.R, 'TASK', self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.helper = B.REMOTE_HELPER.read_bytes()
        self.request = {'job_id': 'upload-' + 'a'*40, 'helper_base64': base64.b64encode(self.helper).decode(),
                        'helper_sha256': U.digest(self.helper)}
        raw = {name: ('SOFTWARE CONTRACT FIXTURE ' + name).encode() for name in B.R.HANDOFF}
        self.request.update(handoff_sha256=U.digest(raw['handoff.json']),
                            files=[{'path': name, 'bytes': len(value), 'sha256': U.digest(value),
                                    'base64': base64.b64encode(value).decode()} for name, value in raw.items()])

    def dispatch(self):
        return B.R.dispatch(self.request)

    def test_dispatch_is_detached_source_bound_and_duplicate_does_not_launch_again(self):
        with mock.patch.object(B.R, 'probe', return_value={'status': 'ready', 'parent': self.parent}), \
             mock.patch.object(B.R, 'parent', return_value=self.parent), \
             mock.patch.object(B.R.subprocess, 'Popen', return_value=mock.Mock(pid=123)) as start:
            one = self.dispatch()
            two = self.dispatch()
        self.assertEqual(one['status'], 'running')
        self.assertEqual(two['status'], 'running')
        self.assertEqual(start.call_count, 1)
        self.assertTrue(start.call_args.kwargs['start_new_session'])
        launch = U.read(self.root / 'upload-bridge-jobs' / self.request['job_id'] / 'launch.json')
        self.assertIn('--nodes=1', launch['argv'])
        self.assertIn('--ntasks=1', launch['argv'])
        self.assertIn('--jobid=16889584', launch['argv'])
        self.assertFalse(launch['parent_allocation_modified'])
        self.assertFalse(launch['automatic_retry'])
        self.assertEqual(launch['parent_before'], self.parent)

    def test_remote_gate_preserves_third_component_without_staging_or_model_attempt(self):
        bridge = self.root / 'upload-bridge-jobs'
        (bridge / 'upload-one').mkdir(parents=True)
        (bridge / 'upload-two').mkdir()
        with mock.patch.object(B.R, 'probe', return_value={'status': 'ready', 'parent': self.parent}), \
             mock.patch.object(B.R.subprocess, 'Popen') as start:
            result = self.dispatch()
        self.assertEqual(result['status'], 'waiting_for_worker_slot')
        start.assert_not_called()
        self.assertFalse((self.root / 'inputs').exists())

    def test_tampered_source_bytes_never_reach_scheduler(self):
        self.request['files'][0]['base64'] = base64.b64encode(b'changed').decode()
        with mock.patch.object(B.R, 'probe', return_value={'status': 'ready', 'parent': self.parent}), \
             mock.patch.object(B.R.subprocess, 'Popen') as start, self.assertRaisesRegex(ValueError, 'Payload'):
            self.dispatch()
        start.assert_not_called()

    def test_unavailable_parent_is_resumable_and_never_changes_scheduler(self):
        output = 'JobId=16889584 JobState=COMPLETED NodeList=ggpu182 EndTime=preserved AllocTRES=gres/gpu=2'
        with mock.patch.object(B.R.subprocess, 'run', return_value=mock.Mock(stdout=output)) as run:
            with self.assertRaises(B.R.AllocationUnavailable):
                B.R.parent()
        self.assertEqual(run.call_args.args[0], ['scontrol', 'show', 'job', '16889584', '-o'])


if __name__ == '__main__':
    unittest.main()
