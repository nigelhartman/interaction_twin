"""Retrieval software fixtures only: no SSH, scheduler, inference or credentials."""
import base64
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts import fetch_submitted_bottle_outputs_20261103_v1 as F
from tests import test_sam3d_cluster_bridge as Fixtures


class FetchOnlyTests(unittest.TestCase):
    def test_dispatch_probe_and_foreign_chunks_never_reach_transport(self):
        fake = Mock()
        wrapper = F.FetchOnlyTransport(fake, 'upload-exact')
        for payload in [
            {'action': 'probe', 'job_id': 'upload-exact'},
            {'action': 'dispatch', 'job_id': 'upload-exact'},
            {'action': 'supervise', 'job_id': 'upload-exact'},
            {'action': 'poll', 'job_id': 'upload-other'},
            {'action': 'poll', 'job_id': 'upload-exact', 'command': 'anything'},
            {'action': 'chunk', 'job_id': 'upload-exact', 'path': '../token', 'offset': 0},
            {'action': 'chunk', 'job_id': 'upload-exact', 'path': 'native/object.glb', 'offset': -1},
        ]:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                wrapper.request(payload)
        fake.request.assert_not_called()

    def test_remote_poll_inventory_and_chunk_do_not_need_live_allocation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            identifier = 'upload-expired'
            bridge = root / 'upload-bridge-jobs' / identifier
            component = root / 'component-jobs' / identifier
            output = root / 'outputs' / identifier
            bridge.mkdir(parents=True)
            component.mkdir(parents=True)
            output.mkdir(parents=True)
            F.U.write(bridge / 'launch.json', {'job_id': identifier})
            F.U.write(bridge / 'supervisor-result.json', {'job_id': identifier, 'returncode': 0})
            execution = {'status': 'native_output_ready_for_independent_import_review', 'model_commands_attempted': 1,
                         'model_inference_executed': True, 'automatic_retry': False}
            F.U.write(component / 'execution.json', execution)
            for n in ['prepare.json', 'plan.json', 'command-spec.json', 'offline-preflight.json', 'offline-run.json']:
                F.U.write(component / n, {'fixture': True})
            F.U.write(root / ('offline-job-' + identifier + '.json'), {'fixture': True})
            for n in F.B.R.OUTPUTS:
                p = output / n
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b'SOFTWARE CONTRACT FIXTURE')
            with patch.object(F.B.R, 'TASK', root), \
                 patch.object(F.B.R, 'probe', side_effect=AssertionError('Expired allocation must not be probed')), \
                 patch.object(F.B.R, 'parent', side_effect=AssertionError('No scontrol')), \
                 patch.object(F.B.R, 'dispatch', side_effect=AssertionError('No model dispatch')):
                wrapper = F.FetchOnlyTransport(Mock(request=Mock(side_effect=F.B.R.serve)), identifier)
                self.assertEqual(wrapper.request({'action': 'poll', 'job_id': identifier})['status'], 'native_output_ready')
                inv = wrapper.request({'action': 'inventory', 'job_id': identifier})
                self.assertEqual(len(inv['files']), len(F.B.NATIVE) + len(F.B.EVIDENCE))
                chunk = wrapper.request({'action': 'chunk', 'job_id': identifier, 'path': 'native/object.glb', 'offset': 0})
                self.assertEqual(base64.b64decode(chunk['base64']), b'SOFTWARE CONTRACT FIXTURE')

    def fixture(self):
        fixture = Fixtures.NativeQueueImportTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        digest = F.U.sha(fixture.handoff)
        row = {'component_id': 'bottle_early', 'job_id': fixture.identifier,
               'handoff': F.ref(fixture.handoff, fixture.root),
               'original_bridge': {'path': f'data/sam3d-upload-jobs/{digest}/bridge.json'},
               'recovery_directory': 'data/sam3d-fetch-only-recoveries/' + digest}
        bridge = fixture.directory / 'bridge.json'
        record = {'format': 'sam3d-upload-cluster-bridge', 'version': 1, 'job_id': fixture.identifier,
                  'handoff': row['handoff'], 'maximum_model_commands': 1, 'automatic_retry': False,
                  'parent_allocation_modified': False, 'preferred_assets_changed': False,
                  'requested_destination': f'data/{F.CASE}/flow-model-1-1',
                  'dispatch_requested': True, 'remote_job_submitted': True, 'status': 'failed_no_retry'}
        F.U.write(bridge, record)
        worker = fixture.worker()
        previous = worker.request.side_effect
        worker.request.side_effect = lambda request: ({'status': 'native_output_ready', 'job_id': fixture.identifier}
                                                      if request['action'] == 'poll' else previous(request))
        return fixture, row, bridge, worker, digest

    def test_verified_fetch_and_import_preserve_original_terminal_bridge(self):
        fixture, row, bridge, worker, digest = self.fixture()
        old = bridge.read_bytes()
        with patch.dict(F.JOBS, {'bottle_early': digest}, clear=True), patch.object(F.B, 'configuration', return_value={}):
            result = F.fetch_job(row, root=fixture.root, transport_factory=lambda _: worker, import_native=True)
        self.assertEqual(result['status'], 'native_candidate_imported_unselected')
        self.assertEqual(old, bridge.read_bytes())
        self.assertEqual(F.U.read(bridge)['status'], 'failed_no_retry')
        self.assertFalse(F.U.read(fixture.root / result['model']['path'])['selected'])
        self.assertEqual(set(result['remote_actions']), {'poll', 'inventory', 'chunk'})
        self.assertEqual(result['new_model_commands'], 0)

    def test_oversized_or_unimportable_output_keeps_verified_raw_no_retry(self):
        fixture, row, bridge, worker, digest = self.fixture()
        old = bridge.read_bytes()
        with patch.dict(F.JOBS, {'bottle_early': digest}, clear=True), patch.object(F.B, 'configuration', return_value={}), \
             patch.object(F.N, 'import_output', side_effect=ValueError('Complete initial candidate exceeds local byte budget')):
            result = F.fetch_job(row, root=fixture.root, transport_factory=lambda _: worker, import_native=True)
        self.assertEqual(result['status'], 'raw_verified_native_import_unavailable')
        self.assertTrue(result['raw_bytes_preserved'])
        self.assertTrue((fixture.root / row['recovery_directory'] / 'returned/native/object.glb').exists())
        self.assertEqual(old, bridge.read_bytes())

    def test_runtime_proof_tamper_blocks_import(self):
        fixture, row, bridge, worker, digest = self.fixture()
        import json
        d = json.loads(fixture.raw['evidence/execution.json'])
        d['model_commands_attempted'] = 2
        fixture.raw['evidence/execution.json'] = F.U.encoded(d)
        with patch.dict(F.JOBS, {'bottle_early': digest}, clear=True), patch.object(F.B, 'configuration', return_value={}), \
             patch.object(F.N, 'import_output') as importer:
            with self.assertRaisesRegex(ValueError, 'successful GPU execution'):
                F.fetch_job(row, root=fixture.root, transport_factory=lambda _: worker, import_native=True)
        importer.assert_not_called()

    def test_unsubmitted_intent_is_rejected(self):
        fixture, row, bridge, worker, digest = self.fixture()
        record = F.U.read(bridge)
        record['remote_job_submitted'] = False
        with self.assertRaisesRegex(ValueError, 'unsubmitted'):
            F.bridge_binding(record, 'bottle_early', digest, require_submitted=True)


if __name__ == '__main__':
    unittest.main()
