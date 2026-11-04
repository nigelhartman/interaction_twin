import copy
import unittest
from unittest.mock import patch

from scripts import finish_laptop_workspace_20261103_v1 as F


class LaptopWorkspaceResume(unittest.TestCase):
    def fixture(self):
        row = {'input_path': 'data/source.mp4', 'source_ref': {'sha256': 'a' * 64, 'bytes': 20}}
        profile = {'publication_mode': 'admin_review'}
        source = {'sha256': 'a' * 64, 'bytes': 20}
        inputs = [{'path': 'data/evidence.json', 'sha256': 'b' * 64, 'bytes': 10}]
        signature = F.P.fingerprint({'profile': profile, 'source': source, 'inputs': inputs,
                                     'runner_code': F.P.LOADED_SOURCE_SHA256})
        phases = [{'name': n, 'status': 'complete', 'inputs': inputs, 'files': inputs,
                   'signature': signature, 'result': {}} for n in F.P.PHASES]
        next(p for p in phases if p['name'] == 'mechanism_topology').update(status='failed', error='known contract failure')
        phases[-1].update(status='failed', error='Phase command exceeded 90 seconds')
        state = {'id': F.RUN_ID, 'status': 'failed', 'phase': 'workspace', 'profile': profile,
                 'source': source, 'artifacts': {}, 'phases': phases,
                 'request': {'profile': F.V1.PROFILE, 'input_path': str(F.ROOT / row['input_path']),
                             'case_id': None, 'overnight_task_budget_plan': F.V1.BUDGET, 'publication_mode': 'admin_review'}}
        return state, row, inputs

    def test_preserves_failed_topology_without_replaying_it(self):
        state, row, inputs = self.fixture()
        before = copy.deepcopy(state)
        with patch.object(F.P, 'verify_receipts', return_value=True), patch.object(F.P, 'file_receipts', return_value=inputs):
            F.verify_predecessor(state, row)
        self.assertEqual(before, state)
        self.assertEqual(next(p for p in state['phases'] if p['name'] == 'mechanism_topology')['status'], 'failed')

    def test_pending_model_is_blocked(self):
        state, row, inputs = self.fixture()
        next(p for p in state['phases'] if p['name'] == 'reconstruct')['result']['pending_components'] = ['lid']
        with patch.object(F.P, 'verify_receipts', return_value=True), patch.object(F.P, 'file_receipts', return_value=inputs):
            with self.assertRaisesRegex(ValueError, 'Pending GPU'):
                F.verify_predecessor(state, row)

    def test_changed_code_or_artifact_stops_before_local_commands(self):
        state, row, inputs = self.fixture()
        state['phases'][2]['signature'] = 'wrong'
        with patch.object(F.P, 'verify_receipts', return_value=True), patch.object(F.P, 'file_receipts', return_value=inputs):
            with self.assertRaisesRegex(ValueError, 'signature changed'):
                F.verify_predecessor(state, row)
        state, row, inputs = self.fixture()
        with patch.object(F.P, 'verify_receipts', return_value=False):
            with self.assertRaisesRegex(ValueError, 'inputs changed'):
                F.verify_predecessor(state, row)

    def test_dispatch_is_fixed_local_only_and_full_execute_forbidden(self):
        runner = object.__new__(F.LocalWorkspaceRunner)
        with patch.object(F.P.Runner, 'command') as command:
            runner.command([F.ROOT / '.venv-sam/bin/python', F.ROOT / 'scripts/build_workspace.py', '--no-replays'], 600)
            self.assertEqual(command.call_count, 1)
            with self.assertRaises(ValueError):
                runner.command([F.ROOT / '.venv-sam/bin/python', F.ROOT / 'scripts/stage5_meta_propose.py'], 600)
            with self.assertRaises(ValueError):
                runner.command([F.ROOT / '.venv-sam/bin/python', F.ROOT / 'scripts/build_workspace.py'], 600)
            self.assertEqual(command.call_count, 1)
        with self.assertRaisesRegex(RuntimeError, 'forbidden'):
            runner.execute('anything')
        with self.assertRaisesRegex(RuntimeError, 'forbidden'):
            runner._infer(None, {})


if __name__ == '__main__':
    unittest.main()
