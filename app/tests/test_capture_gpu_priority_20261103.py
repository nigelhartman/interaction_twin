"""CPU-only contracts for the explicit four-recording GPU-first driver."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('capture_priority',ROOT/'scripts/run_capture_gpu_priority_20261103.py')
D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)


class CapturePriorityTests(unittest.TestCase):
    def test_boundary_occurs_only_after_successful_reconstruction_phase_returns(self):
        runner=object.__new__(D.PriorityRunner)
        with patch.object(D.P.Runner,'phase',return_value={'models':['actual']})as phase:
            self.assertEqual(runner.phase('select',[],lambda _:None),{'models':['actual']})
            with self.assertRaises(D.GPUCheckpoint):runner.phase('reconstruct',[],lambda _:None,optional=True)
            self.assertEqual(phase.call_count,2)
        with patch.object(D.P.Runner,'phase',side_effect=ValueError('failed before boundary')):
            with self.assertRaises(ValueError):runner.phase('reconstruct',[],lambda _:None)

    def test_fresh_execution_keeps_private_budget_identity_and_defers_later_work(self):
        with tempfile.TemporaryDirectory()as tmp,patch.object(D,'ROOT',Path(tmp)),patch.object(D.P,'verify_receipts',return_value=True):
            row={'id':'phone','run_id':'run-capture-phone-20261103-v1','input_path':'data/source.mp4'};seen={}
            class Fake:
                def __init__(self,identifier):
                    self.path=Path(tmp)/'data/flow-runs'/identifier/'run.json';self.path.parent.mkdir(parents=True)
                    self.state={'id':identifier,'phases':[{'name':'reconstruct','status':'complete_with_gaps','files':[{'path':'actual'}],
                        'result':{'components':[{'track_id':'phone','output':{'status':'native_candidate_ready'}}]}}]}
                def execute(self,*args,**kwargs):seen.update(args=args,kwargs=kwargs);raise D.GPUCheckpoint()
                def save(self):self.path.write_text(json.dumps(self.state))
            result=D.run_capture(row,'gpu-first',{'sha256':'fixed'},runner_class=Fake)
            self.assertEqual(result['status'],'paused_after_reconstruct')
            self.assertTrue(result['gpu']['all_selected_native_models_ready'])
            self.assertEqual(seen['args'],(D.PROFILE,))
            self.assertEqual(seen['kwargs']['publication_mode'],'admin_review')
            self.assertEqual(seen['kwargs']['overnight_task_budget_plan'],D.BUDGET)
            state=json.loads((Path(tmp)/'data/flow-runs'/row['run_id']/'run.json').read_text())
            self.assertFalse(state['gpu_priority_checkpoint']['automatic_publication'])
            self.assertFalse(state['gpu_priority_checkpoint']['owner_context_passed_to_models'])

    def test_uncertain_existing_attempt_cannot_start_or_change_its_namespace(self):
        for status in ['running','failed','cancelled','gpu_checkpoint_requires_attention']:
            with self.subTest(status=status),tempfile.TemporaryDirectory()as tmp,patch.object(D,'ROOT',Path(tmp)):
                row={'id':'charger','run_id':'run-capture-charger-20261103-v1','input_path':'data/source.mp4'}
                p=Path(tmp)/'data/flow-runs'/row['run_id']/'run.json';p.parent.mkdir(parents=True);p.write_text(json.dumps({'status':status}))
                before=p.read_bytes()
                with patch.object(D,'PriorityRunner',side_effect=AssertionError('must not construct')):
                    result=D.run_capture(row,'gpu-first',{})
                self.assertEqual(result['status'],'blocked_existing_attempt');self.assertEqual(p.read_bytes(),before)

    def test_finish_requires_an_existing_verified_gpu_checkpoint(self):
        with tempfile.TemporaryDirectory()as tmp,patch.object(D,'ROOT',Path(tmp)):
            result=D.run_capture({'id':'bottle','run_id':'run-capture-bottle-20261103-v1'},'finish',{})
            self.assertEqual(result['status'],'blocked_no_gpu_checkpoint')

    def test_source_reference_tamper_blocks_before_any_runner(self):
        with tempfile.TemporaryDirectory()as tmp,patch.object(D,'ROOT',Path(tmp)):
            p=Path(tmp)/'source.mp4';p.write_bytes(b'original');ref=D.reference(p)
            p.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Pinned batch input changed'):D.checked(ref)


if __name__=='__main__':unittest.main()
