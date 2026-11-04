"""Offline source-bound reconciliation gates; no inference or cluster access."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('capture_reconcile_v2',ROOT/'scripts/reconcile_capture_gpu_priority_20261103_v2.py')
R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)


def pending():
    return {'status':'paused_after_reconstruct','phases':[{'name':'reconstruct','status':'complete_with_gaps',
        'files':[{'path':'original'}],'result':{'pending_components':['cable'],
        'components':[{'track_id':'cable','status':'complete_with_gaps','output':{'status':'awaiting_gpu_worker'}}],
        'failures':[]}}]}


class CaptureGPUReconciliationTests(unittest.TestCase):
    def test_only_exact_pending_components_are_reconcilable(self):
        with patch.object(R.P,'verify_receipts',return_value=True):
            self.assertEqual(R.pending_state(pending()),['cable'])
            bad=pending();bad['phases'][0]['result']['components'][0]['output']['status']='failed_no_retry'
            with self.assertRaisesRegex(ValueError,'failed or unsupported'):R.pending_state(bad)
            bad=pending();bad['phases'][0]['result']['pending_components']=[]
            with self.assertRaisesRegex(ValueError,'pending component identity'):R.pending_state(bad)

    def test_changed_paid_phase_cannot_dispatch_any_action(self):
        runner=object.__new__(R.ReconcileRunner);runner.profile={'id':'same'};runner.state={'source':{'sha256':'source'},
            'phases':[{'name':'discover','status':'complete','signature':'changed','files':[{'path':'original'}]}]}
        with patch.object(R.P,'file_receipts',return_value=[{'path':'same'}]),patch.object(R.P,'verify_receipts',return_value=True),patch.object(R.P.Runner,'phase',side_effect=AssertionError('must never call action')):
            with self.assertRaisesRegex(R.ReconciliationBlocked,'no paid request was resent'):
                runner.phase('discover',[],lambda _:None)

    def test_exact_early_phase_is_reused_and_only_reconstruction_checkpoints(self):
        runner=object.__new__(R.ReconcileRunner);runner.profile={'id':'same'};runner.state={'source':{'sha256':'source'},'phases':[]}
        inputs=[{'path':'original'}];identity=R.P.fingerprint({'profile':runner.profile,'source':runner.state['source'],'inputs':inputs,'runner_code':R.P.LOADED_SOURCE_SHA256})
        runner.state['phases']=[{'name':'segment','status':'complete','signature':identity,'files':inputs}]
        with patch.object(R.P,'file_receipts',return_value=inputs),patch.object(R.P,'verify_receipts',return_value=True),patch.object(R.P.Runner,'phase',return_value={'reused':True})as phase:
            self.assertEqual(runner.phase('segment',[],lambda _:None),{'reused':True})
            with self.assertRaises(R.V1.GPUCheckpoint):runner.phase('reconstruct',[],lambda _:None)
            self.assertEqual(phase.call_count,2)

    def test_original_live_driver_is_unchanged(self):
        self.assertEqual(R.P.sha(Path(R.V1.__file__)),R.V1_SHA)


if __name__=='__main__':unittest.main()
