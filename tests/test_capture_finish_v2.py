"""Finishing must reuse the exact successful compact mask contract, never SAM."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('capture_finish_v2',ROOT/'scripts/finish_capture_gpu_priority_20261103_v2.py')
F=importlib.util.module_from_spec(spec);spec.loader.exec_module(F)


class CaptureFinishTests(unittest.TestCase):
    def runner(self,name,inputs,*,proof=None,status='complete',pending=None):
        r=object.__new__(F.FinishRunner);r.profile={'id':'same'};r.compact_proof=proof;r.state={'source':{'sha256':'same'},'phases':[]}
        identity=F.P.fingerprint({'profile':r.profile,'source':r.state['source'],'inputs':inputs,'runner_code':F.P.LOADED_SOURCE_SHA256})
        r.state['phases']=[{'name':name,'status':status,'signature':identity,'files':['original'],'result':{'pending_components':pending or[]}}]
        return r

    def test_compact_mask_signature_reuses_identical_extra_dependencies(self):
        proof={'request':{'path':'data/request.json'},'encoding':{'path':'data/encoding.json'}}
        expected=[ROOT/'legacy.py',F.COMPACT.ADAPTER,ROOT/'data/request.json',ROOT/'data/encoding.json']
        inputs=[{'path':str(p)}for p in expected];runner=self.runner('segment',inputs,proof=proof)
        with patch.object(F.P,'file_receipts',side_effect=lambda rows:[{'path':str(p)}for p in rows]),patch.object(F.P,'verify_receipts',return_value=True),patch.object(F.P.Runner,'phase',return_value={'reused':True})as phase:
            self.assertEqual(runner.phase('segment',[ROOT/'legacy.py'],lambda _:None),{'reused':True})
            self.assertEqual(phase.call_args.args[1],expected)

    def test_legacy_signature_mismatch_cannot_resend_lossless_sam(self):
        runner=self.runner('segment',[{'path':'compact-inputs'}],proof=None)
        with patch.object(F.P,'file_receipts',return_value=[{'path':'legacy-inputs'}]),patch.object(F.P,'verify_receipts',return_value=True),patch.object(F.P.Runner,'phase',side_effect=AssertionError('SAM action forbidden')):
            with self.assertRaisesRegex(F.ReconciliationBlocked,'no perception/model'):runner.phase('segment',[],lambda _:None)

    def test_pending_gpu_and_uncertain_paid_phase_cannot_be_retried_by_finish(self):
        for name,status,pending in [('reconstruct','complete_with_gaps',['part']),('procedure_proposal','failed',[])]:
            with self.subTest(name=name):
                runner=self.runner(name,[],status=status,pending=pending)
                with patch.object(F.P,'file_receipts',return_value=[]),patch.object(F.P,'verify_receipts',return_value=True),patch.object(F.P.Runner,'phase',side_effect=AssertionError('retry forbidden')):
                    with self.assertRaises(F.ReconciliationBlocked):runner.phase(name,[],lambda _:None)

    def test_only_missing_post_gpu_work_may_execute_fresh(self):
        runner=self.runner('reconstruct',[])
        with patch.object(F.P.Runner,'phase',return_value={'fresh_post_gpu':True})as phase:
            self.assertEqual(runner.phase('procedure_proposal',[],lambda _:None),{'fresh_post_gpu':True})
            phase.assert_called_once()


if __name__=='__main__':unittest.main()
