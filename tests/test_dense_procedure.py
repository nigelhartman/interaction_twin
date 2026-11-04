import copy
import importlib.util
import unittest
from pathlib import Path

spec=importlib.util.spec_from_file_location('dense_procedure_cli',Path(__file__).resolve().parents[1]/'scripts/stage5_dense_compile.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)


class DenseProcedureTests(unittest.TestCase):
    def test_transfer_uses_matching_source_pts(self):
        source=[{'index':1,'timestamp_s':0.0},{'index':2,'timestamp_s':1.000089}]
        dense=[{'index':i+1,'timestamp_s':i*1.000089/5} for i in range(6)]
        result=M.transfer_samples(source,dense,[1,2])
        self.assertEqual([x['target_sample_index'] for x in result],[1,6])
        self.assertTrue(all(x['error_s']<1e-7 for x in result))

    def test_transfer_rejects_missing_timestamps_instead_of_inventing(self):
        with self.assertRaisesRegex(ValueError,'no matching'):
            M.transfer_samples([{'index':1,'timestamp_s':5.0}],[{'index':1,'timestamp_s':0.0},{'index':2,'timestamp_s':.2}],[1])

    def fixtures(self):
        rows=[{'index':1,'timestamp_s':.2},{'index':2,'timestamp_s':.4}]
        event={'id':'w1','first_sample':1,'last_sample':2,'evidence_samples':[1,2],
               'evidence_timestamps_s':[.2,.4],'first_timestamp_s':.2,'last_timestamp_s':.4,
               'physical_identity_across_occlusion':'unverified','final_seating':'unknown','contact_or_latch':'unknown'}
        return rows,{'status':'complete_model_proposals','source_sha256':'source','result':{'events':[event]}}

    def test_windows_require_exact_source_evidence(self):
        rows,events=self.fixtures()
        self.assertTrue(M.validate_windows(events,{'source':{'sha256':'source'}},rows))
        events['result']['events'][0]['evidence_timestamps_s'][1]=.401
        with self.assertRaisesRegex(ValueError,'timestamp/sample'):
            M.validate_windows(events,{'source':{'sha256':'source'}},rows)

    def test_windows_reject_claimed_hidden_completion(self):
        rows,events=self.fixtures();events['result']['events'][0]['final_seating']='seated'
        with self.assertRaisesRegex(ValueError,'hidden physical'):
            M.validate_windows(events,{'source':{'sha256':'source'}},rows)

    def test_all_windows_need_step_and_state_evidence(self):
        _,events=self.fixtures()
        proposal={'steps':[{'evidence_samples':[1]}],'assertions':[{'evidence_sample':2}]}
        coverage=M.validate_window_coverage(proposal,events)
        self.assertEqual(coverage[0]['final_seating'],'unknown')
        proposal['assertions']=[]
        with self.assertRaisesRegex(ValueError,'omitted'):
            M.validate_window_coverage(proposal,events)


if __name__=='__main__':unittest.main()
