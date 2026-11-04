"""Offline executor/import boundaries. No inference, API or GPU contact."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import admin_pipeline as A
import admin_http as H
import pipeline as P
from admin_workflow import AdminWorkflow
from agent_memory import FeedbackMemory
from object_program import ref
from review_agents import verified_ref


class AdminPipelineContractTests(unittest.TestCase):
    def runner(self,status='complete',boundary=None):
        row={'name':'discover','status':status,'files':[{'fixture':'output'}],'inputs':[{'fixture':'input'}],
             'result':{'saved':'exact old output','has_gaps':False},'attempt':1,'signature':'historic-signature'}
        runner=A.AdminRunner.__new__(A.AdminRunner)
        runner.base_state={'id':'run_original','phases':[row]};runner.boundary=boundary
        runner.state={'phases':[]};runner.adoptions=[];runner.save=Mock()
        return runner
    def test_every_prior_paid_state_with_unverified_outputs_stops_without_resend(self):
        for status in ['complete','complete_with_gaps','running','failed','cancelled','repair_pending','unknown_state']:
            with self.subTest(status=status):
                runner=self.runner(status)
                with patch.object(P,'verify_receipts',return_value=False),patch.object(P.Runner,'phase',side_effect=AssertionError('No paid fallback'))as fresh:
                    with self.assertRaisesRegex(ValueError,'no automatic resend'):runner.phase('discover',[],lambda _:None)
                    fresh.assert_not_called()
    def test_same_run_resume_verifies_inputs_and_adopts_exact_historical_output(self):
        runner=self.runner()
        with patch.object(P,'verify_receipts',side_effect=[True,False]),patch.object(P.Runner,'phase')as fresh:
            with self.assertRaisesRegex(ValueError,'input changed'):runner.phase('discover',[],lambda _:None)
            fresh.assert_not_called()
        runner=self.runner()
        with patch.object(P,'verify_receipts',side_effect=[True,True]),patch.object(P.Runner,'phase')as fresh:
            result=runner.phase('discover',[],lambda _:None)
        self.assertEqual(result,runner.base_state['phases'][0]['result']);fresh.assert_not_called()
        self.assertEqual(runner.state['phases'][0]['signature'],'historic-signature')
        self.assertFalse(runner.state['phases'][0]['new_inference'])
        self.assertEqual(runner.state['phases'][0]['adopted_from_run'],'run_original')
    def test_only_explicit_downstream_repair_reaches_new_fixed_stage(self):
        runner=self.runner('failed',boundary='discover')
        with patch.object(P.Runner,'phase',return_value={'new_explicit_stage':True})as fresh:
            self.assertEqual(runner.phase('discover',[],lambda _:None),{'new_explicit_stage':True})
        fresh.assert_called_once()
        self.assertEqual(A.repair_boundary(['tracking','reconstruction']),'discover')
        self.assertEqual(A.repair_boundary(['reconstruction']),'select')
        self.assertEqual(A.repair_boundary(['mechanics']),'procedure_proposal')
        with self.assertRaises(ValueError):A.repair_boundary(['execute_model_code'])
    def test_explicit_repair_records_code_changes_but_never_relaxes_evidence_or_resume(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp).resolve();(root/'scripts').mkdir();(root/'data').mkdir()
            code=root/'scripts/selector.py';code.write_bytes(b'original reviewed code')
            evidence=root/'data/observed.png';evidence.write_bytes(b'original saved observation')
            disguised=root/'data/untrusted.py';disguised.write_bytes(b'saved data is not trusted implementation')
            receipts=[ref(code,root),ref(evidence,root)]
            disguised_ref=ref(disguised,root)
            code.write_bytes(b'updated reviewed implementation')
            original_confined=P.confined
            with patch.object(A,'ROOT',root),patch.object(P,'ROOT',root),\
                 patch.object(P,'confined',side_effect=lambda p:original_confined(p,root=root)):
                changes=A.adoption_inputs(receipts,repair=True)
                self.assertEqual(len(changes),1);self.assertEqual(changes[0]['path'],'scripts/selector.py')
                self.assertEqual(changes[0]['historical_sha256'],receipts[0]['sha256'])
                self.assertFalse(changes[0]['executed_current_code'])
                with self.assertRaisesRegex(ValueError,'evidence input changed'):A.adoption_inputs(receipts,repair=False)
                evidence.write_bytes(b'changed observation')
                with self.assertRaisesRegex(ValueError,'evidence input changed'):A.adoption_inputs(receipts,repair=True)
                disguised.write_bytes(b'changed data posing as code')
                with self.assertRaisesRegex(ValueError,'evidence input changed'):A.adoption_inputs([disguised_ref],repair=True)


class ExistingExampleImportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve()
        for folder in ['data/example','showcase/public']:(self.root/folder).mkdir(parents=True)
        source=self.root/'data/example/video.mp4';source.write_bytes(b'fixture original source video')
        asset=self.root/'data/example/asset.glb';asset.write_bytes(b'glTF fixture only for import reference contract')
        self.source=ref(source,self.root);self.asset=ref(asset,self.root)
        self.program={'case_id':'example','source':self.source,'assets':[{'id':'asset_1','track_id':'part',
            'path':self.asset['path'],'sha256':self.asset['sha256'],'backend':'Test fixture'}]}
        path=self.root/'data/example/interaction.json';path.write_text(json.dumps(self.program));self.interaction=ref(path,self.root)
        (self.root/'showcase/public/library-catalog.json').write_text(json.dumps({'records':[{'id':'example--procedure','interaction':self.interaction}]}))
        (self.root/'showcase/public/aligned-scenes.json').write_text(json.dumps({'scenes':[]}))
        self.store=AdminWorkflow(self.root)
    def test_exact_current_curated_source_assets_import_without_inference_and_require_consent(self):
        with patch.object(P,'list_runs',return_value=[]),patch.object(P.Runner,'execute',side_effect=AssertionError('Import cannot run a pipeline')):
            result=H.existing_example(self.store,self.interaction['path'],root=self.root)
        self.assertEqual(result['candidate']['status'],'pending_consent')
        self.assertEqual(result['job']['state'],'pending_consent')
        self.assertEqual(self.store.public_index()['records'],[])
        path=self.root/'.private/admin-imports'/(result['candidate']['run_id']+'.json')
        saved=json.loads(path.read_text())
        for artifact in saved['result']['artifacts']:
            self.assertEqual(set(artifact),{'path','sha256','bytes'});verified_ref(self.root,artifact)
        self.assertEqual(saved['source_ref'],self.source)
        self.assertEqual(saved['result']['new_api_calls'],0)
        self.assertIn(self.asset,saved['result']['artifacts'])
        self.assertFalse(saved['result'].get('human_approved',False))
    def test_changed_or_noncurated_example_cannot_be_adopted(self):
        with self.assertRaisesRegex(ValueError,'currently curated'):H.existing_example(self.store,'data/not-approved/interaction.json',root=self.root)
        (self.root/self.source['path']).write_bytes(b'changed fixture video')
        with patch.object(P,'list_runs',return_value=[]),self.assertRaisesRegex(ValueError,'recording changed'):
            H.existing_example(self.store,self.interaction['path'],root=self.root)
        self.assertEqual(self.store.snapshot()['candidates'],[])

if __name__=='__main__':unittest.main()
