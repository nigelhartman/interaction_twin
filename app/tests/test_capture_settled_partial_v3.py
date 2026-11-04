"""Settled partial compilation never retries failed or pending reconstructions."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('settled_partial_v3',ROOT/'scripts/finish_capture_settled_partial_20261103_v3.py')
V=importlib.util.module_from_spec(spec);spec.loader.exec_module(V)


def sample():
    row={'id':'phone','input_path':'data/approved.mp4','source_ref':{'sha256':'a'*64,'bytes':12}}
    request={'profile':V.V1.PROFILE,'input_path':str((ROOT/row['input_path']).resolve()),'case_id':None,
        'overnight_task_budget_plan':V.V1.BUDGET,'publication_mode':'admin_review'}
    state={'id':'same-run','request':request,'source':{'sha256':'a'*64,'bytes':12},'status':'paused_after_reconstruct',
        'phases':[{'name':'reconstruct','status':'complete_with_gaps','files':['source-hashed'],
          'result':{'models':['data/imported/metadata.json'],'pending_components':[],
            'components':[{'track_id':'body','status':'complete_with_gaps','output':{'status':'native_candidate_ready',
                'model':'data/imported/metadata.json','native_asset':'data/imported/asset.glb'}},
              {'track_id':'glass','status':'failed','error':'ValueError during import; cause not reinterpreted'}]}}]}
    return row,state


class SettledPartialTests(unittest.TestCase):
    def test_imported_asset_plus_terminal_import_failure_is_admitted_with_gaps(self):
        row,state=sample()
        with patch.object(V.P,'verify_receipts',return_value=True):result=V.admitted_partial(state,row,'same-run')
        self.assertEqual(result['imported_component_ids'],['body']);self.assertEqual(result['terminal_failed_component_ids'],['glass'])
        self.assertFalse(result['reconstruction_action_permitted']);self.assertFalse(result['physical_accuracy_verified'])

    def test_pending_active_empty_or_changed_source_cannot_be_admitted(self):
        for kind in ['pending','running','queued','no_models','source','unverified']:
            row,state=sample();result=state['phases'][0]['result']
            if kind=='pending':result['pending_components']=['glass']
            if kind in ['running','queued']:result['components'][1]['status']=kind
            if kind=='no_models':result['models']=[]
            if kind=='source':state['source']['sha256']='b'*64
            with self.subTest(kind=kind),patch.object(V.P,'verify_receipts',return_value=kind!='unverified'):
                with self.assertRaises(ValueError):V.admitted_partial(state,row,'same-run')

    def test_normal_and_semantic_reconstruction_actions_are_hard_fenced(self):
        for cls,parent in [(V.PartialNormalRunner,V.FINISH.FinishRunner),(V.PartialSemanticRunner,V.SEMANTIC.SemanticRunner)]:
            runner=object.__new__(cls)
            def invoke_action(_name,_deps,action,**kwargs):return action(2)
            with self.subTest(cls=cls.__name__),patch.object(parent,'phase',side_effect=invoke_action):
                with self.assertRaisesRegex(V.ReconciliationBlocked,'forbids reconstruction action'):
                    runner.phase('reconstruct',[],lambda _:self.fail('Model action cannot execute'))

    def test_normal_runner_preserves_compact_dependency_guard(self):
        runner=object.__new__(V.PartialNormalRunner);runner.profile={'id':'same'}
        proof={'request':{'path':'data/rejected.json'},'encoding':{'path':'data/encoding.json'}};runner.compact_proof=proof
        dependencies=[ROOT/'old.py',V.FINISH.COMPACT.ADAPTER,ROOT/'data/rejected.json',ROOT/'data/encoding.json']
        inputs=[{'path':str(p)}for p in dependencies];source={'sha256':'same'}
        signature=V.P.fingerprint({'profile':runner.profile,'source':source,'inputs':inputs,'runner_code':V.P.LOADED_SOURCE_SHA256})
        runner.state={'source':source,'phases':[{'name':'segment','status':'complete','signature':signature,'files':['verified']}]} 
        with patch.object(V.P,'file_receipts',side_effect=lambda p:[{'path':str(x)}for x in p]),patch.object(V.P,'verify_receipts',return_value=True),patch.object(V.P.Runner,'phase',return_value={'reused':True})as phase:
            self.assertEqual(runner.phase('segment',[ROOT/'old.py'],lambda _:None),{'reused':True})
            self.assertEqual(phase.call_args.args[1],dependencies)


if __name__=='__main__':unittest.main()
