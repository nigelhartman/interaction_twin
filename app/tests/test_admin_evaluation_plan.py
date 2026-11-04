"""Frozen lineage comparison sets; no fitting, image inference or paid transport."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent_memory import digest
from admin_evaluation_plan import get_or_create,load,verify_audit,comparison_key,EvaluationBasisChanged

class EvaluationPlanContracts(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve()
        self.program={'source':{'sha256':'a'*64,'original_sha256':'b'*64},'tracklets':[{'id':'base'},{'id':'lid'}]}
        self.scene={'source':{'sha256':'a'*64},'original_program':{'path':'data/program.json','sha256':'c'*64,'bytes':3},
            'components':[{'id':part,'poses':[{'source_sample_index':i,'timestamp_s':(i-1)/3}for i in range(1,31)]}for part in ['base','lid']],
            'evaluation_protocol':{'mode':'retrospective_diagnostic','held_out_sample_indices':[],'protected_native_anchor_indices':[2]}}
        self.job={'candidate_id':'candidate_'+'d'*32,'source_ref':{'sha256':'b'*64},'improvement_round':0}
    def freeze(self):return get_or_create(self.job,self.scene,self.program,root=self.root)
    def test_initial_basis_freezes_exact_inventory_sample_universe_clocks_and_split_without_holdout_claim(self):
        reference,plan=self.freeze();basis=plan['basis']
        self.assertEqual(basis['expected_component_ids'],['base','lid']);self.assertEqual(basis['sample_index_universe_sha256'],digest(list(range(1,31))))
        self.assertFalse(set(basis['development_source_indices'])&set(basis['final_test_source_indices']))
        self.assertIn('retrospective',basis['final_test_scope']);self.assertEqual(basis['motion_holdout_indices'],[])
        self.assertEqual((self.root/reference['path']).stat().st_mode&0o777,0o600)
        self.assertEqual(comparison_key(reference,root=self.root),digest({'evaluation_plan_sha256':reference['sha256'],'expected_component_ids':['base','lid']}))
    def test_repair_uses_pinned_first_reference_and_anchor_change_does_not_move_split(self):
        reference,plan=self.freeze();job=dict(self.job,improvement_round=1,evaluation_plan_ref=reference)
        self.scene['original_program']={'path':'data/new-program.json','sha256':'e'*64,'bytes':4}
        self.scene['evaluation_protocol']['protected_native_anchor_indices']=[13]
        later,saved=get_or_create(job,self.scene,self.program,root=self.root)
        self.assertEqual(later,reference);self.assertEqual(saved,plan)
    def test_index_time_component_and_split_changes_stop_without_rewriting_first_plan(self):
        reference,_=self.freeze();before=(self.root/reference['path']).read_bytes()
        for what in ['index','time','component','split']:
            scene=copy.deepcopy(self.scene);program=copy.deepcopy(self.program)
            if what=='index':
                for c in scene['components']:c['poses']=c['poses'][:-1]
            elif what=='time':
                for c in scene['components']:c['poses'][4]['timestamp_s']+=.01
            elif what=='component':program['tracklets'].append({'id':'new_part'})
            else:scene['evaluation_protocol'].update(mode='predeclared_mask_holdout',held_out_sample_indices=[5,10,15])
            with self.subTest(what=what),self.assertRaises(EvaluationBasisChanged):
                get_or_create(dict(self.job,improvement_round=1,evaluation_plan_ref=reference),scene,program,root=self.root)
            self.assertEqual((self.root/reference['path']).read_bytes(),before)
    def test_later_round_cannot_create_or_replace_a_lost_initial_plan(self):
        with self.assertRaises(EvaluationBasisChanged):get_or_create(dict(self.job,improvement_round=1),self.scene,self.program,root=self.root)
        reference,_=self.freeze()
        with self.assertRaises(EvaluationBasisChanged):get_or_create(dict(self.job,improvement_round=1),self.scene,self.program,root=self.root)
        path=self.root/reference['path'];body=json.loads(path.read_text());body['basis']['expected_component_ids']=['base'];body['basis_sha256']=digest(body['basis']);path.write_text(json.dumps(body))
        with self.assertRaisesRegex(ValueError,'changed'):
            get_or_create(dict(self.job,improvement_round=1,evaluation_plan_ref=reference),self.scene,self.program,root=self.root)
    def test_audit_roles_are_verified_and_final_test_is_not_silently_moved_into_development(self):
        reference,plan=self.freeze();basis=plan['basis']
        samples=[{'source_index':i,'component_id':part,'role':role}for role,ids in [('development',basis['development_source_indices']),('final_test',basis['final_test_source_indices'])]for i in ids for part in ['base','lid']]
        audit={'source':{'sha256':'a'*64},'samples':samples}
        self.assertEqual(verify_audit(reference,audit,root=self.root),plan)
        for row in audit['samples']:
            if row['role']=='final_test':row['role']='development'
        with self.assertRaises(EvaluationBasisChanged):verify_audit(reference,audit,root=self.root)
    def test_initial_upload_and_evidence_source_must_match_as_declared_original_or_normalized_pair(self):
        wrong=dict(self.job,source_ref={'sha256':'f'*64})
        with self.assertRaisesRegex(ValueError,'source differ'):get_or_create(wrong,self.scene,self.program,root=self.root)
    def test_changed_companion_fences_executor_before_a_pipeline_or_paid_dispatch(self):
        import admin_pipeline as A
        import pipeline as P
        reference,_=self.freeze();(self.root/reference['path']).write_text('{}')
        with patch.object(A,'ROOT',self.root),patch.object(P.Runner,'execute',side_effect=AssertionError('No dispatch'))as execute:
            with self.assertRaisesRegex(ValueError,'changed'):
                A.FixedExecutor(root=self.root)({'processing_consent':True,'evaluation_plan_ref':reference})
        execute.assert_not_called()

if __name__=='__main__':unittest.main()
