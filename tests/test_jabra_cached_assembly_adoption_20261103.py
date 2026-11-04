"""Private cached adoption preserves lineage, objections and queue isolation."""
import copy
import unittest
from unittest.mock import patch

from admin_workflow import AdminWorkflow
import jabra_cached_assembly_adoption_20261103 as J
from tests import test_admin_workflow as F


class DeclaredAssemblyAdoptionTests(unittest.TestCase):
    def setUp(self):
        fixture=F.AdminWorkflowTests('test_consent_queue_restart_four_review_roles_and_no_automatic_publication')
        fixture.setUp();self.addCleanup(fixture.doCleanups)
        self.fixture=fixture;self.root=fixture.root;self.store=fixture.store
        self.cid,self.base=fixture.ready();self.old=(self.root/self.base['manifest_ref']['path']).read_bytes()
        manifest=self.base['manifest']
        self.prior={'base_revision_id':self.base['id'],'base_fingerprint':self.base['fingerprint'],
                    'hosted_blocking_findings':3,'original_findings_preserved_in_base_revision':True,
                    'new_hosted_visual_review_performed':False,'objections_are_not_resolved_by_structural_review':True}
        self.plan={'contract':{'candidate_id':self.cid,'base_revision_id':self.base['id'],'base_fingerprint':self.base['fingerprint']},
                   'frozen_memory_context':{'context_sha256':manifest['memory_context_sha256'],'lessons':[]},
                   'retained_prior_hosted_findings':self.prior,
                   'result':{'status':'complete_with_gaps','run_id':manifest['run_id'],
                     'interaction_ref':manifest['interaction_ref'],'object_program_ref':manifest['object_program_ref'],
                     'scene_ref':manifest['scene_ref'],'part_ledger_ref':manifest.get('part_ledger_ref'),
                     'artifacts':manifest['artifacts'],'repair_execution':[{'strategy':'declared_source_mask_refit_assembly_adoption_v1',
                         'retained_prior_hosted_findings':copy.deepcopy(self.prior)}],
                     'diagnostic':'Explicit mathematical fixture; no geometry inference.'}}

    def adopt(self,plan=None):
        with patch.object(J,'build',return_value=copy.deepcopy(self.plan)):
            return J.adopt(copy.deepcopy(self.plan)if plan is None else plan,root=self.root)

    def test_new_private_revision_retains_findings_source_lineage_and_other_queue(self):
        unrelated=self.store.submit_upload(run_id='unrelated',input_path=self.fixture.source,processing_consent=True)
        with patch.object(AdminWorkflow,'worker_once',side_effect=AssertionError('No queue draining')):
            result=self.adopt()
        self.assertFalse(result['automatic_approval']);self.assertFalse(result['automatic_publication'])
        self.assertEqual(result['new_api_calls'],0);self.assertEqual(result['retained_prior_hosted_findings'],self.prior)
        detail=self.store.detail(self.cid);revision=detail['revisions'][0]['manifest']
        self.assertEqual(revision['base_revision_id'],self.base['id'])
        self.assertEqual(revision['repair_execution'][0]['retained_prior_hosted_findings'],self.prior)
        self.assertEqual(revision['interaction_ref'],self.base['manifest']['interaction_ref'])
        self.assertEqual((self.root/self.base['manifest_ref']['path']).read_bytes(),self.old)
        self.assertEqual(detail['decisions'],[]);self.assertEqual(self.store.public_index()['records'],[])
        self.assertEqual(next(j for j in self.store.snapshot()['jobs']if j['id']==unrelated['job']['id'])['state'],'queued')

    def test_stale_or_changed_plan_cannot_create_a_second_adoption(self):
        changed=copy.deepcopy(self.plan);changed['result']['diagnostic']='different'
        with self.assertRaisesRegex(ValueError,'adoption changed'):self.adopt(changed)
        self.adopt()
        with self.assertRaisesRegex(ValueError,'base revision changed'):self.adopt()
        self.assertEqual(len(self.store.detail(self.cid)['revisions']),2)

    def test_failed_finalization_is_recorded_without_model_retry_or_old_revision_loss(self):
        with patch.object(AdminWorkflow,'_finalize_revision',side_effect=ValueError('Fixture failure'))as finalize:
            with self.assertRaisesRegex(ValueError,'Fixture failure'):self.adopt()
            with self.assertRaisesRegex(ValueError,'already attempted'):self.adopt()
            self.assertEqual(finalize.call_count,1)
        self.assertEqual(self.store.detail(self.cid)['candidate']['current_revision_id'],self.base['id'])
        self.assertEqual((self.root/self.base['manifest_ref']['path']).read_bytes(),self.old)


if __name__=='__main__':unittest.main()
