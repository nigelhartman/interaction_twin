"""Real durable workflow lifecycle; no inference, GPU or synthetic approvals."""
import copy
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from admin_workflow import AdminWorkflow
from object_program import ref
from tests import test_twin_catalog as catalog_fixtures
from tests.test_portable_twin import glb


class AdminWorkflowTests(unittest.TestCase):
    def setUp(self):
        fixture=catalog_fixtures.TwinCatalogTests('test_search_limits_and_labels_do_not_claim_validated_physics');fixture.setUp()
        self.addCleanup(fixture.doCleanups);self.fixture=fixture;self.root=fixture.root.resolve()
        fixture.doc['assets'][0]['sha256']=fixture.file('data/case/asset.glb',glb())
        fixture.publish();self.store=AdminWorkflow(self.root)
        self.source='data/case/source/video.mp4'
    def submit(self,consent=True,run='run_test'):
        return self.store.submit_upload(run_id=run,input_path=self.source,processing_consent=consent)
    def executor(self,job):
        return {'status':'complete_with_gaps','run_id':job['run_id'],
            'interaction_ref':ref(self.root/'data/case/procedure/interaction.json',self.root),
            'artifacts':[ref(self.root/'data/case/asset.glb',self.root)],'new_api_calls':0}
    def ready(self):
        queued=self.submit();result=self.store.worker_once(self.executor)
        self.assertEqual(result['status'],'complete')
        detail=self.store.detail(queued['candidate']['id'])
        self.assertEqual(detail['candidate']['status'],'awaiting_review')
        return queued['candidate']['id'],detail['revisions'][0]
    def test_consent_queue_restart_four_review_roles_and_no_automatic_publication(self):
        queued=self.submit(False)
        with patch.object(self,'executor')as executor:
            self.assertEqual(self.store.worker_once(executor)['status'],'idle');executor.assert_not_called()
        self.store.authorize_processing(queued['candidate']['id'])
        restarted=AdminWorkflow(self.root);result=restarted.worker_once(self.executor)
        self.assertEqual(result['status'],'complete')
        detail=restarted.detail(queued['candidate']['id']);revision=detail['revisions'][0]
        self.assertEqual([r['role'] for r in revision['manifest']['review']['roles']],['evidence','geometry','mechanics','semantics'])
        self.assertTrue(revision['manifest']['review']['all_required_roles_completed'])
        self.assertEqual(revision['manifest']['review']['blocking_failures'],0)
        self.assertFalse(revision['manifest']['human_guidance_used'])
        self.assertFalse(revision['manifest']['training_performed'])
        self.assertTrue(revision['manifest']['memory_only_not_weight_training'])
        self.assertEqual(restarted.public_index()['records'],[])
        self.assertEqual(self.submit()['candidate']['id'],queued['candidate']['id'],'Upload run IDs are idempotent')
    def test_only_exact_current_revision_owner_approval_releases_packager_selection(self):
        cid,revision=self.ready()
        with self.assertRaises(ValueError):self.store.approve(cid,revision['id'],'f'*64)
        with self.assertRaises(ValueError):self.store.approve(cid,revision['id'],revision['fingerprint'],actor='agent')
        approved=self.store.approve(cid,revision['id'],revision['fingerprint'])
        selected=self.store.public_index()['records']
        self.assertEqual(len(selected),1);self.assertEqual(selected[0]['revision_id'],revision['id'])
        self.assertEqual(approved['decision']['human_review_kind'],'nigel_curatorial_approval')
        self.assertFalse(approved['decision']['physical_accuracy_verified'])
        self.assertNotIn('feedback',json.dumps(selected))
        (self.root/'data/case/asset.glb').write_bytes(b'changed')
        self.assertEqual(self.store.public_index()['records'],[],'Changed approved artifacts are withheld')
        with self.assertRaises(ValueError):self.store.approve(cid,revision['id'],revision['fingerprint'])
    def test_rejection_scoped_memory_real_repair_revision_and_two_attempt_bound(self):
        cid,revision=self.ready();original=(self.root/revision['manifest']['interaction_ref']['path']).read_bytes()
        rejected=self.store.reject(cid,revision['id'],'The mesh cavity is wrong and the hinge looks wrong.')
        self.assertEqual(rejected['repair_plan']['targets'],['reasoning','reconstruction','mechanics'])
        self.assertEqual(self.store.public_index()['records'],[])
        with self.assertRaises(ValueError):self.store.request_repair(cid,revision['id'],rejected['repair_plan']['id'])
        job=self.store.request_repair(cid,revision['id'],rejected['repair_plan']['id'],processing_consent=True)
        observed=[]
        def repair(payload):
            observed.append(payload)
            self.assertEqual(payload['kind'],'repair');self.assertTrue(payload['memory_context']['lessons'])
            return self.executor(payload)
        self.assertEqual(self.store.worker_once(repair)['status'],'complete')
        current=self.store.detail(cid)['revisions'][0]
        self.assertNotEqual(current['id'],revision['id']);self.assertEqual(current['manifest']['base_revision_id'],revision['id'])
        self.assertTrue(current['manifest']['human_guidance_used'])
        self.assertFalse(current['manifest']['training_performed'])
        self.assertEqual((self.root/revision['manifest']['interaction_ref']['path']).read_bytes(),original)
        with self.assertRaises(ValueError):self.store.approve(cid,revision['id'],revision['fingerprint'])
        rejected2=self.store.reject(cid,current['id'],'Try tracking again.',repair_targets=['tracking'])
        self.store.request_repair(cid,current['id'],rejected2['repair_plan']['id'],processing_consent=True)
        self.store.worker_once(repair)
        current3=self.store.detail(cid)['revisions'][0]
        rejected3=self.store.reject(cid,current3['id'],'One more?',repair_targets=['reasoning'])
        self.assertEqual(rejected3['repair_plan']['status'],'attempt_limit_reached')
        with self.assertRaises(ValueError):self.store.request_repair(cid,current3['id'],rejected3['repair_plan']['id'],processing_consent=True)
    def test_bad_source_program_is_review_blocked_and_cannot_be_approved(self):
        self.fixture.doc['source']['original_sha256']='e'*64;self.fixture.doc['source']['sha256']='e'*64;self.fixture.publish()
        queued=self.submit();done=self.store.worker_once(self.executor)
        detail=self.store.detail(queued['candidate']['id']);revision=detail['revisions'][0]
        self.assertEqual(detail['candidate']['status'],'blocked_review')
        self.assertGreater(revision['manifest']['review']['blocking_failures'],0)
        with self.assertRaises(ValueError):self.store.approve(queued['candidate']['id'],revision['id'],revision['fingerprint'])
    def test_prior_approval_survives_private_rejection_and_repair_until_replacement_approved(self):
        cid,first=self.ready();self.store.approve(cid,first['id'],first['fingerprint'])
        def selected():return [r['revision_id'] for r in self.store.public_index()['records']]
        rejected=self.store.reject(cid,first['id'],'Please improve the result.',repair_targets=['reasoning'])
        self.assertEqual(selected(),[first['id']],'Private feedback does not withdraw a published approval')
        self.store.request_repair(cid,first['id'],rejected['repair_plan']['id'],processing_consent=True)
        self.assertEqual(selected(),[first['id']],'Queued repair must preserve the approved public revision')
        def repair(payload):
            self.assertEqual(selected(),[first['id']],'Processing repair must preserve the approved revision')
            return self.executor(payload)
        self.store.worker_once(repair)
        second=self.store.detail(cid)['revisions'][0]
        self.assertEqual(selected(),[first['id']],'Awaiting review does not auto-publish a replacement')
        self.store.reject(cid,second['id'],'The new result is still wrong.',repair_targets=['reasoning'])
        self.assertEqual(selected(),[first['id']],'Rejecting the unapproved replacement preserves the old public model')
        self.store.approve(cid,second['id'],second['fingerprint'])
        self.assertEqual(selected(),[second['id']],'Only a fresh exact approval advances the public model')
    def test_worker_exception_and_expired_lease_do_not_repeat_paid_jobs(self):
        queued=self.submit()
        def failure(job):raise RuntimeError('fixed adapter unavailable')
        self.assertEqual(self.store.worker_once(failure)['status'],'failed')
        with patch.object(self,'executor')as executor:
            self.assertEqual(self.store.worker_once(executor)['status'],'idle');executor.assert_not_called()
        other=self.submit(run='run_interrupted')
        with self.store.db()as db:
            db.execute("UPDATE jobs SET state='running',lease_token='old',lease_until=? WHERE id=?",(time.time()-1,other['job']['id']))
        self.assertEqual(self.store.recover_expired_jobs(),1)
        self.assertEqual(self.store.worker_once(self.executor)['status'],'idle')
    def test_explicit_initial_resume_retains_run_source_and_frozen_context(self):
        queued=self.submit();seen=[]
        def unavailable(job):seen.append(job);return {'status':'pending','diagnostic':'GPU adapter unavailable; caches preserved.'}
        self.store.worker_once(unavailable)
        with self.assertRaises(ValueError):self.store.resume_initial(queued['candidate']['id'],processing_consent=False)
        self.store.memory.add(candidate_id='other',revision_id='other',source_sha256='a'*64,feedback='New generic preference.',scopes=['global_method'])
        resumed=self.store.resume_initial(queued['candidate']['id'])
        self.assertNotEqual(resumed['job']['id'],queued['job']['id'])
        self.store.worker_once(unavailable)
        self.assertEqual(seen[0]['run_id'],seen[1]['run_id'])
        self.assertEqual(seen[0]['source_ref'],seen[1]['source_ref'])
        self.assertEqual(seen[0]['memory_context'],seen[1]['memory_context'])
        self.assertEqual(seen[1]['resume_of_job_id'],queued['job']['id'])
        self.assertIn('stop_on_ambiguous',seen[1]['resume_policy'])
        self.store.resume_initial(queued['candidate']['id']);self.store.worker_once(unavailable)
        with self.assertRaises(ValueError):self.store.resume_initial(queued['candidate']['id'])

    def test_repair_plan_tampering_and_hidden_paths_are_rejected(self):
        cid,revision=self.ready();rejected=self.store.reject(cid,revision['id'],'Review the cavity.')
        with self.store.db()as db:
            row=db.execute('SELECT * FROM repair_plans WHERE id=?',(rejected['repair_plan']['id'],)).fetchone()
        p=self.root/row['record_path'];raw=json.loads(p.read_text());raw['targets']=['arbitrary_shell'];p.write_text(json.dumps(raw))
        with self.assertRaises(ValueError):self.store.request_repair(cid,revision['id'],rejected['repair_plan']['id'],processing_consent=True)
        with self.assertRaises(ValueError):self.store.submit_upload(run_id='bad',input_path='.secrets/meta.env',processing_consent=True)
    def test_review_only_preserves_original_context_even_after_new_feedback(self):
        cid,revision=self.ready()
        self.store.memory.add(candidate_id=cid,revision_id=revision['id'],source_sha256=revision['manifest']['source_ref']['sha256'],
            feedback='New feedback must require a repair, not bypass a failed review namespace.')
        self.store.request_review(cid,revision['id'],processing_consent=True)
        observed=[]
        def review_only(job):
            observed.append(job)
            return self.executor(job)
        self.store.worker_once(review_only)
        self.assertEqual(observed[0]['kind'],'review')
        self.assertEqual(observed[0]['memory_context']['lessons'],[])
        self.assertEqual(observed[0]['memory_context']['context_sha256'],revision['manifest']['memory_context_sha256'])
        self.assertFalse(observed[0]['allow_review_protocol_upgrade'])

if __name__=='__main__':unittest.main()
