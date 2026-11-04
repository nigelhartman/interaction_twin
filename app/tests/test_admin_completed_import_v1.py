"""Private finished-run adoption never invokes inference or the global queue."""
import copy
import json
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import admin_completed_import_v1 as C
from admin_workflow import AdminWorkflow
from tests import test_admin_workflow as workflow_fixture


class CompletedPrivateImportTests(unittest.TestCase):
    def setUp(self):
        fixture=workflow_fixture.AdminWorkflowTests('test_consent_queue_restart_four_review_roles_and_no_automatic_publication')
        fixture.setUp();self.addCleanup(fixture.doCleanups)
        self.root=fixture.root;self.store=fixture.store;self.fixture=fixture
        self.program='data/case/procedure/interaction.json';self.source=fixture.source
        self.state={'id':'run-finished-fixture','status':'complete_with_gaps','phases':[{'name':'procedure','status':'complete'}],
                    'source':{'path':str(self.root/self.source),'sha256':C.ref(self.root/self.source,self.root)['sha256']},
                    'artifacts':{'interaction':self.program}}
        self.path=self.root/'data/flow-runs/run-finished-fixture/run.json';self.path.parent.mkdir(parents=True)
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.state));self.state_ref=C.ref(self.path,self.root)

    def prepare(self):return C.build(self.state_ref,root=self.root)

    def test_read_only_preflight_has_four_structural_roles_and_no_workflow_mutation(self):
        before={p.relative_to(self.root).as_posix():p.read_bytes()for p in (self.root/'.private').rglob('*')if p.is_file()}
        plan=self.prepare()
        after={p.relative_to(self.root).as_posix():p.read_bytes()for p in (self.root/'.private').rglob('*')if p.is_file()}
        self.assertEqual(before,after);self.assertEqual(self.store.snapshot()['candidates'],[])
        review=plan['structural_preflight'];self.assertEqual(review['role_count'],4)
        self.assertTrue(review['all_required_roles_completed']);self.assertEqual(review['blocking_failures'],0)
        self.assertTrue(all(r['mode']=='deterministic_structural_review'for r in review['roles']))
        self.assertEqual(review['hosted_roles'],[]);self.assertEqual(plan['new_api_calls'],0)
        self.assertIn(self.program,[r['path']for r in plan['result']['artifacts']])
        for frame in self.fixture.fixture.doc['evidence']:
            for observation in frame['observations']:
                self.assertIn(observation['mask_path'],[r['path']for r in plan['result']['artifacts']])

    def test_register_is_idempotent_private_and_cannot_drain_an_unrelated_queued_job(self):
        unrelated=self.store.submit_upload(run_id='run-unrelated',input_path=self.source,processing_consent=True)
        before=(self.root/self.program).read_bytes();plan=self.prepare()
        with patch.object(AdminWorkflow,'worker_once',side_effect=AssertionError('Never drain the global queue')):
            result=C.register(plan,root=self.root)
            again=C.register(plan,root=self.root)
        self.assertEqual(result['status'],'registered_private_review');self.assertEqual(again['status'],'reused_exact_private_revision')
        self.assertEqual(result['revision_id'],again['revision_id'])
        detail=self.store.detail(result['candidate_id']);self.assertEqual(len(detail['revisions']),1)
        self.assertEqual(detail['candidate']['status'],'awaiting_review')
        self.assertEqual(detail['revisions'][0]['manifest']['run_id'],self.state['id'])
        self.assertFalse(detail['candidate']['processing_consent'],'No hosted processing consent invented')
        job=next(j for j in self.store.snapshot()['jobs']if j['id']==unrelated['job']['id'])
        self.assertEqual(job['state'],'queued');self.assertEqual(self.store.public_index()['records'],[])
        self.assertEqual((self.root/self.program).read_bytes(),before)
        self.assertEqual(detail['revisions'][0]['manifest']['review']['hosted_roles'],[])

    def test_unfinished_source_mismatch_and_stale_prepared_bytes_fail_before_registration(self):
        for status in ['running','queued','failed','pending']:
            self.state['status']=status;self.save()
            with self.subTest(status=status),self.assertRaisesRegex(ValueError,'must finish'):self.prepare()
        self.state['status']='complete';self.state['source']['sha256']='f'*64;self.save()
        with self.assertRaisesRegex(ValueError,'source differs'):self.prepare()
        self.state['source']['sha256']=C.ref(self.root/self.source,self.root)['sha256'];self.save()
        plan=self.prepare();changed=copy.deepcopy(plan);changed['result']['new_api_calls']=1
        with self.assertRaisesRegex(ValueError,'Prepared completed import changed'):C.register(changed,root=self.root)
        self.assertEqual(self.store.snapshot()['candidates'],[])

    def test_active_job_is_not_interrupted_and_new_import_stays_nonrunnable(self):
        unrelated=self.store.submit_upload(run_id='run-running',input_path=self.source,processing_consent=True)
        with self.store.db()as db:
            db.execute("UPDATE jobs SET state='running',lease_token='fixture-only',lease_until=? WHERE id=?",(time.time()+100,unrelated['job']['id']))
        with self.assertRaisesRegex(ValueError,'Another owner job is running'):C.register(self.prepare(),root=self.root)
        rows=self.store.snapshot()['jobs'];self.assertEqual(next(j for j in rows if j['id']==unrelated['job']['id'])['state'],'running')
        imported=[j for j in rows if j['id']!=unrelated['job']['id']]
        self.assertEqual(len(imported),1);self.assertEqual(imported[0]['state'],'pending_consent')

    def test_failed_local_finalization_is_preserved_and_not_automatically_retried(self):
        plan=self.prepare()
        with patch.object(AdminWorkflow,'_finalize_revision',side_effect=ValueError('Fixture structural failure'))as finalize:
            with self.assertRaisesRegex(ValueError,'Fixture'):C.register(plan,root=self.root)
            with self.assertRaisesRegex(ValueError,'no automatic retry'):C.register(plan,root=self.root)
            self.assertEqual(finalize.call_count,1)
        self.assertEqual(self.store.snapshot()['jobs'][0]['state'],'failed')

    def test_new_private_program_and_native_asset_are_not_anonymously_served(self):
        from scripts.serve_workspace import public_static
        result=C.register(self.prepare(),root=self.root)
        manifest=self.store.detail(result['candidate_id'])['revisions'][0]['manifest']
        for reference in [manifest['interaction_ref'],manifest['source_ref'],*manifest['artifacts']]:
            self.assertFalse(public_static(self.root/reference['path'],root=self.root))
        self.assertEqual(self.store.public_index()['records'],[])

    def test_confinement_precedes_hashing_any_hidden_or_external_input(self):
        import object_program
        with patch.object(object_program,'sha',side_effect=AssertionError('A forbidden input was read')):
            for p in [self.root/'.secrets/credential.env',self.root/'.private/a.json',self.root/'data/../outside.json',Path('/etc/passwd')]:
                with self.subTest(path=str(p)),self.assertRaisesRegex(ValueError,'ordinary data/output'):C.ref(p,self.root)


if __name__=='__main__':unittest.main()
