"""Offline guarded-review contracts; injected fixtures never invoke Meta."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import admin_muse_reviews as M
from admin_workflow import AdminWorkflow
from agent_memory import FeedbackMemory
from object_program import ref
from tests import test_twin_catalog as fixtures
from tests.test_portable_twin import glb


class OfflineClient:
    def __init__(self,*,fail=False,mutate=None):self.calls=[];self.fail=fail;self.mutate=mutate
    def json_result(self,prompt,schema,images,*,output,model,max_output_tokens):
        self.calls.append({'prompt':prompt,'images':images,'tokens':max_output_tokens,'model':model})
        if self.fail:raise RuntimeError('offline uncertain-charge fixture')
        role=prompt.split('You are the ',1)[1].split(' reviewer',1)[0]
        result={'role':role,'verdict':'review_required','summary':'Text metadata requires owner visual inspection.',
            'issues':[],'repair_targets':[],'physical_accuracy_verified':False,'simulation_ready':False,'visual_observation_performed':False}
        if self.mutate:self.mutate(result)
        output.mkdir(parents=True)
        receipt={'status':'complete_accounted','provider':'meta_api','model':model,'kind':'text','image_inputs':[],
            'estimated_cost_usd':'0.001000','fixture_only':True}
        (output/'meta-request.json').write_text(json.dumps(receipt))
        return result,receipt


class AdminMuseReviewTests(unittest.TestCase):
    def setUp(self):
        fixture=fixtures.TwinCatalogTests('test_search_limits_and_labels_do_not_claim_validated_physics');fixture.setUp()
        self.addCleanup(fixture.doCleanups);self.root=fixture.root.resolve();self.fixture=fixture
        fixture.doc['assets'][0]['sha256']=fixture.file('data/case/asset.glb',glb());fixture.publish()
        self.manifest={'source_ref':ref(self.root/'data/case/source/video.mp4',self.root),
            'interaction_ref':ref(self.root/'data/case/procedure/interaction.json',self.root),'object_program_ref':None,'scene_ref':None}
        self.memory=FeedbackMemory(self.root).retrieve(source_sha256=self.manifest['source_ref']['sha256'])
    def test_disabled_by_default_and_distinct_bounded_text_only_roles_cache_exact_artifacts(self):
        client=OfflineClient();off=M.build_reviewer(self.memory,root=self.root,client=client)
        with self.assertRaisesRegex(ValueError,'disabled'):off('evidence',self.manifest)
        self.assertEqual(client.calls,[])
        reviewer=M.build_reviewer(self.memory,root=self.root,client=client,allow_paid_api=True)
        reports=[reviewer(role,self.manifest)for role in M.ROLES]
        self.assertEqual(len(client.calls),4)
        self.assertEqual(len({call['prompt']for call in client.calls}),4)
        for call,report in zip(client.calls,reports):
            self.assertEqual(call['images'],[]);self.assertEqual(call['tokens'],8000)
            self.assertEqual(call['model'],'muse-spark-1.3');self.assertNotIn('data/case/',call['prompt'])
            self.assertFalse(report['independent_model_review']);self.assertFalse(report['visual_observation_performed'])
            self.assertTrue(M.verify_report(report,self.manifest,root=self.root))
        plan=M.prepare('evidence',self.manifest,self.memory,root=self.root)
        self.assertEqual(plan['summary']['state_variables'],self.fixture.doc['variables'])
        self.assertEqual(plan['summary']['assertions'],self.fixture.doc['assertions'])
        self.assertEqual(plan['summary']['component_observation_stats'][0]['total_saved_samples'],len(self.fixture.doc['evidence']))
        self.assertEqual(plan['version'],3);self.assertEqual(plan['review_protocol'],M.REVIEW_PROTOCOL)
        for role in M.ROLES:reviewer(role,self.manifest)
        self.assertEqual(len(client.calls),4,'Completed paid roles are cached by exact artifact/context identity')
        mutated=copy.deepcopy(self.manifest);mutated['source_ref']['sha256']='f'*64
        with self.assertRaises(ValueError):M.verify_report(reports[0],mutated,root=self.root)
        path=self.root/reports[0]['model_response_ref']['path'];path.write_text('{}')
        with self.assertRaises(ValueError):reviewer('evidence',self.manifest)
        self.assertEqual(len(client.calls),4,'Changed reports never trigger an automatic paid resend')
    def test_uncertain_attempt_and_code_edits_do_not_open_a_new_send_namespace(self):
        client=OfflineClient(fail=True);reviewer=M.build_reviewer(self.memory,root=self.root,client=client,allow_paid_api=True)
        with self.assertRaises(RuntimeError):reviewer('geometry',self.manifest)
        original=M.prepare
        def changed(*args,**kwargs):
            plan=original(*args,**kwargs);plan['implementation_sha256']='b'*64;plan['request_key']='c'*64;return plan
        with patch.object(M,'prepare',side_effect=changed):
            with self.assertRaisesRegex(ValueError,'no paid resend'):reviewer('geometry',self.manifest)
        self.assertEqual(len(client.calls),1)
    def test_invented_citations_are_rejected_and_preserved_without_retry(self):
        client=OfflineClient(mutate=lambda r:r.update(issues=[{'code':'unseen','severity':'warning','message':'No direct observation.',
            'component_ids':['invented_part'],'evidence_ids':[]}]))
        reviewer=M.build_reviewer(self.memory,root=self.root,client=client,allow_paid_api=True)
        with self.assertRaisesRegex(ValueError,'unknown saved ID'):reviewer('semantics',self.manifest)
        with self.assertRaisesRegex(ValueError,'no paid resend'):reviewer('semantics',self.manifest)
        self.assertEqual(len(client.calls),1)
    def test_workflow_stores_actual_role_provenance_without_holding_database_writer_during_transport(self):
        store=AdminWorkflow(self.root);queued=store.submit_upload(run_id='reviewed_upload',input_path=self.manifest['source_ref']['path'],processing_consent=True)
        observed=[]
        class InspectClient(OfflineClient):
            def json_result(inner,*args,**kwargs):
                with store.db()as db:
                    db.execute('BEGIN IMMEDIATE')
                    observed.append(db.execute('SELECT state FROM jobs').fetchone()['state'])
                return super(InspectClient,inner).json_result(*args,**kwargs)
        client=InspectClient();reviewer=M.build_reviewer(self.memory,root=self.root,client=client,allow_paid_api=True)
        def execute(job):return {'status':'complete_with_gaps','run_id':job['run_id'],**self.manifest,'artifacts':[]}
        contexts=[]
        def factory(job):
            contexts.append(job['memory_context'])
            return reviewer
        result=store.worker_once(execute,hosted_reviewer_factory=factory)
        self.assertEqual(result['status'],'complete');self.assertEqual(observed,['running']*4)
        self.assertEqual(contexts,[self.memory])
        revision=store.detail(queued['candidate']['id'])['revisions'][0]
        self.assertEqual(len(revision['manifest']['review']['hosted_roles']),4)
        self.assertEqual(store.public_index()['records'],[])
        store.approve(queued['candidate']['id'],revision['id'],revision['fingerprint'])
        self.assertEqual(len(store.public_index()['records']),1)
        self.assertEqual(len(client.calls),4,'Approval revalidates saved reports without calling Muse')
    def test_blocking_hosted_finding_prevents_automatic_structural_acceptance(self):
        from review_agents import review_revision
        client=OfflineClient(mutate=lambda r:r.update(verdict='blocked'))
        reviewer=M.build_reviewer(self.memory,root=self.root,client=client,allow_paid_api=True)
        review=review_revision(self.manifest,root=self.root,hosted_reviewer=reviewer,processing_consent=True)
        self.assertEqual(review['blocking_failures'],4)
        self.assertEqual(review['role_count'],4)
        self.assertFalse(review['automatic_publication'])
    def test_malformed_executor_asset_references_stop_before_hosted_review(self):
        store=AdminWorkflow(self.root);store.submit_upload(run_id='bad_asset_reference',input_path=self.manifest['source_ref']['path'],processing_consent=True)
        client=OfflineClient();reviewer=M.build_reviewer(self.memory,root=self.root,client=client,allow_paid_api=True)
        def execute(job):return {'status':'complete_with_gaps',**self.manifest,'artifacts':[self.fixture.doc['assets'][0]]}
        result=store.worker_once(execute,hosted_reviewer=reviewer)
        self.assertEqual(result['status'],'complete');self.assertEqual(client.calls,[])
        self.assertEqual(result['revision']['candidate_status'],'blocked_review')
        self.assertGreater(result['revision']['review']['blocking_failures'],0)
    def test_one_uncertain_role_preserves_revision_other_roles_and_successful_report_cache(self):
        from review_agents import review_revision
        class PartialClient(OfflineClient):
            def json_result(inner,prompt,*args,**kwargs):
                if 'You are the evidence reviewer'in prompt:
                    inner.calls.append({'failed_role':'evidence'})
                    raise RuntimeError('transport failed SECRET_MUST_NEVER_APPEAR')
                return super(PartialClient,inner).json_result(prompt,*args,**kwargs)
        store=AdminWorkflow(self.root);queued=store.submit_upload(run_id='partial_review',input_path=self.manifest['source_ref']['path'],processing_consent=True)
        client=PartialClient();reviewer=M.build_reviewer(self.memory,root=self.root,client=client,allow_paid_api=True)
        def execute(job):return {'status':'complete_with_gaps','run_id':job['run_id'],**self.manifest,'artifacts':[]}
        result=store.worker_once(execute,hosted_reviewer=reviewer)
        self.assertEqual(result['status'],'complete')
        revision=store.detail(queued['candidate']['id'])['revisions'][0];review=revision['manifest']['review']
        self.assertEqual(review['hosted_roles_completed'],3);self.assertEqual(review['hosted_roles_expected'],4)
        self.assertEqual(review['hosted_review_status'],'partial');self.assertEqual(review['blocking_failures'],0)
        self.assertEqual(review['hosted_errors'][0]['role'],'evidence')
        self.assertEqual(review['hosted_errors'][0]['code'],'transport_failed_charge_uncertain')
        self.assertNotIn('SECRET',json.dumps(review));self.assertEqual(review['role_count'],4)
        self.assertTrue(review['hosted_warnings']);self.assertEqual(len(client.calls),4)
        again=review_revision(self.manifest,root=self.root,hosted_reviewer=reviewer,processing_consent=True)
        self.assertEqual(again['hosted_roles_completed'],3);self.assertEqual(len(client.calls),4)
        self.assertEqual(again['hosted_errors'][0]['code'],'prior_attempt_preserved_no_retry')
        store.approve(queued['candidate']['id'],revision['id'],revision['fingerprint'])
        self.assertEqual(len(store.public_index()['records']),1)
        self.assertEqual(len(client.calls),4,'Owner approval validates actual saved reports without resending failed roles')
    def test_all_budget_or_factory_failures_remain_reviewable_without_model_claims(self):
        from meta_api import MetaError
        for failure in ['role_guard','factory']:
            with self.subTest(failure=failure):
                store=AdminWorkflow(self.root);queued=store.submit_upload(run_id='unavailable_'+failure,input_path=self.manifest['source_ref']['path'],processing_consent=True)
                calls=[]
                def blocked(role,manifest):
                    calls.append(role);raise MetaError('task_budget_admission_blocked SENSITIVE_NOT_A_DIAGNOSTIC')
                def factory(job):raise RuntimeError('SENSITIVE_FACTORY_ERROR')
                def execute(job):return {'status':'complete_with_gaps',**self.manifest,'artifacts':[]}
                options={'hosted_reviewer':blocked}if failure=='role_guard'else{'hosted_reviewer_factory':factory}
                result=store.worker_once(execute,**options)
                self.assertEqual(result['status'],'complete')
                revision=store.detail(queued['candidate']['id'])['revisions'][0];review=revision['manifest']['review']
                self.assertEqual(review['hosted_roles_completed'],0);self.assertEqual(review['hosted_roles_expected'],4)
                self.assertEqual(len(review['hosted_errors']),4);self.assertEqual(review['hosted_review_status'],'unavailable')
                self.assertNotIn('SENSITIVE',json.dumps(review));self.assertEqual(review['blocking_failures'],0)
                self.assertTrue(all(not e['independent_model_review']and not e['automatic_retry']for e in review['hosted_errors']))
                self.assertEqual(calls,list(M.ROLES)if failure=='role_guard'else[])
                self.assertEqual(revision['manifest']['interaction_ref'],self.manifest['interaction_ref'])
                self.assertEqual(store.public_index()['records'],[])
    def test_explicit_initial_recovery_after_uncertain_hosted_attempt_never_resends_that_role(self):
        store=AdminWorkflow(self.root);queued=store.submit_upload(run_id='recover_review',input_path=self.manifest['source_ref']['path'],processing_consent=True)
        client=OfflineClient(fail=True);reviewer=M.build_reviewer(self.memory,root=self.root,client=client,allow_paid_api=True)
        # Reproduce the old worker's preserved paid attempt and failed job.
        with self.assertRaises(RuntimeError):reviewer('evidence',self.manifest)
        with store.db()as db:db.execute("UPDATE jobs SET state='failed'WHERE id=?",(queued['job']['id'],))
        store.resume_initial(queued['candidate']['id'],processing_consent=True)
        def execute(job):return {'status':'complete_with_gaps',**self.manifest,'artifacts':[]}
        result=store.worker_once(execute,hosted_reviewer=reviewer)
        self.assertEqual(result['status'],'complete')
        revision=store.detail(queued['candidate']['id'])['revisions'][0]
        errors=revision['manifest']['review']['hosted_errors']
        self.assertEqual(errors[0]['role'],'evidence');self.assertEqual(errors[0]['code'],'prior_attempt_preserved_no_retry')
        self.assertEqual(len(client.calls),4,'Only the other three new guarded roles were attempted')
        self.assertEqual(revision['manifest']['interaction_ref'],self.manifest['interaction_ref'])
    def old_v1_attempt(self,role,*,status='complete_accounted',output_tokens=1000,reasoning_tokens=998):
        plan=M.prepare(role,self.manifest,self.memory,root=self.root)
        plan.pop('review_protocol');plan.pop('protocol_upgrade');plan.pop('request_key')
        plan.pop('evidence_packet')
        plan.update(version=1,max_output_tokens=1000,implementation_sha256=M.V1_SOURCE_SHA256)
        plan['request_key']=M.digest(plan)
        folder=self.root/'.private/admin-muse-reviews'/M.review_namespace(role,plan['artifact_binding'],self.memory['context_sha256'])
        request=folder/'attempt/request';request.mkdir(parents=True)
        def save(path,value):path.write_text(json.dumps(value));path.chmod(0o600)
        save(folder/'plan.json',plan)
        receipt={'status':status,'provider':'meta_api','model':M.MODEL,'kind':'text',
            'endpoint':'/v1/chat/completions','automatic_retries':False,'payload_sha256':M.digest(M._text_payload(plan)),
            'usage':{'input_tokens':100,'output_tokens':output_tokens,'reasoning_tokens':reasoning_tokens},
            'estimated_cost_usd':'0.001000'}
        if status!='complete_accounted':receipt.pop('usage')
        save(request/'meta-request.json',receipt)
        save(folder/'attempt/failure.json',{'status':'preserved_no_automatic_retry','error_type':'MetaError'})
        return folder
    def test_v1_accounted_cap_completion_requires_explicit_protocol_upgrade_and_keeps_old_bytes(self):
        old=self.old_v1_attempt('geometry');old_bytes={p.relative_to(old).as_posix():p.read_bytes()for p in old.rglob('*')if p.is_file()}
        client=OfflineClient();default=M.build_reviewer(self.memory,root=self.root,client=client,allow_paid_api=True)
        with self.assertRaisesRegex(ValueError,'explicit reviewed protocol upgrade'):default('geometry',self.manifest)
        self.assertEqual(client.calls,[])
        upgraded=M.build_reviewer(self.memory,root=self.root,client=client,allow_paid_api=True,allow_protocol_upgrade=True)
        result=upgraded('geometry',self.manifest)
        self.assertEqual(len(client.calls),1);self.assertEqual(client.calls[0]['tokens'],8000)
        self.assertEqual(result['review_protocol'],M.REVIEW_PROTOCOL)
        plan=json.loads((self.root/result['plan_ref']['path']).read_text())
        self.assertEqual(plan['protocol_upgrade']['old_output_tokens'],1000)
        self.assertFalse(plan['protocol_upgrade']['old_request_retried'])
        self.assertEqual({p.relative_to(old).as_posix():p.read_bytes()for p in old.rglob('*')if p.is_file()},old_bytes)
        self.assertNotEqual(Path(result['plan_ref']['path']).parent.name,old.name)
        default('geometry',self.manifest)
        self.assertEqual(len(client.calls),1,'Completed V2 may be reused without authorizing another migration')
    def test_uncertain_wrong_usage_or_partial_v1_outputs_never_upgrade_or_resend(self):
        for role,parameters in [('evidence',{'status':'transport_error_charge_uncertain'}),
                                ('geometry',{'output_tokens':999}),('mechanics',{'reasoning_tokens':999})]:
            with self.subTest(role=role):
                self.old_v1_attempt(role,**parameters)
                client=OfflineClient();reviewer=M.build_reviewer(self.memory,root=self.root,client=client,allow_paid_api=True,allow_protocol_upgrade=True)
                with self.assertRaisesRegex(ValueError,'no paid resend'):reviewer(role,self.manifest)
                self.assertEqual(client.calls,[])
        old=self.old_v1_attempt('semantics');(old/'attempt/request/response.pending.json').write_text('{}')
        client=OfflineClient();reviewer=M.build_reviewer(self.memory,root=self.root,client=client,allow_paid_api=True,allow_protocol_upgrade=True)
        with self.assertRaisesRegex(ValueError,'no paid resend'):reviewer('semantics',self.manifest)
        self.assertEqual(client.calls,[])
    def test_owner_reason_can_override_hosted_findings_only_and_stays_private(self):
        store=AdminWorkflow(self.root);queued=store.submit_upload(run_id='hosted_override',input_path=self.manifest['source_ref']['path'],processing_consent=True)
        def mark_geometry(result):
            if result['role']=='geometry':result.update(verdict='blocked')
        client=OfflineClient(mutate=mark_geometry)
        reviewer=M.build_reviewer(self.memory,root=self.root,client=client,allow_paid_api=True)
        def execute(job):return {'status':'complete_with_gaps',**self.manifest,'artifacts':[]}
        done=store.worker_once(execute,hosted_reviewer=reviewer)
        self.assertEqual(done['revision']['candidate_status'],'blocked_review')
        revision=store.detail(queued['candidate']['id'])['revisions'][0];cid=queued['candidate']['id']
        for reason in [None,'','   ']:
            with self.subTest(reason=reason),self.assertRaisesRegex(ValueError,'override reason'):
                store.approve(cid,revision['id'],revision['fingerprint'],review_override_reason=reason)
        with self.assertRaises(ValueError):store.approve(cid,revision['id'],revision['fingerprint'],review_override_reason='x'*2001)
        with self.assertRaises(ValueError):store.approve(cid,revision['id'],revision['fingerprint'],review_override_reason='Inspected prototype.',actor='agent')
        reason='PRIVATE_OWNER_REASON: I inspected the draft and accept estimated geometry for this prototype.'
        approved=store.approve(cid,revision['id'],revision['fingerprint'],review_override_reason=reason)
        self.assertTrue(approved['decision']['human_review_override']);self.assertEqual(approved['decision']['review_override_reason'],reason)
        self.assertFalse(approved['decision']['physical_accuracy_verified'])
        public=store.public_index();self.assertTrue(public['records'][0]['human_review_override'])
        self.assertNotIn(reason,json.dumps(public));self.assertNotIn('review_override_reason',json.dumps(public))
        self.assertEqual(len(client.calls),4,'Approval and public selection only verify saved reports')
        self.assertEqual(store.detail(cid)['revisions'][0]['manifest']['review']['blocking_failures'],1,'Model findings remain intact after override')
        (self.root/'data/case/asset.glb').write_bytes(b'changed asset')
        self.assertEqual(store.public_index()['records'],[],'A human override cannot bypass later asset corruption')
        with self.assertRaises(ValueError):store.approve(cid,revision['id'],revision['fingerprint'],review_override_reason=reason)
    def test_structural_failures_and_tampered_override_decisions_cannot_be_bypassed(self):
        store=AdminWorkflow(self.root);queued=store.submit_upload(run_id='broken_structure',input_path=self.manifest['source_ref']['path'],processing_consent=True)
        def execute(job):return {'status':'complete_with_gaps',**self.manifest,'artifacts':[self.fixture.doc['assets'][0]]}
        done=store.worker_once(execute);revision=store.detail(queued['candidate']['id'])['revisions'][0]
        self.assertEqual(done['revision']['candidate_status'],'blocked_review')
        with self.assertRaises(ValueError):store.approve(queued['candidate']['id'],revision['id'],revision['fingerprint'],review_override_reason='I accept an estimated prototype.')
        other=store.submit_upload(run_id='decision_tamper',input_path=self.manifest['source_ref']['path'],processing_consent=True)
        client=OfflineClient(mutate=lambda r:r.update(verdict='blocked'))
        reviewer=M.build_reviewer(self.memory,root=self.root,client=client,allow_paid_api=True)
        def valid(job):return {'status':'complete_with_gaps',**self.manifest,'artifacts':[]}
        store.worker_once(valid,hosted_reviewer=reviewer)
        rev=store.detail(other['candidate']['id'])['revisions'][0]
        decision=store.approve(other['candidate']['id'],rev['id'],rev['fingerprint'],review_override_reason='Inspected estimated draft.')['decision']
        with store.db()as db:path=store.root/db.execute('SELECT record_path FROM decisions WHERE id=?',(decision['id'],)).fetchone()[0]
        saved=json.loads(path.read_text());saved['review_override_reason']='';path.write_text(json.dumps(saved))
        self.assertEqual(store.public_index()['records'],[],'Changed private approval record fails its immutable hash')

if __name__=='__main__':unittest.main()
