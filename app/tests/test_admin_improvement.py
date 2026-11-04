"""Renewable lease + durable bounded improvement tests; only local injected adapters."""
import copy
import json
from pathlib import Path
import threading
import time
import unittest
from unittest.mock import Mock, patch

from admin_execution_policy import WorkerLease, LeaseLost, improvement_policy, review_assessment, improves, TASK_PLAN
from admin_workflow import AdminWorkflow
from admin_muse_reviews import artifact_binding, build_reviewer
from object_program import ref
from tests import test_admin_workflow as workflow_fixtures
from tests.test_admin_muse_reviews import OfflineClient

ROOT=Path(__file__).resolve().parents[1]

class ImprovementContracts(unittest.TestCase):
    def setUp(self):
        harness=workflow_fixtures.AdminWorkflowTests('test_consent_queue_restart_four_review_roles_and_no_automatic_publication');harness.setUp()
        self.addCleanup(harness.doCleanups);self.h=harness;self.root=harness.root;self.store=harness.store
        budget=self.root/TASK_PLAN;budget.parent.mkdir(parents=True);budget.write_bytes((ROOT/TASK_PLAN).read_bytes())
        self.policy=improvement_policy(self.root);self.client=OfflineClient();self.visual_calls=[]
    def submit(self):
        return self.store.submit_upload(run_id='automatic_test',input_path=self.h.source,processing_consent=True,improvement_policy=self.policy)
    def execute(self,job):
        job['execution_lease'].guard('fixture_build')
        result=self.h.executor(job)
        number=job.get('improvement_round',0)
        if number:
            doc=copy.deepcopy(self.h.fixture.doc);doc['summary']='New immutable fixture revision '+str(number)
            path=self.root/f'data/rounds/{number}/interaction.json';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(doc))
            result['interaction_ref']=ref(path,self.root)
        return result
    def factory(self,job):
        return build_reviewer(job['memory_context'],root=self.root,client=self.client,allow_paid_api=True)
    def visual(self,metrics,*,safe=True,targets=None):
        def factory(job):
            def review(manifest):
                n=job.get('improvement_round',0);self.visual_calls.append(n)
                assessment={'measurement_kind':'deterministic_local_metrics','comparison_key':'a'*64,'metrics':metrics[n],
                    'repair_targets':targets if targets is not None else ['mechanics'],'automatic_repair_safe':safe}
                body={'artifact_binding':artifact_binding(manifest),'improvement_assessment':assessment}
                path=self.root/f'.private/measurements/{job["id"]}.json';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(body));path.chmod(0o600)
                return dict(status='complete',artifact_binding=artifact_binding(manifest),report_ref=ref(path,self.root),improvement_assessment=assessment)
            return review
        return factory
    def metrics(self,errors):
        return [dict(blocking_issues=0,missing_components=0,alignment_error=e,coverage=.8)for e in errors]
    def run_one(self,visual,*,budget=True,execute=None):
        return self.store.worker_once(execute or self.execute,hosted_reviewer_factory=self.factory,
            visual_reviewer_factory=visual,budget_checker=lambda job:budget)
    def test_three_rounds_bound_new_revisions_and_no_human_memory_or_publication(self):
        queued=self.submit();cid=queued['candidate']['id'];visual=self.visual(self.metrics([.5,.3,.1]))
        for _ in range(3):self.assertEqual(self.run_one(visual)['status'],'complete')
        self.assertEqual(self.run_one(visual)['status'],'idle')
        detail=self.store.detail(cid);rounds=detail['improvement_rounds']
        self.assertEqual([r['round_number']for r in rounds],[0,1,2]);self.assertEqual(rounds[-1]['decision'],'round_limit_reached')
        self.assertEqual(len(detail['revisions']),3);self.assertEqual(sum(j['kind']=='repair'for j in detail['jobs']),2)
        self.assertTrue(all(not r['automatic_approval']and not r['automatic_publication']for r in rounds))
        self.assertEqual(detail['feedback'],[]);self.assertEqual(self.store.public_index()['records'],[])
        self.assertEqual(detail['candidate']['current_revision_id'],rounds[-1]['best_revision_id'])
    def test_regression_stops_and_retains_best_exact_revision(self):
        q=self.submit();v=self.visual(self.metrics([.2,.4]))
        self.run_one(v);first=self.store.detail(q['candidate']['id'])['improvement_rounds'][0]['best_revision_id']
        self.run_one(v);detail=self.store.detail(q['candidate']['id'])
        self.assertEqual(detail['improvement_rounds'][-1]['decision'],'plateau_or_regression')
        self.assertEqual(detail['candidate']['current_revision_id'],first);self.assertEqual(len(detail['revisions']),2)
        self.assertEqual(self.run_one(v)['status'],'idle')
    def test_unchanged_artifacts_stop_even_if_report_claims_improved_metric(self):
        q=self.submit();v=self.visual(self.metrics([.5,.1]))
        self.run_one(v,execute=self.h.executor);self.run_one(v,execute=self.h.executor)
        detail=self.store.detail(q['candidate']['id']);self.assertEqual(detail['improvement_rounds'][-1]['decision'],'plateau_or_regression')
    def test_unknown_visual_budget_and_unsafe_repair_stop_without_second_job(self):
        for index,mode in enumerate(['unknown','budget','unsafe','partial']):
            with self.subTest(mode=mode):
                store=AdminWorkflow(self.root)
                q=store.submit_upload(run_id='stop_'+mode,input_path=self.h.source,processing_consent=True,improvement_policy=self.policy)
                v=(lambda job:lambda manifest:{'status':'unavailable'})if mode=='unknown'else self.visual(self.metrics([.2]),safe=mode!='unsafe')
                reviewer=(lambda job:lambda role,manifest: (_ for _ in ()).throw(RuntimeError('unavailable')))if mode=='partial'else self.factory
                store.worker_once(self.execute,hosted_reviewer_factory=reviewer,visual_reviewer_factory=v,budget_checker=lambda job:mode!='budget')
                detail=store.detail(q['candidate']['id']);self.assertEqual(len(detail['jobs']),1)
                expected={'unknown':'unknown_visual_quality','budget':'budget_exhausted','unsafe':'repair_requires_human_inspection','partial':'review_incomplete'}[mode]
                self.assertEqual(detail['improvement_rounds'][0]['decision'],expected)
    def test_post_commit_restart_reconciles_without_repeating_executor_or_reports(self):
        q=self.submit();v=self.visual(self.metrics([.3]))
        with patch.object(self.store,'advance_improvement',side_effect=RuntimeError('simulated interruption after revision commit')):
            self.run_one(v)
        calls=len(self.client.calls);other=AdminWorkflow(self.root)
        other.reconcile_improvements(budget_checker=lambda job:True)
        other.reconcile_improvements(budget_checker=lambda job:True)
        detail=other.detail(q['candidate']['id']);self.assertEqual(len(detail['improvement_rounds']),1)
        self.assertEqual(len(detail['jobs']),2);self.assertEqual(len(self.client.calls),calls)
    def test_active_improvement_cannot_be_human_approved_mid_round(self):
        q=self.submit();self.run_one(self.visual(self.metrics([.3])))
        revision=self.store.detail(q['candidate']['id'])['revisions'][0]
        with self.assertRaisesRegex(ValueError,'active candidate job'):
            self.store.approve(q['candidate']['id'],revision['id'],revision['fingerprint'])
    def test_renewal_is_token_bound_and_expired_lease_cannot_resurrect(self):
        q=self.h.submit();job=q['job']['id'];cancel=Mock()
        with self.store.db()as db:
            db.execute("UPDATE jobs SET state='running',lease_token='first',lease_until=? WHERE id=?",(time.time()+10,job))
        lease=WorkerLease(self.store,job,'first',cancel=cancel,seconds=120);lease.renew();lease.guard('fixture_stage')
        with self.store.db()as db:
            row=db.execute('SELECT * FROM jobs WHERE id=?',(job,)).fetchone();self.assertGreater(row['lease_until'],time.time()+100)
            self.assertEqual(json.loads(row['progress_json'])['stage'],'fixture_stage')
            db.execute("UPDATE jobs SET lease_token='replacement' WHERE id=?",(job,))
        with self.assertRaises(LeaseLost):lease.guard()
        cancel.assert_called_once()
        lease2=WorkerLease(self.store,job,'replacement')
        with self.store.db()as db:db.execute('UPDATE jobs SET lease_until=? WHERE id=?',(time.time()-1,job))
        with self.assertRaises(LeaseLost):lease2.renew()
        self.assertEqual(self.store.recover_expired_jobs(),1)
    def test_heartbeat_renews_live_job_and_competing_worker_is_idle(self):
        q=self.h.submit();job=q['job']['id']
        with self.store.db()as db:db.execute("UPDATE jobs SET state='running',lease_token='live',lease_until=? WHERE id=?",(time.time()+1,job))
        lease=WorkerLease(self.store,job,'live',seconds=1,interval=.03).start();self.addCleanup(lease.close)
        time.sleep(.07);lease.guard()
        self.h.submit(run='other_waiting')
        execute=Mock();self.assertEqual(self.store.worker_once(execute)['status'],'idle');execute.assert_not_called()
    def test_missing_or_changed_policy_and_unsupported_repair_targets_fail_closed(self):
        bad=copy.deepcopy(self.policy);bad['maximum_repairs']=3
        with self.assertRaises(ValueError):self.store.submit_upload(run_id='bad',input_path=self.h.source,improvement_policy=bad)
        self.assertFalse(improves({'coverage':.8,'alignment_error':.4},{'coverage':.9,'alignment_error':.5}))
        q=self.submit();result=self.run_one(self.visual(self.metrics([.2]),targets=['arbitrary_shell']))
        self.assertEqual(result['status'],'failed');self.assertEqual(self.store.detail(q['candidate']['id'])['revisions'],[])
    def test_lease_loss_before_finalization_cannot_commit_revision(self):
        q=self.h.submit()
        def stale(job):
            result=self.h.executor(job)
            with self.store.db()as db:db.execute("UPDATE jobs SET lease_token='replacement' WHERE id=?",(job['id'],))
            return result
        self.assertEqual(self.store.worker_once(stale)['status'],'failed')
        self.assertEqual(self.store.detail(q['candidate']['id'])['revisions'],[])
    def test_pending_failure_is_an_immutable_stop_not_a_fabricated_revision_or_retry(self):
        q=self.submit();cid=q['candidate']['id']
        done=self.store.worker_once(lambda job:{'status':'pending','diagnostic':'GPU not available; no retry'})
        self.assertEqual(done['status'],'pending');self.store.reconcile_improvements()
        detail=self.store.detail(cid);self.assertEqual(detail['revisions'],[]);self.assertEqual(len(detail['jobs']),1)
        self.assertEqual(len(detail['improvement_stops']),1);self.assertTrue(detail['improvement_stops'][0]['no_automatic_retry'])
    def test_human_rejection_between_commit_and_controller_preempts_automatic_repair(self):
        q=self.submit();cid=q['candidate']['id']
        with patch.object(self.store,'advance_improvement',side_effect=RuntimeError('pause after commit')):
            self.run_one(self.visual(self.metrics([.3])))
        revision=self.store.detail(cid)['revisions'][0]
        self.store.reject(cid,revision['id'],'Owner wants to inspect before another run.',repair_targets=['reasoning'])
        self.store.reconcile_improvements(budget_checker=lambda job:True)
        detail=self.store.detail(cid);self.assertEqual(detail['candidate']['status'],'rejected');self.assertEqual(len(detail['jobs']),1)
        self.assertEqual(detail['improvement_rounds'][0]['decision'],'human_or_external_decision_preempted')
    def test_changed_comparison_basis_cannot_claim_quality_improvement(self):
        q=self.submit();metrics=self.metrics([.5,.1]);factory=self.visual(metrics)
        self.run_one(factory)
        def changed(job):
            review=factory(job)
            def callback(manifest):
                value=review(manifest);value['improvement_assessment']['comparison_key']='b'*64
                path=self.root/value['report_ref']['path'];body=json.loads(path.read_text());body['improvement_assessment']=value['improvement_assessment'];path.write_text(json.dumps(body));value['report_ref']=ref(path,self.root)
                return value
            return callback
        self.run_one(changed);detail=self.store.detail(q['candidate']['id'])
        self.assertEqual(detail['improvement_rounds'][-1]['decision'],'plateau_or_regression');self.assertFalse(detail['improvement_rounds'][-1]['measured_improvement'])

if __name__=='__main__':unittest.main()
