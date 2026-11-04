"""Private operations: read-only accounting/queue projections, explicit GPU probe."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import admin_operations as A
from meta_budget import BudgetLedger,TextPricing,estimate_text

class AdminOperationsTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve()
    def config(self,enabled=True):
        p=self.root/'config/sam3d-cluster.json';p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps({'version':1,'enabled':enabled,'ssh_control_path':'/private/tmp/test-approved-master/cm-%C','maximum_parallel_workers':2,'poll_seconds':3,'wait_seconds':1800}))
    def ledger(self):
        ledger=BudgetLedger(self.root/A.LEDGER)
        pricing=TextPricing('muse-test','1','2','https://dev.meta.ai/docs/pricing','2026-11-03')
        for identity in ['private-request-sentinel','held','unknown','bounded']:
            ledger.reserve(identity,estimate_text(pricing,input_tokens=1000000,max_output_tokens=0),model=pricing.model,endpoint='/v1/chat/completions',kind='text')
        ledger.mark_started('private-request-sentinel');ledger.settle('private-request-sentinel','0.1',usage={'input_tokens':100,'output_tokens':20,'reasoning_tokens':5})
        ledger.mark_started('unknown');ledger.mark_started('bounded')
        ledger.mark_uncertain('unknown','missing_usage');ledger.mark_uncertain('bounded','timeout')
        # Status fixture only; no inference, recovery or release. The real audit
        # proof path is covered by meta_budget's independent tests.
        with sqlite3.connect(ledger.path)as db:db.execute("UPDATE requests SET state='bounded_unknown'WHERE request_id='bounded'")
        return ledger
    def workflow(self,empty=False):
        p=self.root/A.WORKFLOW;p.parent.mkdir(parents=True,exist_ok=True)
        with sqlite3.connect(p)as db:
            db.executescript('CREATE TABLE candidates(id TEXT,status TEXT);CREATE TABLE jobs(id TEXT,candidate_id TEXT,kind TEXT,state TEXT,lease_until REAL,created_utc TEXT,updated_utc TEXT,heartbeat_utc TEXT,progress_json TEXT,payload_json TEXT,result_json TEXT,diagnostic TEXT);')
            if not empty:
                cid='candidate_'+'1'*32;db.execute('INSERT INTO candidates VALUES(?,?)',(cid,'processing'))
                for i,state in enumerate(['queued','running','complete']):
                    db.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',('job_'+str(i)*32,cid,'pipeline',state,1 if state=='running'else None,A.now(),A.now(),A.now(),json.dumps({'stage':'phase_segment','message':'NEVER_EXPOSE_PROGRESS_TEXT'}),'NEVER_EXPOSE_PAYLOAD','NEVER_EXPOSE_RESULT','NEVER_EXPOSE_DIAGNOSTIC'))
        return p
    def marketing(self,empty=False):
        path=self.root/A.MARKETING;path.parent.mkdir(parents=True,exist_ok=True)
        with sqlite3.connect(path)as db:
            db.executescript('CREATE TABLE jobs(id TEXT,campaign_id TEXT,revision_id TEXT,format_id TEXT,status TEXT,progress REAL,error TEXT,token TEXT,heartbeat TEXT,result TEXT,created TEXT);CREATE TABLE campaigns(id TEXT,title TEXT,brief TEXT);')
            db.execute('INSERT INTO campaigns VALUES(?,?,?)',('campaign_'+'1'*64,'NEVER_EXPOSE_TITLE','NEVER_EXPOSE_BRIEF'))
            if not empty:
                for i,state in enumerate(['planned','rendering','review_ready','error']):
                    db.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?,?,?,?)',('job_'+str(i)*64,'campaign_'+'1'*64,'revision_'+'2'*64,'portrait',state,.25,'NEVER_EXPOSE_RENDER_ERROR','NEVER_EXPOSE_RENDER_TOKEN',A.now(),'NEVER_EXPOSE_RENDER_RESULT',A.now()))
        return path
    def response(self,queue=None,account=None):
        return SimpleNamespace(returncode=0,stdout=b'banner\n'+json.dumps({'version':1,'squeue':{'ok':True,'rows':queue or[]},'sacct':{'ok':True,'rows':account or[]}}).encode(),stderr=b'NEVER_EXPOSE_REMOTE_ERROR')
    def success(self):
        return {'status':'checked','connection_verified':True,'error_code':None,'allocation':{'job_id':A.PARENT,'state':'RUNNING','allocated_tres':'cpu=32,mem=200G,gres/gpu=2','node_list':'ggpu182','start_time':'2026-11-01T22:00:00','end_time':'2026-11-04T22:00:00','elapsed':'1-02:03:04','time_limit':'3-00:00:00'}}

    def test_missing_get_creates_nothing_and_never_contacts_ssh_or_clients(self):
        with patch.object(A.subprocess,'run',side_effect=AssertionError('No subprocess on GET')),patch('admin_workflow.AdminWorkflow.__init__',side_effect=AssertionError('No database migration on GET')),patch('marketing_studio.MarketingStudio.__init__',side_effect=AssertionError('No studio initialization on GET')),patch('meta_api.MetaClient.__init__',side_effect=AssertionError('No credentials/client on GET')):
            value=A.snapshot(self.root)
        self.assertEqual(list(self.root.iterdir()),[]);self.assertEqual(value['budget']['status'],'not_started');self.assertIsNone(value['budget']['provider_balance_usd'])
        self.assertTrue(all(v is None for v in value['usage']['totals'].values()));self.assertEqual(value['gpu']['status'],'not_checked');self.assertTrue(value['gpu']['stale'])
        self.assertEqual(value['marketing']['status'],'not_started');self.assertEqual(value['marketing']['counts_by_state'],{})

    def test_marketing_queue_is_readonly_and_excludes_creative_private_record_bodies(self):
        path=self.marketing();before=path.read_bytes();files=sorted(self.root.rglob('*'))
        with patch('marketing_studio.MarketingStudio.__init__',side_effect=AssertionError('No studio initialization')),patch.object(A.subprocess,'run',side_effect=AssertionError('No subprocess')):value=A.snapshot(self.root)
        marketing=value['marketing'];self.assertEqual(marketing['status'],'available')
        self.assertEqual(marketing['counts_by_state'],{'planned':1,'rendering':1,'review_ready':1,'error':1})
        self.assertEqual({row['state']for row in marketing['active_jobs']},{'planned','rendering'});self.assertEqual(len(marketing['recent_jobs']),4)
        self.assertTrue(all(row['progress']==.25 and row['format_id']=='portrait'for row in marketing['active_jobs']))
        self.assertNotIn('NEVER_EXPOSE',json.dumps(value));self.assertEqual(before,path.read_bytes());self.assertEqual(files,sorted(self.root.rglob('*')))
        self.assertFalse(marketing['worker_liveness_verified'])

    def test_empty_marketing_queue_reports_known_zeros_but_missing_and_bad_schema_remain_unknown(self):
        self.assertEqual(A.marketing_processing_summary(self.root)['status'],'not_started')
        path=self.marketing(empty=True);value=A.marketing_processing_summary(self.root)
        self.assertEqual(value['counts_by_state'],{'planned':0,'rendering':0,'review_ready':0,'error':0});self.assertEqual(value['active_jobs'],[])
        with sqlite3.connect(path)as db:db.execute('DROP TABLE jobs')
        value=A.marketing_processing_summary(self.root);self.assertEqual(value['status'],'unavailable');self.assertEqual(value['counts_by_state'],{})

    def test_marketing_active_projection_is_independently_bounded_and_rejects_foreign_ids(self):
        path=self.marketing();cid='campaign_'+'1'*64;rid='revision_'+'2'*64
        with sqlite3.connect(path)as db:
            # Newer completed work must not crowd an older active item out of
            # the active view, and arbitrary IDs must never become text channels.
            for i in range(30):
                db.execute('INSERT INTO jobs(id,campaign_id,revision_id,format_id,status,progress,created)VALUES(?,?,?,?,?,?,?)',('job_'+format(i+10,'064x'),cid,rid,'landscape','review_ready',1,'2999-01-01T00:00:00Z'))
            db.execute('INSERT INTO jobs(id,campaign_id,revision_id,format_id,status,progress,created)VALUES(?,?,?,?,?,?,?)',('NEVER_EXPOSE_FOREIGN_JOB',cid,rid,'square','rendering',.5,A.now()))
            db.execute("UPDATE jobs SET progress=2 WHERE status='planned'")
        value=A.marketing_processing_summary(self.root)
        self.assertEqual(len(value['active_jobs']),2);self.assertEqual(len(value['recent_jobs']),20)
        self.assertEqual(value['omitted_invalid_identity_rows'],1);self.assertNotIn('NEVER_EXPOSE',json.dumps(value))
        self.assertIsNone(next(r['progress']for r in value['active_jobs']if r['state']=='planned'))

    def test_marketing_symlink_is_unavailable_without_reading_external_database(self):
        path=self.root/A.MARKETING;path.parent.mkdir(parents=True);target=self.root/'foreign.sqlite3';target.write_bytes(b'NEVER_EXPOSE_OTHER_DB');path.symlink_to(target)
        with patch.object(A.sqlite3,'connect',side_effect=AssertionError('No symlink database read')):value=A.marketing_processing_summary(self.root)
        self.assertEqual(value['status'],'unavailable');self.assertEqual(value['active_jobs'],[])

    def test_known_budget_preserves_unknown_holds_and_counts_only_reported_usage(self):
        ledger=self.ledger();self.workflow(empty=True);before=ledger.path.read_bytes()
        with patch.object(A.subprocess,'run',side_effect=AssertionError('No SSH')):value=A.snapshot(self.root)
        budget=value['budget'];usage=value['usage']
        self.assertEqual(budget['settled_usd'],'0.100000');self.assertEqual(budget['reserved_usd'],'3.000000');self.assertEqual(budget['remaining_estimated_usd'],'96.900000')
        self.assertTrue(budget['blocked']);self.assertEqual(budget['unknown_usage_count'],2);self.assertEqual(budget['bounded_unknown_count'],1);self.assertEqual(budget['bounded_unknown_usd'],'1.000000')
        self.assertEqual(usage['totals']['input_tokens'],100);self.assertEqual(usage['totals']['output_tokens'],20);self.assertEqual(usage['totals']['reasoning_tokens'],5);self.assertIsNone(usage['totals']['images'])
        self.assertEqual(usage['requests_without_reported_usage'],3);self.assertEqual(usage['counts_by_state']['bounded_unknown'],1)
        self.assertNotIn('private-request-sentinel',json.dumps(value));self.assertFalse(budget['actual_meta_credit_balance_verified']);self.assertEqual(before,ledger.path.read_bytes())

    def test_queue_projection_never_exposes_or_changes_payload_feedback_or_stale_lease(self):
        self.ledger();path=self.workflow();before=path.read_bytes();value=A.snapshot(self.root);p=value['processing']
        self.assertEqual(p['counts_by_state']['queued'],1);self.assertEqual(p['counts_by_state']['running'],1);self.assertEqual(p['counts_by_state']['cancelled'],0)
        running=next(j for j in p['active_jobs']if j['state']=='running');self.assertEqual(running['lease_status'],'expired');self.assertEqual(running['stage'],'phase_segment')
        self.assertNotIn('NEVER_EXPOSE',json.dumps(value));self.assertEqual(before,path.read_bytes())
        with sqlite3.connect(path)as db:self.assertEqual(db.execute("SELECT state FROM jobs WHERE id=?",(running['id'],)).fetchone()[0],'running')

    def test_empty_existing_queue_and_ledger_return_explicit_known_zero_counts(self):
        BudgetLedger(self.root/A.LEDGER);self.workflow(empty=True);value=A.snapshot(self.root)
        self.assertEqual(value['processing']['counts_by_state']['queued'],0);self.assertEqual(value['processing']['counts_by_state']['running'],0)
        self.assertEqual(value['budget']['request_count'],0);self.assertEqual(value['usage']['counts_by_state']['settled'],0)
        self.assertIsNone(value['usage']['totals']['input_tokens'])

    def test_malformed_usage_and_unknown_model_are_not_arbitrary_text_channels(self):
        ledger=self.ledger()
        with sqlite3.connect(ledger.path)as db:db.execute("UPDATE requests SET usage_json=?,model=?,endpoint=?WHERE request_id='held'",(json.dumps({'prompt':'NEVER_EXPOSE_BAD_USAGE','api_key':'NEVER_EXPOSE_KEY'}),'NEVER_EXPOSE_MODEL','/NEVER_EXPOSE_ENDPOINT'))
        value=A.snapshot(self.root);self.assertNotIn('NEVER_EXPOSE',json.dumps(value));self.assertTrue(any(m['model']=='other_model'for m in value['usage']['models']))
        (self.root/'.secrets/meta.env').write_text('NEVER_EXPOSE_CREDENTIAL_FILE')
        self.assertNotIn('NEVER_EXPOSE',json.dumps(A.snapshot(self.root)))

    def test_genuinely_reported_zero_tokens_stay_zero(self):
        ledger=self.ledger()
        with sqlite3.connect(ledger.path)as db:db.execute("UPDATE requests SET usage_json=?WHERE request_id='private-request-sentinel'",(json.dumps({'input_tokens':0,'output_tokens':0}),))
        usage=A.snapshot(self.root)['usage'];self.assertEqual(usage['totals']['input_tokens'],0);self.assertEqual(usage['totals']['output_tokens'],0);self.assertIsNone(usage['totals']['reasoning_tokens'])

    def test_foreign_scheduler_allocation_is_rejected_and_never_exposed(self):
        remote=self.response(['99999999|RUNNING|2026-11-01T22:00:00|2026-11-04T22:00:00|FOREIGN_NODE'],[])
        with patch.object(A.subprocess,'run',side_effect=[SimpleNamespace(returncode=0),remote]):value=A._query_gpu({'ssh_control_path':'/private/tmp/test-approved-master/cm-%C'})
        self.assertEqual(value['error_code'],'invalid_response');self.assertNotIn('FOREIGN_NODE',json.dumps(value));self.assertNotIn('allocation',value)

    def test_configured_gpu_get_is_unverified_until_explicit_refresh(self):
        self.config()
        with patch.object(A,'_query_gpu',side_effect=AssertionError('GET never queries')):gpu=A.snapshot(self.root)['gpu']
        self.assertTrue(gpu['configured']);self.assertEqual(gpu['status'],'not_checked');self.assertIsNone(gpu['connection_verified_at']);self.assertIsNone(gpu['allocation']);self.assertFalse((self.root/A.GPU_CACHE).exists())

    def test_explicit_refresh_writes_only_private_gpu_cache_then_get_uses_it(self):
        ledger=self.ledger();workflow=self.workflow();self.config();before=(ledger.path.read_bytes(),workflow.read_bytes())
        with patch.object(A,'_query_gpu',return_value=self.success())as query:value=A.refresh_gpu(self.root)
        query.assert_called_once();gpu=value['gpu'];self.assertEqual(gpu['status'],'checked');self.assertFalse(gpu['stale']);self.assertEqual(gpu['allocation']['state'],'RUNNING');self.assertEqual(gpu['allocation']['gpus'],2)
        self.assertIsNone(gpu['gpu_utilization']);self.assertIsNone(gpu['gpu_memory']);self.assertEqual(gpu['worker_readiness'],'not_probed');self.assertFalse(gpu['allocation_or_holder_modified'])
        self.assertEqual((self.root/A.GPU_CACHE).stat().st_mode&0o777,0o600);self.assertEqual(before,(ledger.path.read_bytes(),workflow.read_bytes()))
        with patch.object(A,'_query_gpu',side_effect=AssertionError('No automatic refresh')):self.assertEqual(A.snapshot(self.root)['gpu']['checked_at'],gpu['checked_at'])
        with patch.object(A.time,'time',return_value=A._time(gpu['checked_at']).timestamp()+121):self.assertTrue(A.snapshot(self.root)['gpu']['stale'])

    def test_missing_or_disabled_config_never_starts_ssh_on_explicit_refresh(self):
        for enabled,error in [(None,'configuration_missing'),(False,'configuration_disabled')]:
            with self.subTest(enabled=enabled):
                if enabled is not None:self.config(enabled)
                with patch.object(A,'_query_gpu',side_effect=AssertionError('Unavailable configuration')):gpu=A.refresh_gpu(self.root)['gpu']
                self.assertEqual(gpu['status'],'unavailable');self.assertEqual(gpu['error_code'],error);self.assertIsNone(gpu['connection_verified_at'])

    def test_existing_master_fence_blocks_fresh_connection_and_exposes_no_stderr(self):
        config={'ssh_control_path':'/private/tmp/test-approved-master/cm-%C'}
        with patch.object(A.subprocess,'run',return_value=SimpleNamespace(returncode=255,stdout=b'',stderr=b'NEVER_EXPOSE_REMOTE_PASSWORD'))as run:
            value=A._query_gpu(config)
        self.assertEqual(value['error_code'],'master_unavailable');self.assertEqual(run.call_count,1);self.assertNotIn('NEVER_EXPOSE',json.dumps(value))
        argv=run.call_args.args[0];self.assertIn('ProxyCommand=false',argv);self.assertIn('ControlMaster=no',argv);self.assertEqual(argv[-3:],['-O','check','KISSKI'])

    def test_fixed_readonly_scheduler_probe_reports_own_running_allocation(self):
        q=[A.PARENT+'|RUNNING|2026-11-01T22:00:00|2026-11-04T22:00:00|ggpu182']
        a=[A.PARENT+'|RUNNING|2026-11-01T22:00:00|2026-11-04T22:00:00|cpu=32,gres/gpu=2|1-02:03:04|3-00:00:00|ggpu182|']
        with patch.object(A.subprocess,'run',side_effect=[SimpleNamespace(returncode=0),self.response(q,a)])as run:value=A._query_gpu({'ssh_control_path':'/private/tmp/test-approved-master/cm-%C'})
        self.assertEqual(value['status'],'checked');self.assertEqual(value['allocation']['gpus'],2);self.assertEqual(value['allocation']['state'],'RUNNING')
        self.assertEqual(run.call_count,2);argv=run.call_args.args[0];self.assertIn('ProxyCommand=false',argv);self.assertEqual(argv[-2],'KISSKI');self.assertIn("'16889584'",A.GPU_QUERY_CODE)
        for forbidden in ['sbatch','scancel','salloc','nvidia-smi','srun']:
            self.assertNotIn(forbidden,A.GPU_QUERY_CODE)
        self.assertEqual(run.call_args.kwargs['timeout'],22);self.assertNotIn('NEVER_EXPOSE',json.dumps(value))

    def test_terminal_not_found_and_partial_unknown_are_distinct(self):
        config={'ssh_control_path':'/private/tmp/test-approved-master/cm-%C'}
        terminal=self.response([], [A.PARENT+'|COMPLETED|2026-11-01T22:00:00|2026-11-02T22:00:00|gres/gpu=2|1-00:00:00|1-00:00:00|ggpu182|'])
        for response,status in [(terminal,'checked'),(self.response(),'not_found')]:
            with patch.object(A.subprocess,'run',side_effect=[SimpleNamespace(returncode=0),response]):result=A._query_gpu(config)
            self.assertEqual(result['status'],status)
        partial=SimpleNamespace(returncode=0,stdout=json.dumps({'version':1,'squeue':{'ok':False,'rows':[]},'sacct':{'ok':True,'rows':[]}}).encode())
        with patch.object(A.subprocess,'run',side_effect=[SimpleNamespace(returncode=0),partial]):result=A._query_gpu(config)
        self.assertEqual(result['status'],'unavailable');self.assertEqual(result['error_code'],'scheduler_partial')

    def test_timeout_and_ambiguous_results_never_fake_readiness(self):
        config={'ssh_control_path':'/private/tmp/test-approved-master/cm-%C'}
        for effects,code in [([subprocess.TimeoutExpired('ssh',6)],'master_timeout'),([SimpleNamespace(returncode=0),subprocess.TimeoutExpired('ssh',22)],'query_timeout')]:
            with patch.object(A.subprocess,'run',side_effect=effects):result=A._query_gpu(config)
            self.assertEqual(result['error_code'],code);self.assertFalse(result['connection_verified'])
        row=A.PARENT+'|RUNNING|2026-11-01T22:00:00|2026-11-04T22:00:00|ggpu182'
        with patch.object(A.subprocess,'run',side_effect=[SimpleNamespace(returncode=0),self.response([row,row],[])]):result=A._query_gpu(config)
        self.assertEqual(result['error_code'],'ambiguous_allocation')

    def test_invalid_cache_symlink_and_future_time_are_not_fresh_status(self):
        self.config();folder=self.root/'.private/admin-operations';folder.mkdir(parents=True)
        p=folder/'gpu-status.json'
        for value in [[],{'format':'interaction-twin-private-gpu-status','version':1,'allocation_id':A.PARENT,'checked_at':'2999-01-01T00:00:00+00:00','status':'checked','connection_verified':True,'allocation':self.success()['allocation']}]:
            p.write_text(json.dumps(value));self.assertEqual(A.snapshot(self.root)['gpu']['status'],'not_checked')
        p.unlink();target=self.root/'elsewhere.json';target.write_text(json.dumps(self.success()));p.symlink_to(target)
        self.assertEqual(A.snapshot(self.root)['gpu']['status'],'not_checked')

    def test_concurrent_refresh_is_not_a_second_ssh_probe(self):
        self.config();A._LOCK.acquire()
        try:
            with patch.object(A,'_query_gpu',side_effect=AssertionError('Already refreshing')):result=A.refresh_gpu(self.root)
            self.assertTrue(result['gpu']['refresh_in_progress'])
        finally:A._LOCK.release()

if __name__=='__main__':unittest.main()
