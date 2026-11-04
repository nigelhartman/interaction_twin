"""Root-explicit admin transport audits retain all money and never send again."""
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from agent_memory import FeedbackMemory, digest
from meta_api import MetaClient, MetaError
from meta_budget import (ADMIN_TEXT_PROOF_FORMAT, BudgetError, BudgetLedger,
    TextPricing, estimate_text, validate_bounded_admin_text_audit)

ROOT=Path(__file__).resolve().parents[1]


class AdminTextUncertaintyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.ledger_path=self.root/'.secrets/test-budget.sqlite3'
        self.ledger=BudgetLedger(self.ledger_path)
        self.memory=FeedbackMemory(self.root).retrieve(source_sha256='a'*64)
        bindings={'source_ref':{'path':'data/source.mp4','sha256':'a'*64,'bytes':10},
            'interaction_ref':{'path':'data/program.json','sha256':'b'*64,'bytes':20},'object_program_ref':None,'scene_ref':None}
        self.plan={'format':'interaction-twin-private-muse-review-plan','version':1,'role':'evidence','model':'muse-spark-1.3',
            'artifact_binding':bindings,'memory_context':self.memory,'summary':{'source_sha256':'a'*64},
            'prompt':'Text-only untrusted saved metadata. Offline test; no real images or provider.',
            'schema':{'type':'object'},'provider_schema':{'type':'object','properties':{},'additionalProperties':False},
            'max_output_tokens':1000,'image_inputs':[],'independent_model_review':False,'training_performed':False,
            'implementation_sha256':hashlib.sha256((ROOT/'admin_muse_reviews.py').read_bytes()).hexdigest()}
        self.plan['request_key']=digest(self.plan)
        namespace=digest({'role':'evidence','artifact_binding':bindings,'context_sha256':self.memory['context_sha256'],'model':'muse-spark-1.3'})
        self.folder=self.root/'.private/admin-muse-reviews'/namespace
        self.folder.mkdir(parents=True);self.write(self.folder/'plan.json',self.plan)
        (self.folder/'attempt').mkdir();self.sent=[]
        def failed_transport(endpoint,payload,timeout):
            self.sent.append(copy.deepcopy(payload));raise OSError('Offline fixture, transport delivery unknown')
        client=MetaClient(ledger_path=self.ledger_path,transport=failed_transport)
        with self.assertRaisesRegex(MetaError,'transport_error_charge_uncertain'):
            client.json_result(self.plan['prompt'],self.plan['provider_schema'],[],output=self.folder/'attempt/request',max_output_tokens=1000)
        self.write(self.folder/'attempt/failure.json',{'status':'preserved_no_automatic_retry','error_type':'MetaError'})
        receipt=json.loads((self.folder/'attempt/request/meta-request.json').read_text())
        self.request_id=receipt['request_id']
        self.proof={'format':ADMIN_TEXT_PROOF_FORMAT,'version':1,
            'review_plan_path':str(self.folder/'plan.json'),'review_plan_sha256':self.sha(self.folder/'plan.json'),
            'request_receipt_path':str(self.folder/'attempt/request/meta-request.json'),
            'request_receipt_sha256':self.sha(self.folder/'attempt/request/meta-request.json'),
            'attempt_failure_path':str(self.folder/'attempt/failure.json'),
            'attempt_failure_sha256':self.sha(self.folder/'attempt/failure.json'),
            'request_payload_sha256':receipt['payload_sha256'],
            'reviewer_source_sha256':self.sha(ROOT/'admin_muse_reviews.py'),'transport_source_sha256':self.sha(ROOT/'meta_api.py')}

    @staticmethod
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    @staticmethod
    def write(path,value):path.write_text(json.dumps(value));path.chmod(0o600)
    def ack(self,proof=None,authorization='root-explicit-admin-text-bound-review'):
        return self.ledger.acknowledge_bounded_admin_text_uncertainty(self.request_id,proof=proof or self.proof,
            authorization_reference=authorization,project_root=self.root)
    def row(self):
        with sqlite3.connect(self.ledger_path)as db:
            db.row_factory=sqlite3.Row;return dict(db.execute('SELECT * FROM requests WHERE request_id=?',(self.request_id,)).fetchone())

    def test_read_only_preflight_and_ack_retain_entire_full_context_bound(self):
        before=self.ledger.status();raw=self.ledger_path.read_bytes()
        check=validate_bounded_admin_text_audit(self.ledger_path,self.request_id,proof=self.proof,project_root=self.root)
        self.assertEqual(self.ledger_path.read_bytes(),raw);self.assertTrue(check['eligible'])
        self.assertTrue(before['blocked']);self.assertEqual(check['reserved_usd'],'1.314970')
        result=self.ack();after=self.ledger.status();row=self.row()
        self.assertTrue(result['newly_acknowledged']);self.assertTrue(result['unknown_usage'])
        self.assertEqual(result['reservation_released_usd'],'0.000000');self.assertEqual(result['new_model_requests'],0)
        for key in ['settled_usd','reserved_usd','remaining_estimated_usd','cap_usd','per_request_cap_usd']:
            self.assertEqual(before[key],after[key])
        self.assertFalse(after['blocked']);self.assertTrue(after['unknown_usage'])
        self.assertEqual(row['state'],'bounded_unknown');self.assertEqual(row['reason'],'transport_error')
        self.assertEqual(row['reserved'],1314970);self.assertIsNone(row['actual']);self.assertIsNone(row['usage_json'])
        with sqlite3.connect(self.ledger_path)as db:
            audit=db.execute('SELECT retained_reservation,authorization_reference FROM admin_text_unknown_audits').fetchone()
            self.assertEqual(audit,(1314970,'root-explicit-admin-text-bound-review'))
            with self.assertRaises(sqlite3.IntegrityError):db.execute('UPDATE admin_text_unknown_audits SET retained_reservation=0')
            with self.assertRaises(sqlite3.IntegrityError):db.execute('DELETE FROM admin_text_unknown_audits')
        self.assertEqual(len(self.sent),1,'The audit must never call a model')

    def test_idempotence_conflicts_no_resend_no_release_and_late_real_usage(self):
        self.ack();self.assertFalse(self.ack()['newly_acknowledged'])
        with self.assertRaisesRegex(BudgetError,'conflict'):self.ack(authorization='different-authorization')
        with self.assertRaises(BudgetError):self.ledger.mark_started(self.request_id)
        for reason in ['provider_confirmed_no_charge','cancelled_before_transport']:
            with self.subTest(reason=reason),self.assertRaises(BudgetError):self.ledger.release_no_charge(self.request_id,reason=reason)
        self.ledger.settle(self.request_id,'0.000210',usage={'input_tokens':100,'output_tokens':20})
        self.assertEqual(self.row()['state'],'settled');self.assertFalse(self.ledger.status()['unknown_usage'])
        with sqlite3.connect(self.ledger_path)as db:self.assertEqual(db.execute('SELECT COUNT(*) FROM admin_text_unknown_audits').fetchone()[0],1)

    def test_payload_code_model_images_output_limits_and_extra_attempts_fail_closed(self):
        for key in ['request_payload_sha256','reviewer_source_sha256','transport_source_sha256']:
            wrong=dict(self.proof,**{key:'f'*64})
            with self.subTest(key=key),self.assertRaises(BudgetError):self.ack(wrong)
        original=json.loads((self.folder/'plan.json').read_text())
        for key,value in [('image_inputs',['image.png']),('max_output_tokens',1001),('model','unpriced-model'),('training_performed',True)]:
            altered=copy.deepcopy(original);altered[key]=value
            altered['request_key']=digest({k:v for k,v in altered.items()if k!='request_key'})
            self.write(self.folder/'plan.json',altered)
            wrong=dict(self.proof,review_plan_sha256=self.sha(self.folder/'plan.json'))
            with self.subTest(key=key),self.assertRaises(BudgetError):self.ack(wrong)
        self.write(self.folder/'plan.json',original)
        (self.folder/'attempt-2').mkdir()
        with self.assertRaisesRegex(BudgetError,'once_only'):self.ack()
        self.assertTrue(self.ledger.status()['blocked']);self.assertEqual(self.row()['state'],'uncertain')

    def test_unverified_reservation_http_status_and_usage_are_ineligible(self):
        old=self.row()['estimate_json']
        with sqlite3.connect(self.ledger_path)as db:db.execute('UPDATE requests SET reserved=reserved-1 WHERE request_id=?',(self.request_id,))
        with self.assertRaisesRegex(BudgetError,'full_context_bound'):self.ack()
        with sqlite3.connect(self.ledger_path)as db:db.execute('UPDATE requests SET reserved=1314970 WHERE request_id=?',(self.request_id,))
        path=self.folder/'attempt/request/meta-request.json';receipt=json.loads(path.read_text())
        for key,value in [('http_status',500),('usage',{'input_tokens':10}),('automatic_retries',True)]:
            self.write(path,{**receipt,key:value})
            wrong=dict(self.proof,request_receipt_sha256=self.sha(path))
            with self.subTest(key=key),self.assertRaises(BudgetError):self.ack(wrong)
        self.assertEqual(self.row()['estimate_json'],old);self.assertTrue(self.ledger.status()['blocked'])

    def test_secret_paths_symlinks_and_changed_evidence_refused_without_read(self):
        for path in [self.root/'.secrets/meta.env',self.root/'data/plan.json']:
            bad=dict(self.proof,review_plan_path=str(path))
            with self.subTest(path=path),patch('os.open',side_effect=AssertionError('No unauthorized evidence read')),self.assertRaises(BudgetError):self.ack(bad)
        original=self.folder/'plan.json';saved=self.folder/'saved.json';original.rename(saved);original.symlink_to(saved)
        with self.assertRaisesRegex(BudgetError,'proof_path'):self.ack()
        original.unlink();saved.rename(original)
        original.write_text('{}')
        with self.assertRaisesRegex(BudgetError,'hash_mismatch'):self.ack()

    def test_audit_state_and_immutable_record_commit_atomically(self):
        with sqlite3.connect(self.ledger_path)as db:
            db.execute("CREATE TRIGGER fail_admin_audit_state BEFORE UPDATE ON requests BEGIN SELECT RAISE(ABORT,'fixture_state_failure'); END")
        with self.assertRaises(sqlite3.IntegrityError):self.ack()
        self.assertEqual(self.row()['state'],'uncertain')
        with sqlite3.connect(self.ledger_path)as db:self.assertEqual(db.execute('SELECT COUNT(*) FROM admin_text_unknown_audits').fetchone()[0],0)


if __name__=='__main__':unittest.main()
