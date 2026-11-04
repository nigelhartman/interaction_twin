"""Offline proof/ledger tests. Retained private evidence is never sent or copied to releases."""
import copy,hashlib,json,shutil,sqlite3,tempfile,unittest
from pathlib import Path
import bounded_visual_transport_audit_v1 as V
from meta_budget import BudgetLedger,BudgetError,BudgetExceeded,TextPricing,estimate_text,read_budget_status

ROOT=Path(__file__).resolve().parents[1]
@unittest.skipUnless((ROOT/V.FOLDER/'plan.json').is_file(),'Preserved once-only request evidence is excluded from source-only releases')
class VisualBoundTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.addCleanup(self.tmp.cleanup)
  plan=json.loads((ROOT/V.FOLDER/'plan.json').read_text())
  files=set(V.CODE)|{V.FOLDER+'/plan.json',V.FOLDER+'/attempt/intent.json',V.FOLDER+'/attempt/request/meta-request.json',V.FOLDER+'/attempt-network-v2/intent.json'}|{r['path'] for r in [plan['packet'],plan['hypothesis'],*plan['images']]}
  for name in files:p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,p)
  self.proof=V.prepare(self.root);self.path=self.root/'ledger.sqlite3';self.ledger=BudgetLedger(self.path,cap_usd='1.35',per_request_cap_usd='1.35')
  self.price=TextPricing('muse-spark-1.3','1.25','4.25','https://dev.meta.ai/docs/pricing-rate-limits','2026-10-30',cached_input_per_million_usd='0.15');self.cost=estimate_text(self.price,input_tokens=1048576,max_output_tokens=6500)
  self.ledger.reserve(V.REQUEST,self.cost,model=self.price.model,endpoint='/v1/chat/completions',kind='text');self.ledger.mark_started(V.REQUEST);self.ledger.mark_uncertain(V.REQUEST,reason='transport_error')
 def acknowledge(self,proof=None):return self.ledger.acknowledge_reviewed_visual_transport(V.REQUEST,proof=proof or self.proof,authorization_reference='user-20261104-confirmed-credits-continue',project_root=self.root)
 def test_readonly_proof_and_explicit_ack_keep_every_microdollar_unknown_and_nonretryable(self):
  before=read_budget_status(self.path);self.assertTrue(before['blocked']);self.assertEqual(V.validate(self.path,self.proof,root=self.root)['retained_reservation_usd'],'1.338345');self.assertEqual(read_budget_status(self.path),before)
  ack=self.acknowledge();after=read_budget_status(self.path);self.assertEqual(ack['reservation_released_usd'],'0.000000');self.assertFalse(after['blocked']);self.assertEqual(after['reserved_usd'],before['reserved_usd']);self.assertEqual(after['remaining_estimated_usd'],before['remaining_estimated_usd']);self.assertEqual(after['settled_usd'],'0.000000');self.assertEqual(after['bounded_unknown_usd'],'1.338345')
  with sqlite3.connect(self.path)as db:self.assertEqual(db.execute('SELECT actual,usage_json FROM requests').fetchone(),(None,None))
  with self.assertRaises(BudgetError):self.ledger.mark_started(V.REQUEST)
  with self.assertRaises(BudgetError):self.ledger.release_no_charge(V.REQUEST,reason='provider_confirmed_no_charge')
  with self.assertRaises(BudgetExceeded):self.ledger.reserve('new',self.cost,model=self.price.model,endpoint='/v1/chat/completions',kind='text')
 def test_ack_is_idempotent_and_audit_is_immutable(self):
  first=self.acknowledge();self.assertTrue(first['newly_acknowledged']);self.assertFalse(self.acknowledge()['newly_acknowledged'])
  with self.assertRaises(BudgetError):self.ledger.acknowledge_reviewed_visual_transport(V.REQUEST,proof=self.proof,authorization_reference='changed-authorization',project_root=self.root)
  with sqlite3.connect(self.path)as db:
   with self.assertRaises(sqlite3.IntegrityError):db.execute('DELETE FROM visual_transport_unknown_audits')
   with self.assertRaises(sqlite3.IntegrityError):db.execute("UPDATE visual_transport_unknown_audits SET retained_reservation=1")
 def test_changed_code_receipts_images_attempt_count_and_proof_cannot_clear_pause(self):
  targets=['meta_api.py','scripts/meta_auth_probe.py',V.FOLDER+'/plan.json',V.FOLDER+'/attempt/request/meta-request.json',json.loads((self.root/V.FOLDER/'plan.json').read_text())['images'][0]['path']]
  for name in targets:
   p=self.root/name;raw=p.read_bytes();p.write_bytes(raw+b'changed')
   with self.subTest(name=name),self.assertRaises(BudgetError):self.acknowledge()
   p.write_bytes(raw);self.assertTrue(read_budget_status(self.path)['blocked'])
  for name in [V.FOLDER+'/attempt/result.json',V.FOLDER+'/attempt-network-v2/request/receipt.json']:
   p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
   with self.subTest(name=name),self.assertRaises(BudgetError):self.acknowledge()
   p.unlink()
   if p.parent.name=='request' and p.parent.parent.name=='attempt-network-v2':p.parent.rmdir()
  proof=copy.deepcopy(self.proof);proof['payload_sha256']='0'*64
  with self.assertRaises(BudgetError):self.acknowledge(proof)
  self.assertTrue(read_budget_status(self.path)['blocked'])
 def test_other_request_underreservation_pricing_and_http_failure_are_not_eligible(self):
  with self.ledger._transaction()as db:row=dict(db.execute('SELECT * FROM requests').fetchone())
  for mutation in [{'request_id':'other'},{'reserved':1},{'reason':'timeout'},{'model':'muse-spark-1.3-contributor'},{'actual':0},{'usage_json':'{}'},{'overage':1},{'state':'reserved'}]:
   with self.subTest(mutation=mutation),self.assertRaises(BudgetError):V.validate_row({**row,**mutation},self.proof,root=self.root)
  with self.assertRaises(BudgetError):self.ledger.acknowledge_reviewed_visual_transport('other',proof=self.proof,authorization_reference='user',project_root=self.root)
  self.assertTrue(read_budget_status(self.path)['blocked'])

if __name__=='__main__':unittest.main()
