"""Exact authority/source fixtures; synthetic features do not prove real motion."""
import copy,json,unittest
from pathlib import Path
from unittest.mock import patch
import incremental_motion_publication_v1 as I
from tests import test_curated_motion_hypotheses_v1 as fixtures
from tests import test_incremental_observed_publication_v1 as publication_fixtures

class MotionPublicationTests(unittest.TestCase):
 def setUp(self):
  self.f=fixtures.MotionReaderTests('runTest');self.f.setUp();self.addCleanup(self.f.doCleanups);self.root=self.f.root;self.f.record.pop('motion_hypothesis');program=json.loads((self.root/self.f.program['path']).read_text());keep={r['path']for r in [self.f.source,self.f.program,self.f.scene,self.f.native]}|{r['frame_path']for r in program['evidence']};self.f.assets={p:r for p,r in self.f.assets.items()if p in keep};self.f.publish();publication_fixtures.IncrementalObservedPublicationTests.canonicalize(self);self.expected=I.capture_baseline(self.root);self.document=copy.deepcopy(self.f.document);self.document['catalog_ref']={**self.expected['catalog'],'path':'showcase/public/library-catalog.json'};self.repin()
 def repin(self):
  self.selection={'descriptor':self.f.write('data/temporal-fixture/private-motion.json',self.document,False)};self.selections={self.f.source['sha256']:self.selection}
 def prepare(self):return I.prepare(expected=self.expected,selections=self.selections,root=self.root)
 def public(self):return{p:p.read_bytes()for p in(self.root/'showcase/public').iterdir()if p.is_file()}
 def test_prepare_is_readonly_exact_source_bound_and_preserves_every_other_field(self):
  before=self.public();plan=self.prepare();self.assertEqual(self.public(),before);self.assertEqual(plan.receipt['status'],'dry_run_validated');old=json.loads(before[self.root/self.expected['catalog']['path']]);new=json.loads(plan.outputs[plan.receipt['catalog']['path']]);self.assertEqual(new['records'][0].pop('motion_hypothesis'),self.selection['descriptor']);self.assertEqual(new,old);self.assertTrue(plan.receipt['original_unknowns_preserved']);self.assertTrue(plan.receipt['original_source_model_program_approval_and_extension_fields_unchanged']);self.assertFalse(plan.receipt['old_native_raster_geometry_recomputed']);self.assertNotIn('showcase/public/cases-index.json',plan.outputs)
 def test_mutable_catalog_alias_is_not_mistaken_for_historical_fingerprint(self):
  (self.root/'showcase/public/library-catalog.json').write_text('{"different":"newer alias"}');plan=self.prepare();self.assertTrue(plan.receipt['fresh_motion_contract_proofs'][0]['historical_catalog_fingerprint_checked']['path'].endswith(self.expected['catalog']['sha256']+'.json'))
 def test_changed_historical_fingerprint_and_wrong_source_are_rejected(self):
  original=copy.deepcopy(self.document)
  for variant in ['history','source','program','scene','id']:
   self.document=copy.deepcopy(original)
   if variant=='history':self.document['catalog_ref']['sha256']='f'*64
   elif variant=='source':self.document['source_ref']['sha256']='f'*64
   elif variant=='program':self.document['evidence_program_ref']['sha256']='f'*64
   elif variant=='scene':self.document['baseline_scene_ref']['sha256']='f'*64
   else:self.document['twin_id']='unapproved-working-record'
   self.repin()
   with self.subTest(variant=variant),self.assertRaises(ValueError):self.prepare()
 def test_original_known_anchor_unknown_ledger_joint_and_pose_forgeries_reject(self):
  original=copy.deepcopy(self.document)
  for variant in ['anchor','unknown','child','joint','camera','claims']:
   self.document=copy.deepcopy(original);e=self.document['groups'][0]['episodes'][0]
   if variant=='anchor':e['anchor_before']['camera_from_native_mesh'][0][3]+=.1
   elif variant=='unknown':e['original_unknown_indices']=[]
   elif variant=='child':e['predictions'][0]['component_poses']['lid'][0][3]+=.1
   elif variant=='joint':e['constant_joint_states'][0]['q_rad']=.2
   elif variant=='camera':self.document['camera']['K'][0][0]+=1
   else:self.document['physical_accuracy_verified']=True
   self.repin()
   with self.subTest(variant=variant),self.assertRaises(ValueError):self.prepare()
 def test_typed_private_path_rejects_and_untyped_runtime_fingerprint_is_not_admitted(self):
  self.document['flow_proofs_ref']['path']='data/../.private/token'
  self.repin()
  with self.assertRaises(ValueError):self.prepare()
  self.document=copy.deepcopy(self.f.document);self.document['catalog_ref']={**self.expected['catalog'],'path':'showcase/public/library-catalog.json'};private=self.f.write('data/private/runtime.bin',b'NOT-A-ROUTE',False);self.document['provider_reply']={'runtime':private};self.repin();plan=self.prepare();paths={r['path']for r in json.loads(plan.outputs['showcase/public/public-assets.json'])['assets']};self.assertNotIn(private['path'],paths);self.assertNotIn(self.document['catalog_ref']['path'],paths);self.assertNotIn(self.document['implementation_ref']['path'],paths)
 def test_all_five_stale_CAS_files_fail_before_any_write(self):
  for role,ref in self.expected.items():
   p=self.root/ref['path'];raw=p.read_bytes();p.write_bytes(raw+b' ');before=self.public()
   try:
    with self.subTest(role=role),self.assertRaises(ValueError):self.prepare()
    self.assertEqual(self.public(),before)
   finally:p.write_bytes(raw)
 def test_changed_old_source_program_scene_or_native_byte_rejects(self):
  for ref in [self.f.source,self.f.program,self.f.scene,self.f.native]:
   p=self.root/ref['path'];raw=p.read_bytes();p.write_bytes(raw+b'!')
   try:
    with self.subTest(path=ref['path']),self.assertRaises(ValueError):self.prepare()
   finally:p.write_bytes(raw)
 def test_current_approval_or_existing_motion_selection_cannot_be_replaced(self):
  self.selections={'b'*64:self.selection}
  with self.assertRaises(ValueError):self.prepare()
  self.selections={self.f.source['sha256']:self.selection};self.f.record['motion_hypothesis']=self.f.native;self.f.publish();publication_fixtures.IncrementalObservedPublicationTests.canonicalize(self);self.expected=I.capture_baseline(self.root)
  with self.assertRaisesRegex(ValueError,'existing motion'):self.prepare()
 def test_explicit_commit_retains_original_bytes_and_approval_index_last(self):
  original_assets={p:(self.root/p).read_bytes()for p in self.f.assets};cases=(self.root/self.expected['cases']['path']).read_bytes();original=I.os.replace;order=[]
  def replace(a,b):order.append(Path(b).name);return original(a,b)
  with patch.object(I.os,'replace',side_effect=replace):receipt=I.commit(expected=self.expected,selections=self.selections,root=self.root)
  self.assertEqual(receipt['status'],'committed');self.assertEqual(order[-1],'library-index.json');self.assertEqual((self.root/self.expected['cases']['path']).read_bytes(),cases);self.assertTrue(all((self.root/p).read_bytes()==raw for p,raw in original_assets.items()));catalog=json.loads((self.root/receipt['catalog']['path']).read_text());self.assertEqual(catalog['records'][0]['interaction'],self.f.program);self.assertEqual(catalog['records'][0]['curation'],self.f.record['curation'])
 def test_cas_race_while_staging_rejects_before_first_replace(self):
  original=I._verify_fingerprints;calls=[0]
  def race(root,identities):
   original(root,identities);calls[0]+=1
   if calls[0]==2:
    p=self.root/self.expected['aligned']['path'];p.write_bytes(p.read_bytes()+b' ')
  with patch.object(I,'_verify_fingerprints',side_effect=race),patch.object(I.os,'replace')as replace,self.assertRaises(ValueError):I.commit(expected=self.expected,selections=self.selections,root=self.root)
  replace.assert_not_called()

if __name__=='__main__':unittest.main()
