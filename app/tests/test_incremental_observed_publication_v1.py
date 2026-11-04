"""CAS/additive authority fixtures; mocked new-pixel proof is not3D certification."""
import copy,hashlib,json,sys,types,unittest
from pathlib import Path
from unittest.mock import Mock,patch
import incremental_observed_publication_v1 as I
from tests import test_curated_observed_surfaces_v1 as observed_fixtures

class IncrementalObservedPublicationTests(unittest.TestCase):
 def setUp(self):
  self.f=observed_fixtures.ObservedReaderTests('runTest');self.f.setUp();self.addCleanup(self.f.doCleanups);self.root=self.f.root;self.selection={'descriptor':copy.deepcopy(self.f.record['observed_surfaces'])};self.f.record.pop('observed_surfaces')
  # Only original approved assets are in this baseline. New observed bytes
  # exist privately but receive no route until explicit validated publication.
  for p in list(self.f.assets):
   if p.startswith('data/fixture/observed/'):self.f.assets.pop(p)
  self.f.publish();self.canonicalize();self.expected=I.capture_baseline(self.root);self.selections={self.f.source['sha256']:self.selection};self.proof=Mock(side_effect=self.valid_proof);self.stub=patch.dict(sys.modules,{'source_visible_flexure_verify_v1':types.SimpleNamespace(validate_surface_files=self.proof)});self.stub.start();self.addCleanup(self.stub.stop)
 def canonicalize(self):
  public=self.root/'showcase/public';index=json.loads((public/'library-index.json').read_text());catalog=json.loads((public/index['catalog']['path']).read_text());raw=I.encoded(catalog);sha=hashlib.sha256(raw).hexdigest();name='library-catalog-'+sha+'.json';(public/name).write_bytes(raw);index['catalog']={'path':name,'sha256':sha,'bytes':len(raw)};(public/'library-index.json').write_bytes(I.encoded(index));(public/'library-catalog.json').write_bytes(raw)
  for f in ['aligned-scenes.json','public-assets.json','cases-index.json']:
   p=public/f;p.write_bytes(I.encoded(json.loads(p.read_text())))
 def valid_proof(self,reference,root):return {'descriptor':reference,'all_passed':True,'independent_geometry_correctness_established':False,'supported_sources':[1,3],'unknown_sources':[2],'synthetic_new_pixel_proof_fixture_only':True}
 def prepare(self):return I.prepare(root=self.root,expected=self.expected,selections=self.selections)
 def current(self):return{p:p.read_bytes()for p in(self.root/'showcase/public').iterdir()if p.is_file()}
 def test_default_dry_run_has_no_writes_and_only_adds_typed_observation_fields(self):
  before=self.current();plan=self.prepare();self.assertEqual(self.current(),before);self.assertEqual(plan.receipt['status'],'dry_run_validated');self.proof.assert_called_once();old=json.loads(before[self.root/self.expected['catalog']['path']]);new=json.loads(plan.outputs[plan.receipt['catalog']['path']]);new['records'][0].pop('observed_surfaces');self.assertEqual(new,old);self.assertNotIn('showcase/public/cases-index.json',plan.outputs);self.assertFalse(plan.receipt['old_native_raster_geometry_recomputed']);self.assertTrue(plan.receipt['old_program_geometry_approval_fields_unchanged'])
 def test_explicit_commit_writes_index_last_and_keeps_program_native_approval_bytes(self):
  before={p:(self.root/p).read_bytes()for p in self.f.assets};original=I.os.replace;order=[]
  def record(a,b):order.append(Path(b).name);return original(a,b)
  with patch.object(I.os,'replace',side_effect=record):receipt=I.commit(root=self.root,expected=self.expected,selections=self.selections)
  self.assertEqual(order[-1],'library-index.json');self.assertEqual(receipt['status'],'committed');self.assertEqual((self.root/self.expected['cases']['path']).read_bytes(),I._read(self.root,self.expected['cases']));self.assertTrue(all((self.root/p).read_bytes()==b for p,b in before.items()));catalog=json.loads((self.root/receipt['catalog']['path']).read_text());self.assertEqual(catalog['records'][0]['curation'],self.f.record['curation']);self.assertEqual(catalog['records'][0]['interaction'],self.f.program)
 def test_each_stale_cas_file_rejects_before_any_output(self):
  for role,r in self.expected.items():
   p=self.root/r['path'];raw=p.read_bytes();p.write_bytes(raw+b' ');before=self.current()
   try:
    with self.subTest(role=role),self.assertRaises(ValueError):self.prepare()
    self.assertEqual(self.current(),before)
   finally:p.write_bytes(raw)
 def test_old_source_or_native_bytes_changed_reject_without_raster_work(self):
  for r in [self.f.program,self.f.scene,self.f.native]:
   p=self.root/r['path'];b=p.read_bytes();p.write_bytes(b[:-1]+b'!')
   try:
    with self.subTest(path=r['path']),self.assertRaises(ValueError):self.prepare()
   finally:p.write_bytes(b)
  self.proof.assert_not_called()
 def test_missing_false_or_different_new_source_proof_is_rejected(self):
  for proof in [None,{}, {'all_passed':False},{**self.valid_proof(self.selection['descriptor'],self.root),'descriptor':self.f.scene},{**self.valid_proof(self.selection['descriptor'],self.root),'supported_sources':[1,2,3]}]:
   self.proof.side_effect=None;self.proof.return_value=proof
   with self.subTest(proof=proof),self.assertRaises(ValueError):self.prepare()
 def test_unrelated_native_and_forged_camera_reject_before_new_pixel_proof(self):
  original=copy.deepcopy(self.f.document)
  for mutation in ['native','camera']:
   d=copy.deepcopy(original)
   if mutation=='native':d['closed_native_assets']=[self.f.other]
   else:d['camera']['K'][0][0]+=1
   r=self.f.write('data/new-observed/forged.json',d,False);self.selection['descriptor']=r
   with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.prepare()
  self.proof.assert_not_called()
 def test_untyped_runtime_and_unrelated_refs_never_expand_new_public_inventory(self):
  d=copy.deepcopy(self.f.document);private=self.f.write('data/private/model-checkpoint.bin',b'NO-ROUTE',False);d['provider_text']={'runtime_ref':private,'unrelated':self.f.other};new=self.f.write('data/fixture/observed/research-descriptor.json',d,False);self.selection['descriptor']=new;plan=self.prepare();assets=json.loads(plan.outputs['showcase/public/public-assets.json'])['assets'];paths={r['path']for r in assets};self.assertNotIn(private['path'],paths);self.assertNotIn(self.f.other['path'],plan.receipt['new_artifact_paths']);self.assertNotIn('models/',json.dumps(plan.receipt['new_artifact_paths']))
 def test_cadence_policy_is_separate_bound_artifact_and_not_geometry_edits(self):
  from observed_surface_presentation_v1 import derive
  policy=derive(self.f.document,self.selection['descriptor']);r=self.f.write('data/fixture/editorial-policy.json',policy,False);self.selection['presentation']=r;plan=self.prepare();record=json.loads(plan.outputs[plan.receipt['catalog']['path']])['records'][0];self.assertEqual(record['observed_surface_presentation'],r);self.assertEqual(record['observed_surfaces'],self.selection['descriptor']);bad=copy.deepcopy(policy);bad['display_repeat_cap_s']=.6;self.selection['presentation']=self.f.write('data/fixture/bad-policy.json',bad,False)
  with self.assertRaises(ValueError):self.prepare()
 def test_source_not_currently_approved_and_existing_selection_replacement_reject(self):
  self.selections={'b'*64:self.selection}
  with self.assertRaises(ValueError):self.prepare()
  self.selections={self.f.source['sha256']:self.selection};self.f.record['observed_surfaces']=self.f.scene;self.f.publish();self.canonicalize();self.expected=I.capture_baseline(self.root)
  with self.assertRaisesRegex(ValueError,'replace an existing'):self.prepare()
 def test_changed_cas_during_staging_rejects_before_first_public_replace(self):
  original=I._verify_fingerprints;calls=[0];before_index=(self.root/self.expected['index']['path']).read_bytes()
  def race(root,identities):
   original(root,identities);calls[0]+=1
   if calls[0]==2:
    p=self.root/self.expected['aligned']['path'];p.write_bytes(p.read_bytes()+b' ')
  with patch.object(I,'_verify_fingerprints',side_effect=race),patch.object(I.os,'replace')as replace,self.assertRaises(ValueError):I.commit(root=self.root,expected=self.expected,selections=self.selections)
  replace.assert_not_called();self.assertEqual((self.root/self.expected['index']['path']).read_bytes(),before_index)
 def test_changed_verified_artifact_before_final_write_fence_rejects(self):
  original=I._assert_snapshot;calls=[0]
  def race(root,expected):
   value=original(root,expected);calls[0]+=1
   if calls[0]==3:
    p=self.root/self.f.native['path'];p.write_bytes(p.read_bytes()[:-1]+b'!')
   return value
  with patch.object(I,'_assert_snapshot',side_effect=race),patch.object(I.os,'replace')as replace,self.assertRaises(ValueError):I.commit(root=self.root,expected=self.expected,selections=self.selections)
  replace.assert_not_called()

if __name__=='__main__':unittest.main()
