"""CAS/authority fixtures only; dummy GLBs never certify real surface geometry."""
import copy,hashlib,json,sys,types,unittest
from pathlib import Path
from unittest.mock import Mock,patch
import incremental_observed_revision_v2 as I
from tests import test_curated_observed_surfaces_v1 as observed_fixtures
from tests import test_incremental_observed_publication_v1 as publication_fixtures
from observed_surface_presentation_v1 import derive

class IncrementalObservedRevisionTests(unittest.TestCase):
 def setUp(self):
  self.f=observed_fixtures.ObservedReaderTests('runTest');self.f.setUp();self.addCleanup(self.f.doCleanups);self.root=self.f.root
  for row in self.f.document['samples']:
   if row['status']!='unknown':row['source_grid_stride_px']=4
  self.f.pin();self.parent=copy.deepcopy(self.f.record['observed_surfaces']);self.f.presentation();self.f.publish();publication_fixtures.IncrementalObservedPublicationTests.canonicalize(self);self.expected=I.capture_baseline(self.root);self.old=copy.deepcopy(self.f.document);self.document=copy.deepcopy(self.old);d=self.document;d['source_footprint_proof_version']=2;d['cached_depth_parent']=self.parent;d['policy']['triangle_support']='all_unit_pixel_squares_intersecting_closed_triangle_inside_original_2px_SAM_core';protocol=json.loads((self.root/self.f.protocol['path']).read_text());protocol['cached_depth_parent']=self.parent;self.protocol=protocol
  for row,previous in zip(d['samples'],self.old['samples']):
   if row['status']=='unknown':continue
   i=row['source_sample_index'];folder=f'data/revised/source-{i:03d}';row.update({'source_footprint_proof_version':2,'topology_refined_from_cached_depth':True,'cached_depth_parent_map':previous['source_pixel_map'],'depth_sampling':{'method':'cached_monocular_prior_barycentric_or_bounded_nearest_not_measured_depth','parent_grid_stride_px':4,'maximum_nearest_sampling_distance_px':9.,'maximum_interpolation_parent_edge_px':10.}})
   for key,suffix in [('asset','mesh.glb'),('source_pixel_map','map.npz')]:row[key]=self.f.write(folder+'/'+suffix,b'EXPLICIT-NEW-SYNTHETIC-AUTHORITY-FIXTURE'+suffix.encode()+bytes([i]),False)
   row.pop('provenance',None);row['provenance']=self.f.write(folder+'/provenance.json',row,False)
  self.repin();self.proof=Mock(side_effect=self.valid_proof);self.stub=patch.dict(sys.modules,{'source_visible_flexure_verify_v1':types.SimpleNamespace(validate_surface_files=self.proof)});self.stub.start();self.addCleanup(self.stub.stop)
 def repin(self):
  self.document['protocol']=self.f.write('data/revised/protocol.json',self.protocol,False);self.selection={'descriptor':self.f.write('data/revised/descriptor.json',self.document,False)};self.selection['presentation']=self.f.write('data/revised/presentation.json',derive(self.document,self.selection['descriptor']),False);self.selections={self.f.source['sha256']:self.selection}
 def valid_proof(self,r,root):
  d=json.loads((root/r['path']).read_text());return {'all_passed':True,'descriptor':r,'independent_geometry_correctness_established':False,'supported_sources':[x['source_sample_index']for x in d['samples']if x['status']!='unknown'],'unknown_sources':[x['source_sample_index']for x in d['samples']if x['status']=='unknown'],'synthetic_new_pixel_proof_fixture_only':True}
 def prepare(self):return I.prepare(root=self.root,expected=self.expected,selections=self.selections)
 def current(self):return{p:p.read_bytes()for p in(self.root/'showcase/public').iterdir()if p.is_file()}
 def test_exact_current_parent_revision_works_and_is_readonly_by_default(self):
  before=self.current();plan=self.prepare();self.assertEqual(self.current(),before);record=json.loads(plan.outputs[plan.receipt['catalog']['path']])['records'][0];self.assertEqual(record['observed_surfaces'],self.selection['descriptor']);self.assertEqual(record['observed_surface_presentation'],self.selection['presentation']);self.assertEqual(record['interaction'],self.f.program);self.assertEqual(record['curation'],self.f.record['curation']);self.assertTrue(plan.receipt['old_program_geometry_approval_fields_unchanged']);self.assertFalse(plan.receipt['old_native_raster_geometry_recomputed'])
 def test_foreign_even_source_matching_parent_is_rejected(self):
  parent=copy.deepcopy(self.old);parent['note']='Different unselected research descriptor';foreign=self.f.write('data/foreign/parent.json',parent,False);self.document['cached_depth_parent']=foreign;self.protocol['cached_depth_parent']=foreign;self.repin()
  with self.assertRaisesRegex(ValueError,'currently selected'):self.prepare()
 def test_unadmitted_parent_pointer_cannot_be_healed_into_public_inventory(self):
  self.f.assets.pop(self.parent['path']);self.f.publish();publication_fixtures.IncrementalObservedPublicationTests.canonicalize(self);self.expected=I.capture_baseline(self.root)
  with self.assertRaises(ValueError):self.prepare()
 def test_known_row_cannot_be_dropped_even_if_new_pixel_proof_agrees(self):
  self.document['samples'][2]={'source_sample_index':3,'timestamp_s':2.,'status':'unknown'};self.repin()
  with self.assertRaisesRegex(ValueError,'known or unknown'):self.prepare()
 def test_original_unknown_cannot_be_promoted(self):
  self.document['samples'][1]=copy.deepcopy(self.document['samples'][0]);self.document['samples'][1].update(source_sample_index=2,timestamp_s=1.);self.repin()
  with self.assertRaisesRegex(ValueError,'unknown barrier'):self.prepare()
 def test_existing_cadence_policy_requires_matching_replacement(self):
  self.selection.pop('presentation')
  with self.assertRaisesRegex(ValueError,'matching explicit presentation'):self.prepare()
 def test_stale_policy_and_wrong_source_or_camera_reject(self):
  original=copy.deepcopy(self.document)
  for mutation in ['policy','source','camera']:
   self.document=copy.deepcopy(original);self.repin()
   if mutation=='policy':self.selection['presentation']=self.f.record['observed_surface_presentation']
   elif mutation=='source':self.document['source']['sha256']='b'*64;self.repin()
   else:self.document['camera']['K'][0][0]+=1;self.repin()
   with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.prepare()
 def test_native_replacement_and_runtime_route_reject(self):
  original=copy.deepcopy(self.document)
  for mutation in ['native','runtime']:
   self.document=copy.deepcopy(original)
   if mutation=='native':self.document['closed_native_assets']=[self.f.other]
   else:self.document['samples'][0]['depth_model']['path']='models/private-runtime.pt'
   self.repin()
   with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.prepare()
 def test_untyped_private_references_are_never_admitted(self):
  private=self.f.write('data/private/runtime.bin',b'NEVER-ROUTE',False);self.document['provider_response']={'runtime':private};self.repin();plan=self.prepare();paths={r['path']for r in json.loads(plan.outputs['showcase/public/public-assets.json'])['assets']};self.assertNotIn(private['path'],paths)
 def test_missing_false_or_mismatched_fresh_proof_rejects(self):
  for result in [None,{}, {'all_passed':False},{**self.valid_proof(self.selection['descriptor'],self.root),'descriptor':self.parent}]:
   self.proof.side_effect=None;self.proof.return_value=result
   with self.subTest(result=result),self.assertRaises(ValueError):self.prepare()
 def test_each_of_five_stale_cas_files_rejects_before_writes(self):
  for role,r in self.expected.items():
   path=self.root/r['path'];raw=path.read_bytes();path.write_bytes(raw+b' ');before=self.current()
   try:
    with self.subTest(role=role),self.assertRaises(ValueError):self.prepare()
    self.assertEqual(self.current(),before)
   finally:path.write_bytes(raw)
 def test_changed_old_program_scene_native_and_parent_bytes_reject(self):
  for r in [self.f.program,self.f.scene,self.f.native,self.parent]:
   p=self.root/r['path'];raw=p.read_bytes();p.write_bytes(raw+b'!')
   try:
    with self.subTest(path=r['path']),self.assertRaises(ValueError):self.prepare()
   finally:p.write_bytes(raw)
 def test_explicit_commit_keeps_all_old_artifacts_and_approvals_index_last(self):
  artifacts={p:(self.root/p).read_bytes()for p in self.f.assets};cases=(self.root/self.expected['cases']['path']).read_bytes();original=I.os.replace;order=[]
  def replace(a,b):order.append(Path(b).name);return original(a,b)
  with patch.object(I.os,'replace',side_effect=replace):result=I.commit(root=self.root,expected=self.expected,selections=self.selections)
  self.assertEqual(result['status'],'committed');self.assertEqual(order[-1],'library-index.json');self.assertTrue(all((self.root/p).read_bytes()==v for p,v in artifacts.items()));self.assertEqual((self.root/self.expected['cases']['path']).read_bytes(),cases)
 def test_final_staging_cas_race_prevents_first_replace(self):
  original=I._verify_fingerprints;calls=[0]
  def race(root,identities):
   original(root,identities);calls[0]+=1
   if calls[0]==2:
    p=self.root/self.expected['aligned']['path'];p.write_bytes(p.read_bytes()+b' ')
  with patch.object(I,'_verify_fingerprints',side_effect=race),patch.object(I.os,'replace')as replace,self.assertRaises(ValueError):I.commit(root=self.root,expected=self.expected,selections=self.selections)
  replace.assert_not_called()
 def test_verified_native_changed_before_final_fence_prevents_writes(self):
  original=I._assert_snapshot;calls=[0]
  def race(root,expected):
   value=original(root,expected);calls[0]+=1
   if calls[0]==3:
    p=self.root/self.f.native['path'];p.write_bytes(p.read_bytes()+b'!')
   return value
  with patch.object(I,'_assert_snapshot',side_effect=race),patch.object(I.os,'replace')as replace,self.assertRaises(ValueError):I.commit(root=self.root,expected=self.expected,selections=self.selections)
  replace.assert_not_called()

if __name__=='__main__':unittest.main()
