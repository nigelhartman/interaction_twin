"""Exact offline observation byte graph, not a geometry/model rerun."""
import copy,json,unittest
from unittest.mock import patch
from scripts import build_judge_preview as B
from tests import test_curated_observed_surfaces_v1 as fixtures
class JudgeObservedSurfaceLineageTests(unittest.TestCase):
 def setUp(self):
  self.f=fixtures.ObservedReaderTests('runTest');self.f.setUp();self.addCleanup(self.f.doCleanups);self.root=self.f.root
 def verify(self,approved=None,catalog=None):return B.verify_display_lineage(self.root,self.f.assets if approved is None else approved,catalog=catalog)
 def test_exact_current_surface_inventory_is_included_without_heavy_geometry_or_model_work(self):
  with patch('source_visible_flexure_verify_v1.validate_surface_files',side_effect=AssertionError('No geometry or model rerun')):proof=self.verify()
  self.assertEqual(len(proof),1);self.assertTrue(proof[0]['observed_surfaces_included']);self.assertEqual(proof[0]['supported_sources'],[1,3]);self.assertEqual(proof[0]['unknown_sources'],[2]);self.assertFalse(proof[0]['presentation_included']);self.assertEqual(proof[0]['new_inference_calls'],0)
 def test_each_typed_surface_artifact_must_be_exactly_admitted(self):
  import source_visible_flexure_contract_v1 as F
  checked=F.validate(self.f.document,lambda r:(self.root/r['path']).read_bytes())
  pointers=[self.f.record['observed_surfaces'],*checked['artifacts']]
  for pointer in pointers:
   with self.subTest(path=pointer['path']),self.assertRaisesRegex(ValueError,'missing an exact approved public reference'):
    self.verify({k:v for k,v in self.f.assets.items()if k!=pointer['path']})
 def test_surface_GLTF_and_pixel_map_bytes_are_checked_as_opaque_data(self):
  row=self.f.document['samples'][0]
  for key in ['asset','source_pixel_map']:
   r=row[key];p=self.root/r['path'];raw=p.read_bytes();p.write_bytes(raw+b'!')
   try:
    with self.subTest(key=key),self.assertRaisesRegex(ValueError,'Pinned source bytes changed'):self.verify()
   finally:p.write_bytes(raw)
 def test_current_catalog_and_aligned_declaration_cannot_disagree(self):
  for mutation in ['declaration','program','source']:
   r=copy.deepcopy(self.f.record)
   if mutation=='declaration':r.pop('observed_surfaces')
   elif mutation=='program':r['interaction']=self.f.other
   else:r['source']['sha256']='b'*64
   with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.verify(catalog={'records':[r]})
 def test_exact_editorial_policy_is_present_and_cannot_widen_the_observation_claim(self):
  self.f.presentation();self.f.publish();proof=self.verify();self.assertTrue(proof[0]['presentation_included'])
  policy=self.f.presentation();policy['display_repeat_cap_s']=.6;self.f.presentation(policy);self.f.publish()
  with self.assertRaisesRegex(ValueError,'Editorial policy differs'):self.verify()
 def test_untyped_private_runtime_references_do_not_enter_the_observed_byte_graph(self):
  private=self.f.write('data/private/runtime.bin',b'PRIVATE-NOT-ADMITTED',False);self.f.document['provider_reply']={'runtime_ref':private};self.f.pin();self.f.publish();proof=self.verify();self.assertTrue(proof[0]['observed_surfaces_included']);self.assertNotIn(private['path'],self.f.assets)
 def test_source_runtime_dependencies_are_explicitly_selected(self):
  for name in ['auto-scene-camera-v1.js','source-visible-flexure-v1.js','source-surface-presentation-v1.js','agent-charge-example.js']:self.assertIn(name,B.STATIC)
if __name__=='__main__':unittest.main()
