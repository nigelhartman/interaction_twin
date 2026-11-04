"""Frozen motion transport tests; synthetic byte fixtures are not CV evidence."""
import copy
import json
import unittest
from scripts import build_judge_preview as B
from tests import test_curated_motion_hypotheses_v1 as fixtures


class JudgeMotionLineageTests(unittest.TestCase):
 def setUp(self):
  self.f=fixtures.MotionReaderTests('runTest');self.f.setUp();self.addCleanup(self.f.doCleanups)
 def verify(self,approved=None,catalog=None):
  return B.verify_display_lineage(self.f.root,self.f.assets if approved is None else approved,
      catalog={'records':self.f.records}if catalog is None else catalog)
 def test_exact_current_prediction_graph_is_included_without_inference(self):
  proof=self.verify();self.assertEqual(len(proof),1);self.assertTrue(proof[0]['motion_hypotheses_included'])
  self.assertFalse(proof[0]['original_observations_modified']);self.assertFalse(proof[0]['physical_accuracy_verified'])
  self.assertEqual(proof[0]['new_inference_calls'],0);self.assertGreater(proof[0]['artifact_count'],0)
 def test_every_typed_data_artifact_requires_exact_admission(self):
  import motion_hypothesis_contract_v1 as T
  checked=T.validate(self.f.document,lambda r,maximum=None:(self.f.root/r['path']).read_bytes())
  for ref in [self.f.record['motion_hypothesis'],*checked['artifacts']]:
   with self.subTest(path=ref['path']),self.assertRaises(ValueError):
    self.verify({k:v for k,v in self.f.assets.items()if k!=ref['path']})
 def test_catalog_program_record_and_declaration_cannot_be_substituted(self):
  for key in ['id','interaction','motion_hypothesis','source']:
   r=copy.deepcopy(self.f.record)
   if key=='id':r[key]='foreign-record'
   elif key=='source':r[key]['sha256']='a'*64
   else:r[key]=self.f.native
   with self.subTest(key=key),self.assertRaises(ValueError):self.verify(catalog={'records':[r]})
 def test_omitted_current_motion_pointer_is_rejected(self):
  path=self.f.root/'showcase/public/aligned-scenes.json';doc=json.loads(path.read_bytes());doc['scenes'][0].pop('motion_hypothesis');path.write_text(json.dumps(doc))
  with self.assertRaisesRegex(ValueError,'omitted'):self.verify()
 def test_changed_native_and_forced_known_pose_reject(self):
  p=self.f.root/self.f.native['path'];p.write_bytes(p.read_bytes()+b'!')
  with self.assertRaisesRegex(ValueError,'Pinned source bytes changed'):self.verify()
 def test_local_bounds_are_measured_and_motion_runtime_is_explicit(self):
  self.assertEqual(B.MAX_PREVIEW_BYTES,768*1024*1024);self.assertIn('estimated-motion-v1.js',B.STATIC)


if __name__=='__main__':unittest.main()
