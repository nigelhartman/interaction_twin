import copy,sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from perception import resolve_masks,validate_event_result
class PerceptionTests(unittest.TestCase):
 def test_child_masks_override_parent_and_peer_conflicts_are_unknown(self):
  parent=np.ones((3,4),bool);a=np.zeros_like(parent);b=a.copy();a[1,1:3]=True;b[1,2:4]=True
  out,unknown=resolve_masks({'camera':parent,'a':a,'b':b},{})
  self.assertTrue(unknown[1,2]);self.assertFalse(any(m[1,2] for m in out.values()))
  self.assertTrue(out['a'][1,1]);self.assertFalse(out['camera'][1,1]);self.assertTrue(out['camera'][0,0])
 def test_withheld_prediction_is_not_exported_as_a_visible_part(self):
  parent=np.ones((2,2),bool);out,unknown=resolve_masks({'camera':parent,'uncertain':parent},{},['uncertain'])
  self.assertNotIn('uncertain',out);self.assertTrue(out['camera'].all())
 def valid(self):
  return {'events':[{'id':'e1','start_s':34.0,'end_s':34.4,'participant_track_ids':['camera'],'evidence_samples':[1,2]}],'identity_links':[]}
 def test_event_evidence_must_reference_real_samples_in_interval(self):
  rows=[{'index':1,'timestamp_s':34.0},{'index':2,'timestamp_s':34.2},{'index':3,'timestamp_s':35.0}]
  result=self.valid();self.assertTrue(validate_event_result(result,rows,['camera']))
  for change in ['wrong_sample','wrong_time','wrong_track','duplicate_id']:
   r=copy.deepcopy(result)
   if change=='wrong_sample':r['events'][0]['evidence_samples']=[999]
   if change=='wrong_time':r['events'][0]['evidence_samples']=[3]
   if change=='wrong_track':r['events'][0]['participant_track_ids']=['invented']
   if change=='duplicate_id':r['events'].append(copy.deepcopy(r['events'][0]))
   with self.subTest(change=change),self.assertRaises(ValueError):validate_event_result(r,rows,['camera'])
 def test_identity_claims_cannot_reference_unknown_tracklets(self):
  r=self.valid();r['identity_links']=[{'from_track_id':'camera','to_track_id':'invented','evidence_samples':[1]}]
  with self.assertRaises(ValueError):validate_event_result(r,[{'index':1,'timestamp_s':34.0},{'index':2,'timestamp_s':34.2}],['camera'])
if __name__=='__main__':unittest.main()
