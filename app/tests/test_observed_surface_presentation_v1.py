"""Editorial sample repetition must not become evidence or bridge unknowns."""
import copy
import unittest
import observed_surface_presentation_v1 as P

class ObservedSurfacePresentationTests(unittest.TestCase):
 def setUp(self):
  self.ref={'path':'data/example/descriptor.json','sha256':'a'*64,'bytes':100}
  self.doc={'format':'interaction-twin-source-visible-flexure-series','kind':'source_observed_monocular_depth_prior','source':{'sha256':'b'*64,'duration_s':5},'policy':{'maximum_source_time_error_s':.18},'samples':[{'source_sample_index':1,'timestamp_s':0.,'status':'source_observed_prior','asset':{'sha256':'c'*64}},{'source_sample_index':2,'timestamp_s':2.,'status':'unknown'},{'source_sample_index':3,'timestamp_s':4.,'status':'source_observed_prior','asset':{'sha256':'d'*64}}]}
  self.policy=P.derive(self.doc,self.ref)
 def test_repeat_declares_original_time_and_not_current_geometry(self):
  row=P.at(self.policy,self.doc,.5)
  self.assertEqual(row['observation_timestamp_s'],0.)
  self.assertEqual(row['source_time_error_s'],.5)
  self.assertEqual(row['requested_video_time_s'],.5)
  self.assertEqual(row['presentation_mode'],'editorial_source_sample_repeat')
  self.assertFalse(row['current_geometry'])
  self.assertFalse(row['interpolated'])
  self.assertFalse(row['new_tracking_or_geometry_recovered'])
 def test_sparse_cadence_never_exceeds_separate_display_cap(self):
  self.assertIsNotNone(P.at(self.policy,self.doc,.55))
  self.assertIsNone(P.at(self.policy,self.doc,.550001))
  self.assertIsNone(P.at(self.policy,self.doc,4.550001))
 def test_unknown_nearest_source_is_never_searched_past(self):
  for t in [1.1,1.9,2.,2.1,2.9]:self.assertIsNone(P.at(self.policy,self.doc,t))
 def test_estimate_toggle_and_invalid_times_withhold(self):
  for t in [True,-1,6,float('nan'),float('inf'),'0']:
   self.assertIsNone(P.at(self.policy,self.doc,t))
  self.assertIsNone(P.at(self.policy,self.doc,0.,False))
 def test_forged_window_or_scope_or_lineage_is_rejected(self):
  for key,value in [('display_repeat_cap_s',.56),('current_geometry',True),('geometry_interpolated',True),('search_past_unknown_rows',True),('strict_observation_query_maximum_source_time_error_s',.55)]:
   bad=copy.deepcopy(self.policy);bad[key]=value
   with self.assertRaises(ValueError):P.validate(bad,self.doc,self.ref)
  bad=copy.deepcopy(self.policy);bad['samples'][0]['display_window_end_s']+=.01
  with self.assertRaises(ValueError):P.validate(bad,self.doc,self.ref)
  other={**self.ref,'sha256':'0'*64}
  with self.assertRaises(ValueError):P.validate(self.policy,self.doc,other)
 def test_actual_asymmetric_neighbor_cadence_controls_each_side(self):
  doc=copy.deepcopy(self.doc);doc['samples'][1]['timestamp_s']=.6;doc['samples'][2]['timestamp_s']=1.4
  policy=P.derive(doc,self.ref)
  self.assertEqual(policy['samples'][0]['display_window_end_s'],.3)
  self.assertAlmostEqual(policy['samples'][1]['display_window_start_s'],.3)
  self.assertAlmostEqual(policy['samples'][1]['display_window_end_s'],1.)
 def test_widened_strict_observation_query_is_not_accepted(self):
  doc=copy.deepcopy(self.doc);doc['policy']['maximum_source_time_error_s']=.55
  with self.assertRaises(ValueError):P.derive(doc,self.ref)

if __name__=='__main__':unittest.main()
