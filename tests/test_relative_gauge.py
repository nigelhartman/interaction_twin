import unittest
import numpy as np
from relative_gauge import log_depth_measurement, pair_edges, solve_gauge


class RelativeGaugeTests(unittest.TestCase):
    def test_frame_scale_cancels_and_known_component_ratios_recover(self):
        frames=[]
        for t,g in enumerate([.3,4.,1.2]):
            frames.append({'timestamp_s':t,'objects':[{'object_id':n,'status':'measured','log_scale':np.log(g*s)} for n,s in [('a',2.),('b',.5),('c',3.)]]})
        edges=pair_edges(frames);result=solve_gauge(['a','b','c'],edges,'a',2.)
        for n,s in [('a',2.),('b',.5),('c',3.)]:self.assertAlmostEqual(result['selected_scales'][n],s,places=7)
        self.assertTrue(result['temporal_stability_verified_for_all_edges'])

    def test_corrupt_whole_heldout_timestamp_is_rejected(self):
        frames=[{'timestamp_s':t,'objects':[{'object_id':'a','status':'measured','log_scale':float(v)}, {'object_id':'b','status':'measured','log_scale':0.}]} for t,v in enumerate([0.,0.,2.])]
        edge=pair_edges(frames)[0]
        self.assertFalse(edge['accepted_for_conditional_graph'])
        self.assertEqual(edge['heldout_timestamps_s'],[2.])

    def test_disconnected_components_are_not_given_equal_depth(self):
        result=solve_gauge(['a','b'],[],'a',.1)
        self.assertEqual(result['unplaced_components'],['b'])
        self.assertNotIn('b',result['selected_scales'])

    def test_sparse_edge_is_explicitly_unvalidated(self):
        edge=pair_edges([{'timestamp_s':1.,'objects':[{'object_id':n,'status':'measured','log_scale':q} for n,q in [('a',0.),('b',1.)]]}])[0]
        result=solve_gauge(['a','b'],[edge],'a',1.)
        self.assertEqual(edge['status'],'single_or_two_frame_candidate')
        self.assertFalse(result['temporal_stability_verified_for_all_edges'])

    def test_measurements_use_same_valid_positive_pixels(self):
        d=np.full((8,8),4.);r=np.full((8,8),2.);support=np.ones((8,8),bool)
        d[0,0]=np.nan;r[0,1]=np.inf;d[0,2]=-1
        result=log_depth_measurement(d,r,support)
        self.assertAlmostEqual(result['scale'],2.)
        self.assertEqual(result['support_pixels'],61)
        with self.assertRaises(ValueError):log_depth_measurement(d[:2],r,support)
        with self.assertRaises(ValueError):log_depth_measurement(d,r,np.zeros_like(support))

    def test_duplicate_frame_times_are_not_independent_checks(self):
        with self.assertRaises(ValueError):pair_edges([{'timestamp_s':1.,'objects':[]},{'timestamp_s':1.,'objects':[]}])

    def test_conflicting_scale_cycle_is_rejected_without_forcing_a_fit(self):
        edges=[{'a':a,'b':b,'log_ratio':q,'accepted_for_conditional_graph':True,'heldout_timestamps_s':[]} for a,b,q in [('a','b',0.),('b','c',0.),('a','c',2.)]]
        result=solve_gauge(['a','b','c'],edges,'a',1.)
        self.assertEqual(result['status'],'rejected_graph_inconsistency')
        self.assertEqual(result['selected_scales'],{})


if __name__=='__main__':unittest.main()
