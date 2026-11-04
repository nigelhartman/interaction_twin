"""Fresh-asset motion safety boundaries; no learned model or remote execution."""
import copy
import unittest
from unittest import mock
import numpy as np
import sam3d_layout_motion as F


class FreshNativeMotionTests(unittest.TestCase):
    def identity(self):
        model={'backend':'Meta SAM 3D Objects','fixture':False,'sam3d_output':True,'learned_inference_executed':True,
               'geometry_reconstructed_by_model':True,'manual_geometry_fallback':False,'manual_pose_used':False,
               'object_id':'earbud_a','source':{'video_sha256':'video'}}
        seed={'native_glb':{'sha256':'new-mesh'},'source_binding':{'track_id':'earbud_a','source':{'video_sha256':'video'}},
              'optimized':False,'old_pose_reused':False}
        return model,seed

    def queries(self):
        uv=np.c_[np.arange(30),np.ones(30)];xyz=np.c_[uv,np.ones(30)];hits=np.ones(30,bool)
        train,held=F.T.split_ids(hits)
        query={'query_uv':uv,'canonical_xyz':xyz,'reference_ray_hits':hits,'training_point_ids':train,'heldout_point_ids':held}
        plan={'queries':{'count':30,'fresh_new_mesh_ray_hits':30,'training_ids':train.tolist(),'heldout_ids':held.tolist()}}
        return query,plan

    def test_genuine_mesh_source_component_and_raw_seed_cannot_be_substituted(self):
        model,seed=self.identity();F.verify_identity(model,'new-mesh',seed)
        for field,value in [('backend','TripoSR'),('fixture',True),('manual_pose_used',True),('manual_geometry_fallback',True)]:
            bad=copy.deepcopy(model);bad[field]=value
            with self.assertRaises(ValueError):F.verify_identity(bad,'new-mesh',seed)
        for change in ['mesh','source','component','old-pose']:
            bad=copy.deepcopy(seed)
            if change=='mesh':bad['native_glb']['sha256']='old-mesh'
            if change=='source':bad['source_binding']['source']['video_sha256']='other-video'
            if change=='component':bad['source_binding']['track_id']='lid'
            if change=='old-pose':bad['old_pose_reused']=True
            with self.assertRaises(ValueError):F.verify_identity(model,'new-mesh',bad)

    def test_fresh_queries_raycast_the_same_new_mesh_in_batches_of_at_most_eight(self):
        mesh=object();seeds=np.c_[np.arange(30),np.arange(30)+100];calls=[]
        def ray(batch,actual_mesh,k,r,t):
            self.assertIs(actual_mesh,mesh);self.assertLessEqual(len(batch),8);calls.append(batch.copy())
            return np.c_[batch,np.ones(len(batch))],np.ones(len(batch),bool)
        xyz,hits,train,held=F.fresh_ray_queries(seeds,mesh,np.eye(3),np.eye(3),np.zeros(3),ray_function=ray)
        np.testing.assert_array_equal(np.concatenate(calls),seeds)
        np.testing.assert_array_equal(xyz[:,:2],seeds);self.assertEqual(len(calls),4)
        self.assertFalse(np.intersect1d(train,held).size);self.assertTrue(np.all(held%3==0))
        with self.assertRaises(ValueError):F.fresh_ray_queries(np.zeros((97,2)),mesh,None,None,None,ray_function=ray)

    def test_heldout_points_never_enter_pose_solver(self):
        query,_=self.queries();train=query['training_point_ids'];held=query['heldout_point_ids'];calls=[]
        def solve(xyz,uv,k,r,t):
            calls.append(xyz.copy());np.testing.assert_array_equal(xyz,query['canonical_xyz'][train])
            self.assertFalse(set(xyz[:,0])&set(query['canonical_xyz'][held,0]))
            return {'success':True,'rotation':np.eye(3),'translation':np.zeros(3)}
        fit,present,checks,error,positive=F.fit_frame(query['canonical_xyz'],query['query_uv'],np.ones(30,bool),train,held,
                                                   np.eye(3),np.eye(3),np.zeros(3),solver=solve)
        self.assertEqual(len(calls),1);np.testing.assert_array_equal(checks,held);self.assertTrue(positive)
        np.testing.assert_allclose(error,0)

    def test_negative_depth_heldout_is_unsupported_even_if_uv_is_finite(self):
        query,_=self.queries();train=query['training_point_ids'];held=query['heldout_point_ids']
        query['canonical_xyz'][held,2]=-1
        solve=lambda *args:{'success':True,'rotation':np.eye(3),'translation':np.zeros(3)}
        _,_,checks,error,positive=F.fit_frame(query['canonical_xyz'],query['query_uv'],np.ones(30,bool),train,held,
                                            np.eye(3),np.eye(3),np.zeros(3),solver=solve)
        self.assertFalse(positive);self.assertTrue(np.isinf(error).all());self.assertEqual(len(checks),10)
        reasons=F.conditional_gate(20,20,error,.9,.9,'plausible_visible_mask',rendered=True,positive=positive,withheld=False)
        self.assertIn('heldout_nonpositive_or_nonfinite_depth',reasons)

    def test_unaudited_or_unrendered_rows_never_pass_conditional_gate(self):
        kw={'rendered':True,'positive':True,'withheld':False}
        self.assertEqual(F.conditional_gate(20,20,np.zeros(10),.9,.9,'plausible_visible_mask',**kw),[])
        reasons=F.conditional_gate(20,20,np.zeros(10),.9,.9,'unassessed',**kw)
        self.assertIn('source_mask_not_plausible_in_model_audit',reasons)
        kw['rendered']=False
        self.assertIn('raw_full_mesh_silhouette_not_evaluated',F.conditional_gate(20,20,np.zeros(10),.9,.9,'plausible_visible_mask',**kw))

    def test_source_pts_join_uses_explicit_ids_never_array_position(self):
        frames=[{'index':40,'timestamp_s':13.001144},{'index':38,'timestamp_s':12.334422}]
        track=[{'sample_index':38,'timestamp_s':12.334422},{'sample_index':40,'timestamp_s':13.001144}]
        self.assertEqual([r['index'] for r in F.source_rows_by_id(frames,track)],[38,40])
        with self.assertRaises(ValueError):F.source_rows_by_id(frames,track+[{'sample_index':39,'timestamp_s':12.6}])
        bad=copy.deepcopy(track);bad[0]['timestamp_s']=12.3
        with self.assertRaises(ValueError):F.source_rows_by_id(frames,bad)
        with self.assertRaises(ValueError):F.source_rows_by_id(frames+[frames[0]],track)

    def test_query_split_cannot_be_changed_after_fitting_or_have_invalid_native_hits(self):
        query,plan=self.queries();F.validate_queries(query,plan)
        query['training_point_ids'][0]=0
        with self.assertRaises(ValueError):F.validate_queries(query,plan)
        query,plan=self.queries();query['canonical_xyz'][0]=np.nan
        with self.assertRaises(ValueError):F.validate_queries(query,plan)

    def test_heldout_overlap_rejected_before_solver(self):
        query,_=self.queries();solver=mock.Mock()
        with self.assertRaises(ValueError):F.fit_frame(query['canonical_xyz'],query['query_uv'],np.ones(30,bool),np.arange(12),np.arange(6),
                                                     np.eye(3),np.eye(3),np.zeros(3),solver=solver)
        solver.assert_not_called()


if __name__=='__main__':unittest.main()
