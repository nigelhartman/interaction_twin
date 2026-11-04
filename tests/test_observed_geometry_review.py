"""Independent degeneracy/validation checks; all data here is synthetic."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

import observed_geometry as G
from tests.test_observed_geometry import synthetic_scene


class ObservedGeometryReviewTests(unittest.TestCase):
    def test_focal_variant_keeps_requested_training_seed_pair(self):
        _,frames,tracks,_,_,_=synthetic_scene(noise=0.)
        K=G.camera_matrix(640,480,.8)
        model,logs=G.find_seed(tracks,frames,K,seed_pair=(10,0))
        self.assertIsNotNone(model)
        self.assertEqual((model['seed']['a'],model['seed']['b']),(0,10))
        self.assertEqual(len(logs),1)
        self.assertEqual((logs[0]['a'],logs[0]['b']),(0,10))

    def test_invalid_fixed_seed_cannot_fall_back_to_another_component(self):
        _,frames,tracks,_,K,_=synthetic_scene(noise=0.)
        unrestricted,_=G.find_seed(tracks,frames,K)
        self.assertIsNotNone(unrestricted)
        # View 2 is held out; a requested seed containing it must not be used,
        # nor silently replaced by a different available training pair.
        restricted,logs=G.find_seed(tracks,frames,K,seed_pair=(0,2))
        self.assertIsNone(restricted)
        self.assertEqual(logs,[])

    def test_vectorized_bundle_residual_matches_pointwise_projection(self):
        xyz,frames,tracks,_,K,truth=synthetic_scene(count=30,views=5,noise=.08)
        poses={fi:truth[fi] for fi in (0,1,3,4)}
        model={'seed':{'a':0,'b':4},'poses':poses,
               'points':{tid:point.copy() for tid,point in zip(tracks,xyz)}}
        expected=[]
        for tid in sorted(model['points']):
            for fi,pixel in tracks[tid].items():
                if fi in poses:
                    uv,_=G.project(model['points'][tid][None],poses[fi],K)
                    expected.extend(uv[0]-pixel)
        inspected=[]
        def inspect_residual(function,initial,**kwargs):
            np.testing.assert_allclose(function(initial),expected,atol=1e-10,rtol=1e-10)
            inspected.append(True)
            return SimpleNamespace(x=initial,success=True,nfev=1)
        with patch.object(G,'least_squares',side_effect=inspect_residual):
            G.bundle_adjust(model,tracks,K,max_nfev=1)
        self.assertEqual(inspected,[True])

    def test_camera_rotation_without_baseline_has_zero_ray_angle(self):
        points=np.array([.4,.2,4.])
        observations={0:np.array([1.,2.]),1:np.array([3.,4.])}
        poses={0:(np.eye(3),np.zeros(3)),
               1:(cv2.Rodrigues(np.array([.3,-.2,.1]))[0],np.zeros(3))}
        self.assertLess(G.maximum_track_angle(points,observations,poses),1e-6)

    def test_post_bundle_adjustment_rejects_low_parallax_despite_perfect_fit(self):
        K=G.camera_matrix(640,480)
        poses={0:(np.eye(3),np.zeros(3)),1:(np.eye(3),np.array([-1.,0.,0.]))}
        xyz=np.array([.2,.1,2000.])
        tracks={'far_point':{fi:G.project(xyz[None],pose,K)[0][0] for fi,pose in poses.items()}}
        model={'seed':{'a':0,'b':1},'poses':poses,'points':{'far_point':xyz}}
        refined=G.bundle_adjust(model,tracks,K,max_nfev=2)
        self.assertEqual(refined['points'],{})
        self.assertEqual(refined['bundle_adjustment']['rejected_low_parallax_points_after_fit'],1)
        self.assertLess(refined['bundle_adjustment']['training_median_after_px'],1e-8)

    def test_tiny_shared_cloud_reports_insufficient_sensitivity_coverage(self):
        rng=np.random.default_rng(17)
        reference={f'p{i}':p for i,p in enumerate(rng.normal(size=(100,3)))}
        candidate={key:point.copy() for key,point in list(reference.items())[:12]}
        comparison=G.similarity_difference(reference,candidate)
        self.assertLess(comparison['normalized_rms'],1e-10)
        self.assertEqual(comparison['common_points'],12)
        self.assertAlmostEqual(comparison['common_reference_fraction'],.12)
        self.assertAlmostEqual(comparison['common_candidate_fraction'],1.)
        self.assertLess(comparison['common_reference_fraction'],
                        G.PROTOCOL['sensitivity_min_common_reference_fraction'])

    def test_behind_camera_scored_observation_is_retained_as_failed_evidence(self):
        rng=np.random.default_rng(23)
        points={f'point_{i:06d}':np.r_[rng.uniform(-1.,1.,2),4.] for i in range(100)}
        scored=[tid for tid in points if G.score_track(tid)]
        self.assertGreaterEqual(len(scored),G.PROTOCOL['heldout_min_score_points_per_frame'])
        points[scored[0]][2]=-4.
        K=G.camera_matrix(640,480)
        pose=(np.eye(3),np.zeros(3))
        observed={tid:G.project(xyz[None],pose,K)[0][0] for tid,xyz in points.items()}
        anchors=[tid for tid in points if not G.score_track(tid)]
        frames=[{'index':0,'split':'train'},
                {'index':1,'split':'heldout','timestamp_s':1.,'mask':np.ones((480,640),np.uint8)*255}]
        model={'points':points,'poses':{0:pose}}
        # Exact matching and an independently anchored pose isolate the evaluator's
        # depth accounting. The failed point is NEVER an anchor or pose input.
        with patch.object(G,'match_heldout',return_value=observed), \
             patch.object(G,'solve_pose',return_value=(pose,anchors)):
            rows,_,errors=G.evaluate_heldout(model,{},frames,{},K)
        self.assertEqual(len(rows),1)
        self.assertFalse(rows[0]['passes'])
        self.assertEqual(rows[0]['scored_behind_camera_points'],1)
        self.assertEqual(rows[0]['scored_nonfinite_error_points'],1)
        self.assertIsNone(rows[0]['score_median_px'])
        self.assertEqual(len(errors),len(scored))
        self.assertEqual(sum(not np.isfinite(e) for e in errors),1)


if __name__=='__main__':
    unittest.main()
