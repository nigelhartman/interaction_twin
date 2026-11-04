import copy
import unittest

import cv2
import numpy as np

import source_temporal_recovery_v1 as T


class TemporalSourceTrackingTests(unittest.TestCase):
    def frames(self):
        rng=np.random.default_rng(93);image=np.full((200,240),35,np.uint8)
        patch=rng.integers(0,256,(120,160),np.uint8);patch=cv2.GaussianBlur(patch,(3,3),.5)
        image[40:160,40:200]=patch
        k=np.array([[210.,0.,120.],[0.,210.,100.],[0.,0.,1.]])
        foreground=np.zeros_like(image,bool);foreground[45:155,45:195]=True
        source=T.Frame(0.,image,{})
        matrix=np.eye(4);matrix[2,3]=1.
        corners=cv2.goodFeaturesToTrack(image,450,.003,3,mask=foreground.astype(np.uint8)*255)[:,0]
        points=(np.c_[corners,np.ones(len(corners))]@np.linalg.inv(k).T)-matrix[:3,3]
        seed={'status':'estimated','uv':corners,'native':points,'matrix':matrix,'reference_mask_area':int(foreground.sum()),
            'reference_coverage':.8,'reference_gray':image,'foreground':foreground,'rendered_depth':np.ones(image.shape),
            'plane':{'point':np.zeros(3),'normal':np.array([0.,0.,1.])}}
        targets=[T.Frame(i/6,cv2.warpAffine(image,np.float32([[1,0,3*i],[0,1,-2*i]]),(240,200)),{})for i in range(1,7)]
        return source,targets,seed,k

    def test_real_LK_on_synthetic_source_images_recovers_translation(self):
        source,targets,seed,k=self.frames();before=copy.deepcopy(seed['matrix'])
        rows=T.track_direction(targets,source,seed,k)
        for number,target in enumerate(targets,1):
            row=rows[round(target.timestamp_s,6)]
            self.assertEqual(row['status'],'estimated')
            fitted=np.asarray(row['camera_from_native_mesh'])
            self.assertLess(abs(fitted[0,3]-number*3/210),.004)
            self.assertLess(abs(fitted[1,3]+number*2/210),.004)
            self.assertGreaterEqual(row['PnP_inliers'],12)
        np.testing.assert_array_equal(seed['matrix'],before)

    def test_mutual_anchor_reidentification_recovers_without_a_target_pose_hint(self):
        source,targets,seed,k=self.frames();row=T.direct_source_features(seed,targets[-1],k)
        self.assertEqual(row['status'],'estimated')
        self.assertLess(abs(row['camera_from_native_mesh'][0][3]-18/210),.004)
        self.assertLess(row['median_reprojection_px'],1.1)

    def test_occlusion_and_new_texture_are_not_reacquired_as_native_motion(self):
        source,targets,seed,k=self.frames()
        hidden=T.Frame(.5,np.full(source.gray.shape,128,np.uint8),{})
        self.assertEqual(T.direct_source_features(seed,hidden,k)['status'],'withheld')
        rows=T.track_direction([hidden,targets[-1]],source,seed,k)
        self.assertEqual(rows[round(targets[-1].timestamp_s,6)]['status'],'withheld')

    def test_dense_photometric_alignment_uses_RGB_to_correct_an_imperfect_prior(self):
        source,targets,seed,k=self.frames();prior=seed['matrix'].copy();prior[0,3]=.065;prior[1,3]=-.045
        result=T.dense_source_alignment(seed,targets[-1],k,prior)
        self.assertEqual(result['status'],'estimated')
        self.assertGreaterEqual(result['dense_correlation'],.93)
        self.assertLess(result['median_warp_cycle_px'],1.25)
        self.assertFalse(result['PnP_grid_points_are_independent_feature_matches'])
        self.assertLess(abs(result['camera_from_native_mesh'][0][3]-18/210),.004)
        hidden=T.Frame(.5,np.full(source.gray.shape,128,np.uint8),{})
        self.assertEqual(T.dense_source_alignment(seed,hidden,k,prior)['status'],'withheld')

    def test_bounded_episode_detection_never_extrapolates_or_changes_unknowns(self):
        poses=[{'timestamp_s':float(i),'source_sample_index':i+1,'status':'estimated'if i in [1,4]else'unknown','camera_from_native_mesh':np.eye(4).tolist()if i in [1,4]else None}for i in range(6)]
        before=copy.deepcopy(poses);result=T.bounded_episodes(poses,2.)
        self.assertEqual(result[0]['original_unknown_indices'],[3,4]);self.assertFalse(result[0]['eligible'])
        self.assertEqual(poses,before)


if __name__=='__main__':unittest.main()
