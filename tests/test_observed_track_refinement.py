import unittest
from unittest.mock import patch
import cv2
import numpy as np
from observed_track_refinement import refine_pair


class CPUTrackRefinementTests(unittest.TestCase):
    def scene(self):
        rng=np.random.default_rng(391)
        a=rng.integers(0,256,(120,160,3),dtype=np.uint8)
        a=cv2.GaussianBlur(a,(3,3),.6)
        b=cv2.warpAffine(a,np.array([[1,0,5],[0,1,-3]],float),(160,120))
        mask=np.zeros((120,160),np.uint8);mask[10:-10,10:-10]=255
        return {'image':a,'mask':mask},{'image':b,'mask':mask.copy()}

    def test_image_flow_refines_coarse_initialization(self):
        a,b=self.scene();seeds=np.array([[50,50],[80,70],[100,45]],np.float32)
        uv,accepted,cycle,score=refine_pair(a,b,seeds,seeds+[3,-1])
        self.assertTrue(accepted.all())
        np.testing.assert_allclose(uv,seeds+[5,-3],atol=.1)
        self.assertTrue((cycle<.1).all())
        self.assertTrue((score>.9).all())

    def test_no_target_mask_support_no_correspondence(self):
        a,b=self.scene();b['mask'][:]=0
        _,accepted,_,_=refine_pair(a,b,[[50,50]],[[55,47]])
        self.assertFalse(accepted.any())

    def test_invalid_or_untextured_input_cannot_be_geometry(self):
        a,b=self.scene();a['image'][:]=100;b['image'][:]=100
        _,accepted,_,_=refine_pair(a,b,[[50,50]],[[55,47]])
        self.assertFalse(accepted.any())

    def test_failed_lk_returns_no_correspondence(self):
        a,b=self.scene()
        with patch('observed_track_refinement.cv2.calcOpticalFlowPyrLK',return_value=(None,None,None)):
            uv,accepted,cycle,score=refine_pair(a,b,[[50,50]],[[55,47]])
        self.assertFalse(accepted.any())
        self.assertTrue(np.isnan(uv).all())
        _,accepted,_,_=refine_pair(a,b,[[np.nan,50]],[[55,47]])
        self.assertFalse(accepted.any())


if __name__=='__main__':unittest.main()
