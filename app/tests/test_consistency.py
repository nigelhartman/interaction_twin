import unittest
import numpy as np
from consistency import pixel_intrinsics,estimate_depth_scale,heldout_shape_error

class ConsistencyTests(unittest.TestCase):
    def test_normalized_pixel_centres_match_integer_image_convention(self):
        normalized=np.array([[1.,0,.5],[0,2.,.5],[0,0,1]])
        k=pixel_intrinsics(normalized,640,320)
        np.testing.assert_allclose(k,[[640,0,319.5],[0,640,159.5],[0,0,1]])

    def test_scale_is_fit_on_training_points_and_evaluated_on_disjoint_points(self):
        ref=np.random.default_rng(5).normal(size=(30,3));ref[:,2]+=5
        valid=np.ones(30,bool);train=np.arange(20);heldout=np.arange(20,30)
        current=ref*1.3
        scale=estimate_depth_scale(ref,current,valid,train)
        self.assertAlmostEqual(scale,1/1.3)
        self.assertLess(heldout_shape_error(ref,current*scale,valid,heldout),1e-12)
        current[heldout]*=1.2
        self.assertGreater(heldout_shape_error(ref,current*scale,valid,heldout),.19)
        self.assertAlmostEqual(estimate_depth_scale(ref,current,valid,train),scale)

if __name__=='__main__':unittest.main()
