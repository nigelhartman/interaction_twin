import unittest
import itertools
import numpy as np
import cv2

import occluded_hinge_recovery_v1 as R
import sam3d_mechanisms as M


class ClosedPanelRecoveryTests(unittest.TestCase):
    def test_closure_selects_folded_slab_without_changing_joint(self):
        vertices = np.array(list(itertools.product([-1.,1.], [0.,.025], [0.,1.5])))
        child = vertices.copy()
        a = np.eye(4); b = np.eye(4)
        # Opposite halves at q0 form a wide flat object; folding by pi places
        # both parts on one footprint. The search must reject the wide pose.
        b[:3,:3] = M.axis_rotation([1.,0.,0.], np.pi)
        joint = {'parent_from_joint': a.tolist(), 'joint_from_child_zero': b.tolist(), 'axis': [1.,0.,0.]}
        original = repr(joint)
        row = R.closure_geometry(vertices, child, joint)
        self.assertAlmostEqual(abs(row['angle_rad']), np.pi, places=4)
        self.assertLess(row['footprint_ratio'], 1.1)
        self.assertLess(row['thickness_native'], .06)
        self.assertEqual(repr(joint), original)
        self.assertFalse(row['physical_closed_angle_verified'])

    def test_no_hidden_part_from_missing_foreground_features(self):
        rgb = np.full((180,240,3), 90, np.uint8)
        mask = np.ones((180,240), bool)
        first = R.template_features(rgb,mask)
        result = R.template_support(first,first)
        self.assertFalse(result['accepted'])
        direct = R.template_support(first,first,same_frame=True)
        self.assertTrue(direct['accepted'])
        self.assertEqual(direct['method'],'direct_accounted_Muse_source_state')

    def test_panel_pnp_uses_mask_and_preserves_exterior_facing(self):
        vertices = np.array(list(itertools.product([-1.,1.], [-.6,.6], [-.01,.01])))
        k = np.array([[200.,0.,160.],[0.,200.,120.],[0.,0.,1.]])
        rotation = M.axis_rotation([1.,0.,0.], .4)
        translation = np.array([.1,.1,4.])
        # Native +z is interior; this pose presents exterior (-z) to camera.
        camera = vertices @ rotation.T+translation
        uv = camera @ k.T; uv = uv[:,:2]/uv[:,2:]
        mask = np.zeros((240,320),np.uint8)
        cv2.fillConvexPoly(mask,cv2.convexHull(np.rint(uv).astype(np.int32)),1)
        prior = np.eye(4); prior[:3,:3]=rotation;prior[:3,3]=translation
        result=R.panel_pose(vertices,mask>0,{'K':k.tolist()},1.,prior,[0.,0.,1.])
        self.assertLess(result['interior_facing_cosine'],-.5)
        self.assertLess(result['quad_reprojection_px'],3.)
        M.rotation_pose(result['camera_from_native_mesh'])


if __name__=='__main__':unittest.main()
