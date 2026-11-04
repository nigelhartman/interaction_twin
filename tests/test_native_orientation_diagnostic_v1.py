"""Independent synthetic appearance checks; no source video/model inference."""
import copy,unittest
import numpy as np

import native_orientation_diagnostic_v1 as D


def textured_fixture():
    y,x=np.mgrid[:112,:144]
    values=75+43*((x//7+y//11)%2)+14*np.sin(x/4)+10*np.cos(y/6)
    values[25:42,30:53]+=25
    rgb=np.repeat(np.rint(values).clip(0,255).astype(np.uint8)[...,None],3,axis=2)
    mask=np.zeros((112,144),bool);mask[12:100,16:128]=True
    return rgb,mask


class NativeOrientationDiagnosticTests(unittest.TestCase):
    def test_exactly_24_distinct_proper_rotations_preserve_lengths_and_identity(self):
        rng=np.random.default_rng(740);vertices=rng.normal(size=(71,3))*[3.,1.6,.3]+[7.,-2.,11.]
        original=vertices.copy();rows=D.pca_rotations(vertices)
        self.assertEqual(len(rows),24);self.assertEqual(sum(r['id']=='identity'for r in rows),1)
        unique=set();v=np.array([.13,-.74,.39])
        for row in rows:
            rotation=np.array(row['rotation']);np.testing.assert_allclose(rotation.T@rotation,np.eye(3),atol=1e-10)
            self.assertAlmostEqual(np.linalg.det(rotation),1.,places=10)
            self.assertAlmostEqual(np.linalg.norm(rotation@v),np.linalg.norm(v),places=10)
            unique.add(tuple(np.round(rotation,9).ravel()))
        self.assertEqual(len(unique),24);np.testing.assert_array_equal(vertices,original)

    def test_centered_rigid_transform_keeps_exact_center_not_origin(self):
        center=np.array([3.,-5.,2.]);rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        transform=D.centered_transform(rotation,center)
        np.testing.assert_allclose(transform@np.r_[center,1.],np.r_[center,1.],atol=1e-12)
        delta=np.array([.4,.6,-.8]);np.testing.assert_allclose((transform@np.r_[center+delta,1.])[:3]-center,rotation@delta)
        np.testing.assert_array_equal(transform[3],[0,0,0,1])
        with self.assertRaisesRegex(ValueError,'proper rigid'):D.centered_transform(np.diag([-1.,1.,1.]),center)

    def test_textured_front_is_informative_while_plain_reverse_face_is_unknown(self):
        rgb,mask=textured_fixture();front=D.appearance_metrics(rgb,mask,rgb.copy(),mask)
        back=D.appearance_metrics(rgb,mask,np.full_like(rgb,100),mask)
        self.assertEqual(front['status'],'diagnostic_available');self.assertLess(front['objective'],1e-5)
        self.assertGreater(front['gray_ncc'],.999);self.assertGreater(front['gradient_ncc'],.999)
        self.assertEqual(back['status'],'unknown');self.assertIsNone(back['objective'])
        self.assertEqual(back['reason'],'insufficient_native_or_source_texture')

    def test_nonclipped_gain_and_offset_do_not_turn_same_texture_into_pose_error(self):
        rgb,mask=textured_fixture();shifted=np.rint(rgb.astype(float)*1.12+11).astype(np.uint8)
        metrics=D.appearance_metrics(rgb,mask,shifted,mask)
        self.assertEqual(metrics['status'],'diagnostic_available')
        self.assertGreater(metrics['gray_ncc'],.995);self.assertGreater(metrics['gradient_ncc'],.995)
        self.assertLess(metrics['objective'],.02)

    def test_constant_or_tiny_matching_foreground_cannot_win(self):
        rgb,mask=textured_fixture();tiny=np.zeros_like(mask);tiny[30:33,40:43]=True
        small=D.appearance_metrics(rgb,mask,rgb,tiny)
        self.assertEqual(small['status'],'unknown');self.assertIsNone(small['objective'])
        self.assertEqual(small['reason'],'insufficient_foreground_overlap')
        plain=np.full_like(rgb,100);constant=D.appearance_metrics(plain,mask,plain,mask)
        self.assertEqual(constant['status'],'unknown');self.assertIsNone(constant['objective'])
        ranked=D.choose_candidate([{'id':'identity','metrics':[constant]},{'id':'pca_other','metrics':[small]}])
        self.assertEqual(ranked['status'],'unknown');self.assertIsNone(ranked['selected_id'])

    def test_reversed_texture_is_distinguished_without_physical_pose_claim(self):
        rgb,mask=textured_fixture();inverse=(255-rgb).astype(np.uint8)
        correct=D.appearance_metrics(rgb,mask,rgb,mask)
        wrong=D.appearance_metrics(rgb,mask,inverse,mask)
        self.assertEqual(wrong['status'],'diagnostic_available');self.assertGreater(wrong['objective']-correct['objective'],.06)
        result=D.choose_candidate([{'id':'identity','metrics':[wrong]},{'id':'pca_front','metrics':[correct]}])
        self.assertEqual(result['selected_id'],'pca_front');self.assertEqual(result['status'],'appearance_preference_single_view')
        self.assertFalse(result['physical_pose_verified']);self.assertFalse(result['fit_or_scene_modified'])

    def test_equal_candidates_remain_ambiguous_and_baseline_can_be_retained(self):
        rgb,mask=textured_fixture();correct=D.appearance_metrics(rgb,mask,rgb,mask)
        tied=D.choose_candidate([{'id':'identity','metrics':[correct]},{'id':'pca_tie','metrics':[copy.deepcopy(correct)]}])
        self.assertEqual(tied['status'],'ambiguous');self.assertIsNone(tied['selected_id'])
        wrong=D.appearance_metrics(rgb,mask,255-rgb,mask)
        kept=D.choose_candidate([{'id':'identity','metrics':[correct]},{'id':'pca_wrong','metrics':[wrong]}])
        self.assertEqual(kept['status'],'identity_preferred');self.assertEqual(kept['selected_id'],'identity')

    def test_same_predeclared_frame_slots_required_missing_support_cannot_be_cherry_picked(self):
        rgb,mask=textured_fixture();correct=D.appearance_metrics(rgb,mask,rgb,mask)
        unknown=D.appearance_metrics(rgb,mask,np.zeros_like(rgb),mask)
        with self.assertRaisesRegex(ValueError,'Identical bounded frame'):D.choose_candidate([
            {'id':'identity','metrics':[correct,correct]},{'id':'pca_other','metrics':[correct]}])
        result=D.choose_candidate([{'id':'identity','metrics':[unknown,correct]},
                                   {'id':'pca_other','metrics':[correct,unknown]}])
        self.assertEqual(result['status'],'unknown');self.assertIsNone(result['selected_id'])

    def test_invalid_geometry_and_mismatched_raster_fail_explicitly(self):
        with self.assertRaises(ValueError):D.pca_rotations([[0,0,np.nan]]*4)
        rgb,mask=textured_fixture()
        with self.assertRaisesRegex(ValueError,'raster coordinates'):D.appearance_metrics(rgb,mask,rgb[:50],mask[:50])
        with self.assertRaisesRegex(ValueError,'byte RGB'):D.appearance_metrics(rgb.astype(float),mask,rgb.astype(float),mask)


if __name__=='__main__':unittest.main()
