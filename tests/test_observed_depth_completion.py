"""Synthetic checks for optional model-prior alignment; never run the model."""
import unittest

import numpy as np

import observed_depth_completion as D


class DepthCompletionTests(unittest.TestCase):
    def scene(self):
        yy,xx=np.indices((120,160))
        # Nonconstant image depth, with structure in both spatial FIT halves.
        inverse=.20+.0004*xx+.0002*yy+.008*np.sin(xx/11)*np.cos(yy/9)
        model=1/inverse
        observed=1/(.65*inverse+.025)
        domain=np.ones(model.shape,bool)
        return model,observed,domain

    def test_robust_affine_inverse_depth_recovery(self):
        model,observed,domain=self.scene()
        aligned,report=D.align_depth(model,observed,domain)
        self.assertTrue(report['passes'],report)
        self.assertAlmostEqual(report['fit_parameters']['a'],.65,places=10)
        self.assertAlmostEqual(report['fit_parameters']['b'],.025,places=10)
        np.testing.assert_allclose(aligned,observed,atol=1e-10)

    def test_check_pixels_do_not_change_fit_coefficients(self):
        model,observed,domain=self.scene()
        _,base=D.align_depth(model,observed,domain)
        yy,xx=np.indices(model.shape)
        changed=observed.copy()
        changed[D.check_split(xx,yy)]*=1.4
        _,corrupted=D.align_depth(model,changed,domain)
        self.assertEqual(base['fit_parameters'],corrupted['fit_parameters'])
        self.assertFalse(corrupted['passes'])
        self.assertIn('source_check_median_relative_depth_error',corrupted['reasons'])

    def test_invalid_check_predictions_remain_failed_evidence(self):
        model,observed,domain=self.scene()
        yy,xx=np.indices(model.shape)
        model[D.check_split(xx,yy)]=np.nan
        _,report=D.align_depth(model,observed,domain)
        self.assertFalse(report['passes'])
        self.assertEqual(report['invalid_check_predictions'],report['check_pixels'])
        self.assertIsNone(report['check_median_relative_depth_error'])
        self.assertIn('aligned_inverse_depth_not_positive',report['reasons'])

    def test_negative_inverse_depth_slope_rejected(self):
        model,observed,domain=self.scene()
        observed=1/(.5-.65/model)
        _,report=D.align_depth(model,observed,domain)
        self.assertFalse(report['passes'])
        self.assertIn('nonpositive_inverse_depth_slope',report['reasons'])

    def test_constant_model_prediction_rejected(self):
        model,observed,domain=self.scene()
        _,report=D.align_depth(np.full_like(model,4.),observed,domain)
        self.assertFalse(report['passes'])
        self.assertIn('insufficient_predictor_depth_variation',report['reasons'])

    def test_spatially_inconsistent_fit_rejected(self):
        model,observed,domain=self.scene()
        observed[:,80:]*=1.4
        _,report=D.align_depth(model,observed,domain)
        self.assertFalse(report['passes'])
        self.assertIn('spatial_fit_instability',report['reasons'])

    def test_insufficient_support_rejected(self):
        model,observed,domain=self.scene()
        domain[:]=False
        domain[10:20,10:20]=True
        aligned,report=D.align_depth(model,observed,domain)
        self.assertIsNone(aligned)
        self.assertFalse(report['passes'])

    def test_masked_geometry_has_model_face_provenance_and_no_bridge(self):
        shape=(18,22)
        depth=np.ones(shape)*4
        inferred=np.ones(shape,bool)
        domain=np.ones(shape,bool)
        domain[:,10:13]=False
        labels=np.where(domain,1,2)
        labels[:,13:]=3
        image=np.zeros((*shape,3),np.uint8)
        image[:]=[11,22,33]
        pose=(np.eye(3),np.zeros(3))
        K=np.array([[80.,0,11],[0,80.,9],[0,0,1.]])
        frame={'id':'source','image_url':'/files/data/test/frame.jpg'}
        mesh=D.visible_prior_mesh(depth,inferred,domain,labels,image,pose,K,frame)
        self.assertIsNotNone(mesh)
        uv=np.array(mesh['vertex_source_pixels'])
        for face in mesh['triangles']:
            pixels=uv[face]
            self.assertEqual(len(np.unique(labels[pixels[:,1],pixels[:,0]])),1)
            self.assertTrue(domain[pixels[:,1],pixels[:,0]].all())
        self.assertEqual(set(mesh['face_provenance']),{'model_prior_aligned_inverse_depth'})
        self.assertEqual(set(mesh['face_confidence']),{'unknown'})
        self.assertFalse(mesh['hidden_surface_completion'])
        self.assertFalse(mesh['watertight'])
        self.assertEqual(set(map(tuple,mesh['colors'])),{(33,22,11)})

    def test_mesh_rejects_depth_discontinuity(self):
        shape=(8,8)
        depth=np.ones(shape)*4
        depth[:,4:]=8
        domain=np.ones(shape,bool)
        mesh=D.visible_prior_mesh(depth,domain,domain,np.ones(shape,int),np.zeros((*shape,3),np.uint8),
                                 (np.eye(3),np.zeros(3)),np.array([[80.,0,4],[0,80.,4],[0,0,1.]]),
                                 {'id':'source','image_url':'/files/data/test/frame.jpg'})
        pixels=np.array(mesh['vertex_source_pixels'])
        for face in mesh['triangles']:
            depths=depth[pixels[face,1],pixels[face,0]]
            self.assertEqual(float(depths.max()),float(depths.min()))


if __name__=='__main__':
    unittest.main()
