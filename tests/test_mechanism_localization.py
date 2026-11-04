"""Reversible automatic crop coordinates and explicit unknown seam behavior."""
import copy
import unittest
import mechanism_localization as L


class LocalizationTests(unittest.TestCase):
    def context(self):
        return {'proposal':{'path':'data/proposal.json','sha256':'a'*64,'bytes':1},'source_sha256':'b'*64,
                'interaction':{'sha256':'c'*64},'observations':[{'joint_id':'hinge','parent_id':'body','child_id':'lid',
                    'evidence_id':'frame_1','source_sample_index':4,'timestamp_s':1.,'source_width':540,'source_height':960,
                    'crop_width':201,'crop_height':101,'crop_box_xyxy_exclusive':[100,200,301,301],
                    'crop_to_source':{'x_offset':100,'y_offset':200,'scale_x':1,'scale_y':1}}]}
    def result(self):
        return {'localizations':[{'joint_id':'hinge','evidence_id':'frame_1','status':'visible_candidate',
            'endpoints_normalized':[[0,.5],[1,.5]],'uncertainty_normalized':.02,'confidence':'low',
            'rationale':'Visible attachment seam candidate.','limitations':['Unverified image localization.']}],
            'limitations':['No 3D joint inference.'],'physical_accuracy_verified':False}
    def test_exact_crop_translation_and_uncertainty_to_full_source_pixels(self):
        value=L.export_result(self.result(),self.context())
        h=value['joint_hints'][0]['hinge_hints'][0]
        self.assertEqual(h['endpoints_px'],[[100.,250.],[300.,250.]])
        self.assertEqual(h['uncertainty_px'],4.0200000000000005)
        self.assertEqual(h['source_sample_index'],4)
        self.assertFalse(value['physical_accuracy_verified'])
        self.assertFalse(value['manual_geometry_or_joint_coordinates_used'])
    def test_occluded_unknown_seam_has_no_implicit_endpoint_fallback(self):
        r=self.result();r['localizations'][0].update(status='unknown',endpoints_normalized=[])
        out=L.export_result(r,self.context())
        self.assertEqual(out['joint_hints'][0]['hinge_hints'],[])
        self.assertEqual(len(out['joint_hints'][0]['unknown_observations']),1)
        r['localizations'][0]['endpoints_normalized']=[[0,.5],[1,.5]]
        with self.assertRaisesRegex(ValueError,'guessed'):L.export_result(r,self.context())
    def test_wrong_joint_missing_frame_duplicate_and_degenerate_are_rejected(self):
        for mutate in [lambda r:r['localizations'][0].update(joint_id='other'),
                       lambda r:r['localizations'][0].update(evidence_id='other'),
                       lambda r:r['localizations'].append(copy.deepcopy(r['localizations'][0])),
                       lambda r:r.update(localizations=[]),
                       lambda r:r['localizations'][0].update(endpoints_normalized=[[0,.5],[0,.5]])]:
            r=self.result();mutate(r)
            with self.subTest(value=r),self.assertRaises(Exception):L.export_result(r,self.context())
    def test_no_3d_or_physical_fields_can_enter_model_contract(self):
        r=self.result();r['localizations'][0]['axis']=[1,0,0]
        with self.assertRaises(Exception):L.export_result(r,self.context())
        r=self.result();r['physical_accuracy_verified']=True
        with self.assertRaises(Exception):L.export_result(r,self.context())

if __name__=='__main__':unittest.main()
