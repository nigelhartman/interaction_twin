"""Mask overlap cannot promote unaudited masks or invent seating evidence."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('approach_evidence',ROOT/'scripts/stage4_approach_evidence.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class ApproachEvidenceTests(unittest.TestCase):
    def test_gap_and_overlap_use_actual_mask_pixels(self):
        a=np.zeros((12,12),bool);b=a.copy();a[2:4,2:4]=True;b[2:4,7:9]=True
        metrics=module.mask_metrics(a,b,4)
        self.assertAlmostEqual(metrics['minimum_pixel_center_gap_over_base_bbox_diagonal'],4/np.hypot(2,2))
        self.assertEqual(metrics['earbud_mask_fraction_overlapping_base'],0)
        metrics=module.mask_metrics(a,a,4)
        self.assertEqual(metrics['minimum_pixel_center_gap_over_base_bbox_diagonal'],0)
        self.assertEqual(metrics['earbud_mask_fraction_overlapping_base'],1)
        self.assertNotIn('successful_seating',metrics)

    def test_area_ratio_is_raw_count_and_may_exceed_one(self):
        a=np.ones((5,5),bool)
        self.assertEqual(module.mask_metrics(a,a,5)['raw_mask_area_over_reference'],5)
        empty=module.mask_metrics(np.zeros_like(a),a,5)
        self.assertIsNone(empty['minimum_pixel_center_gap_over_base_bbox_diagonal'])
        with self.assertRaises(ValueError):module.mask_metrics(a,a,0)

    def test_unassessed_uncertain_or_failed_pair_never_reliable(self):
        good='plausible_visible_mask'
        self.assertEqual(module.cue_state(good,good,True,True),'audited_plausible_sampled_2d_cue')
        for verdict in ['not_model_audited','uncertain']:
            self.assertEqual(module.cue_state(good,verdict,True,True),'unassessed_or_uncertain_pair_diagnostic_only')
        self.assertEqual(module.cue_state(good,'reject_wrong_or_lost_mask',True,True),'known_bad_mask_pair_excluded')
        self.assertEqual(module.cue_state(good,good,True,False),'geometric_mask_gate_failed')

    def test_svg_has_independent_clickable_samples_and_no_path_interpolation(self):
        a=np.ones((2,2),bool);metrics=module.mask_metrics(a,a,4)
        rows=[{'sample_index':1,'timestamp_s':1.,'cue_status':'audited_plausible_sampled_2d_cue','metrics':metrics},
              {'sample_index':2,'timestamp_s':2.,'cue_status':'unassessed_or_uncertain_pair_diagnostic_only','metrics':metrics}]
        svg=module.plot_svg(rows)
        self.assertIn('data-sample="1"',svg);self.assertIn('data-sample="2"',svg)
        self.assertEqual(svg.count('<circle '),8)
        self.assertNotIn('<polyline',svg)

    def test_new_positive_never_clears_prior_rejection_uncertainty_or_scope_veto(self):
        good='plausible_visible_mask'
        self.assertEqual(module.merge_verdict('not_model_audited',good),good)
        self.assertEqual(module.merge_verdict('reject_wrong_or_lost_mask',good),'reject_wrong_or_lost_mask')
        self.assertEqual(module.merge_verdict('uncertain',good),'uncertain')
        self.assertEqual(module.merge_verdict(good,good,False),'reject_wrong_or_lost_mask')
        with self.assertRaises(ValueError):module.merge_verdict('invented',good)

    def test_fresh_audit_cannot_transfer_between_different_image_or_mask_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            case=Path(tmp);segmentation=case/'seg';segmentation.mkdir()
            image=case/'frame.png';mask=segmentation/'mask.png'
            Image.new('RGB',(3,3),'red').save(image);Image.new('L',(3,3),255).save(mask)
            tracks_path=segmentation/'tracks.json';module.write(tracks_path,{'fixture':'exact track-index hash'})
            audit=case/'audit.json';evidence=case/'evidence.json'
            observation={'track_id':'part','sample_index':1,'verdict':'plausible_visible_mask',
                         'single_component_visible':True,'component_boundary_plausible':True,
                         'other_components_inside_mask':False,'reason':'Visible pixels.'}
            module.write(audit,{'source_sha256':'video','source_track_index_sha256':module.sha(tracks_path),
                                'result':{'observations':[observation]}})
            for changed in ['image_sha256','mask_sha256']:
                entry={'track_id':'part','sample_index':1,'timestamp_s':1.,
                       'image_sha256':module.sha(image),'mask_sha256':module.sha(mask)}
                entry[changed]='different pixels';module.write(evidence,[entry]);verdict={}
                with self.assertRaises(ValueError):
                    module.merge_exact_audits([audit],'video',tracks_path,{'part':{1:{'raw_mask_file':'mask.png'}}},
                                              {1:{'file':'frame.png','timestamp_s':1.}},case,segmentation,verdict)
                self.assertEqual(verdict,{})


if __name__=='__main__':
    unittest.main()
