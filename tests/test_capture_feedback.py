"""CPU-only feedback proxies and exact source/plan bindings, no inference."""
import copy
import json
from pathlib import Path
import unittest

import numpy as np
from PIL import Image,ImageFilter

import capture_feedback as feedback
from tests.test_observed_jobs import JobFixture


class PixelMetrics(unittest.TestCase):
    def test_original_pixel_size_bounds_and_translation_are_measured_without_pose_claims(self):
        gray=np.arange(80*80,dtype=np.uint8).reshape(80,80)
        first=np.zeros((80,80),bool);first[20:40,15:45]=True
        second=np.zeros_like(first);second[25:45,25:55]=True
        row=feedback.mask_metrics(gray,first)
        self.assertEqual(row['mask_area_pixels'],600)
        self.assertEqual(row['bbox_xyxy'],[15,20,45,40])
        self.assertEqual(row['minimum_bbox_side_px'],20)
        self.assertFalse(row['border_contact'])
        self.assertAlmostEqual(feedback.translated_mask_iou(first,second),1.)
        self.assertIsNone(feedback.translated_mask_iou(first,np.zeros_like(first)))

    def test_blur_reduces_detail_proxy_but_textureless_surface_is_also_low(self):
        pattern=((np.indices((96,96))//4).sum(0)%2*255).astype(np.uint8)
        mask=np.zeros_like(pattern,dtype=bool);mask[8:-8,8:-8]=True
        sharp=feedback.mask_metrics(pattern,mask)['laplacian_variance']
        blurred=np.array(Image.fromarray(pattern).filter(ImageFilter.GaussianBlur(2)))
        soft=feedback.mask_metrics(blurred,mask)['laplacian_variance']
        flat=feedback.mask_metrics(np.full_like(pattern,128),mask)['laplacian_variance']
        self.assertLess(soft,sharp*.1);self.assertEqual(flat,0.)
        self.assertEqual(feedback.DEFINITIONS['low_detail_proxy']['semantics'],'measured_image_detail_proxy_not_confirmed_blur')
        self.assertFalse(feedback.DEFINITIONS['low_detail_proxy']['cause_verified'])

    def test_only_explicit_saved_occlusion_language_is_tagged_and_negation_is_honored(self):
        for text in ['fingers excluded from mask','no finger occlusion','without hand occlusion','teal covers exterior; fingers nearby']:
            self.assertFalse(feedback.model_hand_occlusion(text),text)
        for text in ['hand heavily occludes edge-on target','partially occluded by fingers','finger occlusion at the top']:
            self.assertTrue(feedback.model_hand_occlusion(text),text)
        self.assertIn('not_depth_order_or_occlusion',feedback.DEFINITIONS['overlapping_model_masks']['semantics'])


class FeedbackContracts(JobFixture,unittest.TestCase):
    def edit_program(self,callback):
        document=json.loads(self.source.read_text());callback(document)
        self.source.write_text(json.dumps(document))
        path=self.root/'workspace/cases.json';catalog=json.loads(path.read_text())
        catalog['cases'][0]['original_sha256']=feedback.sha(self.source);path.write_text(json.dumps(catalog))

    def build(self):
        return feedback.build(self.root,interaction=self.source.relative_to(self.root).as_posix())

    def test_missing_masks_are_not_physical_absence_and_reference_is_not_supported_motion(self):
        report=self.build();part=next(c for c in report['components'] if c['id']=='part')
        holder=next(c for c in report['components'] if c['id']=='holder')
        self.assertEqual(part['summary']['baseline']['supported_nonreference_samples'],1)
        self.assertFalse(holder['summary']['baseline']['attempted'])
        self.assertTrue(all('mask_unusable' in r['diagnostic_codes'] for r in holder['samples']))
        definition=report['diagnostic_definitions']['mask_unusable']
        self.assertIn('not_physical_absence',definition['semantics'])
        self.assertEqual(report['episodes'],[])
        self.assertFalse(report['policy']['used_for_reconstruction_selection_or_fitting'])
        self.assertEqual(report['new_model_inference_calls'],0)
        self.assertFalse(report['mps_used'])
        self.assertFalse(report['viewpoint_assessment']['status']=='established')

    def test_unregistered_changed_source_and_escaping_paths_are_rejected(self):
        for path in ['data/../../../private/file','/private/file','https://example.com/data/file','data/a?query','data/a\\b']:
            with self.subTest(path=path),self.assertRaises(ValueError):feedback.data_path(self.root,path)
        with self.assertRaises(ValueError):feedback.build(self.root,interaction='data/unregistered.json')
        source_image=self.root/json.loads(self.source.read_text())['evidence'][0]['frame_path']
        source_image.write_bytes(b'changed')
        with self.assertRaises(ValueError):self.build()

    def test_scopes_do_not_mark_outside_track_range_as_missing_and_masks_are_not_mutated(self):
        self.edit_program(lambda p:p['tracklets'].append(dict(p['tracklets'][0],id='short_scope',
            candidate_entity_ids=['short_scope_candidate'],source_sample_range=[2,2])))
        program=json.loads(self.source.read_text());paths=[self.root/o['mask_path'] for r in program['evidence'] for o in r['observations']]
        before=[p.read_bytes() for p in paths];report=self.build()
        part=next(c for c in report['components'] if c['id']=='short_scope')
        self.assertEqual([r['source_index'] for r in part['samples']],[2])
        self.assertEqual([p.read_bytes() for p in paths],before)

    def test_saved_model_nonrigidity_requires_a_cited_model_description(self):
        report=self.build();self.assertTrue(all('saved_model_nonrigidity' not in r['diagnostic_codes'] for c in report['components'] for r in c['samples']))
        self.edit_program(lambda p:p['tracklets'][0].update(limitations=['deformable, bending changes shape']))
        report=self.build();part=next(c for c in report['components'] if c['id']=='part')
        self.assertTrue(all('saved_model_nonrigidity' in r['diagnostic_codes'] for r in part['samples']))
        self.assertEqual(part['samples'][0]['model_description_quotes'],['deformable, bending changes shape'])
        self.assertFalse(report['physical_accuracy_verified'])

    def test_frozen_feedback_plan_attachment_checks_exact_frame_pts_and_hashes(self):
        report=self.build();path='data/capture-feedback-fixture-v1.json'
        entry=feedback.save(self.root,report,path);before=(self.root/path).read_bytes()
        part=next(c for c in report['components'] if c['id']=='part');row=part['samples'][1]
        plan={'format':'short-interval-motion-plan','interaction_path':report['interaction_path'],
            'interaction_sha256':report['interaction_sha256'],'source_video_sha256':report['source']['sha256'],
            'component_id':'part','episodes':[{'id':'episode-01','frames':[{'sample_index':row['source_index'],
                'timestamp_s':row['timestamp_s'],'source_image_sha256':row['image_sha256'],'mask_sha256':row['mask_sha256']}]}]}
        plan_path=self.root/'data/plan.json';plan_path.write_text(json.dumps(plan))
        derived=feedback.attach_episodes(self.root,path,['data/plan.json'])
        self.assertEqual(derived['version'],2);self.assertEqual(derived['episodes'][0]['episode_id'],'episode-01')
        self.assertEqual(derived['episodes'][0]['source_sample_ids'],[2])
        self.assertEqual((self.root/path).read_bytes(),before)
        self.assertEqual(feedback.sha(self.root/path),entry['sha256'])
        plan['episodes'][0]['frames'][0]['timestamp_s']+=.01;plan_path.write_text(json.dumps(plan))
        with self.assertRaises(ValueError):feedback.attach_episodes(self.root,path,['data/plan.json'])
        with self.assertRaises(ValueError):feedback.save(self.root,report,path)
        snapshot=self.root/read_json(self.root/path)['implementation']['path']
        self.assertEqual(feedback.sha(snapshot),read_json(self.root/path)['implementation']['sha256'])

    def test_scope_veto_is_separate_cited_judgment_and_changed_raw_judgment_is_rejected(self):
        report=self.build();base='data/capture-feedback-scope-v1.json';feedback.save(self.root,report,base)
        sample=next(c for c in report['components'] if c['id']=='part')['samples'][1]
        frame={'sample_index':2,'timestamp_s':sample['timestamp_s'],
               'source_image_sha256':sample['image_sha256'],'mask_sha256':sample['mask_sha256']}
        plan={'format':'short-interval-motion-plan','interaction_path':report['interaction_path'],
              'interaction_sha256':report['interaction_sha256'],'source_video_sha256':report['source']['sha256'],
              'component_id':'part','episodes':[{'id':'episode-01','reference_sample_index':2,'frames':[frame]}]}
        directory=self.root/'data/scope';directory.mkdir();plan_path=directory/'plan.json';plan_path.write_text(json.dumps(plan))
        judgment={'sample_index':2,'verdict':'plausible_visible_mask','component_boundary_plausible':True,
                  'single_component_visible':False,'other_components_inside_mask':False,
                  'notes':'Target mask plausible; another part is visible outside its mask.'}
        (directory/'raw.json').write_text(json.dumps({'observations':[judgment]}))
        request={'plan_path':'data/scope/plan.json','plan_sha256':feedback.sha(plan_path),
                 'interaction_sha256':report['interaction_sha256'],'track_id':'part','targets':[frame]}
        request_path=directory/'request.json';request_path.write_text(json.dumps(request))
        joined=dict(judgment,track_id='part',source_image_sha256=sample['image_sha256'],mask_sha256=sample['mask_sha256'])
        audit={'format':'short-interval-mask-scope-audit','manual_corrections':False,
               'interaction_sha256':report['interaction_sha256'],'source_video_sha256':report['source']['sha256'],
               'plan_sha256':feedback.sha(plan_path),'request_sha256':feedback.sha(request_path),
               'raw_response':'raw.json','result':{'observations':[joined]}}
        audit_path=directory/'audit.json';audit_path.write_text(json.dumps(audit))
        derived=feedback.attach_episodes(self.root,base,['data/scope/plan.json'],scope_audits=['data/scope/audit.json'])
        episode=derived['episodes'][0]
        self.assertEqual(episode['reference_scope_admission'],'withheld_by_saved_model_scope_gate')
        self.assertTrue(episode['supplemental_scope_observations'][0]['component_boundary_plausible'])
        self.assertFalse(episode['scope_review_action']['cause_verified'])
        self.assertIn('outside',episode['scope_review_action']['audit']['notes'])
        self.assertNotIn('saved_audit_rejected',next(c for c in derived['components'] if c['id']=='part')['samples'][1]['diagnostic_codes'])
        audit['result']['observations'][0]['single_component_visible']=True;audit_path.write_text(json.dumps(audit))
        with self.assertRaisesRegex(ValueError,'judgment was changed'):
            feedback.attach_episodes(self.root,base,['data/scope/plan.json'],scope_audits=['data/scope/audit.json'])


def read_json(path):return json.loads(Path(path).read_text())


if __name__=='__main__':unittest.main()
