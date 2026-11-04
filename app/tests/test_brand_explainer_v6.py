"""Refined-geometry and complete-goal task editorial boundaries."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('brand_explainer_v6',ROOT/'scripts/build_brand_explainer_20261103_v6.py')
B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)

class RefinedExplainerTests(unittest.TestCase):
    def test_45_second_story_has_four_second_json_and_ten_second_complete_task(self):
        self.assertEqual(B.CHAPTER_FRAMES,[0,48,120,216,336,432,576,672,768,1008,1080])
        self.assertEqual(B.CHAPTERS[6][:2],(24,28))
        self.assertEqual(B.CHAPTERS[8][:2],(32,42))
        self.assertEqual(B.CHAPTERS[-1][1],45)
        for i,(start,end,_,_)in enumerate(B.CHAPTERS):
            self.assertEqual(B.source_time(start),B.SOURCE_CHAPTERS[i][0])
            self.assertAlmostEqual(B.source_time(end),B.SOURCE_CHAPTERS[i][1])

    def test_whole_robot_source_is_shown_with_final_goal_hold(self):
        p={'_robot_timing':{'initial_hold_end_s':1,'final_hold_start_s':24.9,'duration_s':26.9}}
        self.assertEqual(B.robot_source_time(p,0),0)
        self.assertEqual(B.robot_source_time(p,.6),1)
        self.assertAlmostEqual(B.robot_source_time(p,8.6),24.9)
        self.assertAlmostEqual(B.robot_source_time(p,10),26.9)
        clocks=[B.robot_source_time(p,n/24)for n in range(241)]
        self.assertTrue(all(a<b for a,b in zip(clocks,clocks[1:])))
        self.assertTrue(all(B.robot_source_time(p,t)>=24.9 for t in [8.6,9,9.5,10]))
        self.assertGreater(B.robot_source_time(p,9.5),6) # No first-six-second truncation.
        for t in [-.01,10.01,float('nan')]:
            with self.assertRaises(ValueError):B.robot_source_time(p,t)

    def test_joint_excerpt_uses_refined_frames_and_retains_original_evidence(self):
        p={'_object_program':{'joints':[{'axis':[1,0,0],'evidence_ids':['source_000021']}]},
           '_refined_scene':{'mechanics':{'joints':[{'parent_id':'case_base','child_id':'case_lid','type':'revolute','status':'estimated','axis':[0,.5,.8660254],'parent_from_joint':[[1,0,0,.12],[0,1,0,-.3],[0,0,1,.04],[0,0,0,1]],
             'samples':[{'source_sample_index':21,'timestamp_s':6.667256,'q_rad':1.9}]}]}}}
        excerpt=json.loads('\n'.join(B.descriptive_lines(p)))
        self.assertEqual(excerpt['axis'],[0,.5,.87]);self.assertEqual(excerpt['q_rad'],1.9)
        self.assertEqual(excerpt['joint_origin_parent'],[.12,-.3,.04])
        self.assertEqual(excerpt['evidence_ids'],['source_000021'])
        self.assertIs(excerpt['physical_accuracy_verified'],False)
        p['_object_program']['joints'][0]['evidence_ids']=[]
        with self.assertRaisesRegex(ValueError,'Original source evidence'):B.descriptive_lines(p)

    def test_blank_margin_crop_keeps_the_complete_camera_envelope(self):
        p={'robot_goal_validation':{'camera':{'recorded_native_bbox_union':{'ndc_min':[-.4,-.86],'ndc_max':[.32,.7]}}}}
        left,top,right,bottom=B.robot_editorial_crop(p)
        for x in [-.4,.32]:
            for y in [-.86,.7]:
                self.assertGreaterEqual((x+1)*1920/2,left+35)
                self.assertLessEqual((x+1)*1920/2,right-35)
                self.assertGreaterEqual((1-y)*1080/2,top+35)
                self.assertLessEqual((1-y)*1080/2,bottom-35)
        self.assertGreater(960/(bottom-top),1) # Case does not shrink below raw render scale.

    def test_missing_refined_preparation_blocks_without_old_model_fallback(self):
        with tempfile.TemporaryDirectory()as tmp,patch.object(B,'OUT',Path(tmp)):
            with self.assertRaisesRegex(ValueError,'no fallback film'):B.load_v6_preparation()

    def test_incomplete_robot_receipt_is_not_used_as_a_complete_goal_capture(self):
        with self.assertRaisesRegex(ValueError,'exact completed robot capture'):
            B.verify_robot_capture({'format':'interaction-twin-complete-robot-browser-capture','version':2,'captured_full_task':False,'manifest':{}},{})
        with self.assertRaisesRegex(ValueError,'Complete bounded robot state trace'):
            B.verify_robot_capture({'format':'interaction-twin-complete-robot-browser-capture','version':3,'captured_full_task':True,'manifest':{},'trace':[]},{})

if __name__=='__main__':unittest.main()
