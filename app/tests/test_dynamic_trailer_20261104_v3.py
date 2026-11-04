"""Source-clock, complete-task and crop guards for the third editorial version."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
import cv2

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('dynamic_film_v3',ROOT/'scripts/build_dynamic_trailer_20261104_v3.py')
F=importlib.util.module_from_spec(spec);spec.loader.exec_module(F)


class DynamicFilmV3Tests(unittest.TestCase):
    def test_actual_native_geometry_clears_new_caption_and_rejects_the_old_covered_region(self):
        path=ROOT/'output/motion-trailer-20261104-v5/interaction-twin-motion-20261104-v5.json'
        if not path.is_file():self.skipTest('Optional actual source-bound V5 master unavailable')
        receipt=json.loads(path.read_text());proof=F.hero_caption_clearance(receipt,F.HERO_CAPTION)
        self.assertEqual(proof['original_capture_hero_rows'],270);self.assertEqual(proof['selected_final_hero_frames'],240)
        self.assertEqual(proof['component_bounds_checks'],2040);self.assertEqual(proof['overlaps'],0);self.assertGreater(proof['minimum_separation_px'],0)
        with self.assertRaisesRegex(ValueError,'Hero caption overlaps'):F.hero_caption_clearance(receipt,{'text':'Old caption region, regression only','rect':[79,914,765,66]})

    def test_editorial_chapter_edges_stay_two_frames_inside_the_actual_capture(self):
        for a,b,sa,sb in F.V.MASTER_MAP:
            self.assertAlmostEqual(F.editorial_master_time(a),sa+2/30)
            samples=[F.editorial_master_time(a+(b-a)*n/120)for n in range(120)]
            self.assertEqual(samples,sorted(samples));self.assertGreaterEqual(samples[0],sa+2/30);self.assertLess(samples[-1],sb-2/30)
        self.assertAlmostEqual(F.editorial_master_time(8),9+2/30)
        with self.assertRaises(ValueError):F.editorial_master_time(29)

    def test_actual_final_frame_clock_uses_trimmed_capture_samples_and_original_trace(self):
        path=ROOT/'output/motion-trailer-20261104-v5/interaction-twin-motion-20261104-v5.json'
        if not path.is_file():self.skipTest('Optional actual source-bound V5 master unavailable')
        receipt=json.loads(path.read_text());rows=[F.master_sample_clock(n/30,1110,receipt)for n in range(870)]
        self.assertEqual(rows[240]['decoded_master_frame_index'],272)
        self.assertEqual(rows[240]['capture_chapter_index'],2)
        self.assertEqual(sum(r['original_draw_trace_index']is not None for r in rows),600)
        for row in rows:
            self.assertEqual(row['decoded_master_frame_index'],round(F.editorial_master_time(row['final_film_s'])*30))
            if row['original_draw_trace_index']is None:continue
            original=receipt['projection_trace'][row['original_draw_trace_index']]
            self.assertEqual(row['original_draw_source_s'],original['source_s'])
            self.assertLessEqual(row['draw_clock_offset_s'],1/30+.005)
            chapter=receipt['chapter_source_intervals'][row['capture_chapter_index']]
            a,b=chapter['film_s'];sa,sb=chapter['source_s']
            self.assertAlmostEqual(row['source_clock_at_decoded_master_time_s'],sa+(row['decoded_master_frame_time_s']-a)*(sb-sa)/(b-a))

    def test_real_previous_empty_first_inset_is_rejected_but_healthy_source_picture_passes(self):
        path=ROOT/'showcase/media/interaction-twin-dynamic-20261104-v2.mp4'
        if not path.is_file():self.skipTest('Optional previous project film unavailable')
        cap=cv2.VideoCapture(str(path))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES,240);ok,bad=cap.read();self.assertTrue(ok)
            with self.assertRaisesRegex(ValueError,'Source evidence picture is empty'):F.check_source_picture(Image.fromarray(cv2.cvtColor(bad,cv2.COLOR_BGR2RGB)),8.)
            cap.set(cv2.CAP_PROP_POS_FRAMES,360);ok,good=cap.read();self.assertTrue(ok)
            stats=F.check_source_picture(Image.fromarray(cv2.cvtColor(good,cv2.COLOR_BGR2RGB)),12.)
            self.assertGreater(stats['non_dark_fraction'],.7)
        finally:cap.release()

    def test_empty_black_or_white_source_panels_are_rejected(self):
        for color in ['black','white','#111823']:
            for time in [0,8,15.99]:
                with self.subTest(color=color,time=time),self.assertRaisesRegex(ValueError,'Source evidence picture is empty'):F.check_source_picture(Image.new('RGB',(1920,1080),color),time)
        self.assertIsNone(F.check_source_picture(Image.new('RGB',(1920,1080),'white'),20))

    def test_multiple_declared_source_intervals_do_not_admit_the_gap_or_missing_components(self):
        clip=ROOT/'output/motion-trailer-20261104-v5/interaction-twin-motion-20261104-v5.webm'
        source_receipt=clip.with_suffix('.json')
        if not clip.is_file()or not source_receipt.is_file():self.skipTest('Optional source-bound master capture unavailable')
        with tempfile.TemporaryDirectory()as folder:
            receipt=Path(folder)/'fixture.json'
            d=json.loads(source_receipt.read_text())
            receipt.write_text(json.dumps(d));self.assertEqual(F.verify_motion(clip,receipt)['version'],5)
            for change in [lambda x:x['projection_trace'][60].update(source_s=22),lambda x:x['projection_trace'][60]['parts'].pop(),lambda x:x['projection_trace'][60]['parts'][0].update(camera_from_native_mesh=None),lambda x:x.update(source_video_seek_fallback_policy='unverified_frame'),lambda x:x['projection_trace'][0].update(source_video_display=None),lambda x:x['projection_trace'][0]['source_video_display'].update(timestamp_s=0),lambda x:x['capture_runtime_source_refs'].pop(),lambda x:x.update(physical_accuracy_verified=True),lambda x:x.update(native_rgb_display_policy='srgb_converted_twice'),lambda x:x['chapter_source_intervals'][2].update(source_s=[17.35,20.3])]:
                bad=copy.deepcopy(d);change(bad);receipt.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):F.verify_motion(clip,receipt)

    def test_real_robot_closeup_begins_after_both_releases_and_preserves_both_projected_goals(self):
        p=ROOT/F.V.ROBOT_JSON['path']
        if not p.is_file():self.skipTest('Optional complete browser capture unavailable')
        receipt=F.verify_robot(p);full=F.V.robot_crop(receipt);changed=0
        for frame in range(301):
            elapsed=frame/30;box=F.robot_box(receipt,elapsed)
            if box!=full:
                changed+=1;row=min(receipt['trace'],key=lambda r:abs(r['time_s']-F.V.robot_time(elapsed)))
                self.assertTrue(all(g['released']and g['reached']for g in row['goals']))
            for goal in receipt['final_goal_canvas']:
                x,y=goal['canvas'][:2];self.assertLess(box[0],x);self.assertGreater(box[2],x);self.assertLess(box[1],y);self.assertGreater(box[3],y)
        self.assertGreater(changed,30)

    def test_real_robot_cannot_claim_completion_while_attached_unplaced_or_open(self):
        p=ROOT/F.V.ROBOT_JSON['path']
        if not p.is_file():self.skipTest('Optional complete browser capture unavailable')
        original=json.loads(p.read_text())
        with tempfile.TemporaryDirectory()as folder:
            target=Path(folder)/'receipt.json'
            for change in [lambda d:d.update(final_gripper_attached='earbud_a'),lambda d:d['final_goals'][0].update(released=False),lambda d:d['final_joint_q_rad'].update(joint=.2),lambda d:d.update(physical_contact_verified=True)]:
                bad=copy.deepcopy(original);change(bad);target.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):F.verify_robot(target)

    def test_physics_starts_with_the_whole_frame_before_an_editorial_push_near_contact(self):
        for elapsed in [0,.5,1,2]:self.assertEqual(F.newton_box(elapsed),(0,0,1920,1080))
        self.assertEqual(F.newton_box(4),(200,340,1750,1080))
        tops=[F.newton_box(n/30)[1]for n in range(121)]
        self.assertEqual(tops,sorted(tops))


if __name__=='__main__':unittest.main()
