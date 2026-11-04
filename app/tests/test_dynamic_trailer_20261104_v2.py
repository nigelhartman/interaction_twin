"""Source-clock, complete-task and crop guards for the second editorial version."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('dynamic_film_v2',ROOT/'scripts/build_dynamic_trailer_20261104_v2.py')
F=importlib.util.module_from_spec(spec);spec.loader.exec_module(F)


class DynamicFilmV2Tests(unittest.TestCase):
    def test_multiple_declared_source_intervals_do_not_admit_the_gap_or_missing_components(self):
        clip=ROOT/'output/motion-trailer-20261104-v3/interaction-twin-motion-20261104-v3.webm'
        source_receipt=clip.with_suffix('.json')
        if not clip.is_file()or not source_receipt.is_file():self.skipTest('Optional source-bound master capture unavailable')
        with tempfile.TemporaryDirectory()as folder:
            receipt=Path(folder)/'fixture.json'
            d=json.loads(source_receipt.read_text())
            receipt.write_text(json.dumps(d));self.assertEqual(F.verify_motion(clip,receipt)['version'],3)
            for change in [lambda x:x['projection_trace'][60].update(source_s=22),lambda x:x['projection_trace'][60]['parts'].pop(),lambda x:x['projection_trace'][60]['parts'][0].update(camera_from_native_mesh=None),lambda x:x.update(physical_accuracy_verified=True),lambda x:x.update(native_rgb_display_policy='srgb_converted_twice'),lambda x:x['chapter_source_intervals'][2].update(source_s=[17.35,20.3])]:
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
