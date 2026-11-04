"""Editorial invariants: full robot outcome, honest capture gate, fixed45s clock."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('dynamic_trailer',ROOT/'scripts/build_dynamic_trailer_20261104_v1.py')
T=importlib.util.module_from_spec(spec);spec.loader.exec_module(T)

class DynamicTrailerTests(unittest.TestCase):
    def test_editorial_clock_covers_every_whole_master_chapter_in_order(self):
        self.assertEqual(T.MASTER_MAP[0][:2],(0,4))
        self.assertEqual(T.MASTER_MAP[-1][1],29)
        samples=[T.master_time(n/30)for n in range(29*30)]
        self.assertTrue(all(b>a for a,b in zip(samples,samples[1:])))
        for a,b,sa,sb in T.MASTER_MAP:
            self.assertAlmostEqual(T.master_time(a),sa)
            self.assertGreater(T.master_time(b-1/30),sb-.06)
        for invalid in [-.1,29,float('nan')]:
            with self.assertRaises(ValueError):T.master_time(invalid)
    def test_complete_robot_chronology_retains_both_releases_and_final_hold(self):
        self.assertAlmostEqual(T.robot_time(0),0)
        self.assertAlmostEqual(T.robot_time(.6),1)
        self.assertAlmostEqual(T.robot_time(8.6),24.9)
        self.assertAlmostEqual(T.robot_time(10),26.9)
        times=[T.robot_time(n/30)for n in range(301)]
        self.assertTrue(all(b>a for a,b in zip(times,times[1:])))
        self.assertEqual(sum(T.robot_time(n/30)>=24.9 for n in range(300)),42)
    def test_actual_robot_crop_keeps_both_goal_positions(self):
        p=ROOT/T.ROBOT_JSON['path']
        if not p.exists():self.skipTest('Optional browser capture unavailable in code-only snapshot')
        d=json.loads(p.read_text());box=T.robot_crop(d)
        for g in d['final_goal_canvas']:
            x,y=g['canvas'][:2];self.assertLess(box[0],x);self.assertGreater(box[2],x);self.assertLess(box[1],y);self.assertGreater(box[3],y)
        d['final_goal_canvas'][0]['canvas'][0]=0
        with self.assertRaisesRegex(ValueError,'hides'):T.robot_crop(d)
    def test_repaired_motion_capture_is_required_with_no_silent_draft_fallback(self):
        with tempfile.TemporaryDirectory()as tmp:
            clip=Path(tmp)/'capture.webm';clip.write_bytes(b'fixture')
            receipt=Path(tmp)/'receipt.json'
            receipt.write_text(json.dumps({'format':'interaction-twin-dynamic-film-capture','version':1}))
            with self.assertRaisesRegex(ValueError,'version2'):T.verify_motion(clip,receipt)
            trace=[{'film_s':i/30,'source_s':17.35+i*.005,'parts':[{'id':p,'camera_from_native_mesh':np.eye(4).tolist()}for p in ['case_base','case_lid','earbud_a','earbud_b']]}for i in range(101)]
            d={'format':'interaction-twin-dynamic-film-capture','version':2,'canvas':[1920,1080],'fps':30,'duration_s':37,'bytes':7,'physical_accuracy_verified':False,'new_api_calls':0,'camera_microphone_capture':False,'inputs':[],'fully_supported_source_interval_s':[17.35,20.3],'projection_trace':trace}
            receipt.write_text(json.dumps(d));self.assertEqual(T.verify_motion(clip,receipt)['version'],2)
            d['physical_accuracy_verified']=True;receipt.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,'claims'):T.verify_motion(clip,receipt)
            d['physical_accuracy_verified']=False;d['projection_trace'][20]['parts'][1]['camera_from_native_mesh']=None;receipt.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,'withheld'):T.verify_motion(clip,receipt)
    def test_end_card_uses_approved_logo_and_contains_real_animation_frames(self):
        from PIL import Image
        p=ROOT/T.LOGO_REF['path']
        if not p.exists():self.skipTest('Optional brand image unavailable')
        logo=Image.open(p).convert('RGBA')
        a=np.asarray(T.end_frame(0,logo));b=np.asarray(T.end_frame(1,logo))
        self.assertEqual(a.shape,(1080,1920,3))
        self.assertGreater(np.count_nonzero(a!=b),20000)
