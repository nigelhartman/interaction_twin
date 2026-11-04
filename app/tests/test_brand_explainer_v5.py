"""The45-second editorial cut preserves evidence and removes requested clutter."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('brand_explainer_v5',ROOT/'scripts/build_brand_explainer_20261103_v5.py')
B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)

class CleanExplainerTests(unittest.TestCase):
    def test_exact_45_second_frame_bound_chapters_preserve_full_progress(self):
        self.assertEqual(B.DURATION*B.FPS,1080)
        self.assertEqual(B.CHAPTER_FRAMES,[0,59,157,275,412,530,687,805,903,1021,1080])
        for i,(start,end,_,_)in enumerate(B.CHAPTERS):
            self.assertEqual(B.source_time(start),B.SOURCE_CHAPTERS[i][0])
            self.assertAlmostEqual(B.source_time(end),B.SOURCE_CHAPTERS[i][1])
        previous=-1
        for frame in range(1080):
            t=frame/24;old_t=B.source_time(t)
            self.assertGreater(old_t,previous);previous=old_t
            self.assertAlmostEqual(B.display_time(old_t),t,places=10)
        self.assertEqual(B.source_time(45),55)
        for invalid in [-1,45.01,float('nan')]:
            with self.assertRaises(ValueError):B.source_time(invalid)

    def test_all_chapters_are_drawn_without_footer_rules_notes_or_removed_copy(self):
        blank=Image.new('RGB',(1920,1080),B.BG)
        prep={'masks':[{'timestamp_s':16,'mask_overlay':{'path':'unused'}}],
              '_framing':{'independent_native':{part:[0,0,1920,1080]for part in B.PARTS},'robot_canvas_crop':[0,0,1920,1080]},
              'clips':{'robot':{}}}
        texts=[];lines=[]
        def record_text(im,xy,value,*args,**kwargs):texts.append(value)
        def record_line(im,points,*args,**kwargs):lines.append(points)
        with patch.object(B,'mark'),patch.object(B,'source_frame',return_value=blank),patch.object(B,'native_frame',return_value=blank),patch.object(B,'connected_picture',return_value=blank),patch.object(B,'image',return_value=blank),patch.object(B,'clip_frame',return_value=blank),patch.object(B,'descriptive_lines',return_value=['{"physical_accuracy_verified": false}']),patch.object(B,'txt',side_effect=record_text),patch.object(B,'line',side_effect=record_line):
            for start,end,_,_ in B.CHAPTERS:
                frame=B.film_frame(prep,(start+end)/2)
                self.assertEqual(frame.size,(1920,1080))
                # This is generated blank background, not an external mask or
                # crop applied over a previously encoded55-second movie.
                self.assertTrue(np.all(np.asarray(frame)[1005:]==np.array([247,248,251])))
        self.assertNotIn('A real demonstration → a reusable interaction',texts)
        self.assertNotIn('Designed and built by Nigel Hartman',texts)
        self.assertNotIn([(72,1006),(1848,1006)],lines)
        self.assertIn('An authored robot rehearsal, built with these parts.',texts)
        self.assertIn('Estimated joint motion, linked to the recording.',texts)

    def test_source_master_and_mutation_boundary_are_explicit(self):
        self.assertEqual(B.SOURCE_MASTER['path'],'output/brand-trailer-20261103/renders/v4/verification.json')
        self.assertEqual(B.SOURCE_MASTER['sha256'],'8aacdfb00a5a02376ec55d46501e562bfabdd78ca2d6b90a0c2365bf6beeca3f')
        code=(ROOT/'scripts/build_brand_explainer_20261103_v5.py').read_text()
        self.assertNotIn('foot(im,',code)
        self.assertIn('afade=t=out:st=43:d=2',code)
        self.assertNotIn('atempo=',code)

if __name__=='__main__':unittest.main()
