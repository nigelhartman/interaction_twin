"""Current visible film contract; older films remain independent evidence."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import unittest

ROOT=Path(__file__).resolve().parents[1]
BASE='media/flagship-20261104-v1/cut-v2/'
VIDEO=BASE+'interaction-twin-flagship-45s.mp4'


class Tags(HTMLParser):
    def __init__(self,text):
        super().__init__();self.tags=[];self.feed(text)
    def handle_starttag(self,tag,attrs):self.tags.append((tag,dict(attrs)))


class FlagshipWebsiteTests(unittest.TestCase):
    def test_both_visible_entry_points_share_movie_caption_and_poster(self):
        for name in ['index.html','film.html']:
            with self.subTest(page=name):
                text=(ROOT/'showcase'/name).read_text();tags=Tags(text).tags
                self.assertIn(('source',{'src':VIDEO,'type':'video/mp4'}),tags)
                self.assertTrue(any(tag=='video' and a.get('poster')==BASE+'poster.jpg' for tag,a in tags))
                self.assertTrue(any(tag=='track' and a.get('src')==BASE+'captions.vtt' and a.get('kind')=='captions'for tag,a in tags))
                self.assertNotIn('43-second',text);self.assertNotIn('film · 43s',text)
                self.assertNotIn('01-origin.mp4',text)

    def test_chapters_match_new_pipeline_robot_and_agent_timing(self):
        tags=Tags((ROOT/'showcase/film.html').read_text()).tags
        times=[float(a['data-film-time'])for tag,a in tags if 'data-film-time'in a]
        self.assertEqual(times,[0,3,6,9,12,27,35,39])
        self.assertTrue(all(a<b for a,b in zip(times,times[1:])) and times[-1]<45)

    def test_footage_notes_keep_partial_surface_draft_and_kinematic_scope(self):
        text=(ROOT/'showcase/film.html').read_text()
        for phrase in ['partial source-visible surface','estimated depth','authored kinematic preview',
                       'release both earbuds','actual saved MCP task draft','Owner-described USB-C',
                       'not executed or verified']:
            self.assertIn(phrase,text)

    def test_gallery_retains_six_alternates_and_links_current_flagship(self):
        text=(ROOT/'showcase/trailers.html').read_text();js=(ROOT/'showcase/trailer-collection.js').read_text()
        self.assertIn('Watch the 45-second flagship',text)
        self.assertEqual(len(re.findall(r"\{id:'[^']+', duration_s:",js)),6)
        self.assertIn("stem:'01-origin'",js)
        self.assertIn("const BASE = 'media/trailer-collection-20261104-v1/';",js)

    def test_no_private_provenance_or_new_raw_text_route_is_linked(self):
        for name in ['index.html','film.html','trailers.html']:
            for tag,attrs in Tags((ROOT/'showcase'/name).read_text()).tags:
                for field in ['href','src']:
                    value=attrs.get(field,'')
                    self.assertNotIn('/output/',value)
                    self.assertNotIn('.secrets',value)
                    self.assertNotIn('flagship-20261104-v1/cut-v2/provenance',value)
                    self.assertNotIn('flagship-20261104-v1/cut-v2/transcript.txt',value)


@unittest.skipUnless((ROOT/'showcase'/VIDEO).is_file(),'Optional saved movie absent in source-only workspace.')
class FlagshipSavedMediaTests(unittest.TestCase):
    def test_exact_final_movie_and_sound_are_bound_by_preserved_provenance(self):
        media=ROOT/'showcase'/BASE;doc=json.loads((media/'provenance.json').read_text())
        for key in ['movie','poster','captions','transcript']:
            ref=doc[key];raw=(ROOT/ref['path']).read_bytes()
            self.assertEqual(len(raw),ref['bytes']);self.assertEqual(hashlib.sha256(raw).hexdigest(),ref['sha256'])
        self.assertEqual(doc['movie']['sha256'],'dc4a241293afcd8d50c723db402bb432c8aa99b98de85e3e0331e6dff23c09a9')
        self.assertEqual(doc['duration_s'],45);self.assertEqual(doc['fps'],30)
        video=next(s for s in doc['probe']['streams']if s['codec_type']=='video')
        audio=next(s for s in doc['probe']['streams']if s['codec_type']=='audio')
        self.assertEqual((video['width'],video['height'],video['nb_read_frames']),(1920,1080,'1350'))
        self.assertEqual((audio['codec_name'],audio['sample_rate'],audio['channels']),('aac','48000',2))
        self.assertFalse(doc['camera_recording_included']);self.assertFalse(doc['physical_accuracy_verified'])
        self.assertEqual(doc['new_API_calls'],0);self.assertEqual(doc['new_GPU_calls'],0)
        self.assertTrue(doc['editorial_robot_pacing']['all_actions_retained_in_original_order'])
        self.assertFalse(doc['editorial_robot_pacing']['new_pose_interpolation'])
        self.assertTrue(all(row['decoded_video']['decoded_pixels_passed']for row in doc['capture_revalidation'].values()))


if __name__=='__main__':unittest.main()
