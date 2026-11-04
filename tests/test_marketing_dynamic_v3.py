"""Actual current-film evidence, complete history and bounded new-campaign selection."""
import json
from pathlib import Path
import unittest

import marketing_sources as M
from marketing_studio import default_storyboard,social_copy,validate_storyboard

ROOT=Path(__file__).resolve().parents[1]
SOURCE='pipeline_dynamic_45s_v3'
AVAILABLE=(ROOT/'showcase/media/interaction-twin-dynamic-20261104-v3.mp4').is_file()


class DynamicFilmV3MarketingTests(unittest.TestCase):
    @unittest.skipUnless(AVAILABLE,'Optional approved V3 project film unavailable')
    def test_actual_source_binds_exact_movie_and_version8_producer(self):
        source=M.MarketingSources().get(SOURCE)
        self.assertEqual(source['file'],{'path':'showcase/media/interaction-twin-dynamic-20261104-v3.mp4','sha256':'6b590c0aa40550208b475f405979696dfb3e865486ad184ec1c8fb401826b404','bytes':5433134})
        self.assertEqual(source['chapters_document_version'],8)
        producer=json.loads(M.checked(source['chapters_document']).read_bytes())
        self.assertEqual(producer['movie_ref'],source['file']);self.assertEqual(producer['duration_s'],45)
        self.assertEqual([(c['start_s'],c['end_s'])for c in source['chapters']],[(c['start_s'],c['end_s'])for c in producer['chapters']])
        self.assertEqual(source['pack_ids'],[M.JABRA,M.WALLET,M.CHARGER,M.LAPTOP])

    @unittest.skipUnless((ROOT/'output/presentation-polish-20261104-v3/historical-marketing-source-pins-v1.json').is_file(),'Optional historical project source snapshots unavailable')
    def test_all_eight_older_declared_and_resolved_sources_remain_exact(self):
        pins=json.loads((ROOT/'output/presentation-polish-20261104-v3/historical-marketing-source-pins-v1.json').read_text());self.assertEqual(len(pins),8)
        for identifier,expected in pins.items():
            self.assertEqual(M.digest(M.DECLARED[identifier]),expected['declared_digest'])
            self.assertEqual(M.digest(M.MarketingSources().get(identifier)),expected['resolved_digest'])

    @unittest.skipUnless(AVAILABLE,'Optional approved V3 project film unavailable')
    def test_v3_preference_requires_selection_and_never_cuts_outside_its_evidence(self):
        store=M.MarketingSources();new=store.get(SOURCE);older=store.get('pipeline_dynamic_45s_v2')
        self.assertEqual({r['source_id']for r in default_storyboard({older['id']:older},'pipeline',30)},{older['id']})
        for duration in [15,30]:
            for prefs in [{},{'pace':'fast'},{'pace':'slow'},{'focus':'robotics'},{'focus':'agents'},{'focus':'library','order':'result_first'}]:
                sources={SOURCE:new,older['id']:older};rows=default_storyboard(sources,'pipeline',duration,prefs)
                self.assertEqual({r['source_id']for r in rows},{SOURCE});self.assertEqual(validate_storyboard(rows,sources,duration),rows)
        copy=social_copy('pipeline',[SOURCE],copy_spec={'hook_id':'overview','body_fact_ids':['physics']})
        self.assertIn('assumed physical parameters',copy['linkedin']);self.assertFalse(copy['publication_performed'])

    @unittest.skipUnless(AVAILABLE,'Optional approved V3 project film unavailable')
    def test_producer_binds_real_nonempty_pixel_caption_geometry_and_trimmed_clock_proofs(self):
        source=M.MarketingSources().get(SOURCE);producer=json.loads(M.checked(source['chapters_document']).read_bytes())
        pixels=json.loads(M.checked(producer['source_picture_checks_ref']).read_bytes());self.assertEqual(pixels['checked_frames'],360)
        self.assertEqual(next(r['display_time_s']for r in pixels['checks']if r['display_time_s']==8),8)
        caption=json.loads(M.checked(producer['hero_caption_clearance_ref']).read_bytes());self.assertEqual(caption['component_bounds_checks'],2040);self.assertEqual(caption['overlaps'],0);self.assertGreater(caption['minimum_separation_px'],10)
        clock=json.loads(M.checked(producer['output_master_clock_ref']).read_bytes());self.assertEqual(clock['selected_frames'],870);self.assertEqual(clock['original_draw_associations'],600)
        self.assertEqual(clock['frames'][240]['decoded_master_frame_index'],272);self.assertLess(clock['maximum_draw_clock_offset_s'],.04)

    @unittest.skipUnless(AVAILABLE,'Optional approved V3 project film unavailable')
    def test_flagship_is_public_default_and_older_marketing_sources_are_preserved(self):
        for name in ['showcase/index.html','showcase/film.html']:
            text=(ROOT/name).read_text();self.assertIn('<source src="media/flagship-20261104-v1/cut-v2/interaction-twin-flagship-45s.mp4"',text)
            self.assertIn('trailers.html',text)
        self.assertEqual(M.file_ref('showcase/media/interaction-twin-dynamic-20261104-v2.mp4')['sha256'],'13d48ac20b5a617b53a74efaf49ee1e5372a775028b539cf6ee73b619ea65dbf')
        self.assertEqual(M.file_ref('showcase/media/interaction-twin-dynamic-20261104-v3.mp4')['sha256'],'6b590c0aa40550208b475f405979696dfb3e865486ad184ec1c8fb401826b404')


if __name__=='__main__':unittest.main()
