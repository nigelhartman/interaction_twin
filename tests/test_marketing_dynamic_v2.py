"""Current film evidence stays exact; historical sources stay independently usable."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import marketing_sources as M
from marketing_studio import default_storyboard,source_chapters,social_copy,validate_storyboard

SOURCE='pipeline_dynamic_45s_v2'
ROOT=Path(__file__).resolve().parents[1]


class DynamicFilmMarketingTests(unittest.TestCase):
    @unittest.skipUnless((ROOT/'showcase/media/interaction-twin-dynamic-20261104-v2.mp4').is_file(),'Optional approved project film unavailable')
    def test_actual_source_and_nine_chapters_bind_exact_current_film(self):
        source=M.MarketingSources().get(SOURCE)
        self.assertEqual(source['file'],{'path':'showcase/media/interaction-twin-dynamic-20261104-v2.mp4','sha256':'13d48ac20b5a617b53a74efaf49ee1e5372a775028b539cf6ee73b619ea65dbf','bytes':5442954})
        self.assertEqual(source['pack_ids'],[M.JABRA,M.WALLET,M.CHARGER,M.LAPTOP])
        chapters=source_chapters(source);self.assertEqual(len(chapters),9)
        self.assertEqual([c['start_s']for c in chapters]+[chapters[-1]['end_s']],[0,4,8,16,20,25,29,39,43,45])
        producer=json.loads(M.checked(source['chapters_document']).read_bytes())
        self.assertEqual(producer['movie_ref'],source['file'])
        self.assertEqual(producer['chapter_frame_boundaries'],[0,120,240,480,600,750,870,1170,1290,1350])
        self.assertIn('assumed',source['limitation']);self.assertIn('authored',source['limitation'])
        self.assertEqual(chapters[6]['evidence_kind'],'robot_preview');self.assertEqual(chapters[6]['fact_ids'],['robotics_limit'])
        self.assertEqual(chapters[7]['evidence_kind'],'physics');self.assertEqual(chapters[7]['fact_ids'],['physics'])

    @unittest.skipUnless((ROOT/'output/presentation-polish-20261104-v2/historical-marketing-source-pins.json').is_file(),'Optional historical project source snapshots unavailable')
    def test_all_historical_sources_and_resolved_snapshots_remain_exact(self):
        pins=json.loads((ROOT/'output/presentation-polish-20261104-v2/historical-marketing-source-pins.json').read_text())
        store=M.MarketingSources();self.assertEqual(len(pins),7)
        for identifier,pin in pins.items():
            with self.subTest(identifier=identifier):
                self.assertEqual(M.digest(M.DECLARED[identifier]),pin['declared_digest'])
                self.assertEqual(M.digest(store.get(identifier)),pin['resolved_digest'])

    @unittest.skipUnless((ROOT/'showcase/media/interaction-twin-dynamic-20261104-v2.mp4').is_file(),'Optional approved project film unavailable')
    def test_new_film_preference_requires_it_to_be_selected_and_each_cut_is_bounded(self):
        store=M.MarketingSources();new=store.get(SOURCE);old=store.get('pipeline_explainer_refined45')
        self.assertEqual({r['source_id']for r in default_storyboard({old['id']:old},'pipeline',30)},{old['id']})
        for duration in [15,30]:
            for prefs in [{},{'pace':'fast'},{'pace':'slow'},{'caption_density':'minimal'},{'focus':'robotics'},{'focus':'agents'},{'focus':'library','order':'result_first'}]:
                sources={old['id']:old,SOURCE:new};rows=default_storyboard(sources,'pipeline',duration,prefs)
                self.assertEqual({r['source_id']for r in rows},{SOURCE})
                self.assertEqual(validate_storyboard(rows,sources,duration),rows)
                for row in rows:self.assertTrue(any(c['start_s']<=row['start_s']+1e-10 and row['start_s']+row['duration_s']<=c['end_s']+1e-10 and row['fact_id']in c['fact_ids']for c in new['chapters']))

    @unittest.skipUnless((ROOT/'showcase/media/interaction-twin-dynamic-20261104-v2.mp4').is_file(),'Optional approved project film unavailable')
    def test_physics_copy_cannot_label_a_robot_or_unchaptered_clip(self):
        source=M.MarketingSources().get(SOURCE)
        row={'source_id':SOURCE,'start_s':39.,'duration_s':4.,'headline_id':'physics','fact_id':'physics','transition':'cut'}
        prefix={**row,'start_s':0.,'duration_s':1.,'headline_id':'source','fact_id':'source'}
        self.assertEqual(validate_storyboard([prefix,row],{SOURCE:source},8.25),[prefix,row])
        for start in [0.,29.]:
            wrong={**row,'start_s':start}
            with self.assertRaisesRegex(ValueError,'Physics copy'):validate_storyboard([prefix,wrong],{SOURCE:source},8.25)
        unchaptered={'kind':'robot_preview','duration_s':45.}
        with self.assertRaisesRegex(ValueError,'Physics copy'):validate_storyboard([prefix,row],{SOURCE:unchaptered},8.25)
        for identifier in ['robot_workbench','pipeline_explainer_refined45']:
            with self.assertRaisesRegex(ValueError,'registered Newton'):social_copy('pipeline',[identifier],copy_spec={'hook_id':'overview','body_fact_ids':['physics']})
        approved=social_copy('pipeline',[SOURCE],copy_spec={'hook_id':'overview','body_fact_ids':['physics']})
        self.assertIn('assumed physical parameters',approved['linkedin']);self.assertFalse(approved['publication_performed'])

    def test_matching_producer_hash_cannot_hide_wrong_chapter_timing(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder).resolve();movie=root/'synthetic.mp4';movie.write_bytes(b'SYNTHETIC registry fixture, not footage')
            chapters=copy.deepcopy(M.DYNAMIC_V2_CHAPTERS)
            producer={'format':'interaction-twin-brand-explainer-storyboard','version':7,'chapters':[{'start_s':c['start_s'],'end_s':c['end_s'],'title':'Synthetic bounded title','scope':'Synthetic registry fixture; not real capture.'}for c in chapters]}
            doc=root/'storyboard.json';doc.write_text(json.dumps(producer))
            item={**M.DECLARED[SOURCE],'path':'synthetic.mp4','sha256':M.file_ref(movie,root)['sha256'],'chapters_document':M.file_ref(doc,root)}
            with patch.dict(M.DECLARED,{SOURCE:item},clear=True):
                self.assertEqual(len(M.MarketingSources(root).get(SOURCE)['chapters']),9)
                producer['chapters'][7]['start_s']=29.;doc.write_text(json.dumps(producer));item['chapters_document']=M.file_ref(doc,root)
                with self.assertRaisesRegex(ValueError,'timing differs'):M.MarketingSources(root).get(SOURCE)


if __name__=='__main__':unittest.main()
