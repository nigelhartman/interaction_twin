"""Versioned refined-film evidence; synthetic IO fixtures are never recordings."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import marketing_sources as M
from marketing_studio import default_storyboard,validate_storyboard

IDENTIFIER='pipeline_explainer_refined45'
BOUNDARIES=[0,48,120,216,336,432,576,672,768,1008,1080]


def synthetic_chapters():
    chapters=copy.deepcopy(M.EXPLAINER_45S_CHAPTERS)
    for i,chapter in enumerate(chapters):
        chapter['start_s']=BOUNDARIES[i]/24;chapter['end_s']=BOUNDARIES[i+1]/24
    chapters[3]['label']='Three native meshes plus a declared lid derivative'
    return chapters


def synthetic_source():
    return {'id':IDENTIFIER,'kind':'pipeline_film','duration_s':45.,'chapters':synthetic_chapters()}


class RefinedFilmEvidenceTests(unittest.TestCase):
    def test_actual_refined_registration_binds_exact_movie_producer_and_declared_derivative(self):
        source=M.MarketingSources().get(IDENTIFIER)
        self.assertEqual(source['file'],{'path':'showcase/media/interaction-twin-explainer-20261103-v6-refined45.mp4',
                                        'sha256':'3545dcb03d2c5101fa3c986e32122617fda205635c9b026aba2ae7bab3526627','bytes':3717480})
        self.assertEqual(source['pack_ids'],[M.JABRA]);self.assertEqual(source['duration_s'],45.)
        self.assertIn('Three original native',source['limitation']);self.assertIn('one documented geometrically refined lid',source['limitation'])
        self.assertIn('authored',source['limitation']);self.assertIn('derivative',source['chapters'][3]['label'])
        producer=json.loads(M.checked(source['chapters_document']).read_bytes())
        self.assertEqual(producer['version'],6);self.assertEqual(producer['chapter_frame_boundaries'],BOUNDARIES)
        self.assertEqual([c['start_s']for c in source['chapters']]+[source['chapters'][-1]['end_s']],[f/24 for f in BOUNDARIES])
        self.assertEqual([(c['start_s'],c['end_s'])for c in source['chapters']],[(c['start_s'],c['end_s'])for c in producer['chapters']])

    def test_new_source_preference_requires_it_to_be_selected(self):
        store=M.MarketingSources();new=store.get(IDENTIFIER);old=store.get('pipeline_explainer_45s')
        historical=default_storyboard({old['id']:old},'pipeline',30)
        selected=default_storyboard({old['id']:old,IDENTIFIER:new},'pipeline',30)
        self.assertEqual({r['source_id']for r in historical},{old['id']})
        self.assertEqual({r['source_id']for r in selected},{IDENTIFIER})
        self.assertEqual(default_storyboard({old['id']:old},'pipeline',30),historical)

    def test_all_six_old_source_declarations_and_resolved_snapshots_remain_exact(self):
        expected={
          'pipeline_demo':('97855a15ff89ed4692d3bf2870775fda2603514049fbb618d87fad95aeb66d15','c75e1d8a3c8a0138ccdf14fa58bd6ded49ca1d306665a3b10d0786ac785e9ae9'),
          'robot_workbench':('2994def10f597108965fa011f0f7c6400dcd3b3f0cf13ac643b9b8e168a49a26','b87b2ce25fb5087fea902dc37e6d74a67ef508a11b30cd6387f76a02ab0b3dee'),
          'jabra_capture':('a005f565c1fe2f1e8e2395c0bb123711f28b9cb27d412ef82dd44934db39a289','be205370fd8c06d7997b2420cdf00d2a23c68f22a0edb2bb759a970337639525'),
          'wallet_capture':('53273b697ee6b34df62d42cee7c5ef826293bef88388bb3192ccff9a87310ba9','5842b1004e3b34ac56668f5ef52cd4fed52bd28b81b3c3e1cd227ead03fe4965'),
          'pipeline_explainer':('7da2da51b067a5dbcc866d9ea9914c42ff60233fd583db7e9109a05cc4c4fdb8','00f8ef81ccb5f2150a143d26a8b02125351f9189d864358800067762dc91a6a4'),
          'pipeline_explainer_45s':('6eb5661f775e88efacb9816b186b276bd80256434160dd6fcc0f42b9a9fb9b70','5d9eddcba42efda1112dd1240cf71d12d863654ed82ebdd7b2604d379cbf49de'),
        }
        store=M.MarketingSources()
        for identifier,(declared,resolved)in expected.items():
            with self.subTest(identifier=identifier):
                self.assertEqual(M.digest(M.DECLARED[identifier]),declared)
                self.assertEqual(M.digest(store.get(identifier)),resolved)

    def test_generic_reader_accepts_only_exact_version6_producer_bytes_and_timing(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp).resolve();movie=root/'synthetic.mp4';movie.write_bytes(b'NOT A RECORDING: synthetic registry IO only')
            document={'format':'interaction-twin-brand-explainer-storyboard','version':6,
                      'chapters':[{'start_s':c['start_s'],'end_s':c['end_s'],'title':'Synthetic bounded title',
                                   'scope':'Synthetic declared lid derivative; no physical validation.'}for c in synthetic_chapters()]}
            path=root/'storyboard.json';path.write_text(json.dumps(document))
            item={**synthetic_source(),'label':'Synthetic source','path':'synthetic.mp4','sha256':M.file_ref(movie,root)['sha256'],
                  'pack_ids':[M.JABRA],'rights':'Synthetic fixture only.','limitation':'One declared lid derivative; physics unverified.',
                  'chapters_document_version':6,'chapters_document':M.file_ref(path,root)}
            with patch.dict(M.DECLARED,{IDENTIFIER:item},clear=True):
                result=M.MarketingSources(root).get(IDENTIFIER)
                self.assertEqual(result['chapters_document_version'],6)
                self.assertIn('declared lid derivative',result['chapters'][3]['editorial_scope'])
                path.write_text(json.dumps(document)+' ')
                with self.assertRaisesRegex(ValueError,'changed'):M.MarketingSources(root).get(IDENTIFIER)
                document['version']=5;path.write_text(json.dumps(document));item['chapters_document']=M.file_ref(path,root)
                with self.assertRaisesRegex(ValueError,'differs'):M.MarketingSources(root).get(IDENTIFIER)
                document['version']=6;document['chapters'][3]['end_s']+=1/24
                path.write_text(json.dumps(document));item['chapters_document']=M.file_ref(path,root)
                with self.assertRaisesRegex(ValueError,'timing differs'):M.MarketingSources(root).get(IDENTIFIER)

    def test_new_film_reuses_only_its_matching_15_and_30_second_evidence_windows(self):
        source=synthetic_source();sources={IDENTIFIER:source}
        for duration in [15,30]:
            for prefs in [{},{'pace':'fast'},{'pace':'slow'},{'caption_density':'minimal'},
                          {'focus':'robotics'},{'focus':'library','order':'result_first'}]:
                with self.subTest(duration=duration,prefs=prefs):
                    rows=default_storyboard(sources,'pipeline',duration,prefs)
                    self.assertEqual(validate_storyboard(rows,sources,duration),rows)
                    self.assertEqual({r['source_id']for r in rows},{IDENTIFIER})
                    for r in rows:
                        self.assertTrue(any(c['start_s']<=r['start_s']+1e-10 and r['start_s']+r['duration_s']<=c['end_s']+1e-10
                                            and r['fact_id']in c['fact_ids']and r['headline_id']in c['headline_ids']for c in source['chapters']))
        with self.assertRaisesRegex(ValueError,'cannot fill'):default_storyboard(sources,'pipeline',45)

    def test_refined_film_does_not_weaken_physical_or_robot_capability_gates(self):
        source=synthetic_source();sources={IDENTIFIER:source}
        rows=default_storyboard(sources,'pipeline',15)
        wrong=copy.deepcopy(rows);wrong[0].update(start_s=32.,fact_id='robotics',headline_id='robotics')
        with self.assertRaisesRegex(ValueError,'actual workbench'):validate_storyboard(wrong,sources,15)
        wrong=copy.deepcopy(rows);wrong[0].update(start_s=9.,fact_id='masks',headline_id='masks')
        with self.assertRaisesRegex(ValueError,'matching source chapter'):validate_storyboard(wrong,sources,15)


if __name__=='__main__':unittest.main()
