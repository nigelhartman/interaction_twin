"""Frozen chapter windows keep footage, displayed captions and Muse context aligned."""
import copy
import json
import shutil
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import marketing_sources as MS
from marketing_sources import ROOT,EXPLAINER_CHAPTERS,EXPLAINER_45S_CHAPTERS,file_ref
from marketing_studio import default_storyboard,validate_storyboard,source_chapters,MarketingStudio


def film(identifier='pipeline_explainer',*,chapters=True):
    return {'id':identifier,'kind':'pipeline_film','duration_s':55.,'chapters':copy.deepcopy(EXPLAINER_CHAPTERS)}if chapters else{'id':identifier,'kind':'pipeline_film','duration_s':55.}


class MarketingChapterTests(unittest.TestCase):
    def test_15_30_45_cuts_and_preferences_stay_in_matching_visual_chapters(self):
        source=film()
        for duration in [15,30,45]:
            for preferences in [{},{'pace':'fast'},{'pace':'slow'},{'caption_density':'minimal'},{'focus':'library','order':'result_first'},{'focus':'robotics'},{'focus':'agents'}]:
                with self.subTest(duration=duration,preferences=preferences):
                    rows=default_storyboard({'pipeline_explainer':source},'pipeline',duration,preferences)
                    self.assertLessEqual(len(rows),8);self.assertGreaterEqual(len(rows),3)
                    self.assertAlmostEqual(4+sum(r['duration_s']for r in rows)-.25*(len(rows)+1),duration)
                    self.assertEqual(validate_storyboard(rows,{'pipeline_explainer':source},duration),rows)
                    for row in rows:
                        self.assertTrue(any(row['start_s']>=c['start_s']-1e-6 and row['start_s']+row['duration_s']<=c['end_s']+1e-6 and row['fact_id']in c['fact_ids']and row['headline_id']in c['headline_ids']for c in EXPLAINER_CHAPTERS))
                    self.assertTrue(all(r['start_s']>=3 and r['start_s']+r['duration_s']<=52+1e-6 for r in rows))

    def test_mismatched_copy_cross_chapter_timing_and_brand_frames_are_rejected(self):
        source=film();rows=default_storyboard({'pipeline_explainer':source},'pipeline',15)
        for patch in [{'start_s':14.,'fact_id':'masks','headline_id':'masks'},{'start_s':12.5,'duration_s':3.0625,'fact_id':'masks','headline_id':'masks'},{'start_s':0.},{'headline_id':'library'}]:
            changed=copy.deepcopy(rows);changed[0].update(patch)
            with self.subTest(patch=patch),self.assertRaisesRegex(ValueError,'matching source chapter'):validate_storyboard(changed,{'pipeline_explainer':source},15)

    def test_new_film_is_used_only_when_selected_and_generic_kind_is_supported(self):
        old=film('pipeline_demo',chapters=False);new=film();rows=default_storyboard({'pipeline_demo':old},'pipeline',30)
        self.assertEqual({r['source_id']for r in rows},{'pipeline_demo'})
        selected=default_storyboard({'pipeline_demo':old,'pipeline_explainer':new},'pipeline',30)
        self.assertEqual({r['source_id']for r in selected},{'pipeline_explainer'})
        other=default_storyboard({'a_new_pipeline':film('a_new_pipeline')},'pipeline',30)
        self.assertEqual({r['source_id']for r in other},{'a_new_pipeline'})
        robotic=default_storyboard({'pipeline_explainer':new,'robot_workbench':{'kind':'robot_preview','duration_s':23.523}},'robotics',15)
        self.assertIn('robotics_limit',[row['fact_id']for row in robotic])

    def test_versioned_short_film_preference_never_changes_a_historical_only_cut(self):
        old=film();historical=default_storyboard({'pipeline_explainer':old},'pipeline',30)
        short=film('pipeline_explainer_45s');short['duration_s']=45.
        # Synthetic compressed chapter fixture; the real registration binds its
        # own producer's immutable timestamps independently of this test.
        for chapter in short['chapters']:
            chapter['start_s']=round(chapter['start_s']*45/55,3);chapter['end_s']=round(chapter['end_s']*45/55,3)
        sources={'pipeline_explainer':old,'pipeline_explainer_45s':short}
        for duration in [15,30]:
            with self.subTest(duration=duration):
                selected=default_storyboard(sources,'pipeline',duration)
                self.assertEqual({row['source_id']for row in selected},{'pipeline_explainer_45s'})
                self.assertEqual(validate_storyboard(selected,sources,duration),selected)
        self.assertEqual(default_storyboard({'pipeline_explainer':old},'pipeline',30),historical)
        self.assertEqual(old,film())
        with self.assertRaisesRegex(ValueError,'cannot fill'):
            default_storyboard({'pipeline_explainer_45s':short},'pipeline',45)

    def test_legacy_unchaptered_recipe_keeps_exact_sources_starts_durations_and_copy(self):
        sources={'pipeline_demo':film('pipeline_demo',chapters=False),'capture':{'id':'capture','kind':'source_video','duration_s':33.},'robot_workbench':{'id':'robot_workbench','kind':'robot_preview','duration_s':23.523}}
        expected=[{'source_id':sid,'start_s':start,'duration_s':5.5,'headline_id':word,'fact_id':word,'transition':'crossfade'}for sid,start,word in [('capture',0.,'source'),('pipeline_demo',5.,'masks'),('pipeline_demo',13.,'connections'),('pipeline_demo',21.,'program'),('robot_workbench',0.,'robotics')]]
        self.assertEqual(default_storyboard(sources,'pipeline',30),expected)
        renamed={('new_film'if k=='pipeline_demo'else k):({**v,'id':'new_film'}if k=='pipeline_demo'else v)for k,v in sources.items()}
        generic=default_storyboard(renamed,'pipeline',30)
        self.assertEqual(generic,[{**row,'source_id':'new_film'if row['source_id']=='pipeline_demo'else row['source_id']}for row in expected])

    def test_malformed_chapter_bounds_ids_and_inadequate_evidence_fail_closed(self):
        for patch in [{'end_s':float('nan')},{'end_s':100.},{'fact_ids':['invented']},{'start_s':True},{'evidence_kind':['source_video']}]:
            source=film();source['chapters'][1].update(patch)
            with self.subTest(patch=patch),self.assertRaises(ValueError):source_chapters(source)
        source=film();source['chapters']=source['chapters'][1:2]
        with self.assertRaisesRegex(ValueError,'cannot fill'):default_storyboard({'pipeline_explainer':source},'pipeline',45)

    def test_muse_inert_packet_includes_frozen_chapters_without_a_model_call(self):
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary).resolve()
            for relative in ['config/marketing-brand.json','showcase/mark.svg']:
                target=root/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy(ROOT/relative,target)
            path=root/'synthetic.mp4';path.write_bytes(b'SYNTHETIC CHAPTER CONTRACT ONLY')
            class Sources:
                def get(self,identifier):
                    if identifier!='pipeline_explainer':raise ValueError('Only selected fixture')
                    return film()|{'file':file_ref(path,root),'label':'Synthetic chapter fixture','pack_ids':[],'limitation':'Synthetic software fixture.'}
            studio=MarketingStudio(root,sources=Sources());saved=studio.create_campaign('Chapter fixture',source_ids=['pipeline_explainer'],duration_s=30,music_id=None)
            packet=studio.propose_with_muse(saved['campaign']['id'],saved['latest_revision']['id'])
            self.assertEqual(packet['new_api_calls'],0);self.assertEqual(packet['context']['sources']['pipeline_explainer']['chapters'],EXPLAINER_CHAPTERS)
            self.assertTrue(packet['context']['no_media_uploaded'])
            self.assertFalse(packet['context']['copy_capability_scope']['positive_robotics_hook_and_fact_available'])

    def test_registered_film_chapters_are_bound_to_exact_producer_document_and_timing(self):
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary).resolve();video=root/'synthetic.mp4';video.write_bytes(b'SYNTHETIC SOURCE REGISTRY ONLY')
            document={'format':'interaction-twin-brand-explainer-storyboard','version':2,'chapters':[{'start_s':c['start_s'],'end_s':c['end_s'],'title':'Synthetic chapter title','scope':'Synthetic chapter scope.'}for c in EXPLAINER_CHAPTERS]}
            path=root/'storyboard.json';path.write_text(json.dumps(document))
            entry={**film(),'path':'synthetic.mp4','sha256':file_ref(video,root)['sha256'],'label':'Synthetic film','pack_ids':[],'limitation':'Synthetic source registry only.','chapters_document':file_ref(path,root)}
            with patch.dict(MS.DECLARED,{'pipeline_explainer':entry},clear=True):
                source=MS.MarketingSources(root).get('pipeline_explainer')
                self.assertEqual(source['chapters'][1]['editorial_title'],'Synthetic chapter title')
                self.assertEqual(source['chapters_sha256'],MS.digest(source['chapters']))
                path.write_text(json.dumps(document)+' ')
                with self.assertRaisesRegex(ValueError,'changed'):MS.MarketingSources(root).get('pipeline_explainer')
                document['chapters'][1]['start_s']=3.5;path.write_text(json.dumps(document));entry['chapters_document']=file_ref(path,root)
                with self.assertRaisesRegex(ValueError,'chapter timing differs'):MS.MarketingSources(root).get('pipeline_explainer')

    def test_actual_versioned_45_second_source_preserves_the_exact_historical_source(self):
        sources=MS.MarketingSources();short=sources.get('pipeline_explainer_45s');old=sources.get('pipeline_explainer')
        self.assertEqual(short['file'],{'path':'showcase/media/interaction-twin-explainer-20261103-v5-45s.mp4','sha256':'1f803994111e144ad2402f4a90ebe01907ddb250b512203f294e04a71c56aac7','bytes':3909354})
        self.assertEqual(short['duration_s'],45.);self.assertEqual(short['pack_ids'],[MS.JABRA])
        producer=json.loads(MS.checked(short['chapters_document']).read_bytes())
        boundaries=[0,59,157,275,412,530,687,805,903,1021,1080]
        self.assertEqual(producer['version'],5);self.assertEqual(producer['chapter_frame_boundaries'],boundaries)
        self.assertEqual([c['start_s']for c in EXPLAINER_45S_CHAPTERS]+[EXPLAINER_45S_CHAPTERS[-1]['end_s']],[f/24 for f in boundaries])
        self.assertEqual([(c['start_s'],c['end_s'])for c in short['chapters']],[(c['start_s'],c['end_s'])for c in producer['chapters']])
        self.assertEqual(MS.digest(MS.DECLARED['pipeline_explainer']),'7da2da51b067a5dbcc866d9ea9914c42ff60233fd583db7e9109a05cc4c4fdb8')
        self.assertEqual(MS.digest(old),'00f8ef81ccb5f2150a143d26a8b02125351f9189d864358800067762dc91a6a4')

    def test_actual_45_second_source_cuts_keep_every_clip_inside_its_recorded_chapter(self):
        source=MS.MarketingSources().get('pipeline_explainer_45s');sources={source['id']:source}
        for duration in [15,30]:
            for preferences in [{},{'pace':'fast'},{'pace':'slow'},{'caption_density':'minimal'},{'focus':'library','order':'result_first'},{'focus':'robotics'},{'focus':'agents'}]:
                with self.subTest(duration=duration,preferences=preferences):
                    rows=default_storyboard(sources,'pipeline',duration,preferences)
                    self.assertEqual(validate_storyboard(rows,sources,duration),rows)
                    for row in rows:
                        self.assertTrue(any(row['start_s']>=c['start_s']-1e-10 and row['start_s']+row['duration_s']<=c['end_s']+1e-10 and row['fact_id']in c['fact_ids']and row['headline_id']in c['headline_ids']for c in source['chapters']))
        with self.assertRaisesRegex(ValueError,'cannot fill'):default_storyboard(sources,'pipeline',45)

    def test_precision_repair_cannot_manufacture_chapter_headroom(self):
        source=film('precision_fixture');source['chapters']=[]
        start=3.
        for name in ['source','masks','parts']:
            end=start+4.-2e-7
            source['chapters'].append({'id':name,'start_s':start,'end_s':end,'fact_ids':[name],'headline_ids':[name],'evidence_kind':'source_video'})
            start=end
        with self.assertRaisesRegex(ValueError,'no evidence headroom'):
            default_storyboard({'precision_fixture':source},'pipeline',15)

    def test_all_historical_55_second_recipe_bytes_stay_frozen(self):
        preferences=[{},{'pace':'fast'},{'pace':'slow'},{'caption_density':'minimal'},{'focus':'library','order':'result_first'},{'focus':'robotics'},{'focus':'agents'}]
        recipes=[default_storyboard({'pipeline_explainer':film()},'pipeline',duration,p)for duration in [15,30,45]for p in preferences]
        self.assertEqual(MS.digest(recipes),'819924b77faf35910a63e01e0e7dffb10168d40565056f48395972430696557b')


if __name__=='__main__':unittest.main()
