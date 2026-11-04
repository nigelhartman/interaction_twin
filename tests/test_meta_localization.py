"""Offline model fixtures prove full-frame localization/provenance contracts."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from meta_api import write
from meta_localization import LocalizationError, localize_discovery


class FixtureClient:
    def __init__(self, transform=None):
        self.calls=[];self.transform=transform

    def json_result(self,prompt,schema,images,*,output,model,max_output_tokens):
        self.calls.append({'prompt':prompt,'schema':schema,'images':list(images),'output':output,'model':model,'maximum':max_output_tokens})
        inputs=json.loads((output/'input.json').read_text())
        raw={'localizations':[{'track_id':item['track_id'],'sample_index':item['sample_index'],
                               'box_xyxy_1000':[100,200,400,600],'reason':'Fixture visible component.',
                               'visibility':'visible'} for item in inputs['requested_pairs']]}
        if self.transform:raw=self.transform(raw)
        receipt={'usage':{'input_tokens':100,'output_tokens':20},'estimated_cost_usd':'0.001000'}
        write(output/'response.pending.json',raw);write(output/'meta-request.json',receipt)
        return raw,receipt


class LocalizationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.case=self.root/'case';self.case.mkdir()
        self.rows=[]
        for index in range(1,33):
            path=self.case/f'frames/{index:03d}.png';path.parent.mkdir(exist_ok=True)
            Image.new('RGB',(80,120),(index,60,90)).save(path)
            self.rows.append({'index':index,'timestamp_s':.123456+index*.19876543,
                              'width':80,'height':120,'file':str(path.relative_to(self.case))})
        self.result={'summary':'Fixture candidates.','tracks':[self.track('part_a',[1,2]),self.track('part_b',[2,3])],
                     'actions':[{'description':'Fixture handling.','participant_track_ids':['part_a','part_b'],
                                 'evidence_samples':[1,2,3],'uncertainties':[]}], 'uncertainties':['Identity unknown.']}

    def track(self,identifier,indexes):
        return {'id':identifier,'label':'fixture component','description':'A visible rigid component.',
                'first_sample':min(indexes),'last_sample':max(indexes),
                'seeds':[{'sample_index':index,'timestamp_s':self.rows[index-1]['timestamp_s'],
                          'box_xyxy_1000':[1,2,900,950],'reason':'Original contact-sheet proposal.'} for index in indexes],
                'uncertainties':['Physical identity unverified.']}

    def run_local(self,result=None,client=None,name='localization'):
        return localize_discovery(self.result if result is None else result,self.rows,self.case,self.root/name,
                                  client or FixtureClient(),'muse-spark-1.3',16000)

    def test_full_frames_explicit_order_and_original_inputs_are_immutable(self):
        original=copy.deepcopy(self.result);client=FixtureClient()
        localized,report=self.run_local(client=client)
        self.assertEqual(self.result,original)
        self.assertEqual(json.loads((self.root/'localization/input-discovery.json').read_text()),original)
        self.assertEqual(len(client.calls),1)
        call=client.calls[0]
        self.assertEqual(call['images'],[(self.case/f'frames/{index:03d}.png').resolve() for index in [1,2,3]])
        self.assertIn('individual FULL-FRAME',call['prompt'])
        self.assertIn('untrusted source data',call['prompt'])
        self.assertIn('"image_order": 1',call['prompt'])
        self.assertIn('"original_width": 80',call['prompt'])
        self.assertNotIn('Original contact-sheet proposal.',call['prompt'])
        self.assertNotIn('[1, 2, 900, 950]',call['prompt'])
        self.assertEqual(localized['tracks'][0]['seeds'][0]['box_xyxy_1000'],[100,200,400,600])
        self.assertEqual(report['model_calls'],1);self.assertEqual(report['estimated_cost_usd'],'0.001000')
        self.assertEqual(report['usage'],{'input_tokens':100,'output_tokens':20})
        self.assertFalse(report['manual_seed_boxes_used'])
        self.assertFalse(report['contact_sheet_coordinates_reused'])
        trace=json.loads((self.root/'localization/transformation-trace.json').read_text())
        self.assertEqual(trace['seed_changes'][0]['original_seed']['box_xyxy_1000'],[1,2,900,950])
        self.assertEqual(trace['seed_changes'][0]['operation'],'replace_with_new_model_box')

    def test_authoritative_source_pts_are_copied_not_model_or_rounded_times(self):
        result=copy.deepcopy(self.result)
        result['tracks'][0]['seeds'][0]['timestamp_s']+=.0005
        localized,_=self.run_local(result=result)
        self.assertEqual(localized['tracks'][0]['seeds'][0]['timestamp_s'],self.rows[0]['timestamp_s'])
        original=json.loads((self.root/'localization/input-discovery.json').read_text())
        self.assertEqual(original['tracks'][0]['seeds'][0]['timestamp_s'],result['tracks'][0]['seeds'][0]['timestamp_s'])

    def test_uncertain_seeds_tracks_and_action_participants_removed_with_trace(self):
        def transform(raw):
            for item in raw['localizations']:
                if item['track_id']=='part_b':item.update(visibility='uncertain',box_xyxy_1000=None)
            return raw
        result=copy.deepcopy(self.result)
        result['actions'].append({'description':'Only omitted component.','participant_track_ids':['part_b'],'evidence_samples':[2],'uncertainties':[]})
        localized,report=self.run_local(result=result,client=FixtureClient(transform))
        self.assertEqual([track['id'] for track in localized['tracks']],['part_a'])
        self.assertEqual(len(localized['actions']),1)
        self.assertEqual(localized['actions'][0]['participant_track_ids'],['part_a'])
        self.assertEqual(report['omitted_seeds'],2);self.assertEqual(report['omitted_tracks'],1)
        self.assertIn('unlocalizable',localized['uncertainties'][-1])
        trace=json.loads((self.root/'localization/transformation-trace.json').read_text())
        self.assertEqual([item['operation'] for item in trace['action_changes']],['filter_participants','omit_action'])
        self.assertEqual(trace['removed_tracks'][0]['reason'],'no_visible_valid_full_frame_seed')

    def test_invalid_visible_box_never_falls_back_to_old_coordinates(self):
        def transform(raw):
            for item in raw['localizations']:
                if item['track_id']=='part_a':item['box_xyxy_1000']=[900,200,100,600]
            return raw
        localized,_=self.run_local(client=FixtureClient(transform))
        self.assertEqual([track['id'] for track in localized['tracks']],['part_b'])
        trace=json.loads((self.root/'localization/transformation-trace.json').read_text())
        self.assertEqual(trace['seed_changes'][0]['reason'],'invalid_visible_box')
        self.assertEqual(trace['seed_changes'][0]['operation'],'omit_seed')

    def test_zero_visible_components_is_explicit_empty_result(self):
        def transform(raw):
            for item in raw['localizations']:item.update(visibility='not_visible',box_xyxy_1000=None)
            return raw
        localized,report=self.run_local(client=FixtureClient(transform))
        self.assertEqual(localized['tracks'],[]);self.assertEqual(localized['actions'],[])
        self.assertEqual(report['retained_tracks'],0);self.assertEqual(report['omitted_tracks'],2)

    def test_missing_duplicate_or_invented_pair_fails_with_raw_evidence_preserved(self):
        mutations=[lambda raw:{'localizations':raw['localizations'][:-1]},
                   lambda raw:{'localizations':raw['localizations']+[raw['localizations'][0]]},
                   lambda raw:{'localizations':[dict(raw['localizations'][0],track_id='invented')]+raw['localizations'][1:]}]
        for index,transform in enumerate(mutations):
            with self.subTest(index=index),self.assertRaises(LocalizationError):
                self.run_local(client=FixtureClient(transform),name='invalid-'+str(index))
            out=self.root/('invalid-'+str(index))
            self.assertTrue((out/'batch-001/raw-localization.json').is_file())
            self.assertTrue((out/'batch-001/meta-request.json').is_file())
            self.assertFalse((out/'localized-discovery.json').exists())
            self.assertEqual(json.loads((out/'report.json').read_text())['status'],'failed_no_localized_result')

    def test_invented_timestamp_field_rejected_by_exact_contract(self):
        def transform(raw):raw['localizations'][0]['timestamp_s']=999;return raw
        with self.assertRaisesRegex(LocalizationError,'invalid_localization_contract'):
            self.run_local(client=FixtureClient(transform))

    def test_two_batches_cover_24_unique_full_frames_without_duplicate_pairs(self):
        result=copy.deepcopy(self.result)
        result['tracks']=[self.track('part_'+str(index),list(range(index*3+1,index*3+4))) for index in range(8)]
        result['actions']=[];client=FixtureClient()
        localized,report=self.run_local(result=result,client=client)
        self.assertEqual([len(call['images']) for call in client.calls],[12,12])
        self.assertEqual(report['model_calls'],2);self.assertEqual(report['estimated_cost_usd'],'0.002000')
        self.assertEqual(report['usage'],{'input_tokens':200,'output_tokens':40})
        self.assertEqual(sum(len(track['seeds']) for track in localized['tracks']),24)
        self.assertEqual(report['requests'][1]['source_sample_indices'],list(range(13,25)))

    def test_more_than_24_images_rejected_before_any_model_call(self):
        result=copy.deepcopy(self.result)
        result['tracks']=[self.track('part_'+str(index),list(range(index*4+1,index*4+5))) for index in range(8)]
        result['actions']=[];client=FixtureClient()
        with self.assertRaisesRegex(LocalizationError,'localization_budget_scope_exceeded'):
            self.run_local(result=result,client=client)
        self.assertEqual(client.calls,[]);self.assertFalse((self.root/'localization').exists())

    def test_zero_tracks_bypasses_client(self):
        result={'summary':'None visible.','tracks':[],'actions':[],'uncertainties':[]}
        class NeverCall:
            def json_result(self,*args,**kwargs):raise AssertionError('must not call')
        localized,report=self.run_local(result=result,client=NeverCall())
        self.assertEqual(localized['tracks'],[])
        self.assertFalse(report['model_call_made']);self.assertEqual(report['model_calls'],0)
        self.assertEqual(report['estimated_cost_usd'],'0.000000')

    def test_dimension_mismatch_source_escape_and_duplicate_input_fail_before_call(self):
        client=FixtureClient()
        self.rows[0]['width']=81
        with self.assertRaisesRegex(LocalizationError,'source_image_dimensions_mismatch'):
            self.run_local(client=client)
        self.rows[0]['width']=80
        outside=self.root/'outside.png';Image.new('RGB',(80,120)).save(outside)
        self.rows[0]['file']='../outside.png'
        with self.assertRaisesRegex(LocalizationError,'source_image_unavailable_or_outside_case'):
            self.run_local(client=client)
        self.rows[0]['file']='frames/001.png'
        result=copy.deepcopy(self.result);result['tracks'][0]['seeds'].append(copy.deepcopy(result['tracks'][0]['seeds'][0]))
        with self.assertRaisesRegex(LocalizationError,'invalid_or_duplicate_seed_pair'):
            self.run_local(result=result,client=client)
        self.assertEqual(client.calls,[])

    def test_existing_output_is_never_overwritten(self):
        client=FixtureClient();self.run_local(client=client)
        original=(self.root/'localization/input-discovery.json').read_bytes()
        with self.assertRaisesRegex(LocalizationError,'localization_output_exists'):
            self.run_local(client=client)
        self.assertEqual(len(client.calls),1)
        self.assertEqual((self.root/'localization/input-discovery.json').read_bytes(),original)


if __name__=='__main__':unittest.main()
