"""Completed-state adoption and actual component-bound private follow-on."""
import copy
import io
import json
import shutil
import unittest
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw

import capture_semantic_batch_v1 as B
import semantic_asset_companion as S
import cable_trace_2d as C
from tests import test_twin_catalog as fixtures


class FixedCaptureSemanticBatchTests(unittest.TestCase):
    def setUp(self):
        fixture=fixtures.TwinCatalogTests(methodName='runTest');fixture.setUp();self.addCleanup(fixture.doCleanups)
        self.fixture=fixture;self.root=fixture.root;self.output='output/fixed-semantic-review'
        specifications={
            'charger':[('adapter_early','white power adapter'),('power_cable','white cable')],
            'phone':[('phone_blue','blue-cased smartphone'),('cable_connector','cable connector plug')],
            'laptop':[('laptop_base','laptop base'),('power_cable','power cable')],
            'bottle':[('bottle_body','plastic bottle body'),('bottle_cap','blue bottle cap')],
        }
        notes={
            'charger':'USB-C charger and cable connected. Cable flexible; both cable ends USB-C.',
            'phone':'Connect USB-C cable to phone USB-C port. Charger is owner-reported same as other recording; candidate reuse only.',
            'laptop':'Laptop opens, power button pressed; charger cable briefly disconnected/reconnected.',
            'bottle':'Plastic water bottle screw cap rotates and detaches, water poured into drinking glass, cap closed.',
        }
        self.states={};self.programs={};captures=[]
        image=Image.new('L',(120,100));ImageDraw.Draw(image).line([(10,20),(40,65),(105,40)],fill=255,width=5)
        binary=io.BytesIO();image.save(binary,format='PNG')
        rgb=io.BytesIO();Image.new('RGB',(120,100)).save(rgb,format='PNG')
        for kind,spec in specifications.items():
            run_id=B.RUNS[kind];case='flow-capture-'+('bottle-consistency-20261103-v2'if kind=='bottle'else kind+'-20261103-v1')
            prefix='data/'+case;shutil.copytree(self.root/'data/case',self.root/prefix)
            replacements={'part':spec[0][0],'holder':spec[1][0]}
            def rewrite(value):
                if isinstance(value,str):return replacements.get(value,value.replace('data/case/',prefix+'/'))
                if isinstance(value,list):return[rewrite(v)for v in value]
                if isinstance(value,dict):return{k:rewrite(v)for k,v in value.items()}
                return value
            document=rewrite(copy.deepcopy(fixture.doc));document['case_id']=case
            for track,(_,label)in zip(document['tracklets'],spec):track['label']=label
            source=self.file(document['source']['path'],('fixture-source-'+kind).encode())
            document['source'].update(sha256=source['sha256'],original_sha256=source['sha256'])
            manifest=next(i for i in document['inputs']if i['kind']=='source_manifest')
            value=json.loads((self.root/manifest['path']).read_text());value['source']['sha256']=source['sha256']
            manifest['sha256']=self.file(manifest['path'],value)['sha256']
            for frame in document['evidence']:
                frame['frame_sha256']=self.file(frame['frame_path'],rgb.getvalue())['sha256']
                path=f"{prefix}/masks/{frame['source_index']}.png";mask=self.file(path,binary.getvalue())
                frame['observations']=[{'track_id':spec[1][0],'mask_path':path,'mask_sha256':mask['sha256'],
                                        'gate_status':'candidate_geometric_gate_only','audit_status':'not_reviewed'}]
            if kind=='bottle':
                track=copy.deepcopy(document['tracklets'][0]);track.update(id='drinking_glass',label='drinking glass',candidate_entity_ids=['glass_candidate'])
                document['tracklets'].append(track)
                document['relationships'] += [
                    {'id':'cap_thread','subject_track_id':'bottle_cap','object_track_id':'bottle_body','predicate':'removable_from',
                     'status':'visual_model_proposal','rationale':'The cap unscrews until the thread disengages.',
                     'limitations':['Thread pitch is unknown.'],'evidence_ids':['source_000002']},
                    {'id':'cap_tether','subject_track_id':'bottle_cap','object_track_id':'bottle_body','predicate':'attached_to',
                     'status':'visual_model_proposal','rationale':'Cap remains connected by a flexible tether and retaining ring.',
                     'limitations':['Tether deformation is not recovered.'],'evidence_ids':['source_000002']},
                ]
            path=f'{prefix}/procedure/interaction.json';self.file(path,document);self.programs[kind]=path
            state_path=f'data/flow-runs/{run_id}/run.json'
            state={'id':run_id,'status':'complete_with_gaps','source':source,'artifacts':{'case':case,'interaction':path}}
            self.file(state_path,state);self.states[kind]=state_path
            captures.append({'id':kind,'run_id':'run-capture-bottle-20261103-v1'if kind=='bottle'else run_id,
                'source_ref':source,'owner_context':notes[kind],'owner_context_is_observation':False})
        self.file(B.BATCH_PATH,{'format':'interaction-twin-gpu-priority-capture-batch','captures':captures})

    def file(self,path,value):
        self.fixture.file(path,value);return S.file_ref(path,self.root)

    def test_complete_four_capture_batch_has_exact_refs_tether_reuse_and_only_whole_cable_traces(self):
        original={path:(self.root/path).read_bytes()for path in[*self.states.values(),*self.programs.values(),B.BATCH_PATH]}
        reference,value=B.build_batch(output_dir=self.output,root=self.root)
        self.assertEqual([r['capture_id']for r in value['records']],list(B.RUNS))
        self.assertEqual(B.build_batch(output_dir=self.output,root=self.root)[0],reference)
        catalog=S.load_catalog(value['relation_catalog_ref']['path'],self.root)
        self.assertEqual(len(catalog['possible_reuse']),1)
        self.assertFalse(catalog['possible_reuse'][0]['identity_verified'])
        records={r['capture_id']:r for r in value['records']}
        for kind in ['charger','laptop']:
            self.assertEqual([r['track_id']for r in records[kind]['visible_2d_cable_traces']],['power_cable'])
            trace=C.load(records[kind]['visible_2d_cable_traces'][0]['trace_ref']['path'],root=self.root)
            self.assertEqual(sum(s['status']=='observed_2d_path_candidate'for s in trace['samples']),3)
            self.assertFalse(trace['claims']['depth_recovered'])
        self.assertEqual(records['phone']['visible_2d_cable_traces'],[])
        phone=S.load(records['phone']['semantic_companion_ref']['path'],self.root)
        self.assertFalse(any(e['kind']=='flexible_cable'for e in phone['entities']))
        self.assertTrue(any(e['presence']=='owner_reported_offscreen'and not e['geometry']['track_assets']for e in phone['entities']))
        bottle=S.load(records['bottle']['semantic_companion_ref']['path'],self.root)
        self.assertTrue(any(r['kind']=='retained_tether'and r['provenance']['original_relation_id']=='cap_tether'for r in bottle['relations']))
        closure=next(r for r in bottle['relations']if r['kind']=='threaded_closure')
        self.assertEqual(closure['motion_contract']['stages'][-1],'retained_by_flexible_tether')
        self.assertTrue(any(r['kind']=='pour_transfer'for r in bottle['relations']))
        self.assertTrue(all(e['geometry']['metric_pose']is None for e in catalog['entities']))
        self.assertFalse(value['publication_performed']);self.assertFalse(value['database_changed'])
        for path,raw in original.items():self.assertEqual((self.root/path).read_bytes(),raw)

    def test_incomplete_foreign_or_old_bottle_state_fails_before_outputs(self):
        state=json.loads((self.root/self.states['laptop']).read_text());state['status']='running';self.file(self.states['laptop'],state)
        with self.assertRaisesRegex(ValueError,'has not completed'):B.build_batch(output_dir=self.output,root=self.root)
        self.assertFalse((self.root/self.output).exists())
        state['status']='complete';self.file(self.states['laptop'],state)
        bottle=json.loads((self.root/self.states['bottle']).read_text());bottle['artifacts']['interaction']=self.programs['bottle'].replace('bottle-consistency-20261103-v2','bottle-20261103-v1')
        self.file(self.states['bottle'],bottle)
        with self.assertRaisesRegex(ValueError,'obsolete'):B.build_batch(output_dir=self.output,root=self.root)
        self.assertFalse((self.root/self.output).exists())

    def test_wrong_source_and_wrong_run_id_are_not_adopted(self):
        state=json.loads((self.root/self.states['phone']).read_text());state['id']='foreign_run';self.file(self.states['phone'],state)
        with self.assertRaises(ValueError):B.build_batch(capture_ids=['phone'],output_dir=self.output,root=self.root)
        state['id']=B.RUNS['phone'];self.file(self.states['phone'],state)
        batch=json.loads((self.root/B.BATCH_PATH).read_text());batch['captures'][1]['source_ref']=batch['captures'][0]['source_ref'];self.file(B.BATCH_PATH,batch)
        with self.assertRaisesRegex(ValueError,'source'):B.build_batch(capture_ids=['phone'],output_dir=self.output,root=self.root)

    def test_foreign_run_state_source_fails_even_if_program_and_batch_source_match(self):
        original=json.loads((self.root/self.states['phone']).read_text())
        for field,value in [('sha256','0'*64),('bytes',original['source']['bytes']+1),('path','data/foreign.mp4')]:
            changed=copy.deepcopy(original);changed['source'][field]=value;self.file(self.states['phone'],changed)
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'state source'):
                B.build_batch(capture_ids=['phone'],output_dir=self.output,root=self.root)
            self.assertFalse((self.root/self.output).exists())
        self.file(self.states['phone'],original)

    def test_skipping_optional_traces_does_not_drop_semantics_or_promote_models(self):
        _,value=B.build_batch(capture_ids=['phone','charger'],output_dir=self.output,trace_cables=False,root=self.root)
        self.assertEqual([r['capture_id']for r in value['records']],['charger','phone'])
        self.assertTrue(all(not r['visible_2d_cable_traces']for r in value['records']))
        self.assertEqual(value['new_model_calls'],0)
        self.assertFalse(value['claims']['physical_identity_verified'])
        self.assertTrue(value['records'][0]['untraced_track_reasons'])
        for captures in [[],['bottle_v1'],['phone','phone']]:
            with self.assertRaises(ValueError):B.build_batch(capture_ids=captures,root=self.root)

    def test_branch_selection_uses_actual_labels_not_filenames_or_plug_ids(self):
        self.assertTrue(B._whole_cable({'label':'white cable','id':'power_cable'}))
        self.assertFalse(B._whole_cable({'label':'cable connector plug','id':'power_cable'}))
        self.assertFalse(B._whole_cable({'label':'tiny white object','id':'power_cable'}))
        self.assertTrue(B._whole_cable({'label':'flexible power cord','id':'cord'}))


if __name__=='__main__':unittest.main()
