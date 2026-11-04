"""Exact visual packet/triangle/source/budget gates; provider stays offline."""
import copy,io,json,shutil,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image,ImageDraw

import generation_visual_judge as J
import semantic_asset_companion as S
import part_ledger as L
from tests import test_twin_catalog as fixtures
from tests import test_native_appearance as glb_fixture


class OfflineClient:
    calls=[]
    failure=False
    def __init__(self,**kwargs):self.root=kwargs['root']
    def json_result(self,prompt,schema,images,*,output,model,max_output_tokens):
        self.calls.append({'prompt':prompt,'images':images,'model':model,'tokens':max_output_tokens})
        if self.failure:raise RuntimeError('offline transport uncertainty')
        ctx=json.loads(prompt.split('CONTEXT (data):\n')[1]);refs=[S.file_ref(p,self.root)for p in images]
        wire=J.V.wire_provenance(prompt,schema,refs,self.root)
        result={'format':J.FORMAT+'-result','version':1,'component_id':ctx['component_id'],'verdict':'uncertain',
            'verdict_scope':'sampled_visible_appearance_only','summary':'Visible evidence needs owner inspection.',
            'evidence':[{'evidence_ids':[ctx['anchor']['id'],'orbit:0'],'observation':'The source and rendered shape are visible.'}],
            'issues':[],'anchor_suggestions':[{'action':'retain_unknown','candidate_evidence_id':'none','reason':'Do not infer hidden surfaces.','needed_visibility':'A clearer opposite view is needed.'}],
            'limitations':['This is not physical validation.'],**J.FLAGS}
        receipt={'status':'complete_accounted','provider':'meta_api','model':model,'automatic_retries':False,
            'payload_sha256':wire['payload_sha256'],'image_inputs':wire['encoded_image_inputs'],'estimated_cost_usd':'0.001000','fixture_only':True}
        output.mkdir(parents=True);(output/'meta-request.json').write_text(json.dumps(receipt));(output/'response.pending.json').write_text(json.dumps(result));return result,receipt


class GenerationVisualJudgeTests(unittest.TestCase):
    def setUp(self):
        fixture=fixtures.TwinCatalogTests(methodName='runTest');fixture.setUp();self.addCleanup(fixture.doCleanups)
        self.fixture=fixture;self.root=fixture.root.resolve();self.program='data/case/procedure/interaction.json';self.packet='output/judge/first'
        self.metadata='data/case/native/metadata.json'
        for e in fixture.doc['evidence']:
            rgb=Image.new('RGB',(120,100),(40+e['source_index']*20,80,100));buffer=io.BytesIO();rgb.save(buffer,format='PNG');e['frame_sha256']=fixture.file(e['frame_path'],buffer.getvalue())
            image=Image.new('L',(120,100));ImageDraw.Draw(image).rectangle((20,25,80+e['source_index'],75),fill=255);buffer=io.BytesIO();image.save(buffer,format='PNG')
            path=f'data/case/masks/{e["source_index"]}.png';sha=fixture.file(path,buffer.getvalue());e['observations']=[{'track_id':'part','mask_path':path,'mask_sha256':sha,'gate_status':'candidate_geometric_gate_only','audit_status':'not_reviewed'}]
        raw=glb_fixture.fixture();asset=S.file_ref(self.write('data/case/native/asset.glb',raw),self.root)
        fixture.doc['assets']=[{'id':'asset_1','track_id':'part','path':asset['path'],'sha256':asset['sha256'],'backend':'Meta SAM 3D Objects','geometry_status':'unreviewed_model_candidate','physical_scale_known':False}]
        source=fixture.doc['evidence'][1];self.write('data/case/native/raw/image.png',(self.root/source['frame_path']).read_bytes());self.write('data/case/native/raw/mask.png',(self.root/source['observations'][0]['mask_path']).read_bytes())
        self.md={'backend':'Meta SAM 3D Objects','object_id':'part','sam3d_output':True,'learned_inference_executed':True,'manual_geometry_fallback':False,'manual_pose_used':False,
            'source':{'video_sha256':fixture.doc['source']['sha256'],'source_sample_index':2,'timestamp_s':1.,'image_path':'raw/image.png','mask_path':'raw/mask.png',
                'image_sha256':S.file_ref('data/case/native/raw/image.png',self.root)['sha256'],'mask_sha256':S.file_ref('data/case/native/raw/mask.png',self.root)['sha256']},
            'artifact_files':[{'file':'asset.glb','sha256':asset['sha256'],'bytes':asset['bytes']}]}
        tracks={'tracks':[{'track_id':t['id'],'frames':[{'status':'candidate_visible'}]}for t in fixture.doc['tracklets']]}
        for identifier,path,kind,value in [('mask_tracks','data/case/segment/tracks.json','actual_sam_masks',tracks),('mask_quality','data/case/segment/frames.json','geometric_mask_gates',{'frames':[]})]:
            sha=fixture.file(path,value);fixture.doc['inputs'].append({'id':identifier,'path':path,'sha256':sha,'kind':kind})
        self.publish()
        budget=self.root/J.PLAN;budget.parent.mkdir(parents=True,exist_ok=True);budget.write_bytes((J.ROOT/J.PLAN).read_bytes())
        asset=fixture.doc['assets'][0];qa={'format':'interaction-twin-private-native-visual-quality-notes','records':[{'part_id':'part','asset':S.file_ref(asset['path'],self.root),'finding':'Prior fallible slab artifact claim.','quality_note_kind':'major_visual_artifact'}]}
        self.qa='output/agent-qa.json';self.write(self.qa,json.dumps(qa).encode());OfflineClient.calls=[];OfflineClient.failure=False

    def write(self,path,raw):
        self.fixture.file(path,raw);return self.root/path

    def publish(self):
        sha=self.fixture.file(self.metadata,self.md)
        existing=next((i for i in self.fixture.doc['inputs']if i['kind']=='learned_asset_metadata'),None)
        if existing:existing['sha256']=sha
        else:self.fixture.doc['inputs'].append({'id':'native_metadata','path':self.metadata,'sha256':sha,'kind':'learned_asset_metadata'})
        self.fixture.publish();self.ledger=L.build(self.root/self.program,root=self.root)['path']

    def prepare(self,output=None):return J.prepare(self.program,self.ledger,'part',output or self.packet,agent_qa_path=self.qa,root=self.root)
    def execute(self,output=None,**kwargs):
        output=output or self.packet
        with patch.object(J,'budget_status',return_value={'can_admit':True}),patch.object(J,'OvernightMetaClient',OfflineClient):
            return J.run_prepared(output,root=self.root,allow_paid_api=True,authorized_plan_sha256=S.file_ref(Path(output)/'plan.json',self.root)['sha256'],**kwargs)

    def test_offline_packet_keeps_original_pixels_full_native_faces_and_blind_agent_comparison(self):
        with patch.object(J,'OvernightMetaClient',side_effect=AssertionError('No client during preparation')):
            result=self.prepare();plan,prompt,_=J.verify_plan(self.packet,root=self.root)
        self.assertEqual(result['new_api_calls'],0);self.assertEqual(result['images'],7)
        self.assertNotIn('Prior fallible slab artifact claim',prompt)
        self.assertEqual(plan['saved_agent_qa']['finding'],'Prior fallible slab artifact claim.')
        render=json.loads((self.root/self.packet/'render-receipt.json').read_text());mesh=J.A.load_native_glb((self.root/'data/case/native/asset.glb').read_bytes())
        self.assertEqual(render['full_triangle_count'],len(mesh.faces));self.assertFalse(render['geometry_decimated']);self.assertFalse(render['convex_hull_used'])
        self.assertFalse((self.root/'.state/generation-visual-judge-attempts').exists())

    def test_foreign_source_pixels_fail_before_packet_writes_even_with_valid_rebound_ledger(self):
        frame=self.fixture.doc['evidence'][0];self.write('data/case/native/raw/image.png',(self.root/frame['frame_path']).read_bytes());self.md['source']['image_sha256']=S.file_ref('data/case/native/raw/image.png',self.root)['sha256'];self.publish()
        with self.assertRaisesRegex(ValueError,'pixels'):self.prepare()
        self.assertFalse((self.root/self.packet).exists())

    def test_foreign_pts_and_non_native_components_are_not_eligible(self):
        self.md['source']['timestamp_s']=2.;self.publish()
        with self.assertRaisesRegex(ValueError,'source PTS'):self.prepare()
        with self.assertRaisesRegex(ValueError,'compiled native'):J.prepare(self.program,self.ledger,'holder',self.packet,root=self.root)

    def test_two_pixel_mask_is_not_proposed_as_a_better_source_anchor(self):
        frame=self.fixture.doc['evidence'][0];mask=np.zeros((100,120),dtype=np.uint8);mask[10,10:12]=255
        output=io.BytesIO();Image.fromarray(mask).save(output,format='PNG');observation=frame['observations'][0]
        observation['mask_sha256']=self.fixture.file(observation['mask_path'],output.getvalue());self.publish()
        self.prepare();plan,_,_=J.verify_plan(self.packet,root=self.root)
        self.assertNotIn('part:1',[c['id']for c in plan['context']['candidate_anchors']])
        self.assertEqual(plan['context']['candidate_eligibility']['too_small_masks_excluded'],1)
        self.assertEqual(len(plan['images']),5)

    def test_exact_approval_and_budget_are_required_before_client_or_attempt_claim(self):
        self.prepare()
        with patch.object(J,'OvernightMetaClient',side_effect=AssertionError('Must not construct client')):
            with self.assertRaisesRegex(ValueError,'disabled'):J.run_prepared(self.packet,root=self.root)
            with self.assertRaisesRegex(ValueError,'SHA authorization'):J.run_prepared(self.packet,root=self.root,allow_paid_api=True,authorized_plan_sha256='0'*64)
            with patch.object(J,'budget_status',return_value={'can_admit':False}),self.assertRaisesRegex(ValueError,'budget cannot admit'):
                J.run_prepared(self.packet,root=self.root,allow_paid_api=True,authorized_plan_sha256=S.file_ref(Path(self.packet)/'plan.json',self.root)['sha256'])
        self.assertFalse((self.root/'.state/generation-visual-judge-attempts').exists())

    def test_changed_upload_image_or_prompt_blocks_without_transport(self):
        self.prepare();plan=json.loads((self.root/self.packet/'plan.json').read_text());image=self.root/plan['images'][0]['path'];image.write_bytes(b'changed')
        with self.assertRaises(ValueError):self.execute()
        self.assertEqual(OfflineClient.calls,[])

    def test_accounted_report_roundtrip_and_namespace_prevents_changed_output_retry(self):
        self.prepare();report=self.execute();path=self.root/self.packet/'attempt/report.json'
        self.assertEqual(J.verify_report(path,root=self.root)['verdict'],'uncertain');self.assertFalse(report['automatic_approval'])
        self.prepare('output/judge/second')
        with self.assertRaisesRegex(ValueError,'already attempted'):self.execute('output/judge/second')
        self.assertEqual(len(OfflineClient.calls),1)

    def test_transport_uncertainty_is_preserved_and_never_resent(self):
        self.prepare();OfflineClient.failure=True
        with self.assertRaisesRegex(RuntimeError,'uncertainty'):self.execute()
        self.assertTrue((self.root/self.packet/'attempt/failure.json').is_file())
        with self.assertRaisesRegex(ValueError,'already attempted'):self.execute()
        self.assertEqual(len(OfflineClient.calls),1)

    def test_invented_evidence_anchor_geometry_and_contradictory_faithful_verdict_rejected(self):
        self.prepare();report=self.execute();result=json.loads((self.root/self.packet/'attempt/model-result.json').read_text());plan,_,_=J.verify_plan(self.packet,root=self.root)
        changed=copy.deepcopy(result);changed['evidence'][0]['evidence_ids']=['foreign:999']
        with self.assertRaisesRegex(ValueError,'unsupplied'):J.validate_result(changed,plan)
        changed=copy.deepcopy(result);changed['anchor_suggestions'][0].update(action='recommend_different_anchor',candidate_evidence_id='part:999')
        with self.assertRaisesRegex(ValueError,'invented'):J.validate_result(changed,plan)
        changed=copy.deepcopy(result);changed['geometry_modified']=True
        with self.assertRaises(Exception):J.validate_result(changed,plan)
        changed=copy.deepcopy(result);changed['verdict']='faithful'
        with self.assertRaisesRegex(ValueError,'contradicts'):J.validate_result(changed,plan)

    def test_agent_notes_cannot_be_replaced_without_invalidating_prepared_comparison(self):
        self.prepare();qa=json.loads((self.root/self.qa).read_text());qa['records'][0]['finding']='changed advice';self.write(self.qa,json.dumps(qa).encode())
        with self.assertRaisesRegex(ValueError,'QA changed'):J.verify_plan(self.packet,root=self.root)

    def test_rehashed_attempt_namespace_cannot_bypass_once_only_identity(self):
        self.prepare();path=self.root/self.packet/'plan.json';plan=json.loads(path.read_text());plan['attempt_namespace']='0'*64
        plan['request_key']=J._digest({k:v for k,v in plan.items()if k!='request_key'});path.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError,'namespace changed'):self.execute()
        self.assertEqual(OfflineClient.calls,[])


if __name__=='__main__':unittest.main()
