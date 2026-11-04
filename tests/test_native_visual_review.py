"""Offline visual packet contracts; no real provider, SSH or GPU is invoked."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

import native_visual_review as V
import overnight_budget as B


class OfflineClient:
    def __init__(self,root,*,failure=False,mutate=None):self.root=root;self.failure=failure;self.mutate=mutate;self.calls=[]
    def json_result(self,prompt,schema,images,*,output,model,max_output_tokens):
        self.calls.append({'prompt':prompt,'images':images,'tokens':max_output_tokens})
        if self.failure:raise RuntimeError('offline transport uncertainty fixture')
        context=json.loads(prompt.split('CONTEXT (data):\n')[1].split('\nPREVIOUS REVIEW')[0])
        role='proposer'if'You are the proposer'in prompt else'critic'
        result={'format':V.FORMAT+'-result','version':1,'role':role,'verdict':'review_required','summary':'Development pixels require owner inspection.',
            'issues':[],'repair_proposals':[],'limitations':['Geometry and physical assembly remain unverified.'],
            'visual_observation_performed':True,'independent_model_review':False,'final_test_evidence_used':False,
            'physical_accuracy_verified':False,'simulation_ready':False,'automatic_publication':False,'training_performed':False}
        if self.mutate:self.mutate(result,context)
        output.mkdir(parents=True)
        wire=V.wire_provenance(prompt,schema,context['images'],self.root)
        receipt={'status':'complete_accounted','provider':'meta_api','model':model,'kind':'text','automatic_retries':False,
            'image_inputs':wire['encoded_image_inputs'],'payload_sha256':wire['payload_sha256'],'estimated_cost_usd':'0.001000','usage':{'input_tokens':100,'output_tokens':100},'fixture_only':True}
        (output/'meta-request.json').write_text(json.dumps(receipt))
        return result,receipt


class NativeVisualReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve()
        self.audit_dir=self.root/'data/native-render-audits/fixture';self.audit_dir.mkdir(parents=True)
        for folder in ['data/case/frames','data/case/procedure','data/scene','output/overnight-pipeline-20261102']:(self.root/folder).mkdir(parents=True,exist_ok=True)
        baseline={'settled_usd':'1.000000','reserved_usd':'2.000000','per_request_cap_usd':'2.000000','request_count':2}
        budget={'version':1,'additional_estimated_api_ceiling_usd':'20.000000','budget_baseline':baseline,
            'uncertain_paid_requests_must_not_be_retried':True,'existing_uncertainty_holds_preserved':True,
            'previous_rejected_localization_request_excluded':True,'global_estimated_guard_usd':'100.000000',
            'maximum_repair_rounds':2,'maximum_total_rounds':3}
        (self.root/B.PLAN).write_text(json.dumps(budget))
        self.doc={'source':{'path':'data/case/video.mp4','sha256':'a'*64,'original_sha256':'a'*64,'time_domain':'source_video_presentation_seconds'},'tracklets':[{'id':'base','label':'Case base'},{'id':'lid','label':'Moving lid'}],'evidence':[]}
        self.samples=[]
        for part in ['base','lid']:
            for index in [1,2,3]:
                rgb=self.root/f'data/case/frames/{index}.png'
                if not rgb.exists():
                    Image.new('RGB',(32,48),(index*40,80,120)).save(rgb)
                    self.doc['evidence'].append({'source_index':index,'timestamp_s':float(index),'frame_path':rgb.relative_to(self.root).as_posix(),'frame_sha256':V.sha(rgb)})
                panel=self.audit_dir/f'{part}-{index}-panel.png';Image.new('RGB',(96,48),(100,index*40,100)).save(panel)
                self.samples.append({'component_id':part,'source_index':index,'timestamp_s':float(index),'role':'final_test'if index==3 else'development',
                    'source_rgb_ref':V.ref(rgb,self.root),'comparison_panel_ref':V.ref(panel,self.root),'metrics':{'mask_iou':.72,'visible_source_coverage':.8},'status':'retrospective_diagnostic'})
        self.program=self.root/'data/case/procedure/interaction.json';self.program.write_text(json.dumps(self.doc))
        self.scene=self.root/'data/scene/scene.json';self.scene.write_text(json.dumps({'source':{'sha256':'a'*64},'original_program':V.ref(self.program,self.root),'components':[{'id':'base'},{'id':'lid'}]}))
        self.audit={'format':'interaction-twin-native-render-audit','version':1,'source':{'sha256':'a'*64},
            'interaction_ref':V.ref(self.program,self.root),'scene_ref':V.ref(self.scene,self.root),'samples':self.samples,'inputs':[],
            'evaluation_protocol':{'mode':'retrospective_diagnostic'},'physical_accuracy_verified':False,'simulation_ready':False}
        self.audit_path=self.audit_dir/'audit.json';self.publish();self.prepared='.private/native-visual-reviews/first'
    def publish(self):self.audit_path.write_text(json.dumps(self.audit))
    def prepare(self,**kwargs):return V.prepare(self.audit_path,self.prepared,root=self.root,**kwargs)
    def execute(self,client,**kwargs):
        return V.run_prepared(self.prepared,root=self.root,client=client,allow_paid_api=True,
            authorized_plan_sha256=V.sha(self.root/self.prepared/'plan.json'),**kwargs)

    def test_offline_packet_contains_only_development_original_rgb_and_new_panels(self):
        prepared=self.prepare();self.assertEqual(prepared['new_api_calls'],0);self.assertEqual(prepared['image_count'],8)
        plan=json.loads((self.root/self.prepared/'plan.json').read_text());self.assertEqual(plan['max_output_tokens'],8000)
        self.assertFalse(plan['final_test_evidence_used']);self.assertFalse(plan['old_rejected_localization_payload_used'])
        self.assertTrue(all(r['role']=='development'for r in plan['context']['samples']))
        self.assertNotIn('3.png',(self.root/self.prepared/'prompt.txt').read_text())
        self.assertFalse((self.root/'.private/native-visual-attempts').exists())
        client=OfflineClient(self.root)
        with self.assertRaisesRegex(ValueError,'disabled'):V.run_prepared(self.prepared,root=self.root,client=client)
        self.assertEqual(client.calls,[])
    def test_exact_plan_hash_required_before_upload_or_attempt_claim(self):
        self.prepare();client=OfflineClient(self.root)
        with self.assertRaisesRegex(ValueError,'Exact prepared'):V.run_prepared(self.prepared,root=self.root,client=client,allow_paid_api=True,authorized_plan_sha256='f'*64)
        self.assertEqual(client.calls,[]);self.assertFalse((self.root/'.private/native-visual-attempts').exists())
    def test_successful_visual_report_keeps_accounted_image_provenance_and_cannot_repeat(self):
        self.prepare();client=OfflineClient(self.root);report=self.execute(client)
        self.assertTrue(V.verify_report(report,root=self.root));self.assertTrue(report['visual_observation_performed'])
        self.assertFalse(report['independent_model_review']);self.assertFalse(report['automatic_publication']);self.assertTrue(report['human_approval_required'])
        self.assertEqual(len(client.calls),1)
        with self.assertRaisesRegex(ValueError,'already attempted'):self.execute(client)
        self.assertEqual(len(client.calls),1)
    def test_transport_uncertainty_remains_once_only_even_under_another_output_folder(self):
        self.prepare();client=OfflineClient(self.root,failure=True)
        with self.assertRaises(RuntimeError):self.execute(client)
        self.assertTrue((self.root/self.prepared/'attempt/failure.json').is_file())
        second='.private/native-visual-reviews/changed-output';V.prepare(self.audit_path,second,root=self.root)
        with self.assertRaisesRegex(ValueError,'already attempted'):V.run_prepared(second,root=self.root,client=client,allow_paid_api=True,authorized_plan_sha256=V.sha(self.root/second/'plan.json'))
        self.assertEqual(len(client.calls),1)
    def test_frozen_source_panel_or_prompt_mutation_blocks_without_a_send(self):
        self.prepare();client=OfflineClient(self.root)
        image=self.root/self.samples[0]['comparison_panel_ref']['path'];image.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'bytes changed'):self.execute(client)
        self.assertEqual(client.calls,[])
    def test_final_test_only_audit_and_old_localization_crop_are_refused(self):
        for row in self.audit['samples']:row['role']='final_test'
        self.publish()
        with self.assertRaisesRegex(ValueError,'No development'):self.prepare()
        self.audit['samples'][0]['role']='development';crop=self.root/'data/old-localization-crop.png';Image.new('RGB',(32,32)).save(crop)
        self.audit['samples'][0]['source_rgb_ref']=V.ref(crop,self.root);self.publish()
        with self.assertRaisesRegex(ValueError,'original exact saved source frame'):self.prepare()
    def test_fabricated_component_or_sample_citation_is_preserved_but_never_adopted(self):
        def mutate(result,context):
            result['issues']=[{'category':'pose_alignment','severity':'blocking','component_ids':['invented'],'sample_ids':[context['samples'][0]['sample_id']],
                'visible_observation':'Bad pose','why_it_matters':'Cannot align','confidence':'low'}]
        self.prepare();client=OfflineClient(self.root,mutate=mutate)
        with self.assertRaisesRegex(ValueError,'unknown submitted'):self.execute(client)
        self.assertTrue((self.root/self.prepared/'attempt/model-result.json').is_file());self.assertFalse((self.root/self.prepared/'attempt/report.json').exists())
        with self.assertRaisesRegex(ValueError,'already attempted'):self.execute(client)
        self.assertEqual(len(client.calls),1)
    def test_proposer_is_a_separate_same_model_prompt_with_an_exact_critic_report(self):
        self.prepare();client=OfflineClient(self.root);critic=self.execute(client)
        report_path=self.root/self.prepared/'attempt/report.json'
        second='.private/native-visual-reviews/proposer';V.prepare(self.audit_path,second,root=self.root,role='proposer',previous_report=report_path)
        report=V.run_prepared(second,root=self.root,allow_paid_api=True,client=client,authorized_plan_sha256=V.sha(self.root/second/'plan.json'))
        self.assertEqual(report['role'],'proposer');self.assertFalse(report['independent_model_review']);self.assertEqual(len(client.calls),2)
        self.assertIn('PREVIOUS REVIEW',client.calls[-1]['prompt'])
    def test_admin_adapter_is_inert_without_exact_root_authorization_and_fences_dispatch(self):
        disabled=V.build_admin_reviewer({},root=self.root);self.assertEqual(disabled({})['status'],'unavailable')
        self.prepare();client=OfflineClient(self.root);fences=[]
        job={'authorized_visual_review':{'prepared':self.prepared,'authorized_plan_sha256':V.sha(self.root/self.prepared/'plan.json')}}
        manifest={'source_ref':{'sha256':'a'*64},'interaction_ref':self.audit['interaction_ref'],'scene_ref':self.audit['scene_ref'],'object_program_ref':None}
        adapter=V.build_admin_reviewer(job,root=self.root,lease_guard=lambda:fences.append('live'),client=client)
        report=adapter(manifest);self.assertEqual(report['status'],'complete');self.assertFalse(report['automatic_repair_safe']);self.assertGreaterEqual(len(fences),3)
        self.assertEqual(len(client.calls),1);again=adapter(manifest);self.assertEqual(again['report_ref'],report['report_ref']);self.assertEqual(len(client.calls),1)
        bad=copy.deepcopy(manifest);bad['scene_ref']['sha256']='f'*64
        with self.assertRaisesRegex(ValueError,'another candidate'):adapter(bad)
    def test_private_prepared_child_symlink_is_rejected_before_read_or_attempt(self):
        self.prepare();client=OfflineClient(self.root);prompt=self.root/self.prepared/'prompt.txt'
        prompt.unlink();prompt.symlink_to(self.program)
        with self.assertRaisesRegex(ValueError,'not confined'):self.execute(client)
        self.assertEqual(client.calls,[])
        self.assertFalse((self.root/'.private/native-visual-attempts').exists())
    def test_mutated_report_policy_or_provider_wire_receipt_cannot_be_reused(self):
        self.prepare();client=OfflineClient(self.root);report=self.execute(client)
        changed=copy.deepcopy(report);changed['human_approval_required']=False
        with self.assertRaisesRegex(ValueError,'changes its policy'):V.verify_report(changed,root=self.root)
        changed=copy.deepcopy(report);changed['source_sha256']='f'*64
        with self.assertRaisesRegex(ValueError,'changes its policy'):V.verify_report(changed,root=self.root)
        receipt_path=self.root/report['request_receipt']['path'];receipt=json.loads(receipt_path.read_text());receipt['payload_sha256']='f'*64;receipt_path.write_text(json.dumps(receipt))
        report['request_receipt']=V.ref(receipt_path,self.root)
        with self.assertRaisesRegex(ValueError,'provider provenance'):V.verify_report(report,root=self.root)
    def test_budget_limit_rounds_symlinks_and_nan_metrics_fail_closed(self):
        with self.assertRaisesRegex(ValueError,'role/round'):self.prepare(round_number=4)
        self.audit['samples'][0]['metrics']['mask_iou']=float('nan');self.publish()
        with self.assertRaisesRegex(ValueError,'finite numbers'):self.prepare()


class OvernightBudgetTests(unittest.TestCase):
    def setUp(self):
        self.fixture=NativeVisualReviewTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups);self.root=self.fixture.root
    def test_guard_counts_settled_and_uncertain_holds_without_resetting_them(self):
        status={'settled_usd':'2.000000','reserved_usd':'5.000000','per_request_cap_usd':'2.000000','request_count':9}
        value=B.budget_status(root=self.root,ledger_status=status);self.assertEqual(value['task_remaining_estimated_usd'],'16.000000');self.assertTrue(value['can_admit'])
        status['reserved_usd']='21.500000'
        with self.assertRaisesRegex(Exception,'overnight_task_estimate_limit'): 
            with B.pass_guard(root=self.root,status_reader=lambda:status):self.fail('Budget should block')
        self.assertEqual(status['reserved_usd'],'21.500000')
    def test_budget_and_dispatch_fence_precede_existing_increment_and_network(self):
        calls=[];status={'settled_usd':'1.000000','reserved_usd':'2.000000','per_request_cap_usd':'2.000000'}
        class Ledger:
            def status(self):return status
        with patch.object(B.IncrementMetaClient,'__init__',lambda instance,**kwargs:setattr(instance,'ledger',Ledger())),patch.object(B.IncrementMetaClient,'request',lambda instance,*a,**k:calls.append('existing_guard_and_transport')):
            client=B.OvernightMetaClient(root=self.root,overnight_plan=B.PLAN,dispatch_guard=lambda:calls.append('live_lease'))
            client.request({});self.assertEqual(calls,['live_lease','existing_guard_and_transport'])
            status['reserved_usd']='23.000000'
            with self.assertRaisesRegex(Exception,'overnight_task_estimate_limit'):client.request({})
            self.assertEqual(calls,['live_lease','existing_guard_and_transport'])
