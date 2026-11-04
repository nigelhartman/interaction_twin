"""Synthetic parser/software fixtures only; no API, credentials or native models."""
import copy,json,shutil,tempfile,unittest
from pathlib import Path
from unittest import mock
import numpy as np
from PIL import Image
import hand_occlusion_evidence as H
from tests.test_meta_sam_api import response,PARSER_PRESENT


class OccluderMathTests(unittest.TestCase):
    def test_confidence_size_and_whole_frame_masks_fail_closed(self):
        mask=np.zeros((20,20),bool);mask[2:10,2:10]=True
        self.assertTrue(H.gate_instance(mask,.9)['eligible_possible_occluder'])
        for value in [None,float('nan'),.5,True]:self.assertFalse(H.gate_instance(mask,value)['eligible_possible_occluder'])
        self.assertFalse(H.gate_instance(np.ones((20,20),bool),.99)['eligible_possible_occluder'])
        self.assertFalse(H.gate_instance(np.eye(20,dtype=bool),.99)['eligible_possible_occluder'])

    def test_combined_large_union_is_rejected_without_erasing_original(self):
        first=np.zeros((20,20),bool);first[:10]=True;second=~first
        raw,eligible,gate=H.hand_unions([first,second],[H.gate_instance(first,.9),H.gate_instance(second,.9)],first.shape)
        self.assertTrue(raw.all());self.assertFalse(eligible.any());self.assertFalse(gate['eligible_possible_occluder'])

    def test_only_render_only_disagreement_can_be_discounted_and_raw_is_retained(self):
        target=np.zeros((20,20),bool);target[1:11,1:11]=True;render=target.copy();render[13:17,13:17]=True;hand=np.zeros_like(target);hand[13:17,13:17]=True
        before=render.copy();row=H.accounted_metrics(target,render,hand)
        self.assertAlmostEqual(row['raw_mask_iou'],100/116);self.assertEqual(row['hand_accounted_mask_iou'],1.);self.assertEqual(row['discounted_render_pixels'],16)
        self.assertTrue(row['discount_applied']);np.testing.assert_array_equal(render,before);self.assertFalse(row['hand_in_front_verified'])
        hand[1:5,1:5]=True;conflict=H.accounted_metrics(target,render,hand)
        self.assertFalse(conflict['discount_applied']);self.assertIn('hand_target_segmentation_conflict',conflict['gate_reasons'])
        self.assertEqual(conflict['raw_mask_iou'],conflict['hand_accounted_mask_iou'])

    def test_unknown_heavily_occluded_and_zero_hand_evidence_cannot_fake_a_gain(self):
        target=np.zeros((20,20),bool);target[1:3,1:3]=True;render=np.zeros_like(target);render[1:11,1:11]=True;hand=render&~target
        row=H.accounted_metrics(target,render,hand);self.assertFalse(row['discount_applied']);self.assertIn('render_too_occluded_for_alignment_selection',row['gate_reasons'])
        row=H.accounted_metrics(render,render,np.zeros_like(render));self.assertFalse(row['discount_applied']);self.assertTrue(row['accounting_eligible'])
        row=H.accounted_metrics(render,render,np.zeros_like(render),source_eligible=False);self.assertFalse(row['accounting_eligible'])


@unittest.skipUnless(PARSER_PRESENT,'Official SAM parser required')
class HandPacketTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve()
        for name in H.CODE:
            target=self.root/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(H.ROOT/name,target)
        budget=self.root/H.BUDGET_PLAN;budget.parent.mkdir(parents=True);budget.write_bytes((H.ROOT/H.BUDGET_PLAN).read_bytes())
        self.folder=self.root/'data/native-render-audits/unit';self.folder.mkdir(parents=True)
        video=self.root/'data/source.mp4';video.write_bytes(b'synthetic-source-recording');source={'path':'data/source.mp4','sha256':H.sha(video),'time_domain':H.SAM.TIME_DOMAIN}
        evidence=[];samples=[]
        for i in range(1,7):
            rgb=self.folder/f'{i}.png';Image.fromarray(np.full((12,10,3),150+i,np.uint8)).save(rgb)
            mask=self.folder/f'{i}-mask.png';pixels=np.zeros((12,10),np.uint8);pixels[1:5,1:5]=255;Image.fromarray(pixels).save(mask)
            render=self.folder/f'{i}-render.png';Image.fromarray(pixels).save(render)
            evidence.append({'source_index':i,'timestamp_s':(i-1)*.3,'frame_path':H.ref(rgb,self.root)['path'],'frame_sha256':H.sha(rgb),
                'observations':[{'track_id':'body','mask_path':H.ref(mask,self.root)['path'],'mask_sha256':H.sha(mask),'gate_status':'candidate_geometric_gate_only','audit_status':'not_model_audited'}]})
            samples.append({'source_index':i,'timestamp_s':(i-1)*.3,'role':'final_test'if i==3 else'development','component_id':'body','sample_id':f'body:{i}',
                'render_mask_ref':H.ref(render,self.root),'comparison_panel_ref':H.ref(rgb,self.root),'status':'diagnostic_available'})
        program=self.root/'data/program.json';H.write_new(program,{'source':source,'evidence':evidence});scene=self.root/'data/scene.json'
        H.write_new(scene,{'source':{**source,'height':12,'width':10},'original_program':H.ref(program,self.root),'components':[{'id':'body'}],'evaluation_protocol':{'held_out_sample_indices':[3]}})
        self.audit={'scene_ref':H.ref(scene,self.root),'interaction_ref':H.ref(program,self.root),'samples':samples};self.audit_path=self.folder/'audit.json';H.write_new(self.audit_path,self.audit)
        patcher=mock.patch.object(H.A,'validate_manifest',return_value=self.audit);patcher.start();self.addCleanup(patcher.stop)
        self.prepared='.private/hand-occlusion/test';self.selected={'training_samples':[1],'development_samples':[2,4,5,6],'root':self.root}

    def prepare(self,output=None):return H.prepare(self.audit_path,output or self.prepared,**self.selected)

    def fake_client(self,*,fail=False):
        client=mock.Mock(spec=H.OvernightMetaClient);self.calls=[]
        def callback(output,*,images=0,video_frames=0):
            self.assertEqual(images,1);self.assertEqual(video_frames,0)
            def request(payload):
                self.calls.append(copy.deepcopy(payload))
                if fail:raise RuntimeError('Synthetic uncertain failure')
                output.mkdir(parents=True);document=response();H.write_new(output/'sam-response.pending.json',document)
                H.write_new(output/'meta-request.json',{'status':'complete_accounted','model':H.SAM.MODEL,'kind':'sam','payload_sha256':H.digest(payload),'automatic_retries':False})
                return document
            return request
        client.sam_callback.side_effect=callback;return client

    def execute(self,metadata,client):
        return H.run_prepared(metadata['prepared'],root=self.root,client=client,allow_paid_api=True,authorized_plan_sha256=metadata['plan_ref']['sha256'])

    def test_prepare_is_offline_pins_original_bytes_and_never_decodes_final_frames(self):
        original=Image.open
        def guarded(path,*args,**kwargs):
            self.assertFalse(Path(path).name in ['3.png','3-mask.png'],'Final test pixels entered hand preparation');return original(path,*args,**kwargs)
        with mock.patch.object(H,'OvernightMetaClient')as constructor,mock.patch.object(Image,'open',side_effect=guarded):metadata=self.prepare()
        constructor.assert_not_called();self.assertEqual(metadata['new_api_calls'],0);self.assertEqual(metadata['estimated_cost_usd'],'0.012500')
        _,plan=H.verified_plan(metadata['prepared'],root=self.root);self.assertEqual([f['source']['sample_index']for f in plan['context']['frames']],[1,2,4,5,6])
        for frame in plan['context']['frames']:self.assertEqual(frame['wire_payload_sha256'],H.digest(H.SAM.build_payload('hand',{'type':'input_image','image_url':H.SAM._data_url(self.root/frame['rgb_ref']['path'],'image')})))

    def test_no_authorization_or_changed_prepared_source_never_calls_transport(self):
        metadata=self.prepare();client=self.fake_client()
        for kwargs in [{},{'allow_paid_api':True,'authorized_plan_sha256':'a'*64}]:
            with self.assertRaisesRegex(ValueError,'authorization'):H.run_prepared(metadata['prepared'],root=self.root,client=client,**kwargs)
        self.assertEqual(self.calls,[]);self.assertFalse((self.root/H.ATTEMPTS).exists())
        frame=self.folder/'1.png';frame.write_bytes(b'changed-source')
        with self.assertRaisesRegex(ValueError,'RGB differs'):self.execute(metadata,client)
        self.assertEqual(self.calls,[])

    def test_final_test_selection_rejected_before_prepared_folder_or_client(self):
        with self.assertRaisesRegex(ValueError,'Final-test'):H.prepare(self.audit_path,self.prepared,training_samples=[3],development_samples=[2,4,5,6],root=self.root)
        self.assertFalse((self.root/self.prepared).exists())

    def test_actual_codec_five_calls_preserves_source_and_raw_native_masks(self):
        metadata=self.prepare();client=self.fake_client();original={p:p.read_bytes()for p in self.folder.glob('*.png')};report=self.execute(metadata,client)
        self.assertEqual(len(self.calls),5);document=H.validate_evidence(report['evidence_ref']['path'],root=self.root)
        self.assertEqual(len(document['frames']),5);self.assertFalse(document['asset_identity_created']);self.assertFalse(document['native_geometry_or_poses_modified'])
        self.assertTrue(all(p.read_bytes()==raw for p,raw in original.items()));comparison=H.read(self.root/report['comparison_ref']['path'])
        self.assertEqual([r['source_index']for r in comparison['samples']],[1,2,4,5,6]);self.assertFalse(comparison['automatic_approval'])
        self.assertEqual([p['input'][0]['content'][0]['text']for p in self.calls],['hand']*5)
        for row in comparison['samples']:self.assertEqual(row['metrics']['raw_mask_iou'],1.)
        metadata2=self.prepare('.private/hand-occlusion/another-output');client2=self.fake_client()
        with self.assertRaisesRegex(ValueError,'already attempted'):self.execute(metadata2,client2)
        self.assertEqual(self.calls,[])

    def test_uncertain_failure_is_preserved_and_cannot_be_retried(self):
        metadata=self.prepare();client=self.fake_client(fail=True)
        with self.assertRaises(RuntimeError):self.execute(metadata,client)
        self.assertEqual(len(self.calls),1);failure=H.read(self.root/metadata['prepared']/'attempt/failure.json');self.assertEqual(failure['status'],'preserved_no_automatic_retry')
        with self.assertRaisesRegex(ValueError,'already attempted'):self.execute(metadata,client)
        self.assertEqual(len(self.calls),1)

    def test_changed_mask_cannot_pass_by_updating_its_hash(self):
        metadata=self.prepare();report=self.execute(metadata,self.fake_client());path=self.root/report['evidence_ref']['path'];document=H.read(path);first=document['frames'][0]['instances'][0]
        mask=self.root/first['mask_ref']['path'];Image.fromarray(np.zeros((12,10),np.uint8)).save(mask);first['mask_ref']=H.ref(mask,self.root)
        path.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError,'original SAM response'):H.validate_evidence(path,root=self.root)

    def test_changed_accounted_response_is_not_accepted_as_decoded_hand_evidence(self):
        metadata=self.prepare();report=self.execute(metadata,self.fake_client());document=H.read(self.root/report['evidence_ref']['path']);item=document['frames'][0]
        path=self.root/item['accounted_sam_response_ref']['path'];path.write_text(json.dumps(response('')))
        with self.assertRaisesRegex(ValueError,'bytes changed'):H.validate_evidence(report['evidence_ref']['path'],root=self.root)

    def test_prepared_code_and_portable_source_time_tamper_stop_before_dispatch(self):
        metadata=self.prepare();client=self.fake_client();path=self.root/metadata['prepared']/'source-000001/input.json';manifest=H.read(path);manifest['source']['timestamp_s']+=1;path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'Portable hand RGB'):self.execute(metadata,client)
        self.assertEqual(self.calls,[]);self.assertFalse((self.root/H.ATTEMPTS).exists())

    def test_changed_frozen_code_and_symlink_inputs_are_not_accepted(self):
        metadata=self.prepare();client=self.fake_client();snapshot=self.root/metadata['prepared']/'implementation/hand_occlusion_evidence.py';snapshot.write_text('changed code')
        with self.assertRaisesRegex(ValueError,'Frozen hand implementation'):self.execute(metadata,client)
        self.assertEqual(self.calls,[]);link=self.root/'data/linked';link.symlink_to(self.folder)
        with self.assertRaisesRegex(ValueError,'not confined'):H.local(self.root,'data/linked/1.png')


if __name__=='__main__':unittest.main()
