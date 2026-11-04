"""Ordinary-camera HTTP/job/workflow boundaries; no inference is executed."""
import json
import shutil
import hashlib
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import observed_jobs as jobs
import observed_pipeline as pipeline
from tests.test_observed_jobs import JobFixture, ROOT
import tests.test_observed_pipeline as workflow_test_fixtures


class CameraWorkflowTests(JobFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        for name in ['observed_geometry.py','observed_stereo.py','observed_surface_validation.py',
                     'observed_surface_completion.py','observed_depth_completion.py',
                     'observed_surface_export.py','observed_completion_export.py','observed_pipeline.py',
                     'automatic_camera_prior.py']:
            shutil.copyfile(ROOT/name,self.root/name)
        shutil.copyfile(ROOT/'scripts/run_observed_workflow.py',self.root/'scripts/run_observed_workflow.py')
        self.extended=dict(self.body,workflow='gated-local-v1',feature_provider='sift',completion='none')

    def workflow(self,body=None):
        request=jobs.canonical_request(body or self.extended,root=self.root)
        output=self.root/pipeline.BASE/'observed-cccccccccccccccccccccccc-workflow'
        return pipeline.Workflow(self.root,request,output)

    def test_old_request_shape_remains_cpu_and_without_learned_camera(self):
        request=jobs.canonical_request(self.body,root=self.root)
        self.assertNotIn('camera_mode',request)
        runner=jobs.ObservedRunner(request,root=self.root)
        self.assertTrue(runner.state['policy']['cpu_only'])
        self.assertFalse(runner.state['policy']['model_inference'])
        self.assertNotIn('camera_prior_engine_sha256',runner.state)

    def test_explicit_assumed_mode_is_cpu_and_optional(self):
        default=jobs.canonical_request(self.extended,root=self.root)
        self.assertNotIn('camera_mode',default)
        request=jobs.canonical_request(dict(self.extended,camera_mode='assumed'),root=self.root)
        self.assertEqual(request['camera_mode'],'assumed')
        runner=jobs.ObservedRunner(request,root=self.root)
        self.assertTrue(runner.state['policy']['cpu_only'])
        self.assertFalse(runner.state['policy']['model_inference'])

    def test_new_camera_option_restricted_to_workflow_and_named_modes(self):
        with self.assertRaises(ValueError):
            jobs.canonical_request(dict(self.body,camera_mode='moge_prior'),root=self.root)
        for bad in ['measured','../../model','',None,True,42,{},[]]:
            with self.subTest(value=bad),self.assertRaises((ValueError,TypeError)):
                jobs.canonical_request(dict(self.extended,camera_mode=bad),root=self.root)

    def test_raw_camera_and_hidden_http_capabilities_rejected(self):
        for field,value in [('K',[[1,0,0],[0,1,0],[0,0,1]]),('camera_K',[]),('intrinsics',{}),
                ('focal_px',123.),('camera_provider','moge'),('camera_report_path','data/a.json'),
                ('camera_model_path','/tmp/model'),('device','cuda'),('source_path','data/a.mp4'),
                ('parent_worker_lock_held',True),('max_views',10000),('sensitivity',False)]:
            with self.subTest(field=field),self.assertRaises(ValueError):
                jobs.canonical_request(dict(self.extended,**{field:value}),root=self.root)

    def test_selected_learned_camera_sets_model_policy_and_pins_engine(self):
        request=jobs.canonical_request(dict(self.extended,camera_mode='moge_prior'),root=self.root)
        runner=jobs.ObservedRunner(request,root=self.root)
        self.assertFalse(runner.state['policy']['cpu_only'])
        self.assertTrue(runner.state['policy']['model_inference'])
        self.assertEqual(runner.state['camera_prior_engine_sha256'],pipeline.sha(self.root/'automatic_camera_prior.py'))
        self.assertEqual(runner.state['policy']['new_api_calls'],0)
        workflow=self.workflow(dict(self.extended,camera_mode='moge_prior'))
        option=json.loads((workflow.output/'camera-option.json').read_text())
        self.assertTrue(option['fresh_reconstruction_required'])
        self.assertFalse(option['physical_scale_known'])

    def test_new_camera_cannot_be_applied_to_reused_sparse_geometry(self):
        workflow_test_fixtures.WorkflowTests.make_sparse_cache(self)
        cache=pipeline.sparse_caches(self.root)[0]
        body=dict(self.extended,feature_provider='reuse_verified',reuse_sparse_id=cache['id'])
        self.assertEqual(jobs.canonical_request(body,root=self.root)['reuse_sparse_id'],cache['id'])
        with self.assertRaises(ValueError):
            jobs.canonical_request(dict(body,camera_mode='moge_prior'),root=self.root)

    def test_prior_code_mutation_before_launch_fails_without_inference(self):
        request=jobs.canonical_request(dict(self.extended,camera_mode='moge_prior'),root=self.root)
        runner=jobs.ObservedRunner(request,root=self.root)
        with (self.root/'automatic_camera_prior.py').open('a') as stream:stream.write('\n# synthetic changed engine\n')
        with patch.object(runner,'_launch') as launch:runner.execute()
        launch.assert_not_called()
        self.assertEqual(runner.state['status'],'failed')
        self.assertEqual(runner.state['failure_type'],'ValueError')
        self.assertFalse(runner.state['automatic_retry'])

    def test_registered_program_mutation_before_launch_fails_without_inference(self):
        request=jobs.canonical_request(dict(self.extended,camera_mode='moge_prior'),root=self.root)
        runner=jobs.ObservedRunner(request,root=self.root)
        doc=json.loads(self.source.read_text());doc['summary']='Changed synthetic source program'
        self.source.write_text(json.dumps(doc))
        with patch.object(runner,'_launch') as launch:runner.execute()
        launch.assert_not_called()
        self.assertEqual(runner.state['status'],'failed')
        self.assertFalse(runner.state['automatic_retry'])

    def test_fresh_sift_reconstruction_receives_provider_no_old_poses_and_sensitivity_disabled(self):
        workflow=self.workflow(dict(self.extended,camera_mode='moge_prior'))
        provider=object()
        with patch('torch.backends.mps.is_available',return_value=True),\
             patch('automatic_camera_prior.CameraPrior',return_value=provider) as constructor,\
             patch('observed_geometry.run',return_value={'fixture_only':True}) as run:
            folder,viewer=workflow.sparse()
        constructor.assert_called_once_with(self.root)
        self.assertEqual(viewer,{'fixture_only':True})
        self.assertIs(run.call_args.kwargs['camera_provider'],provider)
        self.assertFalse(run.call_args.kwargs['sensitivity'])
        self.assertEqual(run.call_args.args,(self.source,'part',folder))
        self.assertEqual(set(run.call_args.kwargs),{'root','max_views','sensitivity','max_nfev','camera_provider'})

    def test_assumed_sift_does_not_construct_provider_or_check_MPS(self):
        workflow=self.workflow()
        with patch('torch.backends.mps.is_available',side_effect=AssertionError('No GPU check for old CPU mode')),\
             patch('automatic_camera_prior.CameraPrior',side_effect=AssertionError('No implicit learned camera')),\
             patch('observed_geometry.run',return_value={}) as run:
            workflow.sparse()
        self.assertNotIn('camera_provider',run.call_args.kwargs)
        self.assertFalse(run.call_args.kwargs['sensitivity'])

    def test_unavailable_MPS_blocks_selected_prior_without_camera_fallback(self):
        workflow=self.workflow(dict(self.extended,camera_mode='moge_prior'))
        with patch('torch.backends.mps.is_available',return_value=False),\
             patch('automatic_camera_prior.CameraPrior') as provider,patch('observed_geometry.run') as run:
            with self.assertRaises(pipeline.StageUnavailable):workflow.sparse()
        provider.assert_not_called();run.assert_not_called()

    def test_cotracker_receives_same_camera_provider_with_separate_device_scope(self):
        workflow=self.workflow(dict(self.extended,feature_provider='cotracker',camera_mode='moge_prior'))
        provider=object()
        original_read=pipeline.read
        with patch('torch.backends.mps.is_available',return_value=True),\
             patch('automatic_camera_prior.CameraPrior',return_value=provider),\
             patch('observed_cotracker.run',return_value={}) as run,\
             patch.object(pipeline,'read',side_effect=lambda p:{} if p.name=='protocol.json' else original_read(p)),\
             patch.object(pipeline,'verify_correspondence_provider'):
            workflow.sparse()
        self.assertIs(run.call_args.kwargs['camera_provider'],provider)
        self.assertEqual(run.call_args.kwargs['device'],'mps')
        self.assertFalse(run.call_args.kwargs['sensitivity'])

    def make_learned_cache(self):
        """Complete temporary synthetic receipts; fake weights never execute."""
        import automatic_camera_prior as P
        import observed_geometry as G
        folder=workflow_test_fixtures.WorkflowTests.make_sparse_cache(self)
        protocol=pipeline.read(folder/'protocol.json')
        declaration=P.describe()
        protocol.update(learned_camera_prior_used=True,camera_provider={'declaration':declaration,
            'result_record_path':str((folder/'camera-assumption.json').relative_to(self.root)),
            'estimation_precedes_geometry':True,'heldout_images_supplied_to_provider':False})
        (folder/'protocol.json').write_text(json.dumps(protocol))
        weights=self.root/'models/moge-2-vits-normal';weights.mkdir(parents=True)
        (weights/'model.pt').write_bytes(b'FAKE CONTRACT TEST WEIGHTS; NEVER EXECUTED')
        (weights/'provenance.json').write_text(json.dumps({'repo':'fake-fixture',
            'checkpoint_sha256':pipeline.sha(weights/'model.pt')}))
        frames=[]
        for row in protocol['input_views']:
            frame=dict(row,image=cv2.imread(str(self.root/row['image_url'].removeprefix('/files/'))))
            frames.append(frame)
        def fake_predict(provider,rows,directory,checkpoint):
            code=b'# SYNTHETIC MODEL CODE FIXTURE; NEVER EXECUTED\n'
            model_sha=hashlib.sha256(code).hexdigest()
            P.write_bytes_once(directory/'model-implementation.py',code)
            P.write_once(directory/'model-code.json',{'module_sha256':model_sha,
                'checkpoint_sha256':P.sha(checkpoint),'torch_version':'fake-no-model-execution'})
            P.write_once(directory/'model-code-integrity.json',{'module_sha256_before':model_sha,
                'module_sha256_after':model_sha,'unchanged':True})
            predictions=[]
            for i,frame in enumerate(rows,1):
                h,w=frame['image'].shape[:2]
                provider._call_started(directory,frame,(w,h))
                value={'frame_id':frame['id'],'input_size':[w,h],
                    'normalized_K':[[1.2,0.,.5],[0.,1.2,.5],[0.,0.,1.]]}
                P.write_once(directory/f'prediction-{i}.json',value);predictions.append(value)
            return predictions
        source=pipeline.read(self.source)
        with patch.object(P.CameraPrior,'_predict',fake_predict):
            K,assumption=G.learned_camera_assumption(P.CameraPrior(self.root),frames,folder,self.root,declaration,
                {'case_id':source.get('case_id'),'source':protocol['source'],'track_id':'part',
                 'program_path':protocol['program_path'],'program_sha256':protocol['program_sha256']})
        viewer=pipeline.read(folder/'viewer.json')
        viewer.update(input_protocol_sha256=pipeline.sha(folder/'protocol.json'),camera_assumption=assumption,
            learned_camera_prior_used=True,
            cameras=[{'frame_id':frames[0]['id'],'K':K.tolist(),'R':np.eye(3).tolist(),'t':[0.,0.,0.],'split':'train'}],
            frames=[dict(row,width=64,height=64) for row in protocol['input_views']])
        (folder/'viewer.json').write_text(json.dumps(viewer))
        return folder

    def rebind_camera(self,folder,report=None,assumption=None):
        """Update local fixture digests so semantic guards are exercised too."""
        record_path=folder/'camera-assumption.json'
        record=pipeline.read(record_path) if assumption is None else assumption
        report_path=folder/'camera-prior/camera-prior.json'
        if report is not None:report_path.write_text(json.dumps(report))
        record.update(report_sha256=pipeline.sha(report_path),
            report={'path':str(report_path.relative_to(self.root)),'sha256':pipeline.sha(report_path)})
        record_path.write_text(json.dumps(record))
        viewer=pipeline.read(folder/'viewer.json')
        viewer['camera_assumption']={**record,'result_record':{'path':str(record_path.relative_to(self.root)),'sha256':pipeline.sha(record_path)}}
        (folder/'viewer.json').write_text(json.dumps(viewer))

    def test_complete_camera_cache_pins_every_nested_receipt_and_reuses_without_inference(self):
        folder=self.make_learned_cache()
        caches=pipeline.sparse_caches(self.root)
        self.assertEqual(len(caches),1)
        cache=caches[0]
        expected={'camera-assumption.json'}|{str(p.relative_to(folder)) for p in (folder/'camera-prior').rglob('*') if p.is_file()}
        self.assertTrue(expected.issubset(cache['files']))
        self.assertIn('camera-prior/model-provenance.json',cache['files'])
        body=dict(self.extended,feature_provider='reuse_verified',reuse_sparse_id=cache['id'])
        workflow=self.workflow(body)
        with patch('automatic_camera_prior.CameraPrior',side_effect=AssertionError('No inference for verified cache')):
            selected,viewer=workflow.sparse()
        self.assertEqual(selected,folder)
        self.assertTrue(viewer['learned_camera_prior_used'])

    def test_changed_camera_model_snapshot_is_not_listed_as_verified_cache(self):
        folder=self.make_learned_cache()
        (folder/'camera-prior/model-implementation.py').write_text('# Changed synthetic model\n')
        self.assertEqual(pipeline.sparse_caches(self.root),[])

    def test_changed_camera_provenance_snapshot_is_rejected(self):
        folder=self.make_learned_cache()
        (folder/'camera-prior/model-provenance.json').write_text('{}')
        with self.assertRaises(ValueError):pipeline._candidate_binding(self.root,folder)

    def test_rebound_report_cannot_claim_metric_calibration(self):
        folder=self.make_learned_cache();report=pipeline.read(folder/'camera-prior/camera-prior.json')
        report['camera_calibrated']=True;self.rebind_camera(folder,report)
        with self.assertRaises(ValueError):pipeline._candidate_binding(self.root,folder)

    def test_saved_geometry_cannot_change_K_under_same_prior(self):
        folder=self.make_learned_cache();viewer=pipeline.read(folder/'viewer.json')
        viewer['cameras'][0]['K'][0][0]+=1.;(folder/'viewer.json').write_text(json.dumps(viewer))
        with self.assertRaises(ValueError):pipeline._candidate_binding(self.root,folder)

    def test_camera_input_pixel_mutation_is_rejected_before_cache_reuse(self):
        folder=self.make_learned_cache();protocol=pipeline.read(folder/'camera-prior/protocol.json')
        path=self.root/protocol['input_views'][0]['image_url'].removeprefix('/files/')
        path.write_bytes(b'CHANGED SOURCE PIXELS')
        self.assertEqual(pipeline.sparse_caches(self.root),[])

    def test_symlink_extra_camera_artifact_is_not_followed_or_cached(self):
        folder=self.make_learned_cache()
        (folder/'camera-prior/extra.json').symlink_to(self.source)
        with self.assertRaises(ValueError):pipeline._candidate_binding(self.root,folder)

    def test_missing_frozen_camera_declaration_rejects_learned_result(self):
        folder=self.make_learned_cache();protocol=pipeline.read(folder/'protocol.json')
        protocol.pop('camera_provider');(folder/'protocol.json').write_text(json.dumps(protocol))
        viewer=pipeline.read(folder/'viewer.json');viewer['input_protocol_sha256']=pipeline.sha(folder/'protocol.json')
        (folder/'viewer.json').write_text(json.dumps(viewer))
        with self.assertRaises(ValueError):pipeline._candidate_binding(self.root,folder)

    def test_receipt_mutation_after_request_admission_is_caught_before_reuse(self):
        folder=self.make_learned_cache();cache=pipeline.sparse_caches(self.root)[0]
        workflow=self.workflow(dict(self.extended,feature_provider='reuse_verified',reuse_sparse_id=cache['id']))
        path=folder/'camera-prior/prediction-1.json';path.write_text('{}')
        with self.assertRaises(ValueError):workflow.sparse()

    def test_malformed_camera_provenance_is_skipped_instead_of_breaking_catalog(self):
        folder=self.make_learned_cache()
        path=folder/'camera-prior/model-provenance.json';path.write_text('[]')
        self.assertEqual(pipeline.sparse_caches(self.root),[])

    def test_malformed_rebound_nested_camera_integrity_is_skipped(self):
        folder=self.make_learned_cache();report=pipeline.read(folder/'camera-prior/camera-prior.json')
        report['input_integrity']={'unchanged':True,'files':[42],'decoded_source_buffers':[]}
        self.rebind_camera(folder,report)
        self.assertEqual(pipeline.sparse_caches(self.root),[])

    def test_malformed_rebound_camera_integrity_record_is_skipped(self):
        folder=self.make_learned_cache();report=pipeline.read(folder/'camera-prior/camera-prior.json')
        report['input_integrity']=[];self.rebind_camera(folder,report)
        self.assertEqual(pipeline.sparse_caches(self.root),[])

    def test_workflow_entrypoint_preserves_canonical_learned_camera_option(self):
        request=jobs.canonical_request(dict(self.extended,camera_mode='moge_prior'),root=self.root)
        output=self.root/pipeline.BASE/'observed-cccccccccccccccccccccccc-workflow'
        with patch.object(pipeline.Workflow,'run',autospec=True,side_effect=lambda workflow:workflow.request) as run:
            result=pipeline.run(self.root,request,output)
        run.assert_called_once()
        self.assertEqual(result,request)
        self.assertEqual(result['camera_mode'],'moge_prior')
        self.assertEqual(pipeline.read(output/'contract.json')['request'],request)

    def test_workflow_entrypoint_preserves_explicit_assumed_option(self):
        request=jobs.canonical_request(dict(self.extended,camera_mode='assumed'),root=self.root)
        output=self.root/pipeline.BASE/'observed-cccccccccccccccccccccccc-workflow'
        with patch.object(pipeline.Workflow,'run',autospec=True,side_effect=lambda workflow:workflow.request):
            result=pipeline.run(self.root,request,output)
        self.assertEqual(result,request)

    def test_workflow_entrypoint_keeps_old_missing_camera_key_unchanged(self):
        request=jobs.canonical_request(self.extended,root=self.root)
        output=self.root/pipeline.BASE/'observed-cccccccccccccccccccccccc-workflow'
        with patch.object(pipeline.Workflow,'run',autospec=True,side_effect=lambda workflow:workflow.request):
            result=pipeline.run(self.root,request,output)
        self.assertEqual(result,request)
        self.assertNotIn('camera_mode',result)

    def test_workflow_entrypoint_does_not_admit_extra_request_capabilities(self):
        request=jobs.canonical_request(dict(self.extended,camera_mode='moge_prior'),root=self.root)
        output=self.root/pipeline.BASE/'observed-cccccccccccccccccccccccc-workflow'
        for field,value in [('K',[]),('camera_provider','unknown'),('model_path','/tmp/model'),('untrusted',True)]:
            with self.subTest(field=field),patch.object(pipeline.Workflow,'run') as run:
                with self.assertRaises(ValueError):pipeline.run(self.root,dict(request,**{field:value}),output)
                run.assert_not_called()
        self.assertFalse(output.exists())

    def unsupported_camera_fixture(self,folder,*,corruption=None,reason=None):
        import automatic_camera_prior as P
        folder.mkdir(parents=True);prior=folder/'camera-prior';prior.mkdir()
        policy=P.describe();doc=pipeline.read(self.source);source=[]
        for i,row in enumerate(doc['evidence']):
            source.append({'id':row['id'],'timestamp_s':row['timestamp_s'],
                'image_url':'/files/'+row['frame_path'],'image_sha256':row['frame_sha256'],
                'split':'heldout' if i==2 else 'train'})
        inputs=[row for row in source if row['split']=='train']
        (prior/'implementation.py').write_bytes(P.LOADED_SOURCE)
        (folder/'protocol.json').write_text(json.dumps({'camera_provider':{'declaration':policy},'input_views':source}))
        (prior/'protocol.json').write_text(json.dumps({'format':'ordinary-video-camera-prior-protocol',
            'version':1,'policy':policy,'native_size':[64,64],'input_views':inputs,
            'implementation_sha256':pipeline.sha(prior/'implementation.py'),
            'frozen_before_inference':True,'heldout_images_used':False}))
        for i,row in enumerate(inputs,1):
            (prior/f'call-{i}-start.json').write_text(json.dumps({'call_number':i,'frame_id':row['id'],
                'split':'train','input_size':[64,64],'status':'inference_invocation_start_recorded'}))
            focal=1.2 if i==1 else 2.4
            (prior/f'prediction-{i}.json').write_text(json.dumps({'frame_id':row['id'],'input_size':[64,64],
                'normalized_K':[[focal,0.,.5],[0.,focal,.5],[0.,0.,1.]]}))
        report={'format':'ordinary-video-camera-prior','version':1,'status':'unsupported','policy':policy,
            'K':None,'reason':pipeline.CAMERA_DISAGREEMENT if reason is None else reason,
            'actual_model_calls':len(inputs),'completed_model_calls':len(inputs),'new_api_calls':0,
            'camera_calibrated':False,'physical_scale_known':False,'physical_accuracy_verified':False,
            'input_integrity':{'unchanged':True,
                'files':[{'path':row['image_url'].removeprefix('/files/'),'sha256_before':row['image_sha256'],'unchanged':True} for row in inputs]+[
                    {'path':'automatic_camera_prior.py','sha256_before':pipeline.sha(prior/'implementation.py'),'unchanged':True}],
                'decoded_source_buffers':[{'frame_id':row['id'],'unchanged':True} for row in inputs]}}
        if corruption:report=corruption(report)
        (prior/'camera-prior.json').write_text(json.dumps(report))

    def camera_failure_stage(self,*,corruption=None,error=None):
        workflow=self.workflow(dict(self.extended,camera_mode='moge_prior'))
        def fail(*args,**kwargs):
            self.unsupported_camera_fixture(args[2],corruption=corruption)
            raise ValueError(pipeline.CAMERA_DISAGREEMENT) if error is None else error
        with patch('torch.backends.mps.is_available',return_value=True),\
             patch('automatic_camera_prior.CameraPrior',return_value=object()),\
             patch('observed_geometry.run',side_effect=fail) as run:
            row=workflow.stage('sparse',workflow.sparse)
        run.assert_called_once()
        return workflow,row

    def test_known_camera_disagreement_is_unsupported_with_hash_bound_report_and_all_predictions(self):
        workflow,row=self.camera_failure_stage()
        self.assertEqual(row['status'],'unsupported')
        self.assertEqual(row['failure_type'],'StageUnavailable')
        self.assertIn('No camera fallback or automatic retry ran',row['reason'])
        report=workflow.location('sparse')/'camera-prior/camera-prior.json'
        self.assertEqual(row['result']['camera_report_url'],'/files/'+str(report.relative_to(self.root)))
        self.assertEqual(row['artifact_sha256'][str(report.relative_to(self.root))],pipeline.sha(report))
        for i in [1,2]:
            for name in [f'prediction-{i}.json',f'call-{i}-start.json']:
                path=report.with_name(name)
                self.assertEqual(row['artifact_sha256'][str(path.relative_to(self.root))],pipeline.sha(path))
        self.assertFalse(row['result']['model_fallback_ran'])
        self.assertFalse(row['result']['automatic_retry'])

    def test_malformed_camera_failure_receipt_remains_failed(self):
        workflow,row=self.camera_failure_stage(corruption=lambda report:[])
        self.assertEqual(row['status'],'failed')
        self.assertNotIn('result',row)
        self.assertEqual(row['failure_type'],'ValueError')

    def test_unrelated_ValueError_remains_failed_despite_disagreement_receipt(self):
        workflow,row=self.camera_failure_stage(error=ValueError('Unrelated source hash mismatch'))
        self.assertEqual(row['status'],'failed')
        self.assertEqual(row['failure_type'],'ValueError')

    def test_unrelated_TypeError_remains_failed_despite_disagreement_receipt(self):
        workflow,row=self.camera_failure_stage(error=TypeError('Unrelated programming error'))
        self.assertEqual(row['status'],'failed')
        self.assertEqual(row['failure_type'],'TypeError')

    def test_changed_input_receipt_does_not_become_supported_uncertainty(self):
        workflow,row=self.camera_failure_stage(corruption=lambda report:dict(report,input_integrity={'unchanged':False}))
        self.assertEqual(row['status'],'failed')

    def test_whole_workflow_marks_camera_disagreement_unsupported_and_blocks_geometry(self):
        workflow=self.workflow(dict(self.extended,camera_mode='moge_prior'))
        def fail(*args,**kwargs):
            self.unsupported_camera_fixture(args[2]);raise ValueError(pipeline.CAMERA_DISAGREEMENT)
        with patch('torch.backends.mps.is_available',return_value=True),\
             patch('automatic_camera_prior.CameraPrior',return_value=object()),\
             patch('observed_geometry.run',side_effect=fail),\
             patch('observed_stereo.run',side_effect=AssertionError('No downstream geometry on unsupported camera')):
            result=workflow.run()
        self.assertEqual(result['status'],'unsupported')
        self.assertEqual([row['status'] for row in result['stages']],['unsupported','blocked','blocked','blocked','blocked'])
        self.assertIsNone(result['result_url'])
        self.assertEqual(result['exports'],{})

    def test_invalid_native_camera_bounds_are_not_classified_as_disagreement(self):
        workflow=self.workflow(dict(self.extended,camera_mode='moge_prior'));folder=workflow.location('sparse')
        self.unsupported_camera_fixture(folder)
        path=folder/'camera-prior/prediction-2.json';prediction=pipeline.read(path)
        prediction['normalized_K'][0][0]=100.;path.write_text(json.dumps(prediction))
        self.assertIsNone(pipeline.camera_disagreement_result(self.root,folder,ValueError(pipeline.CAMERA_DISAGREEMENT)))

    def test_changed_adapter_snapshot_is_not_classified_as_disagreement(self):
        workflow=self.workflow(dict(self.extended,camera_mode='moge_prior'));folder=workflow.location('sparse')
        self.unsupported_camera_fixture(folder)
        (folder/'camera-prior/implementation.py').write_text('# corrupted snapshot\n')
        self.assertIsNone(pipeline.camera_disagreement_result(self.root,folder,ValueError(pipeline.CAMERA_DISAGREEMENT)))

    def test_input_integrity_hash_mismatch_remains_failed(self):
        def corrupt(report):report['input_integrity']['files'][0]['sha256_before']='0'*64;return report
        workflow,row=self.camera_failure_stage(corruption=corrupt)
        self.assertEqual(row['status'],'failed')

    def test_noncanonical_temporal_TRAIN_selection_is_not_classified_as_disagreement(self):
        workflow=self.workflow(dict(self.extended,camera_mode='moge_prior'));folder=workflow.location('sparse')
        self.unsupported_camera_fixture(folder)
        path=folder/'protocol.json';protocol=pipeline.read(path)
        extra=dict(protocol['input_views'][-1],id='another_train',timestamp_s=100.,split='train')
        protocol['input_views'].append(extra);path.write_text(json.dumps(protocol))
        self.assertIsNone(pipeline.camera_disagreement_result(self.root,folder,ValueError(pipeline.CAMERA_DISAGREEMENT)))


if __name__=='__main__':unittest.main()
