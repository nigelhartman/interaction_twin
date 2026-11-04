import copy
import fcntl
import json
import sys
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import upgrade_runner as U


class UpgradeRequestTests(unittest.TestCase):
    def request(self):
        return {'twin_id':'example--procedure','track_id':'part_a','base_sha256':'a'*64,'mode':'refit_existing'}

    def test_request_is_allowlisted_and_requires_import_identity(self):
        request=self.request(); self.assertEqual(U.request_value(request),request)
        self.assertEqual(U.request_value(request|{'mode':'refit_native_cpu'})['mode'],'refit_native_cpu')
        for change in [{'mode':'hosted_reasoning'},{'base_sha256':'A'*64},{'mode':'imported_sam3d'},
                       {'imported_result_id':'../../.secrets/meta.env'},{'credentials':'secret'}, {'track_id':False}]:
            with self.subTest(change=change),self.assertRaises(ValueError): U.request_value(request|change)
        valid=request|{'mode':'imported_sam3d','imported_result_id':'import-123'}
        self.assertEqual(U.request_value(valid),valid)

    def test_data_paths_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve(); (root/'data').mkdir(); (root/'outside').mkdir(); (root/'data/link').symlink_to(root/'outside',target_is_directory=True)
            for value in ['data/../outside','data/.hidden/key','/tmp/file','data/link/file','data/foo/../../file']:
                with self.subTest(path=value),self.assertRaises(ValueError): U.data_path(root,value)
            self.assertEqual(U.data_path(root,'data/model-upgrades/runs'),root/'data/model-upgrades/runs')


class AlignmentConfigurationTests(unittest.TestCase):
    def test_legacy_vertex_selection_keeps_original_adapter_dependencies(self):
        config=U.alignment_configuration({'backend':'TripoSR'})
        self.assertFalse(config['native'])
        self.assertEqual(config['pose_script'],'stage4_generated_pose.py')
        self.assertEqual(config['motion_script'],'stage4_track_generated.py')
        self.assertNotIn('native_appearance.py',config['pose_dependencies'])
        self.assertNotIn('native_alignment.py',config['motion_dependencies'])

    def test_uv_selection_uses_native_pose_and_tracker_with_decoder_and_appearance_sources(self):
        config=U.alignment_configuration({'fit_contract':{'alignment_adapter':'native-base-color-v1'}})
        self.assertTrue(config['native'])
        self.assertEqual(config['pose_script'],'stage4_native_pose.py')
        self.assertEqual(config['motion_script'],'stage4_native_track.py')
        for key in ['pose_dependencies','motion_dependencies']:
            for dependency in ['native_appearance.py','native_alignment.py','portable_twin.py','alignment_render.py']:
                self.assertIn(dependency,config[key])
        self.assertIn('scripts/stage4_generated_pose.py',config['pose_dependencies'])
        self.assertIn('scripts/stage4_track_generated.py',config['motion_dependencies'])

    def test_current_resolved_capability_overrides_immutable_historical_contract(self):
        old={'fit_contract':{'alignment_eligible':False,'alignment_adapter':'legacy-vertex-colour-v1'}}
        config=U.alignment_configuration(old,{'alignment_adapter':'native-base-color-v1'})
        self.assertTrue(config['native'])
        self.assertFalse(old['fit_contract']['alignment_eligible'])
        with self.assertRaisesRegex(ValueError,'Unsupported native alignment'):
            U.alignment_configuration({}, {'alignment_adapter':'unsupported-mode'})


class MatchedComparisonTests(unittest.TestCase):
    def row(self,t,status='supported_conditional_motion_candidate',iou=.6,error=2.):
        return {'timestamp_s':t,'status':status,'raw_full_projection_iou_with_sam':iou,'heldout_median_px':error}

    def test_matches_exact_pts_excludes_union_anchors_and_reports_missing_coverage(self):
        baseline={'frames':[self.row(1,'reference_initialization'),self.row(2),self.row(3),self.row(4),self.row(6)]}
        candidate={'frames':[self.row(1),self.row(2,'reference_unverified'),self.row(3,iou=.9),self.row(4,'withheld'),self.row(5)]}
        result=U.compare_motion(baseline,candidate)
        self.assertEqual(result['timestamps_s'],[3,4]); self.assertEqual(result['matched_sample_count'],2)
        self.assertEqual(result['excluded_reference_timestamps_s'],[1,2]); self.assertEqual(result['baseline']['conditional_supported'],2)
        self.assertEqual(result['candidate']['conditional_supported'],1)
        self.assertEqual(result['candidate']['silhouette_iou_median'],.9)
        self.assertTrue(all(r['denominator']==2 for r in result['metrics']))
        self.assertFalse(result['physical_accuracy_verified']); self.assertFalse(result['automatic_baseline_replacement'])

    def test_no_matching_pts_or_metrics_stay_absent(self):
        result=U.compare_motion({'frames':[self.row(1)]},{'frames':[self.row(1.0000001)]})
        self.assertEqual(result['matched_sample_count'],0); self.assertIsNone(result['candidate']['residual_px_median'])
        self.assertEqual(U.compare_motion(None,None)['timestamps_s'],[])


class CachedLineageTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name).resolve()
        self.case=self.root/'data/case'; self.case.mkdir(parents=True)
        self.seg=self.case/'segmentation/run'; (self.seg/'model-audit-v1').mkdir(parents=True)
        (self.case/'frames').mkdir(); (self.case/'frames/one.jpg').write_bytes(b'pixels')
        (self.case/'source.mp4').write_bytes(b'source'); self.source_sha=U.sha(self.case/'source.mp4')
        U.write(self.case/'manifest.json',{'source':{'sha256':self.source_sha}})
        U.write(self.case/'frames/index.json',{'frames':[{'index':1,'timestamp_s':1.,'file':'frames/one.jpg'}]})
        U.write(self.case/'annotations.json',{'tracks':[{'id':'part'}]})
        U.write(self.case/'discovery.json',{'status':'complete'})
        U.write(self.seg/'annotations.json',{'manual_seed_boxes_used':False,'tracks':[{'id':'part'}]})
        U.write(self.seg/'tracks.json',{'tracks':[{'track_id':'part','frames':[{'sample_index':1,'raw_mask_file':'mask.png'}]}]})
        U.write(self.seg/'frames.json',{'frames':[]}); U.write(self.seg/'quality.json',{'tracks':{'part':{'withheld_samples':[]}}}); (self.seg/'mask.png').write_bytes(b'mask')
        U.write(self.seg/'model-audit-v1/audit.json',{'source_sha256':self.source_sha,'source_track_index_sha256':U.sha(self.seg/'tracks.json')})
        U.write(self.case/'proposal.json',{'status':'complete_validated_model_proposal','source_sha256':self.source_sha,
            'discovery_sha256':U.sha(self.case/'discovery.json'),'annotation_sha256':U.sha(self.case/'annotations.json'),
            'frame_index_sha256':U.sha(self.case/'frames/index.json')})
        selected=self.case/'selected'; selected.mkdir(); (selected/'rgb.png').write_bytes(b'pixels'); (selected/'mask.png').write_bytes(b'mask')
        U.write(selected/'selection.json',{'object_id':'part','selection_method':{'agent_visual_choice':False},'selected':[{'selection_rank':1,'exports':{
            'rgb':'rgb.png','rgb_sha256':U.sha(selected/'rgb.png'),'mask':'mask.png','mask_sha256':U.sha(selected/'mask.png')}}]})
        self.model=self.case/'model'; self.model.mkdir(); (self.model/'asset.glb').write_bytes(b'untouched GLB')
        (self.model/'source-rgb.png').write_bytes(b'pixels'); (self.model/'source-mask.png').write_bytes(b'mask')
        self.metadata=self.model/'metadata.json'
        U.write(self.metadata,{'object_id':'part','geometry_reconstructed_by_model':True,'manual_geometry_fallback':False,'outputs':{'glb_path':'asset.glb'},'source':{
            'video_sha256':self.source_sha,'selection_manifest_path':str(selected/'selection.json'),'selection_manifest_sha256':U.sha(selected/'selection.json'),
            'image_path':'source-rgb.png','mask_path':'source-mask.png','image_sha256':U.sha(self.model/'source-rgb.png'),'mask_sha256':U.sha(self.model/'source-mask.png')}})
        kinds={'source_manifest':self.case/'manifest.json','source_frame_index':self.case/'frames/index.json','model_discovery':self.case/'discovery.json',
               'model_tracklet_proposals':self.case/'annotations.json','model_semantic_proposal':self.case/'proposal.json','actual_sam_masks':self.seg/'tracks.json',
               'geometric_mask_gates':self.seg/'frames.json','model_mask_plausibility_audit':self.seg/'model-audit-v1/audit.json','learned_asset_metadata':self.metadata}
        self.document={'source':{'path':'data/case/source.mp4','sha256':self.source_sha},'assets':[{'track_id':'part','path':'data/case/model/asset.glb','sha256':U.sha(self.model/'asset.glb')}],
                       'inputs':[{'kind':kind,'path':str(path.relative_to(self.root))} for kind,path in kinds.items()]}
        self.request={'twin_id':'test','track_id':'part','base_sha256':'a'*64,'mode':'refit_existing'}
        self.patch=patch.object(U.TwinCatalog,'load',return_value=({'title':'Fixture'},self.document,'a'*64)); self.patch.start()

    def tearDown(self): self.patch.stop(); self.temp.cleanup()
    def context(self): return U._context(self.request,self.root)

    def test_verified_cache_includes_consumed_pixels_and_never_runs_reasoning(self):
        context=self.context(); self.assertIn(self.case/'frames/one.jpg',context['dependencies']); self.assertIn(self.seg/'mask.png',context['dependencies'])
        self.assertEqual(context['tracking_case'],self.case)

    def test_cpu_native_mode_preflights_appearance_and_discloses_no_new_reconstruction(self):
        import native_appearance
        request=self.request|{'mode':'refit_native_cpu'}
        with patch.object(native_appearance,'load_native_glb',return_value=SimpleNamespace(receipt={'contract_version':'native-base-color-v1'})) as loader:
            context=U._context(request,self.root)
            self.assertEqual(context['native_cpu_preflight']['contract_version'],'native-base-color-v1')
            proposed=U.plan(request,self.root)
        loader.assert_called_with(self.model/'asset.glb')
        self.assertEqual(proposed['execution_device'],'cpu')
        self.assertEqual(proposed['tracking_model_maximum_dimension'],256)
        self.assertFalse(proposed['new_reconstruction_inference']); self.assertFalse(proposed['new_reasoning_inference'])
        self.assertEqual(next(s for s in proposed['stages'] if s['id']=='reconstruct')['action'],'reuse')
        self.assertIn('Explicit CPU',next(s for s in proposed['stages'] if s['id']=='motion')['reason'])
        with patch.object(native_appearance,'load_native_glb',side_effect=ValueError('Unsupported synthetic alpha fixture')):
            with self.assertRaisesRegex(ValueError,'Unsupported synthetic alpha'): U._context(request,self.root)

    def test_changed_model_reference_pixels_are_rejected(self):
        (self.model/'source-rgb.png').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError,'reference pixels changed'): self.context()

    def test_changed_proposal_dependencies_are_rejected(self):
        (self.case/'annotations.json').write_text('{"tracks":[]}')
        with self.assertRaisesRegex(ValueError,'Cached reasoning dependency changed'): self.context()

    def test_changed_audit_and_track_source_are_rejected(self):
        value=U.read(self.seg/'model-audit-v1/audit.json'); value['source_track_index_sha256']='f'*64; U.write(self.seg/'model-audit-v1/audit.json',value)
        with self.assertRaisesRegex(ValueError,'Cached mask audit'): self.context()

    def test_manual_seed_dependency_is_rejected(self):
        value=U.read(self.seg/'annotations.json'); value['manual_seed_boxes_used']=True; U.write(self.seg/'annotations.json',value)
        with self.assertRaisesRegex(ValueError,'Model-generated same-track'): self.context()

    def test_changed_base_and_selected_view_bytes_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'original program changed'): U._context(self.request|{'base_sha256':'b'*64},self.root)
        (self.case/'selected/rgb.png').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Selected input pixels changed'): self.context()

    def test_sam31_anchor_import_cannot_be_used_as_sam3d_geometry(self):
        import upgrade_bundles
        with patch.object(upgrade_bundles,'resolve_import',return_value={'backend':'sam31_image','status':'validated_anchor_comparison_only'}):
            with self.assertRaisesRegex(ValueError,'SAM 3.1 anchor comparisons are not a replacement 3D asset'):
                U._context(self.request|{'mode':'imported_sam3d','imported_result_id':'import-anchor'},self.root)

    def test_uv_native_import_is_preserved_but_blocked_for_legacy_alignment(self):
        import upgrade_bundles
        result={'backend':'sam3d_objects','status':'validated_candidate_pending_alignment','alignment_eligible':False,
                'alignment_ineligible_reason':'UV textured asset has no verified vertex colors'}
        with patch.object(upgrade_bundles,'resolve_import',return_value=result):
            with self.assertRaisesRegex(ValueError,'retained for inspection.*UV textured'):
                U._context(self.request|{'mode':'imported_sam3d','imported_result_id':'import-native-uv'},self.root)

    def test_historical_uv_import_uses_current_capability_without_rewriting_metadata(self):
        import upgrade_bundles
        value=U.read(self.metadata); value['fit_contract']={'alignment_eligible':False,'reason':'historical vertex-only adapter'}
        U.write(self.metadata,value); before=self.metadata.read_bytes()
        result={'backend':'sam3d_objects','status':'validated_candidate_pending_alignment','alignment_eligible':True,
                'alignment_adapter':'native-base-color-v1','import_time_alignment_eligible':False,
                'twin_id':self.request['twin_id'],'track_id':self.request['track_id'],'base_sha256':self.request['base_sha256'],
                'model_metadata_path':str(self.metadata.relative_to(self.root))}
        with patch.object(upgrade_bundles,'resolve_import',return_value=result):
            context=U._context(self.request|{'mode':'imported_sam3d','imported_result_id':'import-native-uv'},self.root)
        self.assertTrue(U.alignment_configuration(U.read(self.metadata),context['imported'])['native'])
        self.assertEqual(self.metadata.read_bytes(),before)


class StageReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name).resolve(); (self.root/'data').mkdir()
        self.input=self.root/'data/input.json'; self.input.write_text('original')
        self.runner=U.UpgradeRunner(root=self.root,run_id='upgrade-test')
        self.calls=0

    def tearDown(self): self.temp.cleanup()
    def action(self,directory):
        self.calls+=1; path=directory/'output.json'; U.write(path,{'attempt':self.calls})
        return {'path':str(path.relative_to(self.root))},[path]

    def test_completed_stage_reuses_only_verified_inputs_outputs_and_parameters(self):
        first=self.runner.phase('reconstruct',[self.input],self.action,parameters={'resolution':96})
        self.assertEqual(self.runner.phase('reconstruct',[self.input],self.action,parameters={'resolution':96}),first)
        self.assertEqual(self.calls,1)
        (self.root/first['path']).write_text('changed output')
        second=self.runner.phase('reconstruct',[self.input],self.action,parameters={'resolution':96})
        self.assertEqual(self.calls,2); self.assertNotEqual(first['path'],second['path'])
        self.assertTrue((self.root/first['path']).exists())
        self.input.write_text('changed input'); self.runner.phase('reconstruct',[self.input],self.action,parameters={'resolution':96})
        self.assertEqual(self.calls,3)
        self.runner.phase('reconstruct',[self.input],self.action,parameters={'resolution':128}); self.assertEqual(self.calls,4)

    def test_unrelated_display_file_does_not_invalidate_model(self):
        self.runner.phase('reconstruct',[self.input],self.action)
        (self.root/'display.html').write_text('new presentation')
        self.runner.phase('reconstruct',[self.input],self.action)
        self.assertEqual(self.calls,1)

    def test_frozen_implementation_identity_controls_reuse(self):
        self.runner.phase('reconstruct',[self.input],self.action)
        self.runner.implementation_identities['reconstruct']='b'*64
        self.runner.phase('reconstruct',[self.input],self.action)
        self.assertEqual(self.calls,2)
        receipt=self.runner.state['phases'][-1]
        self.assertEqual(receipt['implementation']['stage_sha256'],'b'*64)
        self.assertEqual(receipt['implementation']['source_sha256'],U.LOADED_SOURCE_SHA256)

    def test_comparison_only_source_edit_changes_only_comparison_identity(self):
        changed=U.LOADED_SOURCE.replace(b"'matched_sample_count':len(times)",b"'matched_sample_count':int(len(times))")
        self.assertNotEqual(changed,U.LOADED_SOURCE)
        old=U.stage_source_identities(); new=U.stage_source_identities(changed)
        self.assertNotEqual(old['comparison'],new['comparison'])
        for stage in old:
            if stage!='comparison': self.assertEqual(old[stage],new[stage],stage)
        self.runner.phase('comparison',[self.input],self.action)
        self.runner.implementation_identities=new
        self.runner.phase('comparison',[self.input],self.action)
        self.assertEqual(self.calls,2)

    def test_common_orchestration_change_invalidates_worker_stage_identity(self):
        changed=U.LOADED_SOURCE.replace(b'Upgrade stage exceeded its local time budget',b'Upgrade stage exceeded its revised local time budget')
        old=U.stage_source_identities(); new=U.stage_source_identities(changed)
        self.assertTrue(all(old[k]!=new[k] for k in old))

    def test_backend_adapter_change_affects_pose_motion_not_reconstruction(self):
        changed=U.LOADED_SOURCE.replace(b'geometry_or_numeric_results_modified',b'geometry_and_numeric_results_modified')
        old=U.stage_source_identities(); new=U.stage_source_identities(changed)
        self.assertNotEqual(old['pose'],new['pose']); self.assertNotEqual(old['motion'],new['motion'])
        self.assertEqual(old['reconstruct'],new['reconstruct'])

    def test_runtime_or_cotracker_source_identity_change_invalidates_motion(self):
        first={'packages':{'torch':{'version':'old'}},'cotracker_source':[{'file':'predictor.py','sha256':'a'*64}]}
        self.runner.phase('motion',[self.input],self.action,parameters={'installed_runtime':first})
        self.runner.phase('motion',[self.input],self.action,parameters={'installed_runtime':copy.deepcopy(first)})
        self.assertEqual(self.calls,1)
        second=copy.deepcopy(first); second['cotracker_source'][0]['sha256']='b'*64
        self.runner.phase('motion',[self.input],self.action,parameters={'installed_runtime':second}); self.assertEqual(self.calls,2)
        third=copy.deepcopy(second); third['packages']['torch']['version']='new'
        self.runner.phase('motion',[self.input],self.action,parameters={'installed_runtime':third}); self.assertEqual(self.calls,3)

    def test_complete_runner_source_is_frozen_and_tampering_rejected(self):
        self.assertEqual(self.runner.code_snapshot.read_bytes(),U.LOADED_SOURCE)
        self.assertEqual(U.sha(self.runner.code_snapshot),U.LOADED_SOURCE_SHA256)
        self.runner.code_snapshot.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'source snapshot changed'): U.UpgradeRunner(root=self.root,run_id=self.runner.id)

    def test_native_routing_implementation_change_is_scoped_to_alignment(self):
        changed=U.LOADED_SOURCE.replace(b'Unsupported native alignment adapter contract',b'Unsupported revised native alignment adapter contract')
        self.assertNotEqual(changed,U.LOADED_SOURCE)
        old=U.stage_source_identities(); new=U.stage_source_identities(changed)
        for key in old:
            self.assertEqual(old[key]!=new[key],key in {'pose','motion'},key)

    def test_native_dependency_edit_invalidates_pose_but_keeps_completed_reconstruction(self):
        native=self.root/'data/native_renderer.py'; native.write_text('renderer v1')
        self.runner.phase('reconstruct',[self.input],self.action)
        self.runner.phase('pose',[self.input,native],self.action)
        self.assertEqual(self.calls,2)
        native.write_text('renderer v2')
        self.runner.phase('reconstruct',[self.input],self.action)
        self.runner.phase('pose',[self.input,native],self.action)
        self.assertEqual(self.calls,3)
        self.assertEqual(self.runner.state['phases'][0]['status'],'reused_verified')

    def test_failed_attempts_are_preserved_and_retry_gets_fresh_directory(self):
        def broken(directory):
            (directory/'partial').write_text('keep me'); raise RuntimeError('useful failure')
        result=self.runner.phase('pose',[self.input],broken,optional=True)
        self.assertTrue(result['has_gaps']); self.assertEqual(self.runner.state['phases'][0]['status'],'failed')
        self.runner.phase('pose',[self.input],self.action)
        self.assertTrue((self.runner.directory/'pose-attempt-1/partial').is_file())
        self.assertTrue((self.runner.directory/'pose-attempt-2/output.json').is_file())

    def test_cancellation_does_not_start_stage(self):
        self.runner.cancel()
        with self.assertRaises(U.Cancelled): self.runner.phase('pose',[self.input],self.action)
        self.assertEqual(self.calls,0)

    def test_cancellation_terminates_a_real_subprocess_group(self):
        ready=self.root/'data/child-ready'; stopped=self.root/'data/child-stopped'
        child=('import signal,time,sys; from pathlib import Path; '
               'signal.signal(signal.SIGTERM,lambda *_:(Path(sys.argv[2]).write_text("terminated"),sys.exit(0))); '
               'Path(sys.argv[1]).write_text("ready"); time.sleep(20)')
        parent=('import subprocess,sys,time; '
                'subprocess.Popen([sys.executable,"-c",sys.argv[1],sys.argv[2],sys.argv[3]]); time.sleep(20)')
        errors=[]
        def run():
            try: self.runner.command([sys.executable,'-c',parent,child,ready,stopped],timeout=25)
            except Exception as exc: errors.append(exc)
        thread=threading.Thread(target=run); thread.start()
        deadline=time.monotonic()+5
        while not ready.exists() and thread.is_alive() and time.monotonic()<deadline: time.sleep(.02)
        try:
            self.assertTrue(ready.exists()); self.runner.cancel(); thread.join(timeout=5)
            self.assertFalse(thread.is_alive()); self.assertTrue(any(isinstance(e,U.Cancelled) for e in errors))
            deadline=time.monotonic()+2
            while not stopped.exists() and time.monotonic()<deadline: time.sleep(.02)
            self.assertEqual(stopped.read_text(),'terminated')
        finally: self.runner.cancel(); thread.join(timeout=5)

    def test_resume_rejects_changed_request_without_inference(self):
        self.runner.state['request']={'twin_id':'test','track_id':'part','base_sha256':'a'*64,'mode':'refit_existing'}
        with self.assertRaises(ValueError): self.runner.execute(self.runner.state['request']|{'track_id':'other'})

    def test_shared_model_worker_lock_blocks_before_actual_work(self):
        request={'twin_id':'test','track_id':'part','base_sha256':'a'*64,'mode':'refit_existing'}
        path=self.root/'data/flow-runs/.model-worker.lock'; path.parent.mkdir()
        with path.open('a+') as lock,patch.object(U,'_context',return_value={}),patch.object(U,'plan',return_value={'request':request,'inputs':[],'blockers':[]}),patch.object(self.runner,'_execute') as execute:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            state=self.runner.execute(request)
        self.assertEqual(state['status'],'failed'); self.assertIn('Another local pipeline',state['error']); execute.assert_not_called()

    def test_backend_adapter_preserves_raw_numeric_receipt(self):
        model=self.root/'data/model.json'; U.write(model,{'backend':'SAM 3D Objects','appearance':'learned UV texture'})
        path=self.root/'data/experiment.json'; value={'candidate':{'rotation':[[1,0,0],[0,1,0],[0,0,1]]},'provenance':{'sam3d_output':False}}
        U.write(path,value); raw=path.read_bytes(); self.runner._adapt_backend(path,model)
        self.assertEqual(path.with_name('raw-experiment.json').read_bytes(),raw)
        result=U.read(path); self.assertEqual(result['candidate'],value['candidate']); self.assertTrue(result['provenance']['sam3d_output'])
        self.assertFalse(result['backend_label_adapter']['geometry_or_numeric_results_modified'])

    def test_run_lookup_rejects_traversal_and_lists_durable_state(self):
        for bad in ['../meta.env','upgrade-../x','upgrade-']:
            with self.assertRaises(ValueError): U.get_run(bad,self.root)
        self.assertEqual(U.list_runs(self.root)[0]['id'],'upgrade-test')


if __name__=='__main__': unittest.main()
