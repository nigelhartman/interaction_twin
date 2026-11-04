"""Generic inert program/import fixtures; no models, GPU, API or geometry fits."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import component_asset_queue as Q
import upgrade_bundles as U


class ComponentQueueTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        for name in Q.DEPENDENCIES:
            target = self.root/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(Q.ROOT/name,target)
        self.program = self.root/'data/unseen-video/procedure/interaction.json'
        self.program.parent.mkdir(parents=True)
        self.program.write_text('inert program fixture')
        self.program_sha = U.sha(self.program)
        self.document = {'source':{'path':'data/unseen-video/source.mp4','sha256':'a'*64,'original_sha256':'b'*64},
          'tracklets':[{'id':'panel_x','label':'unseen hinged panel','identity_status':'model_proposal'},
                       {'id':'insert_y','label':'unseen insert','identity_status':'model_proposal'}],
          'relationships':[{'id':'relation_7','subject_track_id':'panel_x','object_track_id':'insert_y'}],
          'steps':[{'id':'procedure_4','participant_track_ids':['panel_x','insert_y']}],
          'motion':[{'track_id':'panel_x','experiment_input_id':'motion_1','status':'conditional_pose_candidate',
                     'camera_from_native_mesh':[[99,98,97,96]]}],
          'inputs':[{'id':'motion_1','kind':'model_conditional_motion_experiment','path':'data/unseen-video/old-motion.json','sha256':'c'*64}]}
        self.binding = {'twin_id':'never-seen-before','track_id':'panel_x','base_sha256':self.program_sha,
          'selection_path':'data/unseen-video/selection.json','selection_sha256':None,
          'image_path':'data/unseen-video/rgb.png','mask_path':'data/unseen-video/mask.png',
          'mask_provenance':{'backend':'SAM','prompt_origin':'model_generated_prompt','manual_intervention':False},
          'source':{'video_sha256':'a'*64,'sample_index':7,'timestamp_s':2.5,'time_domain':U.TIME_DOMAIN}}
        for role in ('image','mask'):
            path = self.root/self.binding[role+'_path']
            path.write_bytes(('inert '+role).encode())
            self.binding[role+'_sha256'] = U.sha(path)
        selection = self.root/self.binding['selection_path']
        selection.write_text('inert selection')
        self.binding['selection_sha256'] = U.sha(selection)
        catalog = mock.patch.object(Q.TwinCatalog,'load',return_value=({'interaction_path':str(self.program.relative_to(self.root))},self.document,self.program_sha))
        catalog.start();self.addCleanup(catalog.stop)
        context = mock.patch.object(Q.U,'context',side_effect=self.context)
        context.start();self.addCleanup(context.stop)
        self.output = 'data/component-asset-queues/unseen-video-v1'

    def context(self,twin,track,root):
        if track != 'panel_x': raise ValueError('No saved learned source asset for this component')
        return copy.deepcopy(self.binding)

    def prepare(self,**kwargs):
        return Q.prepare('never-seen-before',self.output,root=self.root,**kwargs)

    def candidate(self):
        directory = self.root/'data/genuine-import-contract-fixture'
        directory.mkdir()
        for role in ('image','mask'):
            shutil.copyfile(self.root/self.binding[role+'_path'],directory/(role+'.png'))
        (directory/'asset.glb').write_bytes(b'INERT_GEOMETRY_FIXTURE_NOT_A_MODEL_RESULT')
        (directory/'import.json').write_text('inert existing verifier contract')
        model = {'object_id':'panel_x','backend':'Meta SAM 3D Objects','learned_inference_executed':True,
          'manual_pose_used':False,'manual_geometry_fallback':False,'camera_pose_estimated':False,'metric_scale_verified':False,
          'outputs':{'glb_path':'asset.glb'},'source':dict(self.binding['source'],
            image_path='image.png',image_sha256=self.binding['image_sha256'],mask_path='mask.png',mask_sha256=self.binding['mask_sha256'],
            selection_manifest_path=self.binding['selection_path'],selection_manifest_sha256=self.binding['selection_sha256'])}
        U.write(directory/'metadata.json',model)
        result = {'backend':'sam3d_objects','twin_id':'never-seen-before','track_id':'panel_x','base_sha256':self.program_sha,
          'import_path':str((directory/'import.json').relative_to(self.root)),
          'model_metadata_path':str((directory/'metadata.json').relative_to(self.root)),
          'adapter_capability':{'alignment_eligible':True,'raw_glb_sha256':U.sha(directory/'asset.glb')}}
        patch = mock.patch.object(Q.U,'resolve_import',return_value=result)
        patch.start();self.addCleanup(patch.stop)
        return {'panel_x':{'upgrade_import_id':'import-'+'1'*32}},directory,model,result

    def test_unknown_video_ids_and_missing_anchor_are_explicit(self):
        result = self.prepare()
        self.assertEqual(result['component_counts'],{'proposed':1,'unavailable':1})
        plan = U.read(self.root/self.output/'plan.json')
        part = plan['components'][0]
        self.assertEqual(part['relationships'],['relation_7'])
        self.assertEqual(part['procedure_steps'],['procedure_4'])
        self.assertEqual(part['observed_motion']['record_count'],1)
        self.assertNotIn('camera_from_native_mesh',json.dumps(plan))
        self.assertNotIn('99',json.dumps(part['observed_motion']))
        self.assertTrue(plan['no_inference'])
        self.assertEqual(Q.inspect(self.root/self.output/'queue.json',root=self.root)['status'],'verified_offline_queue')

    def test_assisted_anchor_never_queued_as_automatic(self):
        self.binding['mask_provenance']['manual_intervention'] = True
        self.assertEqual(self.prepare()['component_counts'],{'unavailable':2})

    def test_missing_anchor_cannot_receive_candidate(self):
        with self.assertRaisesRegex(ValueError,'without its current source anchor'):
            self.prepare(candidate_imports={'insert_y':{'upgrade_import_id':'import-'+'1'*32}})

    def test_native_candidate_requires_new_fit_and_keeps_original_program(self):
        refs,directory,_,_ = self.candidate()
        before = self.program.read_bytes()
        self.prepare(candidate_imports=refs)
        queue = Q.inspect(self.root/self.output/'queue.json',root=self.root)
        part = queue['components'][0]
        self.assertEqual(part['state'],'geometry_candidate')
        self.assertEqual(part['attachment']['state'],'motion_alignment_required')
        self.assertFalse(part['attachment']['existing_poses_reused'])
        self.assertIn(str((directory/'metadata.json').relative_to(self.root)),part['attachment']['reference_fit_argv'])
        self.assertEqual(self.program.read_bytes(),before)

    def test_wrong_candidate_source_cannot_attach(self):
        refs,directory,model,_ = self.candidate()
        model['source']['video_sha256'] = 'd'*64
        U.write(directory/'metadata.json',model)
        with self.assertRaisesRegex(ValueError,'source identity'): self.prepare(candidate_imports=refs)

    def test_wrong_candidate_mesh_cannot_attach(self):
        refs,directory,_,_ = self.candidate()
        (directory/'asset.glb').write_bytes(b'other geometry')
        with self.assertRaisesRegex(ValueError,'mesh differs'): self.prepare(candidate_imports=refs)

    def test_wrong_source_pixels_cannot_attach(self):
        refs,directory,_,_ = self.candidate()
        (directory/'image.png').write_bytes(b'other pixels')
        with self.assertRaisesRegex(ValueError,'source pixels'): self.prepare(candidate_imports=refs)

    def test_other_program_candidate_cannot_attach(self):
        refs,_,_,result = self.candidate()
        result['base_sha256'] = 'e'*64
        with self.assertRaisesRegex(ValueError,'another source'): self.prepare(candidate_imports=refs)

    def test_unsupported_material_keeps_geometry_without_fit_command(self):
        refs,_,_,result = self.candidate()
        result['adapter_capability']['alignment_eligible'] = False
        result['adapter_capability']['alignment_ineligible_reason'] = 'unsupported native appearance'
        self.prepare(candidate_imports=refs)
        item = Q.inspect(self.root/self.output/'queue.json',root=self.root)['components'][0]
        self.assertEqual(item['state'],'geometry_candidate')
        self.assertEqual(item['attachment']['state'],'unavailable')
        self.assertNotIn('reference_fit_argv',item['attachment'])

    def test_command_or_state_tamper_fails_closed(self):
        refs,_,_,_ = self.candidate()
        self.prepare(candidate_imports=refs)
        path = self.root/self.output/'queue.json'
        queue = U.read(path)
        queue['components'][0]['attachment']['reference_fit_argv'] = ['untrusted','command']
        U.write(path,queue)
        with self.assertRaisesRegex(ValueError,'fitting command'): Q.inspect(path,root=self.root)

    def test_changed_program_binding_fails_closed(self):
        self.prepare()
        self.binding['base_sha256'] = 'f'*64
        with self.assertRaisesRegex(ValueError,'Program changed'): Q.inspect(self.root/self.output/'queue.json',root=self.root)

    def test_no_output_inside_original_inputs_and_no_reuse(self):
        with self.assertRaises(ValueError): Q.prepare('never-seen-before','data/unseen-video/insert',root=self.root)
        self.prepare()
        with self.assertRaisesRegex(ValueError,'Preserve previous queue'): self.prepare()

    def test_symlink_output_refused(self):
        (self.root/'data/component-asset-queues').symlink_to(self.root/'data/unseen-video',target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'Symlink'): self.prepare()

    def test_native_import_reference_reuses_existing_verifier(self):
        _,directory,_,result = self.candidate()
        from scripts import build_sam3d_milestone_review as existing
        receipt = {'fit_contract':result['adapter_capability']}
        with mock.patch.object(existing,'verified_import',return_value=(directory,receipt,{'binding':self.binding},{},{})) as verify:
            self.prepare(candidate_imports={'panel_x':{'native_import_directory':str(directory.relative_to(self.root))}})
        self.assertEqual(verify.call_count,2)  # Before and after preparing the batch.

    def test_optional_local_handoff_reuses_existing_job_contract(self):
        directory = self.root/'data/upgrade-jobs/job-inert-fixture'
        directory.mkdir(parents=True)
        (directory/'job.json').write_text('inert handoff metadata')
        Path(str(directory)+'.zip').write_bytes(b'inert handoff bytes')
        summary = {'job_id':'job-inert-fixture','job_relative_path':str((directory/'job.json').relative_to(self.root)),
                   'zip_relative_path':str(directory.relative_to(self.root))+'.zip'}
        saved = {'backend':'sam3d_objects','input_binding':self.binding}
        with mock.patch.object(Q.U,'prepare_job',return_value=summary) as prepare,\
             mock.patch.object(Q.U,'_job_read',return_value=(directory/'job.json',saved)) as verify:
            result = self.prepare(prepare_jobs=True)
            self.assertEqual(result['local_handoffs_prepared'],1)
            checked = Q.inspect(self.root/self.output/'queue.json',root=self.root)
            self.assertEqual(checked['components'][0]['job']['state'],'prepared_not_submitted')
            self.assertEqual(prepare.call_count,1)
            self.assertEqual(verify.call_count,2)
            self.assertEqual(prepare.call_args.args[0]['base_sha256'],self.program_sha)
            queue_path = self.root/self.output/'queue.json'
            queue = U.read(queue_path)
            queue['components'][0]['job']['state'] = 'submitted'
            U.write(queue_path,queue)
            with self.assertRaisesRegex(ValueError,'job state'): Q.inspect(queue_path,root=self.root)


if __name__ == '__main__':
    unittest.main()
