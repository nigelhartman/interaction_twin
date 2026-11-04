"""Tiny saved-image/source fixtures: no reconstruction/model/baseline exists."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

import initial_sam3d_handoff as H
import upgrade_bundles as U


class InitialHandoffTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name).resolve()
        for name in H.DEPENDENCIES:
            target=self.root/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(H.ROOT/name,target)
        self.case=self.root/'data/arbitrary-capture'
        self.input=self.case/'portable/input.json'
        self.selection=self.case/'selections/arbitrary-part/selection.json'
        for p in [self.input.parent,self.selection.parent/'rank',self.case/'frames',self.case/'segmentation',self.case/'source']:p.mkdir(parents=True,exist_ok=True)
        self.rgb=np.array([[[10,20,30],[40,50,60],[70,80,90]],[[15,25,35],[45,55,65],[75,85,95]]],dtype=np.uint8)
        self.mask=np.array([[0,255,0],[0,255,0]],dtype=np.uint8)
        self.frame=self.case/'frames/000007.jpg';Image.fromarray(self.rgb).save(self.frame,format='PNG')
        self.raw_mask=self.case/'segmentation/mask.png';Image.fromarray(self.mask).save(self.raw_mask)
        for base in [self.input.parent,self.selection.parent/'rank']:
            Image.fromarray(self.rgb).save(base/'image.png');Image.fromarray(self.mask).save(base/'mask.png')
        self.video=self.case/'source/video.mp4';self.video.write_bytes(b'INERT_SOURCE_VIDEO_IDENTITY_FIXTURE')
        self.source={'file':'source/video.mp4','sha256':U.sha(self.video),'original_sha256':'a'*64}
        U.write(self.case/'manifest.json',{'case_id':'arbitrary-capture','source':self.source})
        U.write(self.case/'frames/index.json',{'time_domain':U.TIME_DOMAIN,'frames':[{'index':7,'file':'frames/000007.jpg','timestamp_s':2.5}]})
        self.row={'selection_rank':1,'score':.9,'sample_index':7,'source_sample_index':7,'timestamp_s':2.5,
          'source_case':'arbitrary-capture','source_track_id':'widget_92','object_id':'widget_92',
          'image_file':'frames/000007.jpg','image_sha256':U.sha(self.frame),
          'mask_file':'segmentation/mask.png','mask_sha256':U.sha(self.raw_mask),
          'mask_provenance':{'prompt_origin':'model_generated_prompt'},'model_provider':'SAM inert contract fixture','mask_scope':'visible widget',
          'exports':{'rgb':'rank/image.png','mask':'rank/mask.png','rgb_sha256':U.sha(self.selection.parent/'rank/image.png'),
                     'mask_sha256':U.sha(self.selection.parent/'rank/mask.png'),'rgb_mask_dimensions':[3,2],
                     'resized':False,'background_invented':False}}
        self.selected={'object_id':'widget_92','source':self.source,'selection_method':{'agent_visual_choice':False,'reads_geometry_or_corner_annotations':False},'selected':[self.row]}
        U.write(self.selection,self.selected)
        self.manifest={'schema_version':'0.1','object_id':'widget_92','image':'image.png','mask':'mask.png',
          'image_sha256':U.sha(self.input.parent/'image.png'),'mask_sha256':U.sha(self.input.parent/'mask.png'),
          'source':{'video_sha256':U.sha(self.video),'sample_index':7,'source_sample_index':7,'source_case':'arbitrary-capture','timestamp_s':2.5,'time_domain':U.TIME_DOMAIN},
          'mask_provenance':dict(self.row['mask_provenance'],backend=self.row['model_provider'],manual_intervention=False,scope='visible widget'),
          'selection':{'rank':1,'score':.9,'selection_sha256':U.sha(self.selection),'agent_visual_selection':False,'manual_geometry_used':False,'manual_camera_corners_used':False}}
        U.write(self.input,self.manifest)
        self.output='data/initial-sam3d-handoffs/arbitrary-capture-v1'

    def prepare(self):return H.prepare(self.input,self.selection,self.output,root=self.root)

    def test_prepares_without_baseline_and_reuses_adapter_input_validator(self):
        with mock.patch.object(U,'context',side_effect=AssertionError('Baseline context must not be used')):
            result=self.prepare()
            checked=H.inspect(self.root/self.output/'handoff.json',root=self.root)
        self.assertEqual(result['status'],'initial_reconstruction_prepared')
        self.assertFalse(checked['baseline_required'])
        self.assertTrue(checked['no_inference'])
        self.assertFalse((self.root/'models').exists())
        binding=U.read(self.root/self.output/'source-binding.json')
        self.assertEqual(binding['decoded_arrays']['rgb_sha256'],U.digest(self.rgb.tobytes()))
        self.assertFalse(binding['baseline_asset_read'])
        self.assertFalse(binding['old_poses_read'])

    def test_changed_video_rejected_before_output(self):
        self.video.write_bytes(b'different source')
        with self.assertRaisesRegex(ValueError,'source video'):self.prepare()
        self.assertFalse((self.root/self.output).exists())

    def test_different_pts_rejected(self):
        self.manifest['source']['timestamp_s']=3.
        U.write(self.input,self.manifest)
        with self.assertRaisesRegex(ValueError,'PTS'):self.prepare()

    def test_assisted_mask_rejected(self):
        self.manifest['mask_provenance']['manual_intervention']=True
        U.write(self.input,self.manifest)
        with self.assertRaisesRegex(ValueError,'automatic'):self.prepare()

    def test_baseline_bundle_cannot_impersonate_initial_input(self):
        self.manifest['baseline']={'backend':'TripoSR'}
        U.write(self.input,self.manifest)
        with self.assertRaisesRegex(ValueError,'baseline'):self.prepare()

    def test_parent_traversing_input_cannot_open_other_artifacts(self):
        self.manifest['image']='../../outside.png'
        U.write(self.input,self.manifest)
        with self.assertRaisesRegex(ValueError,'own full-frame'):self.prepare()

    def test_pixel_changes_rejected(self):
        Image.fromarray(np.zeros_like(self.rgb)).save(self.input.parent/'image.png')
        with self.assertRaisesRegex(ValueError,'pixel file'):self.prepare()

    def test_source_mask_changes_rejected(self):
        Image.fromarray(np.flip(self.mask,axis=0)^255).save(self.raw_mask)
        with self.assertRaisesRegex(ValueError,'frame/SAM'):self.prepare()

    def test_tampered_portable_adapter_rejected(self):
        self.prepare()
        (self.root/self.output/'adapter/stage3_sam3d.py').write_text('different adapter')
        with self.assertRaisesRegex(ValueError,'inventory/hash'):H.inspect(self.root/self.output/'handoff.json',root=self.root)

    def test_rebound_run_policy_rejected_even_with_rehashed_inventory(self):
        self.prepare();directory=self.root/self.output
        run=U.read(directory/'run-plan.json');run['commands_executed']=True;U.write(directory/'run-plan.json',run)
        handoff=U.read(directory/'handoff.json')
        for row in handoff['files']:
            if row['path']=='run-plan.json':row.update(sha256=U.sha(directory/'run-plan.json'),bytes=(directory/'run-plan.json').stat().st_size)
        U.write(directory/'handoff.json',handoff)
        with self.assertRaisesRegex(ValueError,'runtime/command'):H.inspect(directory/'handoff.json',root=self.root)

    def test_unknown_file_refused_without_reading(self):
        self.prepare();directory=self.root/self.output
        hidden=directory/'unexpected-secret';hidden.write_text('must not read')
        actual=U.sha
        def guarded(path):
            if Path(path)==hidden:raise AssertionError('Unexpected artifact read')
            return actual(path)
        with mock.patch.object(U,'sha',side_effect=guarded):
            with self.assertRaisesRegex(ValueError,'file inventory'):H.inspect(directory/'handoff.json',root=self.root)

    def test_source_or_output_symlinks_rejected(self):
        source=self.input.parent/'image.png';source.unlink();source.symlink_to(self.frame)
        with self.assertRaisesRegex(ValueError,'Symlink'):self.prepare()

    def test_repeated_output_preserved(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError,'Preserve previous'):self.prepare()


if __name__ == '__main__':unittest.main()
