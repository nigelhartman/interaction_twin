"""Temporary adapter-contract fixtures; no model, GPU or catalog assets produced."""
import copy
from pathlib import Path
import shutil
import unittest
from unittest import mock

import numpy as np
from PIL import Image

import initial_sam3d_import as N
import sam3d_milestone as M
import upgrade_bundles as U
import tests.test_initial_sam3d_handoff as F
from tests.test_upgrade_bundles import vertex_glb


class InitialNativeImportTests(unittest.TestCase):
    def setUp(self):
        self.anchor = F.InitialHandoffTests()
        self.anchor.setUp()
        self.addCleanup(self.anchor.doCleanups)
        self.anchor.prepare()
        self.root = self.anchor.root
        self.handoff = self.root / self.anchor.output / 'handoff.json'
        self.bundle = self.handoff.parent
        self.output = self.root / 'data/returned-software-fixture'
        self.output.mkdir()
        inputs = self.output / 'inputs'; inputs.mkdir()
        shutil.copyfile(self.bundle / 'input.json', inputs / 'manifest.json')
        for source, target in [('image.png','image-source.png'), ('mask.png','mask-source.png'),
                               ('image.png','rgb.png'), ('mask.png','mask.png')]:
            shutil.copyfile(self.bundle / source, inputs / target)
        (self.output / 'object.glb').write_bytes(vertex_glb())
        names = ['x','y','z','nx','ny','nz','f_dc_0','f_dc_1','f_dc_2','opacity','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3']
        header = 'ply\nformat binary_little_endian 1.0\nelement vertex 1\n' + ''.join('property float '+n+'\n' for n in names) + 'end_header\n'
        (self.output / 'object-splat.ply').write_bytes(header.encode() + np.zeros(len(names),dtype='<f4').tobytes())
        self.layout = {'rotation': np.array([1.,0.,0.,0.]), 'translation': np.array([0.,0.,2.]), 'scale': np.ones(3)}
        record,rgb,mask = M.read_inputs(self.bundle / 'input.json')
        self.metadata = {'backend':'Meta SAM 3D Objects','model_repository':'facebook/sam-3d-objects',
            'status':'candidate_generated_pending_visual_review','seed':42,'appearance_requested':'vertex-color',
            'learned_inference_executed':True,'errors':[],'inputs':record,
            'code':{'commit':M.SAM3D_COMMIT,'expected_commit':M.SAM3D_COMMIT,'origin':M.OFFICIAL,'tracked_changes':''},
            'runtime':{'platform':'Linux','device_requested':'cuda','cuda_available':True},
            'checkpoint':{'weights_hashed':True,'files':[{'path':'/inert-fixture/model.ckpt','sha256':'3'*64}]},
            'model_input_sha256':{n:U.sha(inputs/n) for n in ('rgb.png','mask.png')},
            'model_input_array_sha256':{'rgb_uint8':U.digest(rgb.tobytes()),'mask_bool':U.digest(mask.tobytes())},
            'outputs':{'geometry_reconstructed_by_model':True,'manual_geometry_fallback':False,
                'metric_scale_verified':False,'direct_original_video_texture_projection':False,
                'appearance':'SYNTHETIC SOFTWARE CONTRACT FIXTURE; never retained','appearance_mode':'vertex-color'}}
        self.refresh()

    def refresh(self):
        np.savez_compressed(self.output / 'layout-and-pointmap.npz', **self.layout)
        self.metadata['outputs']['files']=[{'file':n,'sha256':U.sha(self.output/n),'bytes':(self.output/n).stat().st_size}
                                          for n in ('object.glb','object-splat.ply','layout-and-pointmap.npz')]
        self.metadata['outputs']['layout_fields']={k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in self.layout.items()}
        U.write(self.output / 'metadata.json', self.metadata)

    def check(self):
        return N.inspect_output(self.handoff,self.output,root=self.root)

    def test_initial_check_never_reads_a_baseline_program_or_old_poses(self):
        before = M._records(M._tree(self.output))
        with (mock.patch.object(U,'context',side_effect=AssertionError('No baseline context')),
              mock.patch.object(M,'inspect_input_bundle',side_effect=AssertionError('No upgrade bundle'))):
            report = self.check()[0]
        self.assertFalse(report['baseline_required']);self.assertFalse(report['baseline_asset_read']);self.assertFalse(report['old_poses_read'])
        self.assertFalse(report['execution_independently_verified']);self.assertFalse(report['model_selection_changed'])
        self.assertEqual(report['component_id'],'widget_92');self.assertEqual(report['source']['timestamp_s'],2.5)
        self.assertNotIn('comparison',report);self.assertEqual(M._records(M._tree(self.output)),before)

    def test_fresh_import_preserves_entire_raw_and_handoff_and_native_glb(self):
        dest=self.root/'data/initial-native-v1';raw=M._tree(self.output);handoff=M._tree(self.bundle)
        with mock.patch.object(U,'context',side_effect=AssertionError('No baseline context')):
            result=N.import_output(self.handoff,self.output,dest,root=self.root)
        for name,value in raw.items():self.assertEqual((dest/'raw'/name).read_bytes(),value)
        for name,value in handoff.items():self.assertEqual((dest/'initial-handoff'/name).read_bytes(),value)
        self.assertEqual((dest/'asset.glb').read_bytes(),raw['object.glb'])
        model=U.read(dest/'metadata.json')
        self.assertNotIn('upgrade_lineage',model);self.assertNotIn('milestone_lineage',model)
        self.assertFalse(model['selected']);self.assertFalse(model['catalog_changed']);self.assertFalse(model['simulation_ready'])
        self.assertFalse(model['native_layout_candidate']['accepted_video_pose'])
        self.assertFalse(model['initial_reconstruction_lineage']['baseline_asset_read'])
        self.assertNotIn('frames',model);self.assertFalse((self.root/'workspace').exists())
        self.assertFalse(result['source_video_pose_validated']);self.assertFalse(result['simulation_executed'])
        self.assertEqual(result['import_files'],M._records({k:v for k,v in M._tree(dest).items() if k!='import.json'}))

    def test_explicit_fixture_or_failed_model_cannot_import(self):
        for key,value in [('fixture',True),('status','failed_no_fallback'),('learned_inference_executed',False)]:
            old=copy.deepcopy(self.metadata);self.metadata[key]=value;self.refresh()
            with self.subTest(key=key),self.assertRaises(ValueError):self.check()
            self.metadata=old

    def test_source_handoff_or_original_source_changes_fail_closed(self):
        self.anchor.video.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'source video'):self.check()

    def test_input_manifest_mask_and_returned_geometry_are_byte_bound(self):
        for filename in ['inputs/manifest.json','inputs/mask-source.png','object.glb']:
            p=self.output/filename;original=p.read_bytes();p.write_bytes(b'changed')
            with self.subTest(filename=filename),self.assertRaises(ValueError):self.check()
            p.write_bytes(original)

    def test_reencoded_model_pixels_cannot_replace_original_arrays(self):
        rgb=self.anchor.rgb.copy();rgb[0,0,0]^=1;Image.fromarray(rgb).save(self.output/'inputs/rgb.png')
        self.metadata['model_input_sha256']['rgb.png']=U.sha(self.output/'inputs/rgb.png');self.refresh()
        with self.assertRaisesRegex(ValueError,'model arrays'):self.check()

    def test_unexpected_output_file_and_symlink_refused(self):
        extra=self.output/'unexpected.txt';extra.write_text('inert')
        with self.assertRaisesRegex(ValueError,'inventory'):self.check()
        extra.unlink();p=self.output/'object.glb';p.unlink();p.symlink_to(self.bundle/'image.png')
        with self.assertRaisesRegex(ValueError,'Symlink'):self.check()

    def test_wrong_official_pin_unhashed_weights_seed_and_photographic_claim_rejected(self):
        for mutate in [lambda m:m['code'].update(commit='b'*40),lambda m:m['checkpoint'].update(weights_hashed=False),
                       lambda m:m.update(seed=43),lambda m:m['outputs'].update(direct_original_video_texture_projection=True)]:
            old=copy.deepcopy(self.metadata);mutate(self.metadata);self.refresh()
            with self.assertRaises(ValueError):self.check()
            self.metadata=old

    def test_optional_nan_pointmap_is_preserved_without_becoming_video_pose(self):
        self.layout['pointmap']=np.array([[[np.nan,np.nan,np.nan],[1.,2.,3.]]]);self.refresh()
        report=self.check()[0];self.assertEqual(report['optional_pointmap']['missing_points'],1)
        self.assertFalse(report['optional_pointmap']['used_for_video_pose'])
        before=(self.output/'layout-and-pointmap.npz').read_bytes();dest=self.root/'data/nan-pointmap-v1'
        N.import_output(self.handoff,self.output,dest,root=self.root)
        self.assertEqual((dest/'raw/layout-and-pointmap.npz').read_bytes(),before)

    def test_nonfinite_or_anisotropic_transform_refused(self):
        for scale in [np.array([1.,2.,1.]),np.array([1.,np.inf,1.])]:
            self.layout['scale']=scale;self.refresh()
            with self.assertRaises(ValueError):self.check()

    def test_destination_must_be_fresh_separate_and_inside_data(self):
        for destination in [self.bundle/'nested',self.output/'nested',self.root/'outside-data',self.bundle]:
            with self.subTest(destination=destination),self.assertRaises(ValueError):
                N.import_output(self.handoff,self.output,destination,root=self.root)
        dest=self.root/'data/fresh';dest.mkdir();(dest/'preserve').write_bytes(b'keep')
        with self.assertRaisesRegex(ValueError,'fresh'):N.import_output(self.handoff,self.output,dest,root=self.root)
        self.assertEqual((dest/'preserve').read_bytes(),b'keep')

    def test_changed_handoff_or_output_during_import_stops_before_writing(self):
        # This specific mutation is after the first native contract check.
        real_inspect=N.inspect_output
        def inspect_then_change(*args,**kwargs):
            result=real_inspect(*args,**kwargs);(self.output/'object.glb').write_bytes(b'changed after validation');return result
        dest=self.root/'data/stopped-before-write'
        with mock.patch.object(N,'inspect_output',side_effect=inspect_then_change),self.assertRaisesRegex(ValueError,'changed during'):
            N.import_output(self.handoff,self.output,dest,root=self.root)
        self.assertFalse(dest.exists())


if __name__=='__main__':unittest.main()
