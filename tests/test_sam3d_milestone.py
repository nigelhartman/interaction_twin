"""Temporary synthetic contract tests; no GPU/model or retained model fixtures."""
import copy
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import sam3d_milestone as M
import upgrade_bundles as U
from tests.test_upgrade_bundles import vertex_glb


class SAM3DMilestoneTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.bundle = self.root / 'data/input-v1'
        self.bundle.mkdir(parents=True)
        self.rgb = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
        self.mask = np.zeros((4, 4), dtype=np.uint8); self.mask[1:3, 1:3] = 255
        Image.fromarray(self.rgb).save(self.bundle / 'image.png')
        Image.fromarray(self.mask).save(self.bundle / 'mask.png')
        (self.bundle / 'baseline').mkdir()
        (self.bundle / 'baseline/asset.glb').write_bytes(vertex_glb())
        U.write(self.bundle / 'baseline/metadata.json', {'backend': 'SYNTHETIC SOFTWARE TEST', 'source': {'video_sha256': '1'*64, 'sample_index': 1, 'timestamp_s': .25, 'time_domain': U.TIME_DOMAIN}})
        U.write(self.bundle / 'baseline/interaction.json', {'fixture': True})
        self.info = {'twin_id': 'synthetic', 'track_id': 'part', 'base_sha256': U.sha(self.bundle / 'baseline/interaction.json'),
            'mask_provenance': {'manual_intervention': False, 'backend': 'SYNTHETIC', 'prompt_origin': 'model_generated_prompt'},
            'selection_sha256': '7'*64, 'selection_rank': 1, 'selection_score': .75,
            'model_metadata_path': 'data/input-v1/baseline/metadata.json', 'model_metadata_sha256': U.sha(self.bundle / 'baseline/metadata.json'),
            'baseline_asset_path': 'data/input-v1/baseline/asset.glb', 'baseline_asset_sha256': U.sha(self.bundle / 'baseline/asset.glb'),
            'image_sha256': U.sha(self.bundle / 'image.png'), 'mask_sha256': U.sha(self.bundle / 'mask.png'),
            'source': {'video_sha256': '1'*64, 'sample_index': 1, 'timestamp_s': .25, 'time_domain': U.TIME_DOMAIN,
                       'width': 4, 'height': 4, 'rgb_array_sha256': U.digest(self.rgb.tobytes()), 'mask_array_sha256': U.digest((self.mask > 0).tobytes())}}
        U.write(self.bundle / 'input-binding.json', self.info)
        source = {'format': 'sam3d-source-binding', 'version': 1, 'case_id': 'synthetic', 'track_id': 'part', 'source': self.info['source'],
                  'manual_geometry_used': False, 'manual_pose_used': False, 'physical_identity_verified': False, 'physical_shape_verified': False, 'metric_scale_verified': False, 'assembly_verified': False,
                  'pixel_policy': {'full_native_frame': True, 'crop': False, 'rotation': False, 'resize': False, 'inpainting': False, 'pixel_values_modified': False, 'mask_modified': False},
                  'native_rgb': {'path': 'image.png', 'sha256': self.info['image_sha256']}, 'native_mask': {'path': 'mask.png', 'sha256': self.info['mask_sha256']},
                  'baseline': {'same_rgb_and_mask': True, 'physical_accuracy_verified': False}}
        for field, path, sha in [('native_asset','baseline/asset.glb',self.info['baseline_asset_sha256']), ('metadata','baseline/metadata.json',self.info['model_metadata_sha256']), ('program','baseline/interaction.json',self.info['base_sha256'])]:
            source['baseline'][field+'_path'] = path; source['baseline'][field+'_sha256'] = sha
        U.write(self.bundle / 'source-binding.json', source)
        manifest = {'object_id': 'part', 'image': 'image.png', 'mask': 'mask.png', 'image_sha256': self.info['image_sha256'], 'mask_sha256': self.info['mask_sha256'],
                    'source': self.info['source'], 'mask_provenance': {'manual_intervention': False, 'backend': 'SYNTHETIC', 'prompt_origin': 'model_generated_prompt'},
                    'selection': {'selection_sha256': self.info['selection_sha256'], 'rank': 1, 'score': .75, 'agent_visual_selection': False, 'manual_geometry_used': False, 'manual_camera_corners_used': False},
                    'source_binding': {'path': 'source-binding.json', 'sha256': U.sha(self.bundle / 'source-binding.json')},
                    'input_binding': {'path': 'input-binding.json', 'sha256': U.sha(self.bundle / 'input-binding.json')}}
        U.write(self.bundle / 'input.json', manifest)
        (self.bundle / 'adapter').mkdir()
        (self.bundle / 'adapter/stage3_sam3d.py').write_text('# SOFTWARE CONTRACT FIXTURE; never executed\n')
        U.write(self.bundle/'run-plan.json', {'format':'sam3d-single-object-smoke-plan', 'version':1, 'input_manifest':'input.json', 'input_manifest_sha256':U.sha(self.bundle/'input.json'), 'adapter':'adapter/stage3_sam3d.py', 'adapter_sha256':U.sha(self.bundle/'adapter/stage3_sam3d.py'), 'official_code_commit':M.SAM3D_COMMIT, 'device':'cuda', 'seed':42, 'appearance':'vertex-color', 'parts':1, 'automatic_retry':False})
        self.freeze()
        self.patcher = patch.object(U, 'context', return_value=copy.deepcopy(self.info))
        self.patcher.start(); self.addCleanup(self.patcher.stop)
        self.output = self.root / 'data/returned-output-v1'; self.output.mkdir()
        inputs = self.output / 'inputs'; inputs.mkdir()
        shutil.copy2(self.bundle / 'input.json', inputs / 'manifest.json')
        for source, dest in [('image.png','image-source.png'), ('mask.png','mask-source.png'), ('image.png','rgb.png'), ('mask.png','mask.png')]:
            shutil.copy2(self.bundle / source, inputs / dest)
        (self.output / 'object.glb').write_bytes(vertex_glb())
        names = ['x','y','z','nx','ny','nz','f_dc_0','f_dc_1','f_dc_2','opacity','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3']
        header = 'ply\nformat binary_little_endian 1.0\nelement vertex 1\n' + ''.join('property float '+n+'\n' for n in names) + 'end_header\n'
        (self.output / 'object-splat.ply').write_bytes(header.encode() + np.zeros(len(names), dtype='<f4').tobytes())
        self.layout = {'rotation': np.array([1.,0.,0.,0.]), 'translation': np.array([0.,0.,2.]), 'scale': np.ones(3)}
        np.savez_compressed(self.output / 'layout-and-pointmap.npz', **self.layout)
        record, _, _ = M.read_inputs(self.bundle / 'input.json')
        self.meta = {'backend': 'Meta SAM 3D Objects', 'model_repository': 'facebook/sam-3d-objects', 'status': 'candidate_generated_pending_visual_review',
             'seed': 42, 'appearance_requested': 'vertex-color',
             'learned_inference_executed': True, 'errors': [], 'inputs': record,
             'code': {'commit': M.SAM3D_COMMIT, 'expected_commit': M.SAM3D_COMMIT, 'origin': M.OFFICIAL, 'tracked_changes': ''},
             'runtime': {'platform': 'Linux', 'device_requested': 'cuda', 'cuda_available': True},
             'checkpoint': {'weights_hashed': True, 'files': [{'path': '/test/model.ckpt','sha256': '3'*64}]},
             'model_input_sha256': {n: U.sha(inputs / n) for n in ('rgb.png','mask.png')},
             'model_input_array_sha256': {'rgb_uint8': self.info['source']['rgb_array_sha256'], 'mask_bool': self.info['source']['mask_array_sha256']},
             'outputs': {'geometry_reconstructed_by_model': True, 'manual_geometry_fallback': False, 'metric_scale_verified': False, 'direct_original_video_texture_projection': False,
                  'appearance': 'SYNTHETIC CONTRACT TEST; not retained', 'appearance_mode': 'vertex-color'}}
        self.refresh_output()

    def freeze(self):
        files = M._tree(self.bundle)
        files.pop('bundle-manifest.json', None)
        U.write(self.bundle / 'bundle-manifest.json', {'format':'sam3d-gpu-input-bundle','version':1,'files':M._records(files)})

    def refresh_output(self):
        self.meta['outputs']['files'] = [{'file': n, 'sha256': U.sha(self.output / n), 'bytes': (self.output / n).stat().st_size} for n in ('object.glb','object-splat.ply','layout-and-pointmap.npz')]
        self.meta['outputs']['layout_fields'] = {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k,v in self.layout.items()}
        U.write(self.output / 'metadata.json', self.meta)

    def check(self):
        return M.inspect_output(self.bundle, self.output, root=self.root)

    def test_readonly_check_preserves_bytes_and_all_physical_unknowns(self):
        before = M._records(M._tree(self.output)); report = self.check()[0]
        self.assertEqual(before, M._records(M._tree(self.output)))
        self.assertEqual(report['status'], 'validated_native_candidate_pending_review')
        self.assertIsNone(report['comparison']['winner'])
        for k in ('metric_scale_verified','physical_accuracy_verified','source_video_pose_validated','assembly_verified','contact_verified','simulation_executed','model_selection_changed'):
            self.assertIs(report[k], False)
        self.assertFalse(report['comparison']['native_dimensions_comparable'])

    def test_import_writes_new_candidate_without_fake_upgrade_lineage(self):
        dest = self.root / 'data/import-v1'
        original = M._tree(self.output)
        report = M.import_output(self.bundle, self.output, dest, root=self.root)
        self.assertEqual((dest/'asset.glb').read_bytes(), original['object.glb'])
        for name, raw in original.items(): self.assertEqual((dest/'raw'/name).read_bytes(), raw)
        model = U.read(dest/'metadata.json')
        self.assertNotIn('upgrade_lineage', model)
        self.assertFalse(model['sam3d_milestone_complete'])
        self.assertFalse(model['native_layout_candidate']['accepted_video_pose'])
        self.assertEqual(next(x['sha256'] for x in model['artifact_files'] if x['file']=='asset.glb'), U.digest(original['object.glb']))
        self.assertFalse(report['model_selection_changed'])
        with self.assertRaisesRegex(ValueError, 'fresh'): M.import_output(self.bundle, self.output, dest, root=self.root)

    def test_changed_bundle_file_rejected(self):
        (self.bundle/'mask.png').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'manifest'): self.check()

    def test_changed_current_baseline_rejected(self):
        U.context.return_value = dict(self.info, base_sha256='a'*64)
        with self.assertRaisesRegex(ValueError, 'current source'): self.check()

    def test_input_path_rejected_before_generic_reader_can_follow_it(self):
        manifest = U.read(self.bundle/'input.json'); manifest['image'] = '../../private.env'
        U.write(self.bundle/'input.json', manifest)
        plan = U.read(self.bundle/'run-plan.json'); plan['input_manifest_sha256'] = U.sha(self.bundle/'input.json'); U.write(self.bundle/'run-plan.json',plan)
        self.freeze()
        with patch.object(M, 'read_inputs', side_effect=AssertionError('Must not read')):
            with self.assertRaisesRegex(ValueError, 'own exact'): self.check()

    def test_explicit_fixture_rejected(self):
        self.meta['fixture'] = True; self.refresh_output()
        with self.assertRaisesRegex(ValueError, 'Fixture'): self.check()

    def test_failed_inference_cannot_import(self):
        self.meta['status'] = 'failed_no_fallback'; self.refresh_output()
        with self.assertRaisesRegex(ValueError, 'did not finish'): self.check()

    def test_seed_and_appearance_match_predeclared_first_run(self):
        for key, value in (('seed',43), ('seed',True), ('appearance_requested','uv-baked')):
            previous = self.meta[key]; self.meta[key] = value; self.refresh_output()
            with self.subTest(key=key,value=value), self.assertRaisesRegex(ValueError,'first-run plan'): self.check()
            self.meta[key] = previous

    def test_wrong_nested_metadata_types_fail_deterministically(self):
        for key in ('runtime','code','checkpoint','inputs','outputs'):
            previous = self.meta[key]; self.meta[key] = []; U.write(self.output/'metadata.json',self.meta)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError,'object'): self.check()
            self.meta[key] = previous

    def test_mask_provenance_cannot_disown_actual_automatic_provider(self):
        manifest = U.read(self.bundle/'input.json')
        manifest['mask_provenance']['backend'] = 'wrong-model'
        manifest['mask_provenance']['prompt_origin'] = 'assistant_visual_annotation'
        U.write(self.bundle/'input.json',manifest)
        plan = U.read(self.bundle/'run-plan.json'); plan['input_manifest_sha256'] = U.sha(self.bundle/'input.json'); U.write(self.bundle/'run-plan.json',plan)
        self.freeze()
        with self.assertRaisesRegex(ValueError,'manual segmentation'): self.check()

    def test_malformed_nested_source_binding_fails_without_attribute_error(self):
        original_binding = U.read(self.bundle/'source-binding.json')
        original_input = U.read(self.bundle/'input.json')
        original_plan = U.read(self.bundle/'run-plan.json')
        for role in ('source','native_rgb','native_mask','baseline'):
            for invalid in ([], 5, None):
                binding = copy.deepcopy(original_binding); binding[role] = invalid
                U.write(self.bundle/'source-binding.json',binding)
                manifest = copy.deepcopy(original_input); manifest['source_binding']['sha256'] = U.sha(self.bundle/'source-binding.json')
                U.write(self.bundle/'input.json',manifest)
                plan = copy.deepcopy(original_plan); plan['input_manifest_sha256'] = U.sha(self.bundle/'input.json'); U.write(self.bundle/'run-plan.json',plan)
                self.freeze()
                with self.subTest(role=role,invalid=invalid), self.assertRaisesRegex(ValueError,'must be an object'):
                    M.inspect_input_bundle(self.bundle,root=self.root)

    def test_malformed_nested_input_fails_before_generic_reader(self):
        original = U.read(self.bundle/'input.json')
        for role in ('source','mask_provenance','selection'):
            manifest = copy.deepcopy(original); manifest[role] = []
            U.write(self.bundle/'input.json',manifest)
            plan = U.read(self.bundle/'run-plan.json'); plan['input_manifest_sha256'] = U.sha(self.bundle/'input.json'); U.write(self.bundle/'run-plan.json',plan)
            self.freeze()
            with self.subTest(role=role), patch.object(M,'read_inputs',side_effect=AssertionError('Must not call generic reader')):
                with self.assertRaisesRegex(ValueError,'must be an object'):
                    M.inspect_input_bundle(self.bundle,root=self.root)

    def test_frozen_project_adapter_hash_cannot_drift(self):
        (self.bundle/'adapter/stage3_sam3d.py').write_text('# changed\n'); self.freeze()
        with self.assertRaisesRegex(ValueError,'adapter pin'): self.check()

    def test_wrong_pin_and_unhashed_checkpoint_rejected(self):
        self.meta['code']['commit'] = 'b'*40; self.refresh_output()
        with self.assertRaisesRegex(ValueError, 'code pin'): self.check()
        self.meta['code']['commit'] = M.SAM3D_COMMIT
        self.meta['checkpoint']['weights_hashed'] = False; self.refresh_output()
        with self.assertRaisesRegex(ValueError, 'checkpoint'): self.check()

    def test_source_input_bytes_cannot_be_swapped(self):
        (self.output/'inputs/image-source.png').write_bytes(b'other')
        with self.assertRaisesRegex(ValueError, 'source RGB'): self.check()

    def test_reencoded_model_array_must_still_match_source(self):
        rgb = self.rgb.copy(); rgb[0,0,0] ^= 1
        Image.fromarray(rgb).save(self.output/'inputs/rgb.png')
        self.meta['model_input_sha256']['rgb.png'] = U.sha(self.output/'inputs/rgb.png'); self.refresh_output()
        with self.assertRaisesRegex(ValueError, 'Actual model arrays'): self.check()

    def test_geometry_bytes_must_match_adapter_attestation(self):
        (self.output/'object.glb').write_bytes(b'wrong')
        with self.assertRaisesRegex(ValueError, 'inventory/hash'): self.check()

    def test_photographic_texture_claim_rejected(self):
        self.meta['outputs']['direct_original_video_texture_projection'] = True; self.refresh_output()
        with self.assertRaisesRegex(ValueError, 'appearance provenance'): self.check()

    def test_anisotropic_layout_is_not_silently_averaged(self):
        self.layout['scale'] = np.array([1.,2.,1.])
        np.savez_compressed(self.output/'layout-and-pointmap.npz', **self.layout); self.refresh_output()
        with self.assertRaisesRegex(ValueError, 'Anisotropic'): self.check()

    def test_official_nan_pointmap_gaps_preserved_and_not_used_as_pose(self):
        self.layout['pointmap'] = np.array([[[np.nan,np.nan,np.nan], [1.,2.,3.]]])
        np.savez_compressed(self.output/'layout-and-pointmap.npz', **self.layout); self.refresh_output()
        before = (self.output/'layout-and-pointmap.npz').read_bytes()
        report = self.check()[0]
        self.assertEqual(report['optional_pointmap']['finite_points'],1)
        self.assertEqual(report['optional_pointmap']['missing_points'],1)
        self.assertFalse(report['optional_pointmap']['used_for_video_pose'])
        self.assertEqual((self.output/'layout-and-pointmap.npz').read_bytes(),before)

    def test_infinite_pointmap_and_nonfinite_transform_still_rejected(self):
        self.layout['pointmap'] = np.array([[[np.inf,0.,0.]]])
        np.savez_compressed(self.output/'layout-and-pointmap.npz', **self.layout); self.refresh_output()
        with self.assertRaisesRegex(ValueError,'Infinite optional'): self.check()
        self.layout.pop('pointmap'); self.layout['translation'] = np.array([0.,0.,np.nan])
        np.savez_compressed(self.output/'layout-and-pointmap.npz', **self.layout); self.refresh_output()
        with self.assertRaisesRegex(ValueError,'Nonfinite native decoded'): self.check()

    def test_exact_alpha_limits_in_official_ply_preserved_not_render_validated(self):
        raw = (self.output/'object-splat.ply').read_bytes()
        split = raw.index(b'end_header\n') + len(b'end_header\n')
        values = np.frombuffer(raw[split:],dtype='<f4').copy(); values[9] = np.inf
        changed = raw[:split] + values.tobytes(); (self.output/'object-splat.ply').write_bytes(changed); self.refresh_output()
        report = self.check()[0]
        self.assertEqual(report['splat_contract']['infinite_opacity_logits'],1)
        self.assertFalse(report['splat_contract']['splat_rendering_validated'])
        self.assertEqual((self.output/'object-splat.ply').read_bytes(),changed)

    def test_layout_object_array_never_unpickled(self):
        self.layout['pointmap'] = np.array([{'not':'numeric'}],dtype=object)
        np.savez_compressed(self.output/'layout-and-pointmap.npz', **self.layout); self.refresh_output()
        with self.assertRaisesRegex(ValueError, 'allow_pickle=False'): self.check()

    def test_empty_or_malformed_gaussian_ply_rejected(self):
        (self.output/'object-splat.ply').write_bytes(b'ply\nend_header\n'); self.refresh_output()
        with self.assertRaisesRegex(ValueError, 'Gaussian PLY'): self.check()

    def test_unexpected_output_and_symlink_rejected(self):
        (self.output/'extra.txt').write_text('not declared')
        with self.assertRaisesRegex(ValueError, 'inventory'): self.check()
        (self.output/'extra.txt').unlink()
        (self.output/'extra.txt').symlink_to(self.bundle/'image.png')
        with self.assertRaisesRegex(ValueError, 'Symlink'): self.check()

    def test_output_outside_data_tree_rejected(self):
        with self.assertRaisesRegex(ValueError, 'data directories'): M.inspect_output(self.bundle, self.root/'secrets', root=self.root)

    def test_import_cannot_append_into_frozen_input_or_output(self):
        for parent in (self.bundle, self.output):
            with self.subTest(parent=parent), self.assertRaisesRegex(ValueError, 'separate'):
                M.import_output(self.bundle, self.output, parent/'new-import', root=self.root)
            self.assertFalse((parent/'new-import').exists())


class ActualFrozenMilestoneInputTests(unittest.TestCase):
    def test_actual_prepared_input_bundle_only_not_inference(self):
        directory = M.ROOT/'data/sam3d-gpu-20261101/earbud-a-sample40-v1'
        if not directory.exists(): self.skipTest('Optional actual prepared bundle absent')
        value = M.inspect_input_bundle(directory)
        self.assertEqual(value['binding']['track_id'], 'earbud_a')
        self.assertEqual(value['binding']['source']['sample_index'], 40)
        self.assertEqual(value['binding']['source']['timestamp_s'], 13.001144)
        self.assertEqual(U.sha(directory/'bundle-manifest.json'), 'e0eb1b8c1005932098c80f7e5bb6f9bec6127a3be5cfa3901d23201f54fd541f')


if __name__ == '__main__': unittest.main()
