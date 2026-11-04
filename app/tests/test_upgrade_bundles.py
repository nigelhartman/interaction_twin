"""No model inference: synthetic adapter contracts live only in temp workspaces."""
from __future__ import annotations

import copy
import importlib.util
import io
import json
from pathlib import Path
import shutil
import stat
import struct
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image

import upgrade_bundles as U

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('portable_fixture_upgrade', ROOT / 'tests/test_portable_twin.py')
portable_fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(portable_fixture)
spec_pack = importlib.util.spec_from_file_location('pack_upgrade_fixture', ROOT / 'scripts/pack_upgrade_result.py')
packer = importlib.util.module_from_spec(spec_pack)
spec_pack.loader.exec_module(packer)
spec_run = importlib.util.spec_from_file_location('run_upgrade_fixture', ROOT / 'scripts/run_upgrade_job.py')
runner = importlib.util.module_from_spec(spec_run)
spec_run.loader.exec_module(runner)


def vertex_glb():
    """A material-free synthetic GLB that Trimesh actually loads as ColorVisuals."""
    raw = portable_fixture.glb()
    json_length = struct.unpack_from('<I', raw, 12)[0]
    doc = json.loads(raw[20:20 + json_length])
    binary = raw[28 + json_length:]
    doc.pop('materials', None)
    for mesh in doc['meshes']:
        for primitive in mesh['primitives']:
            primitive.pop('material', None)
    return portable_fixture.pack_glb(doc, binary)


class UpgradeBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        source = self.root / 'data/source'
        source.mkdir(parents=True)
        self.rgb = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
        self.mask = np.zeros((4, 4), dtype=np.uint8)
        self.mask[1:3, 1:3] = 255
        Image.fromarray(self.rgb).save(source / 'source-rgb.png')
        Image.fromarray(self.mask).save(source / 'source-mask.png')
        (source / 'asset.glb').write_bytes(portable_fixture.glb())
        selection = {'object_id': 'part', 'source': {'sha256': '1' * 64},
                     'selection_method': {'agent_visual_choice': False},
                     'selected': [{'sample_index': 14, 'timestamp_s': 13.25}]}
        U.write(source / 'selection.json', selection)
        metadata = {'object_id': 'part', 'outputs': {'glb_path': 'asset.glb'},
                    'source': {'image_path': 'source-rgb.png', 'mask_path': 'source-mask.png',
                               'image_sha256': U.sha(source / 'source-rgb.png'),
                               'mask_sha256': U.sha(source / 'source-mask.png'),
                               'video_sha256': '1' * 64, 'sample_index': 14, 'timestamp_s': 13.25,
                               'time_domain': U.TIME_DOMAIN, 'width': 4, 'height': 4,
                               'selection_manifest_path': str(source / 'selection.json'),
                               'selection_manifest_sha256': U.sha(source / 'selection.json'),
                               'selection_rank': 1, 'selection_score': .8,
                               'mask_scope': 'test object',
                               'mask_provenance': {'manual_seed_boxes_used': False, 'model': 'SAM2 fixture', 'prompt_origin': 'model_generated_prompt'}}}
        U.write(source / 'metadata.json', metadata)
        self.doc = {'tracklets': [{'id': 'part', 'label': 'test object'}],
                    'assets': [{'track_id': 'part', 'path': 'data/source/asset.glb', 'sha256': U.sha(source / 'asset.glb')}],
                    'source': {'sha256': '1' * 64},
                    'inputs': [{'kind': 'learned_asset_metadata', 'path': 'data/source/metadata.json', 'sha256': U.sha(source / 'metadata.json')}]}
        outer = self
        class Catalog:
            def __init__(self, root): pass
            def load(self, twin_id):
                if twin_id != 'example': raise ValueError('Unknown twin')
                return {'title': 'Synthetic test only'}, copy.deepcopy(outer.doc), '2' * 64
        self.patcher = patch.object(U, 'TwinCatalog', Catalog)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        (self.root / 'scripts').mkdir()
        for name in ('stage3_sam3d.py', 'sam31_segment.py', 'pack_upgrade_result.py', 'run_upgrade_job.py'):
            shutil.copy2(ROOT / 'scripts' / name, self.root / 'scripts' / name)

    def prepare(self, backend='sam3d_objects', **changes):
        return U.prepare_job({'twin_id': 'example', 'track_id': 'part', 'base_sha256': '2' * 64,
                              'backend': backend, **changes}, root=self.root)

    def model_output(self, job_summary, *, sam31=False):
        """Construct contract test bytes, never retained or described as model output."""
        job_path = Path(job_summary['job_path'])
        job = U.read(job_path)
        output = self.root / ('contract-output-' + job['job_id'])
        output.mkdir()
        info = job['input_binding']
        metadata = {'learned_inference_executed': True, 'errors': [],
                    'model_repository': job['model_repository'],
                    'code': {'commit': job['expected_code_commit'], 'tracked_changes': '',
                             'origin': 'https://github.com/facebookresearch/sam3' if sam31 else 'https://github.com/facebookresearch/sam-3d-objects'},
                    'runtime': {'platform': 'Linux', 'device_requested': 'cuda', 'cuda_available': True},
                    'inputs': {'manifest_sha256': job['input_manifest_sha256'], 'manifest': job['input_manifest'],
                               'files': {'image': {'sha256': info['image_sha256']},
                                         'reference_mask' if sam31 else 'mask': {'sha256': info['mask_sha256']}}}}
        if sam31:
            metadata.update(backend='Meta SAM 3.1 Object Multiplex', status='completed_candidates',
                            checkpoint={'sha256': '3' * 64, 'manifest': {'model_repository': 'facebook/sam3.1',
                                'model_revision': '4' * 40, 'checkpoint_file': 'sam3.1_multiplex.pt', 'checkpoint_sha256': '3' * 64}},
                            checkpoint_coverage={'strict_key_and_shape_coverage': True})
            argv = job['command_argv']
            metadata['prompt'] = {'text': argv[argv.index('--prompt') + 1], 'origin': argv[argv.index('--prompt-origin') + 1], 'reference_mask_used_as_prompt': False}
            Image.fromarray(self.rgb).save(output / 'source-rgb.png')
            Image.fromarray(self.rgb).save(output / 'overlay.png')
            Image.fromarray(self.mask).save(output / 'instance-000001-mask.png')
            Image.fromarray(np.dstack([self.rgb, self.mask])).save(output / 'instance-000001-rgba.png')
            metadata['outputs'] = {'instance_count': 1, 'instances': [{'model_instance_id': 7,
                'physical_identity_verified': False, 'score': .9, 'box_xywh_normalized': [.25, .25, .5, .5],
                'mask_pixels': 4, 'mask_file': 'instance-000001-mask.png', 'rgba_file': 'instance-000001-rgba.png',
                'mask_sha256': U.sha(output / 'instance-000001-mask.png')}]}
        else:
            inputs = output / 'inputs'
            inputs.mkdir()
            shutil.copy2(job_path.parent / 'input/input.json', inputs / 'manifest.json')
            for src, dst in [('rgb.png', 'image-source.png'), ('mask.png', 'mask-source.png'), ('rgb.png', 'rgb.png'), ('mask.png', 'mask.png')]:
                shutil.copy2(job_path.parent / 'input' / src, inputs / dst)
            metadata.update(backend='Meta SAM 3D Objects', status='candidate_generated_pending_visual_review',
                            model_input_sha256={name: U.sha(inputs / name) for name in ('rgb.png', 'mask.png')},
                            model_input_array_sha256={'rgb_uint8': U.digest(self.rgb.tobytes()), 'mask_bool': U.digest((self.mask > 0).tobytes())},
                            checkpoint={'weights_hashed': True, 'files': [{'path': '/provisioned/model.ckpt', 'sha256': '3' * 64}]})
            (output / 'object.glb').write_bytes(vertex_glb())
            (output / 'object-splat.ply').write_bytes(b'ply\nformat ascii 1.0\nelement vertex 0\nend_header\n')
            np.savez_compressed(output / 'layout-and-pointmap.npz', rotation=np.array([1., 0., 0., 0.]), translation=np.array([0., 0., 2.]), scale=np.ones(3))
            metadata['outputs'] = {'geometry_reconstructed_by_model': True, 'manual_geometry_fallback': False,
                                   'metric_scale_verified': False, 'appearance_mode': 'vertex-color',
                                   'appearance': 'Model generated test-contract colours',
                                   'files': [{'file': name, 'bytes': (output / name).stat().st_size, 'sha256': U.sha(output / name)}
                                             for name in ('object.glb', 'object-splat.ply', 'layout-and-pointmap.npz')]}
        U.write(output / 'metadata.json', metadata)
        destination = self.root / (job['job_id'] + '-returned.zip')
        packer.pack(job_path, output, destination)
        return destination

    def mutate_zip(self, source, fn):
        with zipfile.ZipFile(source) as archive:
            files = {name: archive.read(name) for name in archive.namelist()}
        fn(files)
        target = self.root / ('mutated-' + str(len(list(self.root.glob('mutated-*')))) + '.zip')
        with zipfile.ZipFile(target, 'x') as archive:
            for name, raw in files.items(): archive.writestr(name, raw)
        return target

    def refresh_result_hashes(self, files):
        receipt = json.loads(files['result.json'])
        receipt['files'] = [{'path': name, 'bytes': len(raw), 'sha256': U.digest(raw)}
                            for name, raw in files.items() if name != 'result.json']
        files['result.json'] = U.encoded(receipt)

    def test_prepare_copies_exact_inputs_and_preserves_originals(self):
        before = {p: U.sha(p) for p in (self.root / 'data/source').glob('*')}
        result = self.prepare()
        self.assertEqual(result['status'], 'prepared_not_submitted')
        self.assertFalse(result['full_video_included'])
        self.assertFalse(result['learned_inference_executed'])
        job = U.read(result['job_path'])
        self.assertFalse(job['fixture'])
        self.assertEqual(job['expected_code_commit'], U.SAM3D_COMMIT)
        self.assertEqual((Path(result['directory']) / 'input/rgb.png').read_bytes(), (self.root / 'data/source/source-rgb.png').read_bytes())
        self.assertEqual(before, {p: U.sha(p) for p in before})
        self.assertEqual(len(U.list_jobs(root=self.root)), 1)
        with zipfile.ZipFile(result['zip_path']) as archive:
            names = archive.namelist()
            self.assertNotIn('video.mp4', names)
            self.assertIn('scripts/stage3_sam3d.py', names)

    def test_stale_program_and_unknown_fields_rejected_before_output(self):
        for change in ({'base_sha256': '9' * 64}, {'irrelevant': 5}, {'prompt': 'box'}, {'appearance': 'invented'}):
            with self.assertRaises(ValueError): self.prepare(**change)
        self.assertFalse((self.root / 'data/upgrade-jobs').exists())

    def test_uv_handoff_binds_explicit_appearance_without_running_a_model(self):
        summary = self.prepare(appearance='uv-baked')
        job = U.read(summary['job_path'])
        self.assertEqual(job['appearance_mode'], 'uv-baked')
        self.assertEqual(summary['appearance_mode'], 'uv-baked')
        self.assertEqual(job['command_argv'][job['command_argv'].index('--appearance') + 1], 'uv-baked')
        command = runner.command(summary['job_path'], 'run', '/provisioned/repository', config='/provisioned/pipeline.yaml')
        self.assertEqual(command[command.index('--appearance') + 1], 'uv-baked')
        self.assertFalse(summary['learned_inference_executed'])
        with self.assertRaises(ValueError):
            self.prepare('sam31_image', appearance='uv-baked')

    def test_returned_appearance_must_match_explicit_handoff(self):
        summary = self.prepare(appearance='uv-baked')
        with self.assertRaisesRegex(ValueError, 'appearance differs'):
            U.import_result(summary['job_id'], self.model_output(summary), root=self.root)

    def test_sampler_pts_and_source_mask_hash_are_binding(self):
        info = U.context('example', 'part', root=self.root)
        self.assertEqual(info['source']['timestamp_s'], 13.25)
        self.assertEqual(info['mask_sha256'], U.sha(self.root / 'data/source/source-mask.png'))
        (self.root / 'data/source/source-mask.png').write_bytes(b'changed')
        with self.assertRaises(ValueError): U.context('example', 'part', root=self.root)

    def test_job_tampering_is_visible_not_resumed(self):
        job = self.prepare()
        (Path(job['directory']) / 'command.json').write_text('{}')
        rows = U.list_jobs(root=self.root)
        self.assertEqual(rows[0]['status'], 'invalid_saved_job')

    def test_changed_download_zip_is_not_advertised_as_valid(self):
        job = self.prepare()
        path = Path(job['zip_path'])
        with zipfile.ZipFile(path, 'a') as archive:
            archive.writestr('unreviewed.py', b'not executed')
        self.assertEqual(U.list_jobs(root=self.root)[0]['status'], 'invalid_saved_job')

    def test_explicit_fixture_return_is_rejected(self):
        job = self.prepare()
        original = self.model_output(job)
        def change(files):
            receipt = json.loads(files['result.json']); receipt['fixture'] = True
            files['result.json'] = U.encoded(receipt)
        with self.assertRaisesRegex(ValueError, 'Fixtures'):
            U.import_result(job['job_id'], self.mutate_zip(original, change), root=self.root)
        self.assertFalse((self.root / 'data/upgrade-results').exists())

    def test_contract_round_trip_keeps_raw_mesh_bytes_and_unknown_physics(self):
        job = self.prepare()
        result = U.import_result(job['job_id'], self.model_output(job), root=self.root)
        self.assertEqual(result['status'], 'validated_candidate_pending_alignment')
        self.assertFalse(result['physical_accuracy_verified'])
        self.assertFalse(result['execution_independently_verified'])
        model = U.read(self.root / result['model_metadata_path'])
        directory = self.root / result['model_metadata_path']; directory = directory.parent
        self.assertEqual((directory / 'asset.glb').read_bytes(), (directory / 'raw/object.glb').read_bytes())
        self.assertTrue(model['sam3d_output'])
        self.assertTrue(result['alignment_eligible'])
        self.assertEqual(result['alignment_adapter'], 'legacy-vertex-colour-v1')
        self.assertTrue(model['fit_contract']['alignment_eligible'])
        self.assertEqual(model['fit_contract']['visual_kind'], 'vertex')
        self.assertEqual(model['source']['timestamp_s'], 13.25)
        self.assertFalse(model['native_layout_candidate']['convention_verified'])
        self.assertEqual(U.resolve_import(result['import_id'], root=self.root)['model_metadata_path'], result['model_metadata_path'])

    def uv_output(self, job, *, alpha_mode=None):
        source = self.model_output(job)
        def make_uv(files):
            raw = portable_fixture.glb(texture=True)
            if alpha_mode is not None:
                length = struct.unpack_from('<I', raw, 12)[0]
                doc = json.loads(raw[20:20 + length])
                doc['materials'][0]['alphaMode'] = alpha_mode
                raw = portable_fixture.pack_glb(doc, raw[28 + length:])
            files['raw/object.glb'] = raw
            metadata = json.loads(files['raw/metadata.json'])
            metadata['outputs']['appearance_mode'] = 'uv-baked'
            metadata['outputs']['appearance'] = 'Synthetic UV contract test; generated texture provenance'
            mesh = next(r for r in metadata['outputs']['files'] if r['file'] == 'object.glb')
            mesh.update(bytes=len(files['raw/object.glb']), sha256=U.digest(files['raw/object.glb']))
            files['raw/metadata.json'] = U.encoded(metadata)
            self.refresh_result_hashes(files)
        return self.mutate_zip(source, make_uv)

    def test_uv_import_can_align_without_fabricated_vertex_colour_conversion(self):
        job = self.prepare(appearance='uv-baked')
        result = U.import_result(job['job_id'], self.uv_output(job), root=self.root)
        self.assertTrue(result['alignment_eligible'])
        self.assertEqual(result['alignment_adapter'], 'native-base-color-v1')
        directory = self.root / result['model_metadata_path']; directory = directory.parent
        self.assertEqual((directory / 'asset.glb').read_bytes(), portable_fixture.glb(texture=True))
        self.assertEqual((directory / 'asset.glb').read_bytes(), (directory / 'raw/object.glb').read_bytes())
        model = U.read(directory / 'metadata.json')
        self.assertFalse(model['fit_contract']['appearance_conversion_performed'])
        resolved = U.resolve_import(result['import_id'], root=self.root)
        self.assertTrue(resolved['alignment_eligible'])
        self.assertEqual(resolved['alignment_adapter'], 'native-base-color-v1')

    def test_unsupported_texture_import_is_preserved_as_native_only(self):
        job = self.prepare(appearance='uv-baked')
        result = U.import_result(job['job_id'], self.uv_output(job, alpha_mode='BLEND'), root=self.root)
        self.assertFalse(result['alignment_eligible'])
        self.assertIn('opaque', result['alignment_ineligible_reason'].lower())
        self.assertTrue((self.root / result['native_asset_path']).is_file())
        self.assertFalse(U.resolve_import(result['import_id'], root=self.root)['alignment_eligible'])

    def test_historical_v1_import_gains_current_capability_without_rewriting_receipts(self):
        job = self.prepare(appearance='uv-baked')
        with patch.object(U, '_fit_contract', U._fit_contract_v1):
            result = U.import_result(job['job_id'], self.uv_output(job), root=self.root)
        self.assertFalse(result['alignment_eligible'])
        directory = Path(result['directory'])
        before = {p: p.read_bytes() for p in directory.rglob('*') if p.is_file()}
        resolved = U.resolve_import(result['import_id'], root=self.root)
        self.assertTrue(resolved['alignment_eligible'])
        self.assertFalse(resolved['import_time_alignment_eligible'])
        self.assertEqual(resolved['alignment_adapter'], 'native-base-color-v1')
        self.assertEqual(U.read(directory / 'metadata.json')['fit_contract']['version'], '1')
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_current_capability_can_withhold_historical_import_without_rewriting_it(self):
        job = self.prepare()
        with patch.object(U, '_fit_contract', U._fit_contract_v1):
            result = U.import_result(job['job_id'], self.model_output(job), root=self.root)
        directory = Path(result['directory'])
        before = {p: p.read_bytes() for p in directory.rglob('*') if p.is_file()}
        blocked = {'version': '2', 'alignment_eligible': False, 'alignment_adapter': None,
                   'alignment_ineligible_reason': 'Current adapter unavailable'}
        with patch.object(U, '_fit_contract', return_value=blocked):
            resolved = U.resolve_import(result['import_id'], root=self.root)
        self.assertFalse(resolved['alignment_eligible'])
        self.assertTrue(resolved['import_time_alignment_eligible'])
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_vertex_colour_claim_does_not_make_a_textured_glb_alignment_eligible(self):
        job = self.prepare()
        source = self.model_output(job)
        def keep_false_vertex_claim(files):
            files['raw/object.glb'] = portable_fixture.glb(texture=True)
            metadata = json.loads(files['raw/metadata.json'])
            mesh = next(r for r in metadata['outputs']['files'] if r['file'] == 'object.glb')
            mesh.update(bytes=len(files['raw/object.glb']), sha256=U.digest(files['raw/object.glb']))
            files['raw/metadata.json'] = U.encoded(metadata)
            self.refresh_result_hashes(files)
        result = U.import_result(job['job_id'], self.mutate_zip(source, keep_false_vertex_claim), root=self.root)
        self.assertFalse(result['alignment_eligible'])
        self.assertIn('exclusively vertex-coloured', result['alignment_ineligible_reason'])

    def test_material_vertex_tint_uses_native_adapter_when_legacy_loader_loses_it(self):
        raw = portable_fixture.glb()
        decoded = U._validate_glb(raw)
        self.assertTrue(all(p['colors'] and not p['texture'] for p in decoded['primitives']))
        contract = U._fit_contract(raw, 'vertex-color', decoded)
        self.assertTrue(contract['alignment_eligible'])
        self.assertEqual(contract['alignment_adapter'], 'native-base-color-v1')
        self.assertFalse(contract['appearance_conversion_performed'])

    def test_saved_import_hashes_are_revalidated(self):
        job = self.prepare()
        result = U.import_result(job['job_id'], self.model_output(job), root=self.root)
        path = self.root / result['model_metadata_path']
        (path.parent / 'asset.glb').write_bytes(b'bad mesh')
        self.assertEqual(U.list_imports(root=self.root)[0]['status'], 'invalid_saved_import')

    def test_wrong_job_or_source_never_imports(self):
        job = self.prepare()
        original = self.model_output(job)
        def change(files):
            receipt = json.loads(files['result.json']); receipt['input_binding']['source']['timestamp_s'] = 99
            files['result.json'] = U.encoded(receipt)
        with self.assertRaises(ValueError): U.import_result(job['job_id'], self.mutate_zip(original, change), root=self.root)

    def test_bad_asset_hash_is_rejected(self):
        job = self.prepare(); original = self.model_output(job)
        with self.assertRaises(ValueError):
            U.import_result(job['job_id'], self.mutate_zip(original, lambda files: files.update({'raw/object.glb': b'changed'})), root=self.root)

    def test_missing_layout_or_unexpected_file_is_rejected(self):
        job = self.prepare(); original = self.model_output(job)
        for transform in (lambda files: files.pop('raw/layout-and-pointmap.npz'), lambda files: files.update({'raw/secret.env': b'fake'})):
            with self.assertRaises(ValueError): U.import_result(job['job_id'], self.mutate_zip(original, transform), root=self.root)

    def test_archive_traversal_and_symlink_rejected(self):
        job = self.prepare()
        for name in ('../oops', '/absolute', 'raw/../oops', 'raw\\oops', '.hidden'):
            path = self.root / 'unsafe.zip'
            with zipfile.ZipFile(path, 'w') as archive: archive.writestr(name, b'bad')
            with self.assertRaises(ValueError): U.import_result(job['job_id'], path, root=self.root)
        with zipfile.ZipFile(path, 'w') as archive:
            info = zipfile.ZipInfo('raw/symlink'); info.create_system = 3; info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, 'elsewhere')
        with self.assertRaisesRegex(ValueError, 'symlinks'): U.import_result(job['job_id'], path, root=self.root)

    def test_nested_layout_budget_and_pickle_rejected(self):
        raw = io.BytesIO()
        np.savez_compressed(raw, rotation=np.array([object()], dtype=object), translation=np.zeros(3), scale=np.ones(3))
        with U._safe_layout(raw.getvalue()) as archive:
            with self.assertRaises(ValueError): archive['rotation']

    def test_sam31_preserves_instances_does_not_promote_video_identity(self):
        job = self.prepare('sam31_image')
        result = U.import_result(job['job_id'], self.model_output(job, sam31=True), root=self.root)
        self.assertEqual(result['status'], 'validated_anchor_comparison_only')
        self.assertNotIn('model_metadata_path', result)
        comparison = U.read(self.root / result['comparison_path'])
        self.assertEqual(comparison['instances'][0]['model_instance_id'], 7)
        self.assertEqual(comparison['instances'][0]['iou_with_saved_sam_mask_not_accuracy'], 1)
        self.assertFalse(comparison['promoted_to_video_segmentation'])
        self.assertFalse(comparison['old_sam2_audit_reused'])
        self.assertFalse(comparison['cross_frame_identity_inherited'])
        self.assertEqual(U.resolve_import(result['import_id'], root=self.root)['backend'], 'sam31_image')

    def test_manual_text_prompt_is_explicit_and_not_shell_expanded(self):
        prompt = 'earbud $(touch secret) `echo no`'
        job = U.read(self.prepare('sam31_image', prompt=prompt)['job_path'])
        self.assertIn(prompt, job['command_argv'])
        self.assertIn('user_text_concept', job['command_argv'])
        self.assertFalse((self.root / 'secret').exists())

    def test_storage_symlink_cannot_redirect_handoff(self):
        target = self.root / 'outside'; target.mkdir()
        (self.root / 'data/upgrade-jobs').symlink_to(target, target_is_directory=True)
        with self.assertRaises(ValueError): self.prepare()

    def test_gpu_runner_has_explicit_mode_and_no_shell_command(self):
        job = self.prepare()
        argv = runner.command(job['job_path'], 'preflight', '/provisioned/code', config='/provisioned/pipeline.yaml')
        self.assertEqual(argv[2], 'preflight')
        self.assertIn(U.SAM3D_COMMIT, argv)
        self.assertTrue(argv[argv.index('--output') + 1].endswith('/preflight.json'))
        with self.assertRaises(ValueError): runner.command(job['job_path'], 'download', '/code', config='/weights')
        with self.assertRaises(ValueError): runner.command(job['job_path'], 'run', '/code')


if __name__ == '__main__':
    unittest.main()
