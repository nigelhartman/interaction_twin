"""Local verification/import of a fresh SAM3D run without invented job history.

No inference, downloads, network, credentials, catalog mutation or model choice.
Adapter attestations bind bytes and provenance; they do not prove execution or
physical accuracy. Imported candidates keep their independent native gauge.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import io
from pathlib import Path, PurePosixPath

import numpy as np

import upgrade_bundles as U
from sam3d_coordinates import SAM3D_COMMIT, DECODED_SCALE_SEMANTICS, native_layout_to_cv
from scripts.stage3_sam3d import read_inputs

ROOT = Path(__file__).resolve().parent
OUTPUT_FILES = {
    'metadata.json', 'object.glb', 'object-splat.ply', 'layout-and-pointmap.npz',
    'inputs/manifest.json', 'inputs/rgb.png', 'inputs/mask.png',
    'inputs/image-source.png', 'inputs/mask-source.png',
}
OFFICIAL = 'https://github.com/facebookresearch/sam-3d-objects'


def _object(value, label):
    if not isinstance(value, dict):
        raise ValueError(label + ' must be an object')
    return value


def _record_list(value, label):
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(label + ' must be an array of objects')
    return value


def _directory(path, root, *, exists=True):
    root = Path(root).resolve()
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    try:
        relative = path.relative_to(root)
    except ValueError:
        raise ValueError('Only workspace data directories are permitted') from None
    if not relative.parts or relative.parts[0] != 'data' or any(p.startswith('.') or '\\' in p for p in relative.parts):
        raise ValueError('Only ordinary workspace data directories are permitted')
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError('Symlink directories are not permitted')
    if path.resolve() != path or (exists and not path.is_dir()):
        raise ValueError('Data directory unavailable')
    return path


def _tree(directory):
    files, total = {}, 0
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():
            raise ValueError('Symlink artifacts are not permitted')
        if path.is_dir():
            continue
        relative = path.relative_to(directory).as_posix()
        if any(p.startswith('.') for p in PurePosixPath(relative).parts) or '\\' in relative:
            raise ValueError('Hidden/unsafe artifact path')
        size = path.stat().st_size
        total += size
        if not path.is_file() or size <= 0 or size > U.MAX_FILE or total > U.MAX_TOTAL or len(files) >= U.MAX_FILES:
            raise ValueError('Artifact tree exceeds local file/byte budget')
        files[relative] = path.read_bytes()
    return files


def _records(files):
    return [{'path': name, 'sha256': U.digest(raw), 'bytes': len(raw)} for name, raw in sorted(files.items())]


def inspect_input_bundle(input_bundle, *, root=ROOT):
    """Validate the fresh frozen bundle against its actual current native anchor."""
    directory = _directory(input_bundle, root)
    files = _tree(directory)
    required = {'input.json', 'input-binding.json', 'source-binding.json', 'bundle-manifest.json', 'run-plan.json', 'adapter/stage3_sam3d.py', 'image.png', 'mask.png'}
    if not required <= files.keys():
        raise ValueError('Fresh bundle is missing required binding files')
    manifest = _object(U._json_bytes(files['bundle-manifest.json']), 'Bundle manifest')
    if manifest.get('format') != 'sam3d-gpu-input-bundle' or type(manifest.get('version')) is not int or manifest['version'] != 1:
        raise ValueError('Unsupported fresh input bundle')
    expected = manifest.get('files')
    if not isinstance(expected, list) or expected != _records({k: v for k, v in files.items() if k != 'bundle-manifest.json'}):
        raise ValueError('Input bundle manifest hashes/file inventory differ')
    plan = _object(U._json_bytes(files['run-plan.json']), 'Run plan')
    if (plan.get('format') != 'sam3d-single-object-smoke-plan' or type(plan.get('version')) is not int or plan['version'] != 1
            or plan.get('input_manifest') != 'input.json' or plan.get('input_manifest_sha256') != U.digest(files['input.json'])
            or plan.get('adapter') != 'adapter/stage3_sam3d.py' or plan.get('adapter_sha256') != U.digest(files['adapter/stage3_sam3d.py'])
            or plan.get('official_code_commit') != SAM3D_COMMIT or plan.get('device') != 'cuda'
            or type(plan.get('seed')) is not int or plan['seed'] != 42 or plan.get('appearance') != 'vertex-color'
            or type(plan.get('parts')) is not int or plan['parts'] != 1 or plan.get('automatic_retry') is not False):
        raise ValueError('Frozen first-run plan/input/adapter pin differs')
    info = _object(U._json_bytes(files['input-binding.json']), 'Input binding')
    current = U.context(info.get('twin_id'), info.get('track_id'), root=root)
    if current != info:
        raise ValueError('Frozen input binding differs from current source/baseline')
    binding = _object(U._json_bytes(files['source-binding.json']), 'Source binding')
    if binding.get('format') != 'sam3d-source-binding' or type(binding.get('version')) is not int or binding['version'] != 1:
        raise ValueError('Unsupported source binding')
    actual = _object(U._json_bytes(files['input.json']), 'Input manifest')
    if actual.get('image') != 'image.png' or actual.get('mask') != 'mask.png':
        raise ValueError('Fresh bundle must use its own exact full-frame RGB and mask')
    # Validate nested types and paths before the generic reader calls .get.
    actual_source = _object(actual.get('source'), 'Input source')
    provenance = _object(actual.get('mask_provenance'), 'Mask provenance')
    selection = _object(actual.get('selection'), 'Automatic selection')
    binding_source = _object(binding.get('source'), 'Source binding source')
    native_rgb = _object(binding.get('native_rgb'), 'Source binding native RGB')
    native_mask = _object(binding.get('native_mask'), 'Source binding native mask')
    baseline_binding = _object(binding.get('baseline'), 'Source binding baseline')
    bound_source = _object(info.get('source'), 'Frozen input source')
    record, rgb, mask = read_inputs(directory / 'input.json')
    for role, filename in (('source_binding', 'source-binding.json'), ('input_binding', 'input-binding.json')):
        if actual.get(role) != {'path': filename, 'sha256': U.digest(files[filename])}:
            raise ValueError('Input manifest binding hash differs')
    if binding.get('case_id') != info['twin_id'] or binding.get('track_id') != info['track_id'] or any(binding_source.get(k) != v for k, v in bound_source.items()):
        raise ValueError('Source binding identity differs')
    for pixels, filename, field in ((native_rgb, 'image.png', 'image_sha256'), (native_mask, 'mask.png', 'mask_sha256')):
        if pixels.get('path') != filename or pixels.get('sha256') != info[field]:
            raise ValueError('Source binding pixel bytes differ')
    for field, filename, expected_sha in (
        ('native_asset', 'baseline/asset.glb', info['baseline_asset_sha256']),
        ('metadata', 'baseline/metadata.json', info['model_metadata_sha256']),
        ('program', 'baseline/interaction.json', info['base_sha256']),
    ):
        if baseline_binding.get(field + '_path') != filename or baseline_binding.get(field + '_sha256') != expected_sha or U.digest(files.get(filename, b'')) != expected_sha:
            raise ValueError('Portable baseline provenance differs')
    if baseline_binding.get('same_rgb_and_mask') is not True or baseline_binding.get('physical_accuracy_verified') is not False:
        raise ValueError('Baseline anchor/physical scope differs')
    if actual.get('object_id') != info['track_id'] or actual_source.get('video_sha256') != bound_source['video_sha256']:
        raise ValueError('Fresh input object/source differs from binding')
    for field in ('sample_index', 'timestamp_s', 'time_domain'):
        if actual_source.get(field) != bound_source[field]:
            raise ValueError('Fresh source PTS/sample differs from binding')
    if provenance != info.get('mask_provenance') or provenance.get('manual_intervention') is not False:
        raise ValueError('First automatic milestone cannot inherit manual segmentation')
    if (selection.get('selection_sha256') != info['selection_sha256'] or selection.get('rank') != info['selection_rank']
            or selection.get('score') != info['selection_score'] or selection.get('agent_visual_selection') is not False
            or selection.get('manual_geometry_used') is not False or selection.get('manual_camera_corners_used') is not False):
        raise ValueError('Automatic selection provenance differs')
    if binding.get('pixel_policy') != {'full_native_frame': True, 'crop': False, 'rotation': False, 'resize': False, 'inpainting': False, 'pixel_values_modified': False, 'mask_modified': False}:
        raise ValueError('Native input pixel policy differs')
    for field in ('manual_geometry_used','manual_pose_used','physical_identity_verified','physical_shape_verified','metric_scale_verified','assembly_verified'):
        if binding.get(field) is not False:
            raise ValueError('Source binding manual/physical scope differs')
    if U.digest(rgb.tobytes()) != info['source']['rgb_array_sha256'] or U.digest(mask.tobytes()) != info['source']['mask_array_sha256']:
        raise ValueError('Fresh model arrays differ from native baseline anchor')
    if record['files']['image']['sha256'] != info['image_sha256'] or record['files']['mask']['sha256'] != info['mask_sha256']:
        raise ValueError('Fresh model source bytes differ from native baseline')
    baseline = U.read(U._local(info['model_metadata_path'], root))
    baseline_asset = U._local(info['baseline_asset_path'], root).read_bytes()
    return {
        'directory': directory, 'files': files, 'input_record': record,
        'binding': info, 'source_binding': binding, 'run_plan': plan, 'rgb': rgb, 'mask': mask,
        'baseline_metadata': baseline, 'baseline_asset': baseline_asset,
    }


def _splat_header(raw):
    """Bounded check of the pinned official Gaussian PLY representation."""
    marker = b'end_header\n'
    end = raw.find(marker, 0, 65536)
    if end < 0:
        raise ValueError('Gaussian PLY header missing or oversized')
    lines = raw[:end + len(marker)].decode('ascii').splitlines()
    if lines[:2] != ['ply', 'format binary_little_endian 1.0']:
        raise ValueError('Expected pinned official binary Gaussian PLY')
    rows = [line.split() for line in lines if line.startswith('element ')]
    if len(rows) != 1 or rows[0][:2] != ['element', 'vertex']:
        raise ValueError('Unexpected Gaussian PLY elements')
    try:
        count = int(rows[0][2])
    except (ValueError, IndexError):
        raise ValueError('Invalid Gaussian count') from None
    props = [line.split() for line in lines if line.startswith('property ')]
    names = [p[2] for p in props if len(p) == 3 and p[1] == 'float']
    required = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
    if count <= 0 or count > 5_000_000 or names != required or len(props) != len(required):
        raise ValueError('Gaussian count/properties differ from pinned exporter')
    payload = raw[end + len(marker):]
    if len(payload) != count * len(names) * 4:
        raise ValueError('Gaussian PLY payload length differs')
    values = np.frombuffer(payload, dtype='<f4').reshape(count, len(names))
    opacity_index = names.index('opacity')
    # Pinned save_ply computes log(alpha/(1-alpha)); exact alpha 0/1 can
    # legitimately serialize +/-Inf. Preserve those values without rendering
    # claims. Geometry/scale/features/rotation and every NaN still fail closed.
    if np.isnan(values).any() or not np.isfinite(np.delete(values, opacity_index, axis=1)).all():
        raise ValueError('Gaussian PLY contains unsupported nonfinite values')
    return {'gaussians': count, 'property_names': names, 'appearance_origin': 'model_generated_gaussian_coefficients',
            'infinite_opacity_logits': int(np.isinf(values[:, opacity_index]).sum()),
            'splat_rendering_validated': False}


def inspect_native_output(*, input_files, input_record, run_plan, expected_arrays, adapter_output, root=ROOT):
    """Shared strict adapter-byte contract; no baseline, catalog or old pose read.

    Callers must first validate their frozen source/plan. Returned provenance is
    an attestation, not independent proof that an inference was executed.
    """
    expected_input = input_record
    directory = _directory(adapter_output, root)
    files = _tree(directory)
    if files.keys() != OUTPUT_FILES:
        raise ValueError('Adapter output file inventory is incomplete or unexpected')
    metadata = _object(U._json_bytes(files['metadata.json']), 'Adapter metadata')
    if metadata.get('fixture') not in (None, False):
        raise ValueError('Fixture output cannot enter native candidate imports')
    if metadata.get('status') != 'candidate_generated_pending_visual_review' or metadata.get('backend') != 'Meta SAM 3D Objects' or metadata.get('learned_inference_executed') is not True:
        raise ValueError('Adapter did not finish genuine learned candidate attestation')
    if metadata.get('model_repository') != 'facebook/sam-3d-objects' or metadata.get('errors'):
        raise ValueError('Model repository/preflight attestation differs')
    if type(metadata.get('seed')) is not int or metadata['seed'] != run_plan['seed'] or metadata.get('appearance_requested') != run_plan['appearance']:
        raise ValueError('Returned seed/appearance differs from frozen first-run plan')
    code = _object(metadata.get('code'), 'Adapter code provenance')
    if code.get('commit') != SAM3D_COMMIT or code.get('expected_commit') != SAM3D_COMMIT or code.get('origin') != OFFICIAL or code.get('tracked_changes') != '':
        raise ValueError('Adapter code pin/origin/cleanliness differs')
    runtime = _object(metadata.get('runtime'), 'Adapter runtime')
    if runtime.get('platform') != 'Linux' or runtime.get('device_requested') != 'cuda' or runtime.get('cuda_available') is not True:
        raise ValueError('CUDA runtime attestation incomplete')
    checkpoints = _object(metadata.get('checkpoint'), 'Adapter checkpoint provenance')
    checkpoint_files = _record_list(checkpoints.get('files'), 'Checkpoint files')
    if checkpoints.get('weights_hashed') is not True or not any(U.HEX64.fullmatch(str(r.get('sha256', ''))) and isinstance(r.get('path'), str) and Path(r['path']).suffix in {'.ckpt', '.pt', '.pth', '.safetensors', '.bin'} for r in checkpoint_files):
        raise ValueError('Hashed checkpoint attestation missing')
    inputs = _object(metadata.get('inputs'), 'Adapter inputs')
    if inputs.get('manifest') != expected_input['manifest'] or inputs.get('manifest_sha256') != expected_input['manifest_sha256'] or files['inputs/manifest.json'] != input_files['input.json']:
        raise ValueError('Adapter input manifest differs from frozen bundle')
    for role in ('image', 'mask'):
        original = input_files[role + '.png']
        source_file = _object(_object(inputs.get('files'), 'Adapter input files').get(role), 'Adapter ' + role + ' file')
        if source_file.get('sha256') != U.digest(original) or files['inputs/' + role + '-source.png'] != original:
            raise ValueError('Adapter source RGB/mask bytes differ')
    rgb, mask = U._pixels(io.BytesIO(files['inputs/rgb.png']), io.BytesIO(files['inputs/mask.png']))
    arrays = {'rgb_uint8': U.digest(rgb.tobytes()), 'mask_bool': U.digest(mask.tobytes())}
    if arrays != expected_arrays or metadata.get('model_input_array_sha256') != arrays:
        raise ValueError('Actual model arrays differ from frozen source')
    if metadata.get('model_input_sha256') != {name: U.digest(files['inputs/' + name]) for name in ('rgb.png', 'mask.png')}:
        raise ValueError('Actual model input file hashes differ')
    outputs = _object(metadata.get('outputs'), 'Adapter outputs')
    if outputs.get('geometry_reconstructed_by_model') is not True or outputs.get('manual_geometry_fallback') is not False or outputs.get('metric_scale_verified') is not False or outputs.get('direct_original_video_texture_projection') is not False:
        raise ValueError('Generated geometry/appearance provenance incomplete')
    records = _record_list(outputs.get('files'), 'Adapter output files')
    expected = [{'file': name, 'sha256': U.digest(files[name]), 'bytes': len(files[name])} for name in ('object.glb', 'object-splat.ply', 'layout-and-pointmap.npz')]
    if records != expected:
        raise ValueError('Native output inventory/hash/size differs')
    decoded = U._validate_glb(files['object.glb'])
    mode = outputs.get('appearance_mode')
    if mode == 'vertex-color' and not any(p.get('colors') for p in decoded['primitives']):
        raise ValueError('Declared generated vertex colours missing')
    if mode == 'uv-baked' and not any(p.get('texture') and p.get('uv') for p in decoded['primitives']):
        raise ValueError('Declared generated UV texture missing')
    if mode != run_plan['appearance']:
        raise ValueError('Generated appearance differs from planned first-run output')
    with U._safe_layout(files['layout-and-pointmap.npz']) as layout:
        fields = {}
        optional_pointmap = {'present': False, 'used_for_video_pose': False, 'filled_or_modified': False}
        for name in layout.files:
            value = layout[name]
            if value.dtype.kind not in 'fiu':
                raise ValueError('Non-numeric native layout arrays')
            if name in {'pointmap','pointmap_colors'}:
                if value.ndim != 3 or value.shape[-1] != 3 or min(value.shape[:2]) < 1:
                    raise ValueError('Expected optional pointmap/color HxWx3 arrays')
                if name == 'pointmap':
                    # The pinned pipeline deliberately clips invalid/outlier
                    # points to NaN; those pixels are missing, never filled.
                    if np.isinf(value).any():
                        raise ValueError('Infinite optional pointmap coordinates')
                    valid_points = np.isfinite(value).all(axis=-1)
                    optional_pointmap.update(present=True, shape=list(value.shape),
                                             finite_points=int(valid_points.sum()), missing_points=int((~valid_points).sum()),
                                             missing_value_policy='preserved NaN pixels; no geometry/pose evidence')
                elif not np.isfinite(value).all():
                    raise ValueError('Nonfinite pointmap colors')
            elif not np.isfinite(value).all():
                raise ValueError('Nonfinite native decoded layout transform')
            fields[name] = {'shape': list(value.shape), 'dtype': str(value.dtype)}
        if 'pointmap_colors' in fields and 'pointmap' in fields and fields['pointmap_colors']['shape'] != fields['pointmap']['shape']:
            raise ValueError('Optional pointmap/color dimensions differ')
        if fields != outputs.get('layout_fields'):
            raise ValueError('Native layout array declaration differs')
        pose = native_layout_to_cv(rotation_wxyz=layout['rotation'], translation=layout['translation'], scale_xyz=layout['scale'], source_commit=SAM3D_COMMIT, scale_semantics=DECODED_SCALE_SEMANTICS)
    coordinates = pose.metadata()
    coordinates.update(use_for_initialization_only=True, accepted_video_pose=False)
    return {'directory': directory, 'files': files, 'metadata': metadata, 'decoded': decoded,
            'coordinates': coordinates, 'optional_pointmap': optional_pointmap, 'mode': mode,
            'fit_contract': U._fit_contract(files['object.glb'], mode, decoded),
            'splat_contract': _splat_header(files['object-splat.ply'])}


def inspect_output(input_bundle, adapter_output, *, root=ROOT):
    """Verify output attestations and native contracts; never promote a model."""
    source = inspect_input_bundle(input_bundle, root=root)
    info = source['binding']
    native = inspect_native_output(input_files=source['files'], input_record=source['input_record'],
                                   run_plan=source['run_plan'],
                                   expected_arrays={'rgb_uint8': info['source']['rgb_array_sha256'],
                                                    'mask_bool': info['source']['mask_array_sha256']},
                                   adapter_output=adapter_output, root=root)
    directory, files, metadata = native['directory'], native['files'], native['metadata']
    decoded, coordinates, optional_pointmap, mode = (native[k] for k in ('decoded', 'coordinates', 'optional_pointmap', 'mode'))
    outputs = metadata['outputs']
    baseline_decoded = U._validate_glb(source['baseline_asset'])
    report = {
        'format': 'sam3d-native-milestone-readiness', 'version': '1', 'fixture': False,
        'status': 'validated_native_candidate_pending_review',
        'input_bundle': {'path': str(source['directory'].relative_to(Path(root).resolve())), 'manifest_sha256': U.digest(source['files']['bundle-manifest.json'])},
        'adapter_output': {'path': str(directory.relative_to(Path(root).resolve())), 'metadata_sha256': U.digest(files['metadata.json']), 'files': _records(files)},
        'adapter_binding': {'planned_adapter_sha256': source['run_plan']['adapter_sha256'], 'run_plan_sha256': U.digest(source['files']['run-plan.json']), 'seed': metadata['seed'], 'appearance': mode, 'adapter_self_hash_attested_in_output': False, 'launch_receipt_review_required': True},
        'source_binding': copy.deepcopy(info), 'native_layout_candidate': coordinates,
        'optional_pointmap': optional_pointmap,
        'fit_contract': native['fit_contract'],
        'splat_contract': native['splat_contract'],
        'comparison': {'same_source_rgb_and_mask': True, 'baseline': {'asset_path': info['baseline_asset_path'], 'asset_sha256': info['baseline_asset_sha256'], 'metadata_path': info['model_metadata_path'], 'metadata_sha256': info['model_metadata_sha256'], 'backend': source['baseline_metadata'].get('backend'), 'native_mesh_stats': baseline_decoded['mesh_stats']},
                       'candidate': {'asset_sha256': U.digest(files['object.glb']), 'native_mesh_stats': decoded['mesh_stats']},
                       'geometry_accuracy_comparison_performed': False, 'native_dimensions_comparable': False, 'common_gauge_established': False, 'winner': None},
        'appearance': {'mode': mode, 'origin': 'model_generated', 'description': outputs.get('appearance'), 'direct_original_video_texture_projection': False, 'hidden_surfaces': 'model_inferred_not_observed'},
        'physical_accuracy_verified': False, 'metric_scale_verified': False, 'source_video_pose_validated': False,
        'assembly_verified': False, 'contact_verified': False, 'simulation_executed': False,
        'execution_independently_verified': False, 'model_selection_changed': False,
        'limitations': ['Adapter attestations are contract-checked, not cryptographic proof of execution.', 'One visible source mask does not observe hidden surfaces.', 'Native model units and inferred layout remain independent of baseline gauge.', 'Topology/detail counts and low projection residual are not shape accuracy.', 'No original program, asset, pose or preferred model is replaced.'],
    }
    return report, source, files, metadata


def import_output(input_bundle, adapter_output, destination, *, root=ROOT):
    """Write a new native-only candidate with unchanged source/model bytes."""
    destination = _directory(destination, root, exists=False)
    if destination.exists():
        raise ValueError('Choose a fresh import version; originals are preserved')
    report, source, files, metadata = inspect_output(input_bundle, adapter_output, root=root)
    if any(destination.is_relative_to(parent) for parent in (source['directory'], _directory(adapter_output, root))):
        raise ValueError('Import must be separate from frozen input/output trees')
    destination.mkdir(parents=True, exist_ok=False)
    try:
        for name, raw in files.items():
            target = destination / 'raw' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        for name, raw in source['files'].items():
            target = destination / 'input-bundle' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        (destination / 'asset.glb').write_bytes(files['object.glb'])
        info = source['binding']
        native_source = copy.deepcopy(source['baseline_metadata']['source'])
        native_source.update(image_path='raw/inputs/image-source.png', mask_path='raw/inputs/mask-source.png')
        model = {
            'schema_version': '0.1', 'status': 'imported_native_candidate_pending_alignment',
            'object_id': info['track_id'], 'backend': 'Meta SAM 3D Objects', 'provider': 'Meta',
            'sam3d_output': True, 'sam3d_milestone_complete': False, 'learned_inference_executed': True,
            'source': native_source, 'outputs': {'glb_path': 'asset.glb'},
            'geometry_reconstructed_by_model': True, 'manual_geometry_fallback': False, 'manual_pose_used': False,
            'appearance': metadata['outputs']['appearance'], 'appearance_mode': metadata['outputs']['appearance_mode'],
            'direct_original_video_texture_projection': False, 'hidden_surfaces': 'model-inferred, not observed or measured',
            'metric_scale_verified': False, 'camera_pose_estimated': False,
            'coordinates': 'Unchanged official native GLB in its independent model gauge; no shared metric frame',
            'validation_status': 'pending_independent_source_projection_review',
            'model_provenance': {'model_repository': metadata['model_repository'], 'code': metadata['code'], 'checkpoint': metadata['checkpoint'], 'adapter_metadata_path': 'raw/metadata.json', 'adapter_metadata_sha256': U.digest(files['metadata.json'])},
            'runtime': metadata['runtime'], 'native_layout_candidate': report['native_layout_candidate'],
            'fit_contract': report['fit_contract'],
            'artifact_files': [{'file': name, 'sha256': U.digest(raw), 'bytes': len(raw)} for name, raw in (
                ('asset.glb', files['object.glb']),
                ('raw/inputs/image-source.png', files['inputs/image-source.png']),
                ('raw/inputs/mask-source.png', files['inputs/mask-source.png']),
            )],
            'milestone_lineage': {'input_bundle_manifest_sha256': report['input_bundle']['manifest_sha256'], 'baseline_asset_sha256': info['baseline_asset_sha256'], 'baseline_metadata_sha256': info['model_metadata_sha256'], 'raw_asset_sha256': U.digest(files['object.glb']), 'geometry_bytes_modified': False, 'execution_independently_verified': False},
            'limitations': report['limitations'],
        }
        U.write(destination / 'metadata.json', model)
        report['created_at'] = datetime.now(timezone.utc).isoformat()
        report['import_directory'] = str(destination.relative_to(Path(root).resolve()))
        report['import_files'] = _records(_tree(destination))
        U.write(destination / 'import.json', report)
        return report
    except Exception:
        # Partial evidence is retained rather than removed or reused.
        raise
