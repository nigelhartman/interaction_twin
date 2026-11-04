"""Import a first SAM3D part from a verified RGB/SAM handoff, without a baseline.

No inference, remote commands, catalog mutation, pose reuse or physics. Adapter
metadata is a checked attestation; independent execution review remains required.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path

import initial_sam3d_handoff as H
import sam3d_milestone as M
import upgrade_bundles as U

ROOT = Path(__file__).resolve().parent


def inspect_input(handoff, *, root=ROOT):
    """Validate the frozen initial anchor against its current source, never a mesh."""
    root = Path(root).resolve()
    path = U._local(handoff, root)
    verified = H.inspect(path, root=root)
    H.require(verified.get('status') == 'verified_initial_reconstruction_prepared', 'Initial handoff did not verify')
    files = M._tree(path.parent)
    binding = U.read(path.parent / 'source-binding.json')
    plan = U.read(path.parent / 'run-plan.json')
    record, _, _ = M.read_inputs(path.parent / 'input.json')
    return {'directory': path.parent, 'handoff': verified['handoff'], 'files': files,
            'binding': binding, 'run_plan': plan, 'input_record': record}


def inspect_output(handoff, adapter_output, *, root=ROOT):
    """Check genuine-result attestations and native contracts; leave selection alone."""
    root = Path(root).resolve()
    source = inspect_input(handoff, root=root)
    binding = source['binding']
    native = M.inspect_native_output(input_files=source['files'], input_record=source['input_record'],
        run_plan=source['run_plan'],
        expected_arrays={'rgb_uint8': binding['decoded_arrays']['rgb_sha256'],
                         'mask_bool': binding['decoded_arrays']['mask_bool_sha256']},
        adapter_output=adapter_output, root=root)
    files, metadata = native['files'], native['metadata']
    report = {
        'format': 'sam3d-initial-native-readiness', 'version': 1, 'fixture': False,
        'status': 'validated_initial_native_candidate_pending_review',
        'initial_handoff': copy.deepcopy(source['handoff']),
        'source_binding_sha256': U.digest(source['files']['source-binding.json']),
        'source': copy.deepcopy(binding['source']), 'component_id': binding['component_id'],
        'adapter_output': {'path': native['directory'].relative_to(root).as_posix(),
                           'metadata_sha256': U.digest(files['metadata.json']), 'files': M._records(files)},
        'adapter_binding': {'planned_adapter_sha256': source['run_plan']['adapter_sha256'],
                            'run_plan_sha256': U.digest(source['files']['run-plan.json']),
                            'seed': metadata['seed'], 'appearance': native['mode'],
                            'adapter_self_hash_attested_in_output': False, 'launch_receipt_review_required': True},
        'native_layout_candidate': native['coordinates'], 'optional_pointmap': native['optional_pointmap'],
        'fit_contract': native['fit_contract'], 'splat_contract': native['splat_contract'],
        'native_mesh_stats': native['decoded']['mesh_stats'],
        'appearance': {'mode': native['mode'], 'origin': 'model_generated',
                       'description': metadata['outputs'].get('appearance'),
                       'direct_original_video_texture_projection': False, 'hidden_surfaces': 'model_inferred_not_observed'},
        'baseline_required': False, 'baseline_asset_read': False, 'old_poses_read': False,
        'model_inference_ran_during_import': False, 'execution_independently_verified': False,
        'physical_accuracy_verified': False, 'metric_scale_verified': False,
        'source_video_pose_validated': False, 'assembly_verified': False, 'contact_verified': False,
        'simulation_executed': False, 'simulation_ready': False, 'model_selection_changed': False,
        'catalog_changed': False,
        'implementation': {'initial_sam3d_import.py': U.sha(Path(__file__).resolve()),
                           **{name: U.sha(Path(module.__file__).resolve()) for name, module in
                              [('initial_sam3d_handoff.py', H), ('sam3d_milestone.py', M), ('upgrade_bundles.py', U)]}},
        'limitations': ['Adapter attestations are contract-checked, not independent proof of execution.',
                        'One visible RGB/mask anchor does not observe hidden surfaces.',
                        'The unchanged native model and inferred layout have an independent, nonmetric gauge.',
                        'A fresh source projection and motion fit are required; no old poses are read or copied.',
                        'No original source, program, asset or preferred model is replaced.'],
    }
    return report, source, native


def import_output(handoff, adapter_output, destination, *, root=ROOT):
    """Preserve every returned byte in a fresh, unselected native candidate."""
    root = Path(root).resolve()
    destination = M._directory(destination, root, exists=False)
    H.require(not destination.exists(), 'Choose a fresh initial import version; originals are preserved')
    report, source, native = inspect_output(handoff, adapter_output, root=root)
    H.require(not any(destination.is_relative_to(p) for p in (source['directory'], native['directory'])),
              'Initial import must be separate from frozen input/output trees')
    files, metadata, binding = native['files'], native['metadata'], source['binding']
    native_source = copy.deepcopy(binding['source'])
    native_source.update(image_path='raw/inputs/image-source.png', mask_path='raw/inputs/mask-source.png',
                         image_sha256=U.digest(files['inputs/image-source.png']), mask_sha256=U.digest(files['inputs/mask-source.png']),
                         width=source['input_record']['width'], height=source['input_record']['height'],
                         mask_provenance=copy.deepcopy(binding['mask_provenance']))
    model = {
        'schema_version': '0.1', 'status': 'imported_initial_native_candidate_pending_alignment',
        'object_id': binding['component_id'], 'backend': 'Meta SAM 3D Objects', 'provider': 'Meta',
        'sam3d_output': True, 'sam3d_milestone_complete': False, 'learned_inference_executed': True,
        'source': native_source, 'outputs': {'glb_path': 'asset.glb'},
        'geometry_reconstructed_by_model': True, 'manual_geometry_fallback': False, 'manual_pose_used': False,
        'appearance': metadata['outputs']['appearance'], 'appearance_mode': native['mode'],
        'direct_original_video_texture_projection': False, 'hidden_surfaces': 'model-inferred, not observed or measured',
        'metric_scale_verified': False, 'physical_accuracy_verified': False, 'camera_pose_estimated': False,
        'coordinates': 'Unchanged official native GLB; independent nonmetric model gauge',
        'validation_status': 'pending_independent_execution_and_source_projection_review',
        'model_provenance': {'model_repository': metadata['model_repository'], 'code': metadata['code'],
                             'checkpoint': metadata['checkpoint'], 'adapter_metadata_path': 'raw/metadata.json',
                             'adapter_metadata_sha256': U.digest(files['metadata.json'])},
        'runtime': metadata['runtime'], 'native_layout_candidate': report['native_layout_candidate'],
        'fit_contract': report['fit_contract'], 'artifact_files': [
            {'file': name, 'sha256': U.digest(raw), 'bytes': len(raw)} for name, raw in
            [('asset.glb', files['object.glb']), ('raw/inputs/image-source.png', files['inputs/image-source.png']),
             ('raw/inputs/mask-source.png', files['inputs/mask-source.png'])]],
        'initial_reconstruction_lineage': {'handoff': copy.deepcopy(source['handoff']),
            'source_binding_sha256': report['source_binding_sha256'], 'raw_asset_sha256': U.digest(files['object.glb']),
            'geometry_bytes_modified': False, 'baseline_required': False, 'baseline_asset_read': False,
            'old_poses_read': False, 'execution_independently_verified': False},
        'simulation_ready': False, 'selected': False, 'catalog_changed': False, 'limitations': report['limitations'],
    }
    package = {prefix + '/' + name: raw for prefix, inventory in [('raw', files), ('initial-handoff', source['files'])]
               for name, raw in inventory.items()}
    package.update({'asset.glb': files['object.glb'], 'metadata.json': U.encoded(model)})
    report['created_at_utc'] = datetime.now(timezone.utc).isoformat()
    report['import_directory'] = destination.relative_to(root).as_posix()
    report['import_files'] = M._records(package)
    package['import.json'] = U.encoded(report)
    H.require(len(package) <= U.MAX_FILES and all(0 < len(raw) <= U.MAX_FILE for raw in package.values())
              and sum(map(len, package.values())) <= U.MAX_TOTAL, 'Complete initial candidate exceeds local byte budget')
    H.inspect(handoff, root=root)
    H.require(M._tree(source['directory']) == source['files'] and M._tree(native['directory']) == files,
              'Initial source/output changed during validation')
    destination.mkdir(parents=True, exist_ok=False)
    for name, raw in package.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    H.require(M._tree(destination) == package, 'Written initial candidate bytes differ; partial evidence retained')
    return report
