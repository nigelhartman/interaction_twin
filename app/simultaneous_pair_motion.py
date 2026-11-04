"""Single unchanged-reference tracking context and saved-peer compatibility.

This creates simultaneous source evidence, not a cadence benchmark, fitted
joint, repaired mesh, calibrated assembly or simulator trajectory.
"""
from __future__ import annotations
import copy
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
from PIL import Image

import interval_motion as M
import interval_motion_density as D
import assembly_scale_feasibility as A
from native_appearance import load_native_glb


POLICY = {'version': 'single-context-simultaneous-pair-motion-v1', 'maximum_frames': 25,
          'maximum_runtime_s': 900, 'maximum_actual_cotracker_calls': 2,
          'purpose': 'simultaneous_pair_source_evidence_not_cadence_accuracy',
          'reference_search_repeated': False, 'saved_reference_R_t_K_crop_queries_unchanged': True,
          'score_gates_and_point_splits_unchanged': True, 'new_api_calls': 0,
          'manual_geometry_or_poses': False, 'physical_assembly_established': False}


def same_source_rows(first, second):
    """One-to-one source joining never interpolates or copies another mask."""
    if len(first) != len(second) or len({r['sample_index'] for r in first}) != len(first):
        raise ValueError('Require exactly matching unique native source rows')
    index = {r['sample_index']: r for r in second}
    for a in first:
        b = index.get(a['sample_index'])
        keys = ['timestamp_s', 'native_pts_exact_s', 'native_decoded_frame_index_zero_based',
                'source_image_sha256', 'width', 'height', 'temporal_split']
        if b is None or any(a[k] != b[k] for k in keys):
            raise ValueError('Paired source bytes/PTS/native IDs/pixel convention differ')
    return True


def compatibility_from_named_formats(first, second):
    """In-memory field adapter; original files retain their actual format/hash."""
    if first.get('format') != 'short-interval-pair-motion-run' or second.get('format') != 'short-interval-density-run':
        raise ValueError('Only the explicit new pair run and existing peer context are accepted')
    if first['status'] != 'complete_with_gaps' or second['status'] != 'complete_with_gaps':
        return {'status': 'unsupported_missing_completed_pose_context', 'accepted_conditional_hypotheses': 0,
                'physical_assembly_established': False, 'pairs': []}
    copies = copy.deepcopy([first, second])
    for doc in copies:
        doc['format'] = 'short-interval-motion'
    groups = A.prepare_episode_pairs(*copies)
    reports = []
    for group in groups:
        report = A.fit_compatibility(group['rows'])
        report.update(a_episode_id=group['a_episode_id'], b_episode_id=group['b_episode_id'],
                      common_source_sample_indices=group['common_source_sample_indices'],
                      unavailable=group['unavailable'])
        reports.append(report)
    return {'status': 'complete_diagnostic', 'accepted_conditional_hypotheses': sum(
                r['status'] == 'conditionally_compatible_scale_hypothesis' for r in reports),
            'format_adaptation': 'in-memory field-preserving named-format adapter; original source rows, poses and bytes unchanged',
            'source_formats': [first['format'], second['format']], 'policy': A.POLICY,
            'physical_assembly_established': False, 'metric_scale_known': False,
            'joint_or_contact_established': False, 'pairs': reports}


def run(plan_path, audit_path, output, *, root):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists() or not output.is_relative_to(root / 'data'):
        raise ValueError('Require a fresh separate pair-motion output directory')
    M.verify_worker_lock(root)
    started = time.monotonic(); deadline = started + POLICY['maximum_runtime_s']
    plan_path, audit_path = M.file_path(root, plan_path), M.file_path(root, audit_path)
    plan = M.read(plan_path); manifest_path = M.file_path(root, plan['input_manifest']['path'])
    if M.sha(manifest_path) != plan['input_manifest']['sha256']:
        raise ValueError('Prepared component input changed')
    manifest = M.read(manifest_path); D.validate_plan_manifest(plan, manifest)
    if manifest.get('test_intent') != 'simultaneous_component_pair_evidence_not_cadence_accuracy' or manifest['status'] != 'complete_masks_unaudited':
        raise ValueError('Require the prepared exact peer-window input')
    if len(manifest['frames']) > POLICY['maximum_frames'] or not manifest['reference_analysis_mask_byte_identity_verified']:
        raise ValueError('Bounded source window and unchanged original mask required')
    parent_path = M.file_path(root, manifest['parent_interval_result']['path'])
    peer_path = M.file_path(root, manifest['peer_interval_result']['path'])
    pair_protocol_path = M.file_path(root, manifest['pair_protocol']['path'])
    for path, sha in [(parent_path, manifest['parent_interval_result']['sha256']),
                      (peer_path, manifest['peer_interval_result']['sha256']),
                      (pair_protocol_path, manifest['pair_protocol']['sha256'])]:
        if M.sha(path) != sha: raise ValueError('Frozen component/peer/protocol changed')
    parent, peer = M.read(parent_path), M.read(peer_path)
    old = next(e for e in parent['episodes'] if e['id'] == manifest['selected_episode_id'])
    if parent['policy'] != M.POLICY or M.read(pair_protocol_path)['assembly_compatibility_policy'] != A.POLICY:
        raise ValueError('Prospective score/assembly gates changed')
    for k, prior in [('fixed_reference_fit', old['reference_fit']), ('fixed_intrinsics', old['intrinsics']),
                     ('fixed_original_query_uv_source', old['queries']['automatic_query_uv_source']),
                     ('fixed_original_crop_xyxy', old['crop_xyxy'])]:
        if manifest[k] != prior: raise ValueError('Frozen automatic reference settings changed')
    old_ref = next(r for r in old['frames'] if r['reference'])
    best_saved = manifest['fixed_reference_fit']['candidates'][0]
    if not np.array_equal(best_saved['rotation'], old_ref['candidate_rotation']) or not np.array_equal(best_saved['translation'], old_ref['candidate_translation']):
        raise ValueError('Original fitted candidate does not equal saved reference pose')
    if len(peer['episodes']) != 1: raise ValueError('Exactly one frozen peer context required')
    same_source_rows(manifest['frames'], peer['episodes'][0]['frames'])
    mesh_path = M.file_path(root, manifest['asset']['path'])
    for path, sha in [(mesh_path, manifest['mesh_sha256']),
                      (M.file_path(root, manifest['source']['path']), manifest['source_video_sha256']),
                      (M.file_path(root, manifest['interaction_path']), manifest['interaction_sha256'])]:
        if M.sha(path) != sha: raise ValueError('Original mesh/source/program changed')
    for row in manifest['frames']:
        for name, sha in [(row['source_image_url'], row['source_image_sha256']), (row['mask_url'], row['mask_sha256'])]:
            if M.sha(M.file_path(root, name)) != sha: raise ValueError('Prepared source pixels or mask changed')
    assessment, audit = M.load_audit(plan, plan_path, audit_path, root)
    output.mkdir(parents=True)
    (output / 'pair-motion-implementation.py').write_bytes(Path(__file__).read_bytes())
    (output / 'assembly-implementation.py').write_bytes(Path(A.__file__).read_bytes())
    record = {'format': 'short-interval-pair-motion-run', 'version': 1, 'status': 'running',
              'policy': POLICY, 'motion_policy': M.POLICY, 'assembly_policy': A.POLICY,
              'case_id': manifest['case_id'], 'component_id': manifest['track_id'], 'track_id': manifest['track_id'],
              'label': manifest['label'], 'source': manifest['source'], 'source_video_sha256': manifest['source_video_sha256'],
              'asset': manifest['asset'], 'mesh_sha256': manifest['mesh_sha256'],
              'interaction_path': manifest['interaction_path'], 'interaction_sha256': manifest['interaction_sha256'],
              'source_extension': {'path': str(manifest_path.relative_to(root)), 'sha256': M.sha(manifest_path)},
              'plan': {'path': str(plan_path.relative_to(root)), 'sha256': M.sha(plan_path)},
              'audit_binding': audit, 'parent_interval_result': manifest['parent_interval_result'],
              'peer_interval_result': manifest['peer_interval_result'], 'pair_protocol': manifest['pair_protocol'],
              'implementation': M.snapshot(root, output), 'pair_implementation_sha256': M.sha(__file__),
              'coordinate_convention': parent['coordinate_convention'],
              'inherited_automatic_reference_pose_used': True, 'reference_search_repeated': False,
              'manual_geometry_or_poses': False, 'new_api_calls': 0, 'physical_pose_accuracy_measured': False,
              'physical_scale_known': False, 'common_assembly_established': False,
              'original_semantic_program_extended': False, 'episodes': [],
              'host_started_at_utc': datetime.now(timezone.utc).isoformat()}
    M.write(output / 'interval-motion.json', record)
    mesh = load_native_glb(mesh_path); k = np.asarray(manifest['fixed_intrinsics']['matrix'], float)
    tracker = M.OfflineTracker(root, output, deadline)
    seed_image = np.asarray(Image.open(M.file_path(root, manifest['seed']['source_image_url'])).convert('RGB'))
    seed_mask = np.asarray(Image.open(M.file_path(root, manifest['seed']['mask_url'])).convert('L')) > 0
    def frozen_reference(mesh_arg, image, mask, camera, _deadline):
        if not np.array_equal(image, seed_image) or not np.array_equal(mask, seed_mask) or not np.array_equal(camera, k):
            raise ValueError('Inherited reference pixels/mask/K changed')
        best = copy.deepcopy(best_saved)
        best['rotation'], best['translation'] = np.asarray(best['rotation']), np.asarray(best['translation'])
        receipt = copy.deepcopy(manifest['fixed_reference_fit'])
        receipt.update(method='Inherited exact earlier automatic component reference; no new pose search',
                       inherited_automatic_reference_pose_used=True, reference_search_repeated=False,
                       old_pose_initialization_used=True, old_pose_source='saved automatic base interval reference',
                       reference_fit_is_physical_truth=False, manual_pose_used=False)
        return best, receipt
    try:
        episode = copy.deepcopy(plan['episodes'][0]); episode['id'] = 'simultaneous-base-12fps'
        target = output / 'component'; target.mkdir()
        result = M.track_episode(episode, assessment, mesh, k, tracker, root, target, deadline,
                                 reference_provider=frozen_reference, fixed_crop=manifest['fixed_original_crop_xyxy'],
                                 expected_query_uv=manifest['fixed_original_query_uv_source'])
        if len(tracker.calls) > POLICY['maximum_actual_cotracker_calls']:
            raise ValueError('Fixed single-context model-call bound exceeded')
        record.update(status='complete_with_gaps' if any(f['status'] == M.SUPPORTED for f in result['frames']) else 'unsupported',
                      episodes=[result], frame_count=len(result['frames']),
                      supported_nonreference_frames=sum(f['status'] == M.SUPPORTED for f in result['frames']),
                      reference_frames=sum(f['reference'] for f in result['frames']),
                      actual_cotracker_model_calls=len(tracker.calls), wall_seconds=time.monotonic() - started)
        if 'tracks_url' in result:
            checks, delta = D.same_point_arrays(M.file_path(root, result['tracks_url']), M.file_path(root, old['tracks_url']))
            if not all(checks.values()): raise ValueError('Original canonical query/ray/split identity changed')
            record['original_point_identity_proof'] = {'checks': checks, 'maximum_canonical_difference': delta,
                                                      'original_tracks_sha256': old['tracks_sha256']}
        M.write(output / 'interval-motion.json', M.serial(record))
        compatibility = compatibility_from_named_formats(record, peer)
        compatibility.update(first={'path': str((output / 'interval-motion.json').relative_to(root)),
                                    'sha256': M.sha(output / 'interval-motion.json')},
                             second=manifest['peer_interval_result'], new_api_calls=0,
                             new_model_inference_calls=0)
        M.write(output / 'assembly-compatibility.json', M.serial(compatibility))
        if M.sha(manifest_path) != record['source_extension']['sha256'] or M.sha(peer_path) != record['peer_interval_result']['sha256'] or M.sha(mesh_path) != record['mesh_sha256']:
            raise ValueError('Original source/peer/model changed during experiment')
        return record, compatibility
    except Exception as error:
        record.update(status='failed', error={'type': type(error).__name__, 'message': str(error)},
                      actual_cotracker_model_calls=len(tracker.calls), wall_seconds=time.monotonic() - started)
        M.write(output / 'interval-motion.json', M.serial(record)); raise
