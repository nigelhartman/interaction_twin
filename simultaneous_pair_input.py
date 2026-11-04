"""One unchanged-reference component on an existing peer's native source window.

The purpose is simultaneous paired evidence, not a cadence accuracy comparison.
Only exact saved peer JPEGs are copied. The sole prompt is the selected earlier
automatic reference's original SAM3.1 mask. No poses are refitted or invented.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

import interval_motion as M
import interval_motion_dense_input as D
from assembly_scale_feasibility import POLICY as ASSEMBLY_POLICY


POLICY = {
    'version': 'simultaneous-pair-evidence-input-v1',
    'purpose': 'simultaneous_component_pair_evidence_not_cadence_accuracy',
    'maximum_frames': 25, 'maximum_runtime_s': 900,
    'reference': 'exact earlier automatic component reference inside the already frozen peer window',
    'source_frames': 'byte-identical existing peer native-PTS JPEGs; no new extraction or interpolation',
    'mask_prompt': 'one byte-identical saved hosted SAM3.1 mask; no other mask or coordinate prompts',
    'future_tracker': 'single native-frame context with original reference R/t/K/crop/query IDs and unchanged score gates',
    'other_component_geometry_or_masks_used_as_target': False,
    'new_api_calls': 0, 'downloads': False, 'manual_boxes_points_geometry_poses': False,
    'saved_geometry_or_poses_modified': False, 'physical_assembly_established': False,
    'original_program_evidence_extended_or_overwritten': False,
}


def validate_window(parent_plan, parent_result, peer, peer_result, episode_id):
    for key in ['interaction_sha256', 'source_video_sha256']:
        if any(x[key] != parent_plan[key] for x in [parent_result, peer, peer_result]):
            raise ValueError('One unchanged source video/program is required')
    if peer['track_id'] == parent_plan['track_id'] or peer_result['component_id'] != peer['track_id']:
        raise ValueError('Require a distinct preserved peer component')
    if parent_result['track_id'] != parent_plan['track_id'] or parent_result['mesh_sha256'] != parent_plan['mesh_sha256']:
        raise ValueError('Parent component or geometry changed')
    episodes = [e for e in parent_result['episodes'] if e['id'] == episode_id]
    plans = [e for e in parent_plan['episodes'] if e['id'] == episode_id]
    if len(episodes) != 1 or len(plans) != 1:
        raise ValueError('Require one explicitly frozen parent episode')
    episode, original = episodes[0], plans[0]
    if not episode.get('reference_fit') or episode['reference_sample_index'] != original['reference_sample_index']:
        raise ValueError('Original automatic reference fit is required')
    frames = peer['frames']
    if not 3 <= len(frames) <= POLICY['maximum_frames'] or len({f['sample_index'] for f in frames}) != len(frames):
        raise ValueError('Require a bounded unique frozen peer source window')
    if any(b['timestamp_s'] <= a['timestamp_s'] for a, b in zip(frames, frames[1:])):
        raise ValueError('Frozen source window must be strictly chronological')
    reference = next(f for f in original['frames'] if f['sample_index'] == original['reference_sample_index'])
    candidates = [f for f in frames if f.get('original_program_sample_index') == reference['sample_index']]
    if len(candidates) != 1 or candidates[0]['source_image_sha256'] != reference['source_image_sha256'] or abs(candidates[0]['timestamp_s'] - reference['timestamp_s']) > .000002:
        raise ValueError('Frozen component reference must exactly match one existing peer native source image')
    if peer['fixed_intrinsics'] != episode['intrinsics']:
        raise ValueError('Peer and component need identical camera/pixel convention')
    if parent_result['policy'] != M.POLICY:
        raise ValueError('Earlier component score gates differ from fixed current policy')
    return episode, reference, candidates[0]


def prepare(parent_plan_path, parent_result_path, peer_input_path, peer_result_path, output, *, root, episode_id):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists() or not output.is_relative_to(root / 'data'):
        raise ValueError('Require a fresh separate component-pair input directory')
    paths = [M.file_path(root, p) for p in [parent_plan_path, parent_result_path, peer_input_path, peer_result_path]]
    parent_plan, parent, peer, peer_result = [M.read(p) for p in paths]
    if parent['plan']['sha256'] != M.sha(paths[0]):
        raise ValueError('Preserved parent plan identity changed')
    if peer_result['source_extension']['sha256'] != M.sha(paths[2]):
        raise ValueError('Existing peer result is not bound to this peer input')
    previous, reference, matching = validate_window(parent_plan, parent, peer, peer_result, episode_id)
    for doc in [parent, peer_result]:
        for path, sha in [(doc['source']['path'], doc['source_video_sha256']), (doc['asset']['path'], doc['mesh_sha256']),
                          (doc['interaction_path'], doc['interaction_sha256'])]:
            if M.sha(M.file_path(root, path)) != sha:
                raise ValueError('Saved source/program/native asset changed')
    seed_path = M.file_path(root, reference['mask_url'])
    if M.sha(seed_path) != reference['mask_sha256']:
        raise ValueError('Saved original component SAM3.1 seed changed')
    seed = np.asarray(Image.open(seed_path).convert('L'))
    if seed.shape != (reference['height'], reference['width']) or not set(np.unique(seed)).issubset({0, 255}) or not np.any(seed):
        raise ValueError('Require an unchanged full-frame binary component seed')
    for row in peer['frames']:
        image = M.file_path(root, row['source_image_url'])
        if M.sha(image) != row['source_image_sha256'] or Image.open(image).size != (reference['width'], reference['height']):
            raise ValueError('Existing peer native source JPEG changed')
    output.mkdir(parents=True)
    records = [{'path': str(p.relative_to(root)), 'sha256': M.sha(p)} for p in paths]
    protocol = {'format': 'simultaneous-pair-evidence-protocol', 'version': 1,
                'created_at_utc': datetime.now(timezone.utc).isoformat(), 'policy': POLICY,
                'inputs': records, 'parent_component_id': parent['track_id'], 'peer_component_id': peer['track_id'],
                'parent_episode_id': episode_id, 'original_reference_sample_index': reference['sample_index'],
                'source_video_sha256': parent['source_video_sha256'], 'mesh_sha256': parent['mesh_sha256'],
                'peer_mesh_sha256': peer_result['mesh_sha256'], 'interaction_sha256': parent['interaction_sha256'],
                'planned_native_frames': [{k: f[k] for k in ['sample_index', 'timestamp_s', 'native_pts_exact_s',
                                         'native_decoded_frame_index_zero_based', 'source_image_sha256', 'temporal_split']}
                                        for f in peer['frames']],
                'fixed_reference_fit': previous['reference_fit'], 'fixed_intrinsics': previous['intrinsics'],
                'fixed_original_query_uv_source': previous['queries']['automatic_query_uv_source'],
                'fixed_original_crop_xyxy': previous['crop_xyxy'], 'motion_policy': M.POLICY,
                'assembly_compatibility_policy': ASSEMBLY_POLICY,
                'implementation_sha256': M.sha(__file__), 'sam2_adapter_sha256': M.sha(D.__file__),
                'tracking_not_yet_run': True, 'new_api_calls': 0,
                'future_audit': 'root-owned hosted exact-mask scope audit before any tracker outcomes'}
    M.write(output / 'pair-protocol.json', protocol)
    (output / 'preparation-implementation.py').write_bytes(Path(__file__).read_bytes())
    (output / 'sam2-adapter-at-prepare.py').write_bytes(Path(D.__file__).read_bytes())
    frames_dir = output / 'frames'; frames_dir.mkdir()
    frames = []
    fields = ['sample_index', 'timestamp_s', 'native_pts_exact_s', 'native_decoded_frame_index_zero_based',
              'width', 'height', 'original_program_sample_index', 'original_program_timestamp_s',
              'program_pts_rounding_difference_s', 'temporal_split']
    for i, old in enumerate(peer['frames']):
        target = frames_dir / f'{i + 1:06d}.jpg'
        shutil.copyfile(M.file_path(root, old['source_image_url']), target)
        row = {k: old[k] for k in fields}
        row.update(local_index_zero_based=i, source_image_url=M.public_url(root, target), source_image_sha256=M.sha(target),
                   peer_source_image_url=old['source_image_url'], peer_source_image_sha256=old['source_image_sha256'],
                   byte_identical_peer_native_jpeg=True)
        frames.append(row)
    anchor = next(f for f in frames if f['sample_index'] == matching['sample_index'])
    metadata = {'format': 'short-interval-density-input', 'version': 1, 'status': 'frames_prepared',
                'test_intent': 'simultaneous_component_pair_evidence_not_cadence_accuracy', 'policy': copy.deepcopy(POLICY),
                'pair_protocol': {'path': str((output / 'pair-protocol.json').relative_to(root)), 'sha256': M.sha(output / 'pair-protocol.json')},
                'case_id': parent['case_id'], 'track_id': parent['track_id'], 'label': parent['label'],
                'parent_plan': records[0], 'parent_interval_result': records[1], 'peer_input': records[2], 'peer_interval_result': records[3],
                'interaction_path': parent['interaction_path'], 'interaction_sha256': parent['interaction_sha256'],
                'source': parent['source'], 'source_video_sha256': parent['source_video_sha256'],
                'asset': parent['asset'], 'mesh_sha256': parent['mesh_sha256'],
                'published_interaction_program': False, 'separate_sampling_domain': True,
                'original_program_evidence_was_extended_or_overwritten': False,
                'selected_episode_id': episode_id, 'selected_interval_s': [frames[0]['timestamp_s'], frames[-1]['timestamp_s']],
                'original_reference_sample_index': reference['sample_index'],
                'fixed_reference_fit': copy.deepcopy(previous['reference_fit']), 'fixed_intrinsics': previous['intrinsics'],
                'fixed_original_query_uv_source': previous['queries']['automatic_query_uv_source'],
                'fixed_original_crop_xyxy': previous['crop_xyxy'],
                'seed': {'source_image_url': reference['source_image_url'], 'source_image_sha256': reference['source_image_sha256'],
                         'mask_url': reference['mask_url'], 'mask_sha256': reference['mask_sha256'],
                         'origin': 'exact_saved_hosted_Meta_SAM3.1_mask', 'manual_points_or_boxes': False},
                'host_started_at_utc': datetime.now(timezone.utc).isoformat(), 'client_task_date': '2026-11-01',
                'new_api_calls': 0, 'new_model_inference_calls': 0, 'physical_pose_accuracy_measured': False,
                'implementation_sha256': M.sha(__file__), 'frames': frames, 'frame_count': len(frames),
                'reference_sample_index': anchor['sample_index'], 'reference_local_index_zero_based': anchor['local_index_zero_based'],
                'source_frame_domain': 'existing peer native decoded frame IDs and PTS; original semantic evidence untouched',
                'reference_RGB_byte_identity_verified': anchor['source_image_sha256'] == reference['source_image_sha256']}
    M.write(output / 'dense-input.json', metadata)
    return metadata
