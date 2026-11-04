"""Source-bound, conditional co-rigid scale compatibility; no assembly repair.

For camera-from-native poses, a pair that is temporarily rigid must satisfy
    p = s R_a.T t_b - R_a.T t_a
and R_a.T R_b must be constant.  A changing camera or a rigidly moving assembly
can supply the excitation needed to identify s.  Motion of one part relative to
the other, shape/camera bias and unstable poses can invalidate this hypothesis.
V2 corrects the V1 excitation gate's mixed native units. Positive rescaling of
either native gauge leaves excitation/eligibility invariant; the independent
rank, positive-scale, stability and reserved-time checks remain necessary.
Nothing here changes geometry or any saved pose, estimates physical units,
infers attachment, or fills absent observations.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np


POLICY = {
    'version': 'co-rigid-scale-compatibility-v2-gauge-invariant-excitation',
    'excitation_definition': 'TRAIN RMS of centred R_A.T*t_A divided by TRAIN median t_A.z; both A-native units',
    'excitation_is_necessary_not_sufficient': True,
    'minimum_training_timestamps': 3,
    'minimum_reserved_timestamps': 1,
    'maximum_relative_rotation_error_degrees': 5.,
    'minimum_translation_excitation_over_camera_depth': .01,
    'maximum_column_normalized_condition': 100.,
    'maximum_leave_one_training_out_relative_scale_spread': .10,
    'maximum_offset_error_over_anchor_camera_depth': .02,
    'maximum_episode_pairs': 16,
    'maximum_rows_per_pair': 128,
    'temporal_split': 'preserve each saved pre-reserved evaluation-timestamp flag; no new selection',
    'fit': 'ordinary least squares scalar scale and constant offset on TRAIN only',
    'rotation_reference': 'proper polar mean of TRAIN relative rotations only',
    'reference_pose_policy': 'may condition fit, remains unverified, never a validation success',
    'gauge': 'component A unchanged native units; B scale ratio only; no metric units',
    'model_geometry_changed': False, 'saved_poses_changed': False,
    'manual_geometry_or_poses': False, 'new_api_calls': 0,
    'new_model_inference_calls': 0, 'physical_scale_known': False,
    'physical_assembly_established': False, 'joint_or_contact_established': False,
    'selection_from_fit_or_reserved_results': False,
}
SUPPORTED = 'supported_nonreference_candidate'
REFERENCE = 'reference_unverified'
CONVENTION = 'camera_from_native_mesh; OpenCV x-right,y-down,z-forward; native mesh @ R.T+t'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def data_file(root, value):
    root = Path(root).resolve()
    text = str(value).removeprefix('/files/')
    rel = Path(text)
    if rel.is_absolute() or '..' in rel.parts or not rel.parts or rel.parts[0] != 'data':
        raise ValueError('Require a saved project data path')
    path = root / rel
    if path.resolve() != path or not path.is_file():
        raise ValueError('Saved artifact missing or symlinked')
    return path


def valid_rotation(value):
    r = np.asarray(value, dtype=float)
    return bool(r.shape == (3, 3) and np.isfinite(r).all()
                and np.linalg.norm(r @ r.T - np.eye(3)) < 1e-5
                and abs(np.linalg.det(r) - 1.) < 1e-5)


def angle_degrees(rotation):
    return float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1.) / 2., -1., 1.))))


def proper_mean(rotations):
    u, _, vt = np.linalg.svd(np.sum(rotations, axis=0))
    return u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt


def solve_train_scale(u, v):
    """u_i = s v_i - p; translation elimination exposes the scalar gauge."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    if u.ndim != 2 or u.shape[1] != 3 or v.shape != u.shape or len(u) < 2:
        raise ValueError('Need matching Nx3 observations from at least two times')
    if not np.isfinite(u).all() or not np.isfinite(v).all():
        raise ValueError('Nonfinite scale observations')
    uc, vc = u - u.mean(axis=0), v - v.mean(axis=0)
    denominator = float(np.sum(vc * vc))
    if denominator <= 1e-20:
        raise ValueError('No translational excitation; scale remains unobservable')
    scale = float(np.sum(vc * uc) / denominator)
    offset = scale * v.mean(axis=0) - u.mean(axis=0)
    a = np.column_stack((-np.tile(np.eye(3), (len(u), 1)), v.reshape(-1)))
    columns = np.linalg.norm(a, axis=0)
    if np.any(columns <= 1e-20):
        condition = math.inf
    else:
        condition = float(np.linalg.cond(a / columns))
    return scale, offset, condition


def fit_compatibility(rows, *, policy=None):
    """Fit one episode pair. Reserved rows are never passed to either fit."""
    policy = dict(POLICY if policy is None else policy)
    if len(rows) > policy['maximum_rows_per_pair']:
        raise ValueError('Pair exceeds fixed row bound')
    if len({r['sample_index'] for r in rows}) != len(rows) or len({round(float(r['timestamp_s']), 7) for r in rows}) != len(rows):
        raise ValueError('Duplicate source IDs or timestamps are not independent evidence')
    train = [r for r in rows if not r['reserved']]
    check = [r for r in rows if r['reserved']]
    result = {'status': 'unsupported', 'reasons': [], 'training_count': len(train),
              'reserved_count': len(check), 'training_reference_conditioned_count': sum(r['a_reference'] or r['b_reference'] for r in train),
              'reserved_reference_rows_excluded': 0, 'training_sample_indices': [r['sample_index'] for r in train],
              'reserved_sample_indices': [r['sample_index'] for r in check], 'diagnostic_fit': None,
              'common_frame_transform_a_from_b': None, 'common_frame_transform_is_physical_truth': False,
              'physical_scale_known': False, 'physical_assembly_established': False,
              'joint_or_contact_established': False, 'geometry_or_saved_poses_changed': False,
              'metrics': [], 'reserved_used_to_fit': False}
    if len(train) < policy['minimum_training_timestamps']:
        result['reasons'].append('insufficient_training_timestamps')
    if len(check) < policy['minimum_reserved_timestamps']:
        result['reasons'].append('insufficient_pre_reserved_timestamps')
    if result['reasons']:
        return result
    for row in rows:
        if not valid_rotation(row['a_R']) or not valid_rotation(row['b_R']):
            raise ValueError('Proper finite rigid rotations required')
        for name in ['a_t', 'b_t']:
            t = np.asarray(row[name], float)
            if t.shape != (3,) or not np.isfinite(t).all() or t[2] <= 0:
                raise ValueError('Finite positive-depth camera-relative translation required')
    def coordinates(rs):
        return (np.array([np.asarray(r['a_R']).T @ r['a_t'] for r in rs]),
                np.array([np.asarray(r['a_R']).T @ r['b_t'] for r in rs]))
    u, v = coordinates(train)
    reference = proper_mean([np.asarray(r['a_R']).T @ r['b_R'] for r in train])
    depth = float(np.median([r['a_t'][2] for r in train]))
    excitation = float(np.sqrt(np.mean(np.sum((u - u.mean(axis=0)) ** 2, axis=1))) / depth)
    if excitation < policy['minimum_translation_excitation_over_camera_depth']:
        result['reasons'].append('insufficient_translation_excitation')
    try:
        scale, offset, condition = solve_train_scale(u, v)
    except ValueError:
        result['reasons'].append('scale_unobservable')
        return result
    if not math.isfinite(condition) or condition > policy['maximum_column_normalized_condition']:
        result['reasons'].append('ill_conditioned_scale_and_offset')
    if not math.isfinite(scale) or scale <= 0:
        result['reasons'].append('nonpositive_or_nonfinite_scale')
    leave_scales = []
    for i in range(len(train)):
        keep = np.arange(len(train)) != i
        try:
            leave_scales.append(solve_train_scale(u[keep], v[keep])[0])
        except ValueError:
            leave_scales.append(None)
    valid = [x for x in leave_scales if x is not None and math.isfinite(x) and x > 0]
    spread = ((max(valid) - min(valid)) / float(np.median(valid))) if len(valid) == len(train) else None
    if spread is None or spread > policy['maximum_leave_one_training_out_relative_scale_spread']:
        result['reasons'].append('leave_one_training_out_scale_unstable')
    for row in rows:
        ra, rb = np.asarray(row['a_R']), np.asarray(row['b_R'])
        relative_rotation = ra.T @ rb
        observed_offset = scale * (ra.T @ row['b_t']) - ra.T @ row['a_t']
        rotation_error = angle_degrees(relative_rotation @ reference.T)
        offset_error = float(np.linalg.norm(observed_offset - offset) / depth)
        group = 'reserved' if row['reserved'] else 'training'
        if rotation_error > policy['maximum_relative_rotation_error_degrees']:
            result['reasons'].append(group + '_relative_rotation_inconsistent')
        if offset_error > policy['maximum_offset_error_over_anchor_camera_depth']:
            result['reasons'].append(group + '_relative_translation_inconsistent')
        result['metrics'].append({'sample_index': row['sample_index'], 'timestamp_s': row['timestamp_s'],
                                 'split': group, 'a_reference': row['a_reference'], 'b_reference': row['b_reference'],
                                 'relative_rotation_error_deg': rotation_error,
                                 'offset_error_over_anchor_camera_depth': offset_error,
                                 'observed_relative_offset_in_a_units': observed_offset.tolist()})
    result['reasons'] = sorted(set(result['reasons']))
    result['diagnostic_fit'] = {'scale_b_in_a_native_units': scale, 'offset_b_origin_in_a_native_units': offset.tolist(),
                                'relative_orientation_train_mean': reference.tolist(),
                                'column_normalized_condition': condition if math.isfinite(condition) else None,
                                'translation_excitation_over_anchor_depth': excitation,
                                'leave_one_training_out_scales': leave_scales,
                                'leave_one_training_out_relative_spread': spread,
                                'normalizing_anchor_camera_depth': depth,
                                'fit_status': 'accepted_compatibility_only' if not result['reasons'] else 'rejected_diagnostic_only'}
    if not result['reasons']:
        matrix = np.eye(4)
        matrix[:3, :3], matrix[:3, 3] = scale * reference, offset
        result.update(status='conditionally_compatible_scale_hypothesis', common_frame_transform_a_from_b=matrix.tolist())
    return result


def scope_safe(frame):
    s = frame.get('scope_assessment', {})
    return (s.get('scope_reviewed') is True and s.get('scope_safe') is True
            and s.get('single_component_visible') is True
            and s.get('component_boundary_plausible') is True
            and s.get('other_components_inside_mask') is False)


def prepare_episode_pairs(first, second):
    """Join only exact source bytes/PTS and each unchanged episode gauge."""
    for doc in [first, second]:
        if doc.get('format') not in {'short-interval-motion', 'short-interval-density-run', 'short-interval-pair-motion-run'} or doc.get('version') != 1 or doc.get('status') != 'complete_with_gaps':
            raise ValueError('Require preserved completed short-interval experiments')
        if doc.get('coordinate_convention') != CONVENTION or doc.get('manual_geometry_or_poses') is not False:
            raise ValueError('Unchanged automatic native camera pose convention required')
    if first['source_video_sha256'] != second['source_video_sha256'] or first['interaction_sha256'] != second['interaction_sha256']:
        raise ValueError('Require one exact source video and source program')
    if first['component_id'] == second['component_id']:
        raise ValueError('Require distinct component identities')
    groups = []
    for a in first['episodes']:
        for b in second['episodes']:
            if a['intrinsics'] != b['intrinsics']:
                raise ValueError('A shared source camera/pixel convention is required')
            fa = {f['sample_index']: f for f in a['frames']}
            fb = {f['sample_index']: f for f in b['frames']}
            if len(fa) != len(a['frames']) or len(fb) != len(b['frames']):
                raise ValueError('Duplicate episode source ID')
            common = sorted(set(fa) & set(fb))
            if not common:
                continue
            rows, unavailable = [], []
            for i in common:
                x, y = fa[i], fb[i]
                if abs(x['timestamp_s'] - y['timestamp_s']) > 1e-7 or x['source_image_sha256'] != y['source_image_sha256']:
                    raise ValueError('Matching source sample must have identical image bytes and PTS')
                if x['width'] != y['width'] or x['height'] != y['height']:
                    raise ValueError('Source dimensions differ')
                reserve_a, reserve_b = [f.get('evaluation_timestamp_reserved_before_reference_selection') for f in [x, y]]
                if type(reserve_a) is not bool or reserve_a != reserve_b:
                    raise ValueError('Preserved temporal split differs')
                reasons = []
                if any(f.get('status') not in {SUPPORTED, REFERENCE} for f in [x, y]):
                    reasons.append('missing_two_available_saved_pose_candidates')
                if any((f.get('status') == REFERENCE and f.get('reference') is not True)
                       or (f.get('status') == SUPPORTED and f.get('reference') is not False) for f in [x, y]):
                    raise ValueError('Reference status and flag disagree')
                if any(not scope_safe(f) for f in [x, y]):
                    reasons.append('two_exact_scope_safe_masks_required')
                if reserve_a and (x.get('reference') or y.get('reference')):
                    reasons.append('reference_cannot_be_reserved_validation')
                if reasons:
                    unavailable.append({'sample_index': i, 'timestamp_s': x['timestamp_s'], 'reasons': reasons})
                    continue
                rows.append({'sample_index': i, 'timestamp_s': x['timestamp_s'], 'reserved': reserve_a,
                             'a_reference': x.get('reference') is True, 'b_reference': y.get('reference') is True,
                             'a_R': x['candidate_rotation'], 'a_t': x['candidate_translation'],
                             'b_R': y['candidate_rotation'], 'b_t': y['candidate_translation'],
                             'source_image_sha256': x['source_image_sha256'],
                             'a_mask_sha256': x['mask_sha256'], 'b_mask_sha256': y['mask_sha256']})
            groups.append({'a_episode_id': a['id'], 'b_episode_id': b['id'], 'intrinsics': a['intrinsics'],
                           'common_source_sample_indices': common, 'rows': rows, 'unavailable': unavailable})
    if len(groups) > POLICY['maximum_episode_pairs']:
        raise ValueError('Too many independent episode pairs')
    return groups


def verify_saved_document(root, path):
    path = data_file(root, path)
    doc = json.loads(path.read_text())
    pins = [(doc['source']['path'], doc['source_video_sha256']),
            (doc['asset']['path'], doc['mesh_sha256']), (doc['interaction_path'], doc['interaction_sha256']),
            (doc['audit_binding']['path'], doc['audit_binding']['sha256'])]
    for episode in doc['episodes']:
        for frame in episode['frames']:
            pins += [(frame['source_image_url'], frame['source_image_sha256']), (frame['mask_url'], frame['mask_sha256'])]
    verified = []
    for name, sha in sorted(set(pins)):
        saved = data_file(root, name)
        if digest(saved) != sha:
            raise ValueError('Changed saved input artifact: ' + str(name))
        verified.append({'path': str(saved.relative_to(root)), 'sha256': sha})
    return doc, {'path': str(path.relative_to(root)), 'sha256': digest(path), 'verified_artifacts': verified}


def run(input_pairs, output, *, root):
    root, output = Path(root).resolve(), Path(output).absolute()
    if output.exists() or not output.is_relative_to(root / 'data') or '..' in output.parts or output.resolve() != output:
        raise ValueError('Require a fresh project data output directory')
    prepared, bindings = [], []
    for first_path, second_path in input_pairs:
        a, ba = verify_saved_document(root, first_path)
        b, bb = verify_saved_document(root, second_path)
        groups = prepare_episode_pairs(a, b)
        prepared.append((a, b, groups)); bindings.append({'first': ba, 'second': bb})
    output.mkdir(parents=True)
    protocol = {'format': 'co-rigid-scale-compatibility-protocol', 'version': 2,
                'created_at_utc': datetime.now(timezone.utc).isoformat(), 'policy': POLICY,
                'input_pairs': bindings, 'implementation_sha256': digest(__file__),
                'selection_before_fit': [{'component_a': a['component_id'], 'component_b': b['component_id'],
                                         'episode_pairs': [{'a_episode_id': g['a_episode_id'], 'b_episode_id': g['b_episode_id'],
                                                            'common_source_sample_indices': g['common_source_sample_indices'],
                                                            'train_samples': [r['sample_index'] for r in g['rows'] if not r['reserved']],
                                                            'reserved_samples': [r['sample_index'] for r in g['rows'] if r['reserved']],
                                                            'unavailable': g['unavailable']} for g in groups]}
                                        for a, b, groups in prepared],
                'new_api_calls': 0, 'new_model_inference_calls': 0, 'originals_modified': False}
    write(output / 'protocol.json', protocol)
    (output / 'implementation.py').write_bytes(Path(__file__).read_bytes())
    results = []
    for a, b, groups in prepared:
        for group in groups:
            report = fit_compatibility(group['rows'])
            report.update(component_a=a['component_id'], component_b=b['component_id'],
                          a_episode_id=group['a_episode_id'], b_episode_id=group['b_episode_id'],
                          common_source_sample_indices=group['common_source_sample_indices'], unavailable=group['unavailable'],
                          source_video_sha256=a['source_video_sha256'], interaction_sha256=a['interaction_sha256'],
                          mesh_a_sha256=a['mesh_sha256'], mesh_b_sha256=b['mesh_sha256'])
            results.append(report)
    final = {'format': 'co-rigid-scale-compatibility-report', 'version': 2, 'status': 'complete_diagnostic',
             'protocol_sha256': digest(output / 'protocol.json'), 'policy': POLICY, 'pairs': results,
             'accepted_conditional_hypotheses': sum(r['status'] == 'conditionally_compatible_scale_hypothesis' for r in results),
             'physical_assembly_established': False, 'metric_scale_known': False,
             'new_api_calls': 0, 'new_model_inference_calls': 0,
             'limitations': ['Temporarily rigid relative motion is a tested hypothesis, not an observed attachment or contact.',
                             'Unverified automatic references and conditional mesh poses are not ground truth.',
                             'Assumed intrinsics and model shape bias can produce apparent consistency or disagreement.',
                             'Native meshes, saved poses, source programs and independent episode gauges remain unchanged.',
                             'A single reserved timestamp is only one compatibility check, not physical or independent repeated validation.',
                             'No joints, common metric world, hidden insertion, collision or simulator execution are inferred.']}
    write(output / 'assembly-compatibility.json', final)
    return final
