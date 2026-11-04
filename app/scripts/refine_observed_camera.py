#!/usr/bin/env python3
"""Run a separate bounded local calibration experiment from saved raw tracks."""
from pathlib import Path
import argparse
import copy
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import observed_geometry as G
import observed_calibration as C


def dump(path, item):
    path.write_text(json.dumps(item, indent=2, allow_nan=False) + '\n')


def pack_model(model):
    return {'seed': model['seed'], 'points': {t: x.tolist() for t, x in model['points'].items()},
            'training_poses': {str(fi): {'R': p[0].tolist(), 't': p[1].tolist()} for fi, p in model['poses'].items()}}


def evaluate_fixed_heldout(model, K, original):
    """Reuse exact frozen raw held-out pixels and deterministic original roles."""
    rows, predictions, errors = [], {}, []
    oldrows = {r['frame_index']: r for r in original['heldout_evaluation']}
    for fi, frame in enumerate(original['frames']):
        if frame['split'] != 'heldout':
            continue
        obs = {o['track_id']: np.array(o['xy']) for o in frame['observations'] if o['track_id'] in model['points']}
        roles = {o['track_id']: o['role'] for o in frame['observations'] if o['track_id'] in obs}
        if any((r == 'heldout') != G.score_track(t) for t, r in roles.items()):
            raise ValueError('Original frozen score/anchor role mismatch')
        anchors = {t: uv for t, uv in obs.items() if roles[t] == 'pose_anchor'}
        scored = {t: uv for t, uv in obs.items() if roles[t] == 'heldout'}
        found = G.solve_pose(model['points'], anchors, K)
        row = {'frame_index': fi, 'timestamp_s': frame['timestamp_s'], 'matched_3d_tracks': len(obs),
               'pose_anchor_count': len(anchors), 'scored_track_count': len(scored), 'pose_registered': False,
               'passes': False, 'scored_behind_camera_points': 0, 'scored_nonfinite_error_points': 0,
               'original_baseline_passes': oldrows[fi]['passes']}
        pose = found[0] if found is not None else None
        if pose is not None:
            tids = sorted(scored)
            e = np.array([])
            if tids:
                uv, z = G.project(np.array([model['points'][t] for t in tids]), pose, K)
                e = np.linalg.norm(uv - np.array([scored[t] for t in tids]), axis=1)
                invalid = (z <= 0) | ~np.isfinite(z)
                e[invalid] = np.inf
                row['scored_behind_camera_points'] = int(invalid.sum())
                row['scored_nonfinite_error_points'] = int((~np.isfinite(e)).sum())
            # This coverage fraction depends only on the exact frozen scored pixels.
            coverage = oldrows[fi].get('score_coverage_fraction', 0.)
            row.update(pose_registered=True, pose_anchor_inliers=len(found[1]),
                       score_coverage_fraction=coverage,
                       score_median_px=float(np.median(e)) if len(e) and np.isfinite(e).all() else None,
                       score_p90_px=float(np.percentile(e, 90)) if len(e) and np.isfinite(e).all() else None)
            row['passes'] = bool(len(scored) >= G.PROTOCOL['heldout_min_score_points_per_frame'] and
                                 row['score_median_px'] is not None and row['score_median_px'] <= G.PROTOCOL['heldout_median_px_max'] and
                                 row['score_p90_px'] <= G.PROTOCOL['heldout_p90_px_max'] and coverage >= .08)
            errors.extend(float(x) for x in e)
        predictions[fi] = {'pose': pose, 'observations': obs, 'roles': roles}
        rows.append(row)
    return rows, predictions, errors


def export_viewer(output, original, chosen, K, calibration, policy_hash):
    model = chosen['model']
    rows, predictions, errors = evaluate_fixed_heldout(model, K, original)
    result = copy.deepcopy(original)
    result['id'] = output.name
    result['title'] = 'Camera body · free-focal observed geometry'
    result['status'] = 'self_calibration_sparse_candidate' if calibration['supported'] else 'self_calibration_ambiguous_sparse_candidate'
    result['reason'] = ('Training-only focal diagnostics pass provisional gates; intrinsics remain unverified.' if calibration['supported'] else
                        'Shared-focal refinement is not stably observable under the predeclared training-only checks; this is not a calibrated scan.')
    result['kind'] = 'self_calibration'
    result['calibration'] = {**calibration, 'estimated': True, 'measured': False, 'observable': calibration['supported'], 'focal_px': float(K[0, 0])}
    result['bundle_adjustment'] = chosen['metrics']
    result['input_protocol_sha256'] = policy_hash
    result['heldout_evaluation'] = rows
    result['focal_sensitivity'] = []
    result['seed'] = {**original['seed'], 'first_pose_fixed': True, 'second_rotation_and_direction_free': True}
    result['limitations'] = [
        'The shared focal estimate is derived only from the original training pixels; no measured camera calibration is available.',
        'The image centre, square pixels and zero distortion are assumptions; crop/stabilization may violate them.',
        'All starts/profiles use the same initial observed model and fixed correspondence support; a shared local basin is still possible.',
        'Low training reprojection error and held-out projections do not establish accurate 3D shape.',
        'Original held-out frame split, raw pixels and disjoint anchor/scored track roles are unchanged and were not used for selection.',
        'Scale is arbitrary with unit seed baseline; hidden surfaces, complete meshes, assembly and physical contact are not established.',
        'No generated geometry, depth, learned inference, paid API, or manual poses were used in this experiment.']
    support = {t: [] for t in model['points']}
    passing = {r['frame_index'] for r in rows if r['passes']}
    for fi, record in predictions.items():
        if fi not in passing or record['pose'] is None:
            continue
        for t, uv in record['observations'].items():
            if record['roles'][t] != 'heldout':
                continue
            prediction, z = G.project(model['points'][t][None], record['pose'], K)
            error = float(np.linalg.norm(prediction[0] - uv))
            if z[0] > 0 and np.isfinite(error) and error <= G.PROTOCOL['heldout_median_px_max']:
                support[t].append({'frame_id': original['frames'][fi]['id'], 'timestamp_s': original['frames'][fi]['timestamp_s'], 'error_px': error})
    for point in result['points']:
        t = point['id']
        point['xyz'] = model['points'][t].tolist()
        point['heldout_scored_projection_support'] = support[t]
        point['heldout_scored_projection_supported_in_at_least_two_views'] = len(support[t]) >= 2
        point['max_training_triangulation_angle_deg'] = G.maximum_track_angle(model['points'][t], chosen['tracks'][t], model['poses'])
    result['cameras'] = []
    for fi, frame in enumerate(result['frames']):
        pose = model['poses'].get(fi) if frame['split'] == 'train' else predictions[fi]['pose']
        frame['registered'] = pose is not None
        if pose is not None:
            result['cameras'].append({'frame_id': frame['id'], 'R': pose[0].tolist(), 't': pose[1].tolist(),
                                      'K': K.tolist(), 'split': 'train' if frame['split'] == 'train' else 'heldout_anchor_pose'})
        for observation in frame['observations']:
            t = observation['track_id']
            observation['predicted_xy'] = None
            observation['error_px'] = None
            if pose is not None and t in model['points']:
                uv, z = G.project(model['points'][t][None], pose, K)
                if np.isfinite(uv).all():
                    observation['predicted_xy'] = uv[0].tolist()
                    observation['error_px'] = float(np.linalg.norm(uv[0] - observation['xy']))
    finite = [e for e in errors if np.isfinite(e)]
    invalid = len(errors) - len(finite)
    result['metrics'].update(focal_px=float(K[0, 0]), focal_sensitivity_stable=calibration['supported'],
                             final_training_point_max_angle_median_deg=float(np.median([p['max_training_triangulation_angle_deg'] for p in result['points']])),
                             heldout_views_passing=sum(r['passes'] for r in rows), heldout_scored_observations=len(errors),
                             heldout_invalid_or_behind_camera_observations=invalid,
                             heldout_median_px=float(np.median(errors)) if errors and not invalid else None,
                             heldout_p90_px=float(np.percentile(errors, 90)) if errors and not invalid else None,
                             heldout_finite_error_median_px=float(np.median(finite)) if finite else None,
                             heldout_finite_error_p90_px=float(np.percentile(finite, 90)) if finite else None,
                             aggregate_sparse_projection_gates_passed=False,
                             points_independently_scored_in_at_least_two_heldout_views=sum(len(v) >= 2 for v in support.values()),
                             **G.shape_metrics(model['points']))
    dump(output / 'viewer.json', result)
    G.write_ply(output / 'points.ply', result['points'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='data/observed-geometry-experiment-20261031/camera-body-core-v5')
    parser.add_argument('--output', default='data/observed-geometry-calibration-20261031/shared-focal-v1')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = (root / args.source).resolve(); output = (root / args.output).resolve()
    if not source.is_relative_to(root / 'data') or not output.is_relative_to(root / 'data') or output.exists():
        raise ValueError('Read data and use a fresh output under project data')
    output.mkdir(parents=True)
    protocol = json.loads((source / 'protocol.json').read_text())
    raw = json.loads((source / 'geometry.json').read_text())
    original = json.loads((source / 'viewer.json').read_text())
    if original['input_protocol_sha256'] != G.digest(source / 'protocol.json'):
        raise ValueError('Original viewer protocol hash mismatch')
    if G.digest(source / 'implementation.py') != protocol['implementation_sha256']:
        raise ValueError('Original frozen implementation hash mismatch')
    for f in protocol['input_views']:
        G.safe_file(root, f['image_url'].removeprefix('/files/'), f['image_sha256'])
        G.safe_file(root, f['mask_url'].removeprefix('/files/'), f['mask_sha256'])
    G.safe_file(root, protocol['program_path'], protocol['program_sha256'])
    code_files = {'observed_calibration.py': Path(C.__file__), 'refine_observed_camera.py': Path(__file__),
                  'observed_geometry.py': Path(G.__file__)}
    policy = {'protocol': C.PROTOCOL, 'source_directory': str(source.relative_to(root)),
              'source_files': {n: G.digest(source / n) for n in ['protocol.json', 'geometry.json', 'viewer.json', 'implementation.py']},
              'code_files': {n: G.digest(p) for n, p in code_files.items()},
              'inputs': protocol['input_views'], 'source_program': protocol['program_path'],
              'source_program_sha256': protocol['program_sha256'], 'new_api_calls': 0,
              'read_generated_mesh': False, 'read_predicted_depth': False, 'read_manual_pose': False,
              'geometry_initialization': 'saved_observed_v5_training_only_mathematical_estimate',
              'heldout_policy': 'freeze_train_selection_before_evaluation; preserve_original_pixels_and_roles'}
    dump(output / 'protocol.json', policy)
    for name, path in code_files.items():
        (output / name).write_bytes(path.read_bytes()); (output / name).chmod(0o444)
    tracks = {t: {int(fi): np.array(uv) for fi, uv in obs.items()} for t, obs in raw['tracks'].items()}
    model = {'seed': original['seed'], 'points': {t: np.array(x) for t, x in raw['points'].items()},
             'poses': {int(fi): (np.array(p['R']), np.array(p['t'])) for fi, p in raw['training_poses'].items()}}
    train = {i for i, f in enumerate(protocol['input_views']) if f['split'] == 'train'}
    width, height = original['frames'][0]['width'], original['frames'][0]['height']
    problem = C.BundleProblem(model, tracks, width, height, train)
    cv2.setNumThreads(2)
    attempts, profiles = [], []
    for free, factors, maximum, target in [(True, C.PROTOCOL['focal_start_factors'], C.PROTOCOL['max_evaluations_per_start'], attempts),
                                           (False, C.PROTOCOL['fixed_focal_profile_factors'], C.PROTOCOL['max_evaluations_per_profile'], profiles)]:
        for i, factor in enumerate(factors):
            begin = time.monotonic()
            print(json.dumps({'phase': 'free_focal' if free else 'fixed_profile', 'factor': factor, 'start': i}), flush=True)
            fit = problem.fit(factor * max(width, height), free_focal=free, max_nfev=maximum)
            fit['metrics']['runtime_s'] = time.monotonic() - begin
            target.append(fit)
            stem = ('start' if free else 'profile') + f'-{i + 1}'
            dump(output / (stem + '-metrics.json'), fit['metrics'])
            dump(output / (stem + '-geometry.json'), pack_model(fit['model']))
            print(json.dumps({'phase': stem + '_complete', **fit['metrics']}), flush=True)
    selected = C.select_training_winner(attempts)
    diagnostic = C.observability(attempts, profiles, selected)
    support_counts = {t: len(set(tracks[t]) & set(model['poses'])) for t in model['points']}
    xyz_initial = np.array(list(problem.points.values()))
    shape_comparison = []
    if selected is not None:
        reference = attempts[selected]['model']['points']
        for i, fit in enumerate(attempts):
            shape_comparison.append({'start_index': i, **G.similarity_difference(reference, fit['model']['points'])})
    diagnostic['common_point_shape_comparisons'] = shape_comparison
    diagnostic['fixed_support'] = {'points': len(problem.ids), 'training_observations': len(problem.observations),
        'points_with_two_registered_training_views': sum(n == 2 for n in support_counts.values()),
        'points_with_at_least_three_registered_training_views': sum(n >= 3 for n in support_counts.values()),
        'initial_first_camera_depth_percentiles': np.percentile(xyz_initial[:, 2], [0, 25, 50, 75, 99, 100]).tolist(),
        'untrimmed_PCA_is_not_nonplanarity_proof': True}
    selection = {'selected_start_index': selected, 'selection_rule': C.PROTOCOL['selection'],
                 'observability': diagnostic, 'starts': [a['metrics'] for a in attempts],
                 'fixed_profiles': [p['metrics'] for p in profiles], 'heldout_evaluation_started': False}
    dump(output / 'training-selection.json', selection)
    # Held-out is first consulted for reporting only after this durable selection.
    if selected is not None:
        chosen = attempts[selected]
        chosen['tracks'] = tracks
        K = G.camera_matrix(width, height, chosen['metrics']['focal_factor'])
        result = export_viewer(output, original, chosen, K, diagnostic, G.digest(output / 'protocol.json'))
        geometry = pack_model(chosen['model']); geometry['tracks'] = raw['tracks']
        dump(output / 'geometry.json', geometry)
        print(json.dumps({'phase': 'complete', 'status': result['status'], 'metrics': result['metrics'],
                          'observability': diagnostic}), flush=True)
    else:
        print(json.dumps({'phase': 'complete', 'status': 'no_valid_training_candidate', 'observability': diagnostic}), flush=True)


if __name__ == '__main__':
    main()
