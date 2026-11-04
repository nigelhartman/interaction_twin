#!/usr/bin/env python3
"""Independently audit saved camera fits; never refit or modify their outputs."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import automatic_camera_independent_audit as A


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def raw_observations(viewer):
    return [{'id': frame['id'], 'timestamp_s': frame['timestamp_s'], 'split': frame['split'],
             'width': frame['width'], 'height': frame['height'], 'image_url': frame['image_url'],
             'mask_url': frame['mask_url'], 'observations': [
                 {key: observation.get(key) for key in ['track_id', 'xy', 'role']}
                 for observation in frame['observations']]}
            for frame in viewer['frames']]


def robust_cost(residual, scale):
    # Independent expression of SciPy's soft_l1 cost (without global 1/2).
    residual = np.asarray(residual)
    return float(np.sum(2 * scale**2 * ((1 + (residual / scale)**2)**.5 - 1)) / 2)


def audit(root, directory, expected):
    protocol_file = directory / 'protocol.json'
    if sha(protocol_file) != expected:
        raise ValueError('Frozen protocol SHA mismatch')
    protocol = read(protocol_file)
    source = root / protocol['source_directory']
    original = read(source / 'viewer.json')
    raw = read(source / 'geometry.json')
    train = {i for i, frame in enumerate(protocol['source_frames']) if frame['split'] == 'train'}
    tracks = {tid: {int(fi): np.asarray(xy) for fi, xy in track.items()} for tid, track in raw['tracks'].items()}
    inputs = []
    for name, expected_hash in protocol['input_files'].items():
        path = root / name
        if not name.startswith(('data/', 'models/')) or path.resolve() != path:
            raise ValueError('Unexpected or symlinked frozen input')
        inputs.append({'path': name, 'matches': sha(path) == expected_hash})
    code = {name: sha(directory / name) == expected_hash for name, expected_hash in protocol['code_files'].items()}
    if not all(row['matches'] for row in inputs) or not all(code.values()):
        raise ValueError('Frozen source inputs or implementation copies changed')
    report = read(directory / 'report.json')
    decision = read(directory / 'training-decision.json')
    if report['input_protocol_sha256'] != expected or decision['protocol_sha256'] != expected:
        raise ValueError('Report/decision protocol mismatch')
    if report['training_decision'] != decision:
        raise ValueError('Report changed the durable pre-reserved decision')
    summaries = []
    for candidate in report['hypotheses']:
        if 'geometry' not in candidate:
            continue
        geometry_path = root / candidate['geometry']['path']
        viewer_path = root / candidate['viewer']['path']
        if sha(geometry_path) != candidate['geometry']['sha256'] or sha(viewer_path) != candidate['viewer']['sha256']:
            raise ValueError('Candidate output hash mismatch')
        geometry, viewer = read(geometry_path), read(viewer_path)
        points = geometry['points']; poses = {int(fi): pose for fi, pose in geometry['training_poses'].items()}
        support = A.raw_support(points, poses, tracks, train)
        if set(points) != set(raw['points']) or set(poses) != set(map(int, raw['training_poses'])):
            raise ValueError('Candidate changed frozen raw support')
        if raw_observations(viewer) != raw_observations(original):
            raise ValueError('Candidate changed source observations or reserved roles')
        prior_candidate = next(row for row in protocol['candidates'] if row['id'] == candidate['id'])
        np.testing.assert_array_equal(candidate['K'], prior_candidate['K'])
        K = np.asarray(candidate['K'])
        residuals, depths = [], []
        for tid in sorted(points):
            for fi, observed in sorted(tracks[tid].items()):
                if fi not in poses:
                    continue
                pose = poses[fi]
                projected, depth = A.opencv_projection([points[tid]], pose['R'], pose['t'], K)
                residuals.extend(projected[0] - observed); depths.append(depth[0])
        residuals = np.asarray(residuals)
        errors = np.linalg.norm(residuals.reshape(-1, 2), axis=1)
        metrics = {'robust_train_cost': robust_cost(residuals, protocol['protocol']['robust_scale_px']),
                   'mean_robust_train_cost': robust_cost(residuals, protocol['protocol']['robust_scale_px']) / len(residuals),
                   'train_median_px': float(np.median(errors)), 'train_p90_px': float(np.percentile(errors, 90)),
                   'positive_training_depth_fraction': float(np.mean(np.asarray(depths) > 0)),
                   'training_observations': len(depths)}
        deltas = {key: abs(value - candidate['training'][key]) for key, value in metrics.items()}
        if any(delta > 1e-7 for delta in deltas.values()):
            raise ValueError('Saved TRAIN metrics do not match original pixels and OpenCV projections')
        first, second = int(geometry['seed']['a']), int(geometry['seed']['b'])
        np.testing.assert_allclose(poses[first]['R'], np.eye(3), atol=1e-12, rtol=0)
        np.testing.assert_allclose(poses[first]['t'], np.zeros(3), atol=1e-12, rtol=0)
        if abs(np.linalg.norm(poses[second]['t']) - 1.) > 1e-12:
            raise ValueError('Unit baseline gauge changed')
        projections = A.viewer_projection_audit(viewer)
        if not projections['passed']:
            raise ValueError('Saved viewer projection audit failed')
        reserved_errors, reserved_rows = [], []
        frame_by_id = {frame['id']: frame for frame in viewer['frames']}
        for fi, original_frame in enumerate(original['frames']):
            if original_frame['split'] != 'heldout':
                continue
            frame = frame_by_id[original_frame['id']]
            camera = next((camera for camera in viewer['cameras'] if camera['frame_id'] == frame['id']), None)
            if camera is None:
                continue
            scores = [o for o in frame['observations'] if o['role'] == 'heldout' and o['track_id'] in points]
            errors_here = []
            for observation in scores:
                pixel, depth = A.opencv_projection([points[observation['track_id']]], camera['R'], camera['t'], camera['K'])
                error = float(np.linalg.norm(pixel[0] - observation['xy'])) if depth[0] > 0 else float('inf')
                errors_here.append(error)
            saved_row = next(row for row in candidate['reserved']['rows'] if row['frame_index'] == fi)
            median = float(np.median(errors_here)) if errors_here and np.isfinite(errors_here).all() else None
            p90 = float(np.percentile(errors_here, 90)) if errors_here and np.isfinite(errors_here).all() else None
            for key, value in [('score_median_px', median), ('score_p90_px', p90)]:
                saved = saved_row[key]
                if (saved is None) != (value is None) or (value is not None and abs(saved - value) > 1e-7):
                    raise ValueError('Reserved scoring metrics changed')
            reserved_errors.extend(errors_here)
            reserved_rows.append({'frame_index': fi, 'scored_observations': len(scores), 'median_px': median, 'p90_px': p90})
        finite = [error for error in reserved_errors if np.isfinite(error)]
        stale = {}
        for key, value in [('heldout_finite_error_median_px', float(np.median(finite)) if finite else None),
                           ('heldout_finite_error_p90_px', float(np.percentile(finite, 90)) if finite else None)]:
            saved = viewer['metrics'].get(key)
            if (saved is None) != (value is None) or (value is not None and abs(saved - value) > 1e-7):
                stale[key] = {'raw_viewer_value': saved, 'recomputed_value': value}
        summaries.append({'id': candidate['id'], 'status': candidate['status'], 'support': support,
                          'independent_train_metrics': metrics, 'maximum_train_metric_difference': max(deltas.values()),
                          'viewer_projection_audit': projections, 'reserved_rows': reserved_rows,
                          'stale_raw_viewer_fields': stale,
                          'geometry_sha256': candidate['geometry']['sha256'], 'raw_viewer_sha256': candidate['viewer']['sha256']})
    expected_preference = None
    fixed = [row for row in report['hypotheses'] if row['id'] in protocol['protocol']['candidate_order']]
    valid = [row for row in fixed if row.get('training', {}).get('optimizer_success')
             and row['training'].get('finite')
             and row['training'].get('positive_training_depth_fraction', 0) >= protocol['protocol']['minimum_positive_training_depth_fraction']]
    if len(fixed) == len(valid) == 2:
        ordered = sorted(valid, key=lambda row: (row['training']['robust_train_cost'], row['id']))
        best, other = ordered
        required = max(protocol['protocol']['minimum_mean_train_cost_separation'],
                       best['training']['mean_robust_train_cost'] * protocol['protocol']['minimum_relative_train_cost_separation'])
        if other['training']['mean_robust_train_cost'] - best['training']['mean_robust_train_cost'] >= required:
            expected_preference = best['id']
    if decision['training_preferred_hypothesis_id'] != expected_preference or decision['selected_production_camera_id'] is not None:
        raise ValueError('Durable TRAIN decision does not follow frozen rules')
    expected_K = A.normalized_camera_to_pixels(protocol['cached_prior_binding']['normalized_K'], *protocol['image_size_wh'])
    np.testing.assert_array_equal(protocol['cached_prior_binding']['native_pixel_K'], expected_K)
    forbidden_true = ['physical_scale_known', 'physical_shape_verified', 'camera_calibrated', 'production_camera_changed']
    if any(report[field] is not False for field in forbidden_true) or report['new_api_calls'] != 0 or report['new_learned_inference_calls'] != 0:
        raise ValueError('Invalid physical/inference claim')
    return {'format': 'independent-automatic-camera-result-audit', 'version': 1,
            'status': 'passed_internal_consistency_with_display_corrections_needed' if any(row['stale_raw_viewer_fields'] for row in summaries) else 'passed_internal_consistency',
            'protocol_sha256': expected, 'raw_report_sha256': sha(directory / 'report.json'),
            'training_decision_sha256': sha(directory / 'training-decision.json'),
            'source_files_checked': len(inputs), 'source_files_unchanged': all(row['matches'] for row in inputs),
            'frozen_code_checked': code, 'candidates': summaries, 'training_decision_reproduced': True,
            'training_preferred_hypothesis_id': expected_preference, 'new_api_calls': 0,
            'new_model_inference_calls': 0, 'new_geometry_or_pose_fits': 0,
            'physical_accuracy_established': False,
            'limitations': ['Projection agreement is internal consistency, not geometric ground truth.',
                            'Reserved scoring is conditional on separately estimated anchor poses; unregistered views remain failures.',
                            'Independent world and metric gauges, hidden shape and physical dynamics are not resolved.']}


def audit_corrected(root, raw_directory, directory):
    """Verify the presentation correction leaves every fitted value unchanged."""
    manifest = read(directory / 'display-correction.json')
    parent = root / manifest['parent']['path']
    if parent != raw_directory / 'report.json' or sha(parent) != manifest['parent']['sha256']:
        raise ValueError('Corrected report parent hash mismatch')
    if sha(directory / 'implementation.py') != manifest['implementation_sha256']:
        raise ValueError('Display correction implementation hash mismatch')
    old_report, corrected_report = read(parent), read(directory / 'report.json')
    if old_report['training_decision'] != corrected_report['training_decision']:
        raise ValueError('Display correction changed camera selection')
    rows = []
    for correction in manifest['corrections']:
        before, after = root / correction['parent']['path'], root / correction['viewer']['path']
        if sha(before) != correction['parent']['sha256'] or sha(after) != correction['viewer']['sha256']:
            raise ValueError('Corrected viewer hash mismatch')
        raw_view, corrected = read(before), read(after)
        if (raw_view['cameras'] != corrected['cameras'] or raw_observations(raw_view) != raw_observations(corrected)
                or raw_view['heldout_evaluation'] != corrected['heldout_evaluation']
                or raw_view['bundle_adjustment'] != corrected['bundle_adjustment']):
            raise ValueError('Display correction changed poses, K, inputs or fit results')
        old_points = {point['id']: point for point in raw_view['points']}
        new_points = {point['id']: point for point in corrected['points']}
        if set(old_points) != set(new_points):
            raise ValueError('Display correction added or removed geometry')
        for tid in old_points:
            for key in ['xyz', 'rgb', 'validated', 'validation_scope', 'max_training_triangulation_angle_deg']:
                if old_points[tid][key] != new_points[tid][key]:
                    raise ValueError('Display correction changed observed geometry/point provenance')
        projection = A.viewer_projection_audit(corrected)
        if not projection['passed']:
            raise ValueError('Corrected projection audit failed')
        passing = {row['frame_index'] for row in corrected['heldout_evaluation'] if row['passes']}
        cameras = {camera['frame_id']: camera for camera in corrected['cameras']}
        finite_errors, errors = [], []
        expected_support = {tid: [] for tid in new_points}
        for fi, frame in enumerate(corrected['frames']):
            if frame['split'] != 'heldout' or frame['id'] not in cameras:
                continue
            camera = cameras[frame['id']]
            for observation in frame['observations']:
                tid = observation['track_id']
                if tid not in new_points or observation['role'] != 'heldout':
                    continue
                pixel, depth = A.opencv_projection([new_points[tid]['xyz']], camera['R'], camera['t'], camera['K'])
                error = float(np.linalg.norm(pixel[0] - observation['xy'])) if depth[0] > 0 else float('inf')
                errors.append(error)
                if np.isfinite(error):
                    finite_errors.append(error)
                if fi in passing and error <= 3.:
                    expected_support[tid].append({'frame_id': frame['id'], 'timestamp_s': frame['timestamp_s'], 'error_px': error})
        for tid, expected in expected_support.items():
            saved = new_points[tid]['heldout_scored_projection_support']
            if len(saved) != len(expected):
                raise ValueError('Corrected point support count mismatch')
            for x, y in zip(saved, expected):
                if x['frame_id'] != y['frame_id'] or x['timestamp_s'] != y['timestamp_s'] or abs(x['error_px'] - y['error_px']) > 1e-7:
                    raise ValueError('Corrected point support changed source or scoring error')
            if new_points[tid]['heldout_scored_projection_supported_in_at_least_two_views'] != (len(expected) >= 2):
                raise ValueError('Corrected point support flag mismatch')
        checked_metrics = {
            'heldout_finite_error_median_px': float(np.median(finite_errors)) if finite_errors else None,
            'heldout_finite_error_p90_px': float(np.percentile(finite_errors, 90)) if finite_errors else None,
            'heldout_scored_observations': len(errors),
            'heldout_invalid_or_behind_camera_observations': len(errors) - len(finite_errors),
            'points_independently_scored_in_at_least_two_heldout_views': sum(len(support) >= 2 for support in expected_support.values())}
        for key, value in checked_metrics.items():
            if (value is None) != (corrected['metrics'][key] is None) or (value is not None and abs(value - corrected['metrics'][key]) > 1e-7):
                raise ValueError('Corrected derived metric mismatch')
        rows.append({'id': correction['id'], 'corrected_viewer_sha256': sha(after),
                     'projection_audit': projection, 'recomputed_metrics': checked_metrics,
                     'observed_geometry_poses_K_fit_and_source_unchanged': True})
    ui_report = read(directory / 'ui-report.json')
    if ui_report['provenance']['report']['sha256'] != sha(directory / 'report.json'):
        raise ValueError('UI report parent hash mismatch')
    for row in ui_report['hypotheses']:
        if row['viewer'] is None:
            continue
        if sha(root / row['viewer']['path']) != row['viewer']['sha256']:
            raise ValueError('UI report viewer hash mismatch')
    return {'status': 'corrected_display_consistent', 'candidates': rows,
            'raw_report_sha256': sha(parent), 'display_report_sha256': sha(directory / 'report.json'),
            'ui_report_sha256': sha(directory / 'ui-report.json'), 'new_fits_or_inference': 0,
            'camera_selection_unchanged': True, 'physical_accuracy_established': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--corrected-display')
    args = parser.parse_args()
    comparison = (ROOT / args.comparison).resolve()
    output = (ROOT / args.output).resolve()
    if not comparison.is_relative_to(ROOT / 'data/automatic-camera-20261101') or not output.is_relative_to(ROOT / 'output/automatic-camera-20261101') or output.exists():
        raise ValueError('Read one saved comparison and use a fresh independent output path')
    result = audit(ROOT, comparison, args.protocol_sha256)
    if args.corrected_display:
        corrected = (ROOT / args.corrected_display).resolve()
        if not corrected.is_relative_to(ROOT / 'data/automatic-camera-20261101'):
            raise ValueError('Corrected display must remain in the comparison data root')
        result['corrected_display_audit'] = audit_corrected(ROOT, comparison, corrected)
        result['status'] = 'passed_internal_consistency_and_corrected_display'
    result['independent_implementation_sha256'] = {
        'automatic_camera_independent_audit.py': sha(ROOT / 'automatic_camera_independent_audit.py'),
        'scripts/audit_automatic_camera_result.py': sha(Path(__file__).resolve())}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'source_files_checked': result['source_files_checked'],
                      'candidates': len(result['candidates']), 'audit_path': str(output.relative_to(ROOT)), 'sha256': sha(output)}))


if __name__ == '__main__':
    main()
