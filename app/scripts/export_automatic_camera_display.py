#!/usr/bin/env python3
"""Correct derived display metrics without refitting or replacing frozen evidence."""
from pathlib import Path
import argparse
import copy
import json
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import automatic_camera as A


def corrected_view(view):
    """Recompute only projection summaries/support from saved XYZ, K and R/t."""
    view = copy.deepcopy(view)
    cameras = {c['frame_id']: c for c in view['cameras']}
    points = {p['id']: p for p in view['points']}
    support = {tid: [] for tid in points}
    passing = {r['frame_index'] for r in view['heldout_evaluation'] if r['passes']}
    finite_scored, all_scored = [], []
    max_disagreement = 0.
    for fi, frame in enumerate(view['frames']):
        camera = cameras.get(frame['id'])
        if camera is None:
            continue
        R, t, K = [np.array(camera[k], dtype=float) for k in ['R', 't', 'K']]
        for observation in frame['observations']:
            tid = observation['track_id']
            if tid not in points:
                continue
            xyz = np.array(points[tid]['xyz'])
            uv = cv2.projectPoints(xyz[None], cv2.Rodrigues(R)[0], t, K, None)[0].reshape(2)
            depth = float((R @ xyz + t)[2])
            if observation.get('predicted_xy') is not None:
                max_disagreement = max(max_disagreement, float(np.max(np.abs(uv - observation['predicted_xy']))))
            error = float(np.linalg.norm(uv - observation['xy']))
            observation['predicted_xy'] = uv.tolist()
            observation['error_px'] = error if np.isfinite(error) else None
            if frame['split'] == 'heldout' and observation['role'] == 'heldout':
                valid = depth > 0 and np.isfinite(error)
                all_scored.append(error if valid else float('inf'))
                if valid:
                    finite_scored.append(error)
                if fi in passing and valid and error <= 3.:
                    support[tid].append({'frame_id': frame['id'], 'timestamp_s': frame['timestamp_s'], 'error_px': error})
    for tid, point in points.items():
        point['heldout_scored_projection_support'] = support[tid]
        point['heldout_scored_projection_supported_in_at_least_two_views'] = len(support[tid]) >= 2
    view['metrics'].update(heldout_finite_error_median_px=float(np.median(finite_scored)) if finite_scored else None,
        heldout_finite_error_p90_px=float(np.percentile(finite_scored, 90)) if finite_scored else None,
        heldout_scored_observations=len(all_scored),
        heldout_invalid_or_behind_camera_observations=len(all_scored)-len(finite_scored),
        points_independently_scored_in_at_least_two_heldout_views=sum(len(x) >= 2 for x in support.values()))
    return view, {'projection_max_absolute_disagreement_px': max_disagreement,
        'reserved_scored_observations': len(all_scored), 'finite_reserved_scored_observations': len(finite_scored),
        'points_with_two_reserved_scored_supports': sum(len(x) >= 2 for x in support.values()),
        'geometry_or_pose_or_K_changes': False, 'refit_or_new_inference': False}


def export(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_relative_to(ROOT / 'data/automatic-camera-20261101') or not output.is_relative_to(ROOT / 'data/automatic-camera-20261101') or output.exists():
        raise ValueError('Use a fresh correction under the fixed camera-comparison data root')
    report = json.loads((source / 'report.json').read_text())
    parent = {'path': str((source / 'report.json').relative_to(ROOT)), 'sha256': A.digest(source / 'report.json')}
    result = copy.deepcopy(report)
    output.mkdir(parents=True)
    corrections = []
    for row in result['hypotheses']:
        if not row.get('viewer'):
            continue
        old = row['viewer']
        path = A.data_file(ROOT, old['path'], old['sha256'])
        view, metrics = corrected_view(json.loads(path.read_text()))
        view['display_correction'] = {'parent': old, 'derived_metrics_only': True,
            'reason': 'Recompute inherited finite reserved error metrics and scored support from unchanged camera, poses and points.'}
        target = output / (row['id'] + '-viewer.json')
        A.dump(target, view)
        row['viewer'] = {'path': str(target.relative_to(ROOT)), 'sha256': A.digest(target)}
        corrections.append({'id': row['id'], 'parent': old, 'viewer': row['viewer'], **metrics})
    result['display_correction'] = {'parent': parent, 'raw_fit_results_unchanged': True,
        'corrections': corrections, 'implementation_sha256': A.digest(Path(__file__))}
    A.dump(output / 'report.json', result)
    A.write_ui_report(ROOT, output, result)
    (output / 'implementation.py').write_bytes(Path(__file__).read_bytes())
    A.dump(output / 'display-correction.json', result['display_correction'])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = export(ROOT / args.source, ROOT / args.output)
    print(json.dumps(result['display_correction']))
