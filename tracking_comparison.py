"""Compare sparse/dense tracking on a common source clock, without 3D truth claims."""
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def kind(frame):
    status = frame.get('status', '')
    if status.startswith('reference_'):
        return 'reference'
    if status.startswith('supported_'):
        return 'supported'
    return 'withheld'


def evidence_key(pack, field):
    return pack.get('provenance', {}).get(field)


def validate_frames(pack):
    rows = pack['frames']
    if len(rows) < 2:
        raise ValueError('At least two timestamped observations required')
    times = np.array([r['timestamp_s'] for r in rows], float)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Tracking PTS must be finite and strictly increasing')
    return rows, times


def describe(pack, start, end):
    rows, times = validate_frames(pack)
    delta = float(np.median(np.diff(times)))
    in_range = [r for r in rows if start-1e-6 <= r['timestamp_s'] <= end+1e-6]
    counts = {k: sum(kind(r) == k for r in in_range) for k in ['supported', 'reference', 'withheld']}
    # A display-time diagnostic only. It assumes holding each sampled label until
    # the next observation, never farther than the normal sampling gap.
    held_seconds = 0.
    for a, b in zip(rows, rows[1:]):
        lo, hi = max(start, a['timestamp_s']), min(end, b['timestamp_s'])
        if hi > lo and b['timestamp_s']-a['timestamp_s'] <= delta*1.15 and kind(a) in ['supported', 'reference']:
            held_seconds += hi-lo
    return {'counts': counts, 'sample_count': len(in_range), 'median_sample_interval_s': delta,
            'supported_or_reference_sample_fraction': (counts['supported']+counts['reference']) / max(1, len(in_range)),
            'held_label_seconds_diagnostic_only': held_seconds,
            'held_label_seconds_is_verified_continuous_tracking': False,
            'samples': [{'sample_index': r['sample_index'], 'timestamp_s': r['timestamp_s'], 'kind': kind(r)} for r in in_range]}


def compare(baseline, candidate):
    for field in ['source_video_sha256', 'mesh_sha256']:
        a, b = evidence_key(baseline, field), evidence_key(candidate, field)
        if not a or a != b:
            raise ValueError('Comparison requires matching ' + field)
    if baseline.get('object_id') != candidate.get('object_id'):
        raise ValueError('Comparison requires the same candidate object ID')
    old, ot = validate_frames(baseline)
    new, nt = validate_frames(candidate)
    start, end = float(max(ot[0], nt[0])), float(min(ot[-1], nt[-1]))
    if end <= start:
        raise ValueError('Tracking intervals do not overlap')
    tolerance = min(.15, float(np.median(np.diff(nt)))*.55)
    pairs = []
    for row in old:
        if not start-1e-6 <= row['timestamp_s'] <= end+1e-6:
            continue
        index = int(np.argmin(abs(nt-row['timestamp_s'])))
        if abs(nt[index]-row['timestamp_s']) > tolerance:
            pairs.append({'timestamp_s': row['timestamp_s'], 'baseline': kind(row),
                          'candidate': 'unmatched', 'candidate_timestamp_s': None})
        else:
            pairs.append({'timestamp_s': row['timestamp_s'], 'baseline': kind(row),
                          'candidate': kind(new[index]), 'candidate_timestamp_s': float(nt[index]),
                          'pts_difference_s': float(abs(nt[index]-row['timestamp_s']))})
    valid_pairs = [r for r in pairs if r['candidate'] != 'unmatched' and r['baseline'] != 'reference' and r['candidate'] != 'reference']
    valid = {'supported'}
    transitions = {name: sum(test(r) for r in valid_pairs) for name, test in {
        'improved_at_matched_time': lambda r: r['baseline'] not in valid and r['candidate'] in valid,
        'regressed_at_matched_time': lambda r: r['baseline'] in valid and r['candidate'] not in valid,
        'both_supported': lambda r: r['baseline'] in valid and r['candidate'] in valid,
        'both_withheld': lambda r: r['baseline'] not in valid and r['candidate'] not in valid,
    }.items()}
    return {'schema_version': '0.1', 'status': 'internal_evidence_comparison_not_accuracy_evaluation',
            'source_video_sha256': evidence_key(baseline, 'source_video_sha256'),
            'mesh_sha256': evidence_key(baseline, 'mesh_sha256'), 'object_id': baseline['object_id'],
            'common_interval_s': [start, end], 'baseline': describe(baseline, start, end),
            'candidate': describe(candidate, start, end), 'matched_time_tolerance_s': tolerance,
            'matched_times': pairs, 'nonreference_matched_time_transitions': transitions,
            'limits': ['More samples do not establish more accurate tracking.',
                       'The two runs may use different gate/audit coverage; compare their reasons as well as counts.',
                       'Held-label seconds describe a sampled display assumption, not verified motion between frames.',
                       'Reference-conditioned reprojection does not establish physical 6D accuracy.']}


def svg(record):
    start, end = record['common_interval_s']
    x = lambda t: 185+(t-start)/(end-start)*930
    colors = {'supported': '#4dd8b2', 'reference': '#edc76d', 'withheld': '#ee7f91'}
    lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="280" viewBox="0 0 1200 280">',
             '<rect width="1200" height="280" fill="#111c29"/>',
             '<g font-family="system-ui,sans-serif" fill="#edf4fb"><text x="24" y="35" font-size="20">Tracking evidence at original source timestamps</text>',
             '<text x="24" y="60" fill="#b3c3d2" font-size="13">Each tick is an observation; continuous motion is not inferred.</text>']
    for name, label, y in [('baseline', 'Baseline ~1 fps', 110), ('candidate', 'Dense crop run', 180)]:
        lines.append(f'<text x="24" y="{y+5}" font-size="15">{label}</text><path d="M185 {y} H1115" stroke="#40536b"/>')
        for sample in record[name]['samples']:
            px = x(sample['timestamp_s'])
            lines.append(f'<path d="M{px:.2f} {y-13} V{y+13}" stroke="{colors[sample["kind"]]}" stroke-width="3"/>')
    lines += [f'<text x="185" y="225" font-size="13">{start:.3f} s</text>',
              f'<text x="1115" y="225" text-anchor="end" font-size="13">{end:.3f} s</text>']
    for k, label, px in [('supported', 'supported candidate', 185), ('reference', 'reference', 500), ('withheld', 'withheld', 720)]:
        lines.append(f'<circle cx="{px}" cy="255" r="5" fill="{colors[k]}"/><text x="{px+12}" y="260" font-size="13">{label}</text>')
    return '\n'.join(lines+['</g></svg>'])
