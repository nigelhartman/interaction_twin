#!/usr/bin/env python3
"""Read-only comparison of two validated saved perception/procedure runs.

Writes JSON to stdout only. No inference, network, credential access, resampling,
nearest-frame matching or physical-accuracy measurement is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import benchmark as B
import stage5

SUPPORTED = 'supported_conditional_motion_candidate'
REFERENCE = 'reference_candidate_unverified_geometry_and_pose'
TIME_DOMAIN = 'source_video_presentation_seconds'


def exact_time(value):
    """Exact stored numeric PTS, with no rounding tolerance or interpolation."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('Source PTS must be a finite nonnegative number')
    return Decimal(str(value))


def source_identity(document):
    source = document['source']
    if source.get('time_domain') != TIME_DOMAIN:
        raise ValueError('Comparison requires original source presentation time')
    original = source.get('original_sha256')
    digest = original or source.get('sha256')
    if not isinstance(digest, str) or not re.fullmatch('[a-f0-9]{64}', digest):
        raise ValueError('A valid source content hash is required')
    return ('original_video' if original else 'declared_source_video', digest)


def evidence_index(document):
    result = {}
    for row in document['evidence']:
        key = exact_time(row['timestamp_s'])
        if key in result:
            raise ValueError('Duplicate exact source PTS in procedure evidence')
        result[key] = row
    return result


def mask_record_index(document, tracks):
    """Use only mask records actually cited by this validated program."""
    index = {}
    for track in tracks.get('tracks', []):
        for row in track['frames']:
            key = (track['track_id'], exact_time(row['timestamp_s']))
            if key in index:
                raise ValueError('Duplicate mask track at one exact source PTS')
            index[key] = row
    bound = {}
    for evidence in document['evidence']:
        for observation in evidence['observations']:
            key = (observation['track_id'], exact_time(evidence['timestamp_s']))
            if key in bound:
                raise ValueError('Duplicate mask observation in procedure evidence')
            if key not in index:
                raise ValueError('Cited mask has no matching exact-PTS upstream record')
            row = index[key]
            if row['sample_index'] != evidence['source_index']:
                raise ValueError('Mask source index differs from cited source evidence')
            if row.get('raw_mask_sha256') and row['raw_mask_sha256'] != observation['mask_sha256']:
                raise ValueError('Mask record hash differs from cited source evidence')
            bound[key] = row
    return bound


def summarize(document, masks, times=None):
    """Internal record counts, not semantic truth or distinct physical objects."""
    selected = lambda t: times is None or exact_time(t) in times
    evidence = [e for e in document['evidence'] if selected(e['timestamp_s'])]
    observations = [(e, o) for e in evidence for o in e['observations']]
    motion = [m for m in document['motion'] if selected(m['timestamp_s'])]
    records = [masks[(o['track_id'], exact_time(e['timestamp_s']))] for e, o in observations]
    result = {
        'source_evidence_samples': len(evidence),
        'cited_mask_observations': len(observations),
        'recorded_nonempty_masks': sum(r.get('raw_area_pixels', 0) > 0 for r in records),
        'native_preserved_mask_observations': sum(r.get('native_returned_instance_preserved') is True for r in records),
        'native_classification_unknown': sum('native_returned_instance_preserved' not in r for r in records),
        'zero_placeholders_for_missing_or_withheld': sum(r.get('zero_placeholder_for_missing_or_withheld') is True for r in records),
        'model_audit_verdicts': dict(sorted(Counter(o.get('audit_status', 'not_model_audited') for _, o in observations).items())),
        'conditional_nonreference_pose_samples': sum(m['status'] == SUPPORTED for m in motion),
        'unverified_reference_pose_samples': sum(m['status'] == REFERENCE for m in motion),
        'unsupported_or_withheld_motion_samples': sum(m['status'] not in {SUPPORTED, REFERENCE} for m in motion),
        'physical_pose_accuracy': None,
        'model_audit_accuracy': None,
    }
    return result


def compare_documents(baseline, candidate, baseline_tracks, candidate_tracks):
    """Compare already validated programs; caller must validate referenced files."""
    a_id, b_id = source_identity(baseline), source_identity(candidate)
    if a_id != b_id:
        raise ValueError('Source content hash or lineage kind differs; cross-video comparison is withheld')
    a_evidence, b_evidence = evidence_index(baseline), evidence_index(candidate)
    common = set(a_evidence) & set(b_evidence)
    a_masks, b_masks = mask_record_index(baseline, baseline_tracks), mask_record_index(candidate, candidate_tracks)
    a_all, b_all = summarize(baseline, a_masks), summarize(candidate, b_masks)
    a_common, b_common = summarize(baseline, a_masks, common), summarize(candidate, b_masks, common)
    def per_track(doc, masks):
        result = []
        for track in doc['tracklets']:
            subset = dict(doc, evidence=[dict(e, observations=[o for o in e['observations'] if o['track_id'] == track['id']]) for e in doc['evidence']],
                          motion=[m for m in doc['motion'] if m['track_id'] == track['id']])
            result.append({'track_id': track['id'], 'label': track['label'], 'identity_status': track['identity_status'],
                           'all_source_pts': summarize(subset, masks), 'exact_common_source_pts': summarize(subset, masks, common)})
        return result
    def delta(a, b):
        return {k: b[k]-a[k] for k in a if isinstance(a[k], int) and not isinstance(a[k], bool)}
    return {
        'format': 'interaction-twin-perception-run-comparison', 'version': '0.1',
        'source_identity': {'kind': a_id[0], 'sha256': a_id[1], 'same_original_source': a_id[0] == 'original_video'},
        'pts_join': {'method': 'strict_Decimal_of_stored_source_numeric_PTS', 'rounding_tolerance_s': 0,
                     'nearest_frame_matching': False, 'interpolation': False, 'resampling': False,
                     'baseline_samples': len(a_evidence), 'candidate_samples': len(b_evidence),
                     'exact_common_samples': len(common), 'baseline_only_samples': len(set(a_evidence)-common),
                     'candidate_only_samples': len(set(b_evidence)-common),
                     'exact_common_source_pts_decimal': [str(t) for t in sorted(common)]},
        'baseline': {'case_id': baseline['case_id'], 'reasoning': baseline['reasoning'], 'all_source_pts': a_all,
                     'exact_common_source_pts': a_common, 'tracklets': per_track(baseline, a_masks)},
        'candidate': {'case_id': candidate['case_id'], 'reasoning': candidate['reasoning'], 'all_source_pts': b_all,
                      'exact_common_source_pts': b_common, 'tracklets': per_track(candidate, b_masks)},
        'candidate_minus_baseline_record_counts': {'all_source_pts': delta(a_all, b_all), 'exact_common_source_pts': delta(a_common, b_common)},
        'interpretation': {'internal_record_comparison_only': True, 'physical_accuracy_measured': False,
                           'model_plausibility_is_measured_accuracy': False, 'track_ids_matched_as_physical_identities': False,
                           'denser_sampling_implies_improved_quality': False, 'reference_initializations_count_as_motion_success': False},
        'limitations': ['Fresh discovery may select different observation tracklets; counts do not match physical identities.',
                       'Additional masks or conditional poses do not establish more accurate geometry, pose, contact or seating.',
                       'Model audit verdicts are fallible judgments, not human-labelled correctness.',
                       'Native mask counts include only preserved records bound to this procedure; provider results outside tracklet ranges are excluded.',
                       'Conditional pose counts exclude unverified reference initializations.',
                       'Exact common PTS are compared without accepting nearby timestamps or filling missing observations.'],
        'new_inference': False, 'network_access': False,
    }


def _guard_paths(document, root):
    # Validate visible, confined paths before the old validator hashes any bytes.
    paths = [document['source']['path']]
    paths += [r['path'] for r in document['inputs'] + document['assets']]
    for e in document['evidence']:
        paths.append(e['frame_path'])
        paths += [o['mask_path'] for o in e['observations']]
    paths += [p for p in document['links'].values() if p]
    for path in paths:
        B.path_in(root, path)


def load_validated(root, relative):
    root = Path(root).resolve()
    target = B.path_in(root, relative)
    if target.suffix != '.json':
        raise ValueError('A saved JSON interaction program is required')
    document = json.loads(target.read_text())
    stage5.validate_interaction(document, root=root, verify_files=False)
    _guard_paths(document, root)
    stage5.validate_interaction(document, root=root, verify_files=True)
    manifests = [i for i in document['inputs'] if i['kind'] == 'source_manifest']
    if len(manifests) != 1:
        raise ValueError('Exactly one verified source manifest is required')
    manifest_path = B.path_in(root, manifests[0]['path'])
    manifest = json.loads(manifest_path.read_text())
    if document['source'].get('original_sha256'):
        source = manifest['source']
        if source.get('original_sha256') != document['source']['original_sha256'] or not source.get('original_file'):
            raise ValueError('Original source lineage differs from verified manifest')
        original_relative = str((manifest_path.parent/source['original_file']).relative_to(root))
        original = B.path_in(root, original_relative)
        if not original.is_file() or B.digest(original) != document['source']['original_sha256']:
            raise ValueError('Original source bytes are unavailable or changed')
    inputs = [i for i in document['inputs'] if i['kind'] == 'actual_sam_masks']
    if len(inputs) != 1:
        raise ValueError('Exactly one verified SAM mask-tracks input is required')
    masks = B.load(root, inputs[0]['path'])
    return document, masks, {'path': str(target.relative_to(root)), 'sha256': B.digest(target),
                             'source_and_referenced_files_validated': True, 'original_source_bytes_verified': bool(document['source'].get('original_sha256'))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, help='Project-relative validated interaction JSON')
    parser.add_argument('--candidate', required=True, help='Project-relative validated interaction JSON')
    args = parser.parse_args()
    a, a_masks, a_receipt = load_validated(ROOT, args.baseline)
    b, b_masks, b_receipt = load_validated(ROOT, args.candidate)
    result = compare_documents(a, b, a_masks, b_masks)
    result['validated_programs'] = {'baseline': a_receipt, 'candidate': b_receipt}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
