"""Conservatively omit model point claims outside their subject's tracklet.

The original response is never changed. This deterministic derived artifact may
only remove unsupported assertions/conditions and add omission disclosures. It
cannot invent a state, move evidence to another sample, add a relationship or
change geometry. The unchanged Stage 5 validator must accept the final result.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

import stage5

POLICY = 'omit_only_known_subject_outside_its_declared_tracklet_at_an_existing_source_sample'
STEP_DISCLOSURE = ('Unsupported point conditions outside their subject tracklets were omitted. '
                   'No condition was retargeted to another source sample; inspect the normalization trace.')
ASSERTION_DISCLOSURE = ('Unsupported point assertions outside their subject tracklets were omitted. '
                        'No replacement state or source observation was added; inspect the normalization trace.')


def _sha(document):
    raw = json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _preflight(proposal, rows, track_specs):
    """Reject other assertion/condition defects before any record can be omitted."""
    import jsonschema
    stage5._finite(proposal)
    jsonschema.validate(proposal, stage5.PROPOSAL_SCHEMA)
    if not rows or any(type(row.get('index')) is not int or type(row.get('timestamp_s')) not in (int, float)
                       or not math.isfinite(row['timestamp_s']) for row in rows):
        raise ValueError('Invalid source sample index or timestamp')
    frames = {row['index']: row for row in rows}
    if (len(frames) != len(rows) or any(index < 1 for index in frames)
            or any(b['timestamp_s'] <= a['timestamp_s'] for a, b in zip(rows, rows[1:]))):
        raise ValueError('Source index must be unique and chronological')
    tracks = {track['id']: track for track in track_specs}
    if len(tracks) != len(track_specs):
        raise ValueError('Duplicate track identity')
    for track in track_specs:
        span = track.get('sample_range')
        if (not isinstance(span, list) or len(span) != 2 or any(type(x) is not int for x in span)
                or not 1 <= span[0] <= span[1] <= max(frames)):
            raise ValueError('Invalid tracklet source sampling range')
    variables = stage5._unique(proposal['variables'])
    stage5._unique(proposal['assertions'])
    stage5._unique(proposal['steps'])
    stage5._unique(proposal['relationships'])
    for variable in variables.values():
        if (variable['subject_track_id'] not in tracks or len(variable['values']) != len(set(variable['values']))
                or 'unknown' not in variable['values']):
            raise ValueError('Invalid state variable, values or unknown default')

    def point(item):
        variable = variables.get(item['variable_id'])
        if variable is None or item['value'] not in variable['values']:
            raise ValueError('Unknown variable or illegal state value')
        if item['evidence_sample'] not in frames:
            raise ValueError('Unknown evidence sample')
        return variable, tracks[variable['subject_track_id']]

    assignments = {}
    for assertion in proposal['assertions']:
        point(assertion)
        key = (assertion['variable_id'], assertion['evidence_sample'])
        if key in assignments:
            raise ValueError('Duplicate or contradictory state assertion')
        if (assertion['status'] == 'unknown') != (assertion['value'] == 'unknown'):
            raise ValueError('Unknown status/value must agree')
        assignments[key] = assertion['value']
    for step in proposal['steps']:
        samples = step['evidence_samples']
        participants = step['participant_track_ids']
        if not samples or len(samples) != len(set(samples)) or any(sample not in frames for sample in samples):
            raise ValueError('Unknown, empty or duplicate step evidence samples')
        if not participants or len(participants) != len(set(participants)) or any(tid not in tracks for tid in participants):
            raise ValueError('Step needs known unique participants')
        for field in ['preconditions', 'postconditions']:
            seen = set()
            for condition in step[field]:
                variable, _track = point(condition)
                if variable['subject_track_id'] not in participants:
                    raise ValueError('Condition subject is not a step participant')
                sample, vid = condition['evidence_sample'], condition['variable_id']
                if sample != (min(samples) if field == 'preconditions' else max(samples)):
                    raise ValueError('Conditions must cite the actual first/last step evidence')
                if vid in seen:
                    raise ValueError('Duplicate or contradictory condition')
                seen.add(vid)
                known = assignments.get((vid, sample), 'unknown')
                if known != 'unknown' and known != condition['value']:
                    raise ValueError('Condition contradicts model state evidence')
    return variables, tracks


def normalize_proposal(proposal, rows, track_specs):
    """Return a validated derived copy and a complete omission-only trace.

    Unknown IDs, invented samples, contradictory states/conditions and every
    remaining Stage 5 contract failure are rejected, never repaired.
    """
    variables, tracks = _preflight(proposal, rows, track_specs)
    normalized = copy.deepcopy(proposal)
    removed = []

    def outside(item, *, kind, item_index, step_id=None, field=None):
        variable = variables[item['variable_id']]
        subject = variable['subject_track_id']
        span = tracks[subject]['sample_range']
        if span[0] <= item['evidence_sample'] <= span[1]:
            return False
        record = {'kind': kind, 'item_index': item_index, 'original_item': copy.deepcopy(item),
                  'variable_id': item['variable_id'], 'subject_track_id': subject,
                  'evidence_sample': item['evidence_sample'], 'subject_tracklet_sample_range': list(span),
                  'reason': 'variable_subject_has_no_tracklet_observation_at_this_existing_source_sample'}
        if step_id is not None:
            record.update(step_id=step_id, field=field)
        removed.append(record)
        return True

    normalized['assertions'] = [item for index, item in enumerate(normalized['assertions'])
                                if not outside(item, kind='point_assertion', item_index=index)]
    assertions_removed = len(removed)
    for step in normalized['steps']:
        before = len(removed)
        for field in ['preconditions', 'postconditions']:
            step[field] = [item for index, item in enumerate(step[field])
                           if not outside(item, kind='step_condition', item_index=index, step_id=step['id'], field=field)]
        if len(removed) > before and STEP_DISCLOSURE not in step['limitations']:
            step['limitations'].append(STEP_DISCLOSURE)
    if assertions_removed and ASSERTION_DISCLOSURE not in normalized['limitations']:
        normalized['limitations'].append(ASSERTION_DISCLOSURE)

    # The public, unchanged validator is the mandatory final gate. No widened
    # track ranges or substituted source samples are supplied to this check.
    stage5.validate_proposal(normalized, rows, track_specs)
    trace = {'format': 'interaction-twin-semantic-scope-normalization', 'version': '0.1',
             'policy': POLICY, 'status': 'omitted_unsupported_points' if removed else 'unchanged_valid_proposal',
             'original_proposal_sha256': _sha(proposal), 'normalized_proposal_sha256': _sha(normalized),
             'removed': removed, 'removed_count': len(removed), 'removed_assertions': assertions_removed,
             'removed_conditions': len(removed) - assertions_removed,
             'original_input_mutated': False, 'manual_semantic_story_used': False,
             'new_semantic_claims_added': False, 'source_evidence_added': False, 'source_samples_retargeted': False,
             'state_values_changed': False, 'relationships_changed': False, 'geometry_or_poses_changed': False,
             'disclosures_added': bool(removed), 'physical_completion_verified': False,
             'validator': 'unchanged_stage5.validate_proposal', 'validation_passed': True,
             'limitations': ['A deterministic omission does not confirm the remaining model interpretations.',
                             'The caller must preserve the original raw model response separately from this derived proposal.']}
    return normalized, trace
