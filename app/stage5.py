"""Evidence-linked procedure contracts and deterministic, non-physical replay.

The model proposes semantics. This module validates their references and executes
only source-time lookup; it never supplies geometry, motion or hidden state.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ALLOWED_POSES = {'supported_conditional_motion_candidate', 'reference_candidate_unverified_geometry_and_pose'}
RELATIONS = ['component_of', 'attached_to', 'removable_from', 'possibly_same_entity']
CERTAINTY = ['visual_model_proposal', 'inferred', 'unknown']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def obj(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}


def arr(item):
    return {'type': 'array', 'items': item}


S = {'type': 'string', 'minLength': 1}
I = {'type': 'integer', 'minimum': 1}
N = {'type': 'number', 'minimum': 0}
STATUS = {'type': 'string', 'enum': CERTAINTY}
COND = obj({'variable_id': S, 'value': S, 'evidence_sample': I})
PROPOSAL_SCHEMA = obj({
    'summary': S,
    'relationships': arr(obj({'id': S, 'subject_track_id': S, 'predicate': {'type': 'string', 'enum': RELATIONS},
                             'object_track_id': S, 'status': STATUS, 'evidence_samples': arr(I), 'rationale': S,
                             'limitations': arr(S)})),
    'variables': arr(obj({'id': S, 'subject_track_id': S, 'name': S, 'values': arr(S)})),
    'assertions': arr(obj({'id': S, 'variable_id': S, 'value': S, 'evidence_sample': I,
                          'status': STATUS, 'rationale': S})),
    'steps': arr(obj({'id': S, 'title': S, 'description': S, 'participant_track_ids': arr(S),
                     'evidence_samples': arr(I), 'preconditions': arr(COND), 'postconditions': arr(COND),
                     'limitations': arr(S)})),
    'limitations': arr(S),
})


def safe_path(root, value):
    root = Path(root).resolve()
    path = Path(value)
    if path.is_absolute():
        raise ValueError('Artifact paths must be project-relative')
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError('Artifact path escapes project')
    return resolved


def relative(root, path):
    return str(Path(path).resolve().relative_to(Path(root).resolve()))


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Non-finite number')
    if isinstance(value, dict):
        for child in value.values():
            _finite(child)
    elif isinstance(value, list):
        for child in value:
            _finite(child)


def _unique(items, key='id'):
    found = {}
    for item in items:
        value = item[key]
        if not re.fullmatch(r'[a-z][a-z0-9_]{0,79}', value) or value in found:
            raise ValueError(f'Invalid or duplicate {key}: {value}')
        found[value] = item
    return found


def _no_cycles(edges):
    children = {}
    for a, b in edges:
        children.setdefault(a, []).append(b)
    active, done = set(), set()
    def visit(node):
        if node in active:
            raise ValueError('Structural component hierarchy contains a cycle')
        if node in done:
            return
        active.add(node)
        for child in children.get(node, []):
            visit(child)
        active.remove(node)
        done.add(node)
    for node in children:
        visit(node)


def validate_proposal(proposal, rows, track_specs):
    import jsonschema
    _finite(proposal)
    jsonschema.validate(proposal, PROPOSAL_SCHEMA)
    frames = {r['index']: r for r in rows}
    if len(frames) != len(rows) or any(b['timestamp_s'] <= a['timestamp_s'] for a, b in zip(rows, rows[1:])):
        raise ValueError('Source index must be unique and chronological')
    tracks = {t['id']: t for t in track_specs}
    if len(tracks) != len(track_specs) or any(not 1 <= t['sample_range'][0] <= t['sample_range'][1] <= max(frames) for t in track_specs):
        raise ValueError('Duplicate track or invalid source range')
    variables = _unique(proposal['variables'])
    _unique(proposal['assertions'])
    _unique(proposal['steps'])
    _unique(proposal['relationships'])
    def evidence(samples, participants=()):
        if not samples or len(samples) != len(set(samples)) or any(i not in frames for i in samples):
            raise ValueError('Unknown, empty or duplicate evidence samples')
        for tid in participants:
            if tid not in tracks:
                raise ValueError('Unknown participant track')
            lo, hi = tracks[tid]['sample_range']
            if not any(lo <= i <= hi for i in samples):
                raise ValueError('Participant has no evidence in its tracklet range')
    structural = []
    for relation in proposal['relationships']:
        a, b = relation['subject_track_id'], relation['object_track_id']
        if a == b:
            raise ValueError('Self relation')
        evidence(relation['evidence_samples'], [a, b])
        if relation['predicate'] == 'component_of':
            structural.append((a, b))
    _no_cycles(structural)
    for var in variables.values():
        if var['subject_track_id'] not in tracks or len(var['values']) != len(set(var['values'])) or 'unknown' not in var['values']:
            raise ValueError('Invalid state variable, values or unknown default')
    assignments = {}
    for assertion in proposal['assertions']:
        vid, sample = assertion['variable_id'], assertion['evidence_sample']
        if vid not in variables or assertion['value'] not in variables[vid]['values']:
            raise ValueError('Unknown variable or illegal state value')
        evidence([sample], [variables[vid]['subject_track_id']])
        pair = (vid, sample)
        if pair in assignments:
            raise ValueError('Duplicate or contradictory state assertion')
        if (assertion['status'] == 'unknown') != (assertion['value'] == 'unknown'):
            raise ValueError('Unknown status/value must agree')
        assignments[pair] = assertion['value']
    previous_start = -1
    for step in proposal['steps']:
        participants = step['participant_track_ids']
        if not participants or len(participants) != len(set(participants)):
            raise ValueError('Step needs unique participants')
        evidence(step['evidence_samples'], participants)
        low, high = min(step['evidence_samples']), max(step['evidence_samples'])
        if frames[low]['timestamp_s'] < previous_start:
            raise ValueError('Procedure steps must be in chronological source order')
        previous_start = frames[low]['timestamp_s']
        for key in ['preconditions', 'postconditions']:
            seen = set()
            for condition in step[key]:
                vid, sample = condition['variable_id'], condition['evidence_sample']
                if vid not in variables or condition['value'] not in variables[vid]['values']:
                    raise ValueError('Condition references unknown variable/value')
                if variables[vid]['subject_track_id'] not in participants:
                    raise ValueError('Condition subject is not a step participant')
                evidence([sample], [variables[vid]['subject_track_id']])
                if sample not in step['evidence_samples'] or (key == 'preconditions' and sample != low) or (key == 'postconditions' and sample != high):
                    raise ValueError('Conditions must cite the actual first/last step evidence')
                if vid in seen:
                    raise ValueError('Duplicate or contradictory condition')
                seen.add(vid)
                known = assignments.get((vid, sample), 'unknown')
                if known != 'unknown' and known != condition['value']:
                    raise ValueError('Condition contradicts model state evidence')
    return True


def state_at(document, timestamp_s):
    """Point-sampled semantics. No persistence or nearest-future state is assumed."""
    if not math.isfinite(timestamp_s) or not 0 <= timestamp_s <= document['source']['duration_s']:
        raise ValueError('Replay time outside source')
    evidence = {e['id']: e for e in document['evidence']}
    result = {v['id']: {'value': 'unknown', 'status': 'unknown', 'evidence_ids': []} for v in document['variables']}
    for assertion in document['assertions']:
        frame = evidence[assertion['evidence_id']]
        if abs(timestamp_s - frame['timestamp_s']) <= 1e-7:
            result[assertion['variable_id']] = {'value': assertion['value'], 'status': assertion['status'], 'evidence_ids': [frame['id']]}
    return result


def condition_result(document, condition):
    frame = next(e for e in document['evidence'] if e['id'] == condition['evidence_id'])
    actual = state_at(document, frame['timestamp_s'])[condition['variable_id']]
    return {'variable_id': condition['variable_id'], 'expected': condition['value'], 'actual': actual['value'],
            'result': 'unknown' if actual['value'] == 'unknown' else ('matches_model_proposal' if actual['value'] == condition['value'] else 'contradiction'),
            'status': actual['status'], 'evidence_id': frame['id']}


def poses_at(document, timestamp_s):
    """Only exact supported samples; gaps and unobserved geometry stay absent."""
    result = []
    for pose in document['motion']:
        if abs(timestamp_s - pose['timestamp_s']) <= 1e-7 and pose['status'] in ALLOWED_POSES and pose['camera_from_native_mesh'] is not None:
            result.append(copy.deepcopy(pose))
    return result


def replay(document):
    return {'format': 'interaction-twin-evidence-replay', 'version': '0.1', 'simulation_ready': False,
            'policy': document['runtime_policy'],
            'steps': [{'id': s['id'], 'title': s['title'], 'seek_s': s['start_s'], 'end_s': s['end_s'],
                       'preconditions': [condition_result(document, c) for c in s['preconditions']],
                       'postconditions': [condition_result(document, c) for c in s['postconditions']],
                       'evidence_ids': s['evidence_ids'],
                       'supported_pose_samples_in_interval': sum(p['track_id'] in s['participant_track_ids'] and s['start_s'] <= p['timestamp_s'] <= s['end_s'] and p['status'] == 'supported_conditional_motion_candidate' for p in document['motion']),
                       'physical_completion_verified': False} for s in document['steps']]}


def validate_motion_join(pack, pose, assets, root=ROOT):
    """A pose is meaningful only for the same untouched learned mesh."""
    provenance = pack['provenance']
    if any(provenance.get(k) is not False for k in ['manual_geometry_used', 'manual_pose_used', 'manual_corners_used']):
        raise ValueError('Motion manual geometry/pose/corner flags must be explicitly false')
    matching = [a for a in assets if a['track_id'] == pose['track_id']]
    if len(matching) != 1 or matching[0]['sha256'] != provenance['mesh_sha256']:
        raise ValueError('Motion mesh hash does not join the same-track generated asset')
    if Path(provenance['mesh_file']).resolve() != safe_path(root, matching[0]['path']):
        raise ValueError('Motion mesh path differs from same-track asset')
    return True


def validate_interaction(document, root=None, verify_files=True):
    import jsonschema
    _finite(document)
    jsonschema.validate(document, read(ROOT/'schemas/interaction.schema.json'))
    root = Path(root or ROOT)
    inputs = _unique(document['inputs'])
    tracks = _unique(document['tracklets'])
    assets = _unique(document['assets'])
    evidence = _unique(document['evidence'])
    source = document['source']
    for item in [source, *inputs.values(), *assets.values()]:
        path = safe_path(root, item['path'])
        if verify_files and (not path.is_file() or sha(path) != item['sha256']):
            raise ValueError(f"Missing or changed evidence artifact: {item['path']}")
    if document['reasoning']['proposal_input_id'] not in inputs:
        raise ValueError('Unknown reasoning proposal input')
    source_rows = None
    if verify_files:
        frame_input = next((i for i in inputs.values() if i['kind'] == 'source_frame_index'), None)
        manifest_input = next((i for i in inputs.values() if i['kind'] == 'source_manifest'), None)
        if not frame_input or not manifest_input:
            raise ValueError('Source manifest and frame index required')
        source_rows = {r['index']: r for r in read(safe_path(root, frame_input['path']))['frames']}
        manifest = read(safe_path(root, manifest_input['path']))
        if source['sha256'] != manifest['source']['sha256'] or source['duration_s'] != manifest['source']['video']['duration_s']:
            raise ValueError('Source contract differs from saved manifest')
        source_case = safe_path(root, manifest_input['path']).parent
    if len({e['source_index'] for e in evidence.values()}) != len(evidence):
        raise ValueError('Duplicate source evidence sample')
    for track in tracks.values():
        if 'unverified' not in track['identity_status'] or not track['candidate_entity_ids']:
            raise ValueError('Track identity must remain explicitly candidate and unverified')
    previous = -1
    for frame in evidence.values():
        if not previous < frame['timestamp_s'] <= source['duration_s']:
            raise ValueError('Evidence must preserve chronological source PTS')
        previous = frame['timestamp_s']
        path = safe_path(root, frame['frame_path'])
        if source_rows is not None:
            row = source_rows.get(frame['source_index'])
            if not row or row['timestamp_s'] != frame['timestamp_s'] or (source_case/row['file']).resolve() != path:
                raise ValueError('Evidence frame differs from saved source PTS/index')
        if verify_files and (not path.is_file() or sha(path) != frame['frame_sha256']):
            raise ValueError('Source frame hash mismatch')
        for observation in frame['observations']:
            if observation['track_id'] not in tracks:
                raise ValueError('Observation references unknown track')
            path = safe_path(root, observation['mask_path'])
            if verify_files and (not path.is_file() or sha(path) != observation['mask_sha256']):
                raise ValueError('Mask hash mismatch')
    for asset in assets.values():
        if asset['track_id'] not in tracks:
            raise ValueError('Asset references unknown track')
    proposal = {'summary': document['summary'], 'relationships': [], 'variables': document['variables'], 'assertions': [], 'steps': [], 'limitations': document['limitations']}
    for relation in document['relationships']:
        item = copy.deepcopy(relation)
        item['evidence_samples'] = [evidence[e]['source_index'] for e in item.pop('evidence_ids') if e in evidence]
        if len(item['evidence_samples']) != len(relation['evidence_ids']):
            raise ValueError('Unknown relationship evidence')
        proposal['relationships'].append(item)
    for assertion in document['assertions']:
        if assertion['evidence_id'] not in evidence:
            raise ValueError('Unknown state evidence')
        item = copy.deepcopy(assertion)
        item['evidence_sample'] = evidence[item.pop('evidence_id')]['source_index']
        proposal['assertions'].append(item)
    for step in document['steps']:
        item = copy.deepcopy(step)
        if any(e not in evidence for e in item['evidence_ids']):
            raise ValueError('Unknown step evidence')
        ev = [evidence[e] for e in item.pop('evidence_ids')]
        if step['start_s'] != min(e['timestamp_s'] for e in ev) or step['end_s'] != max(e['timestamp_s'] for e in ev):
            raise ValueError('Step time differs from its source evidence')
        item.pop('start_s'); item.pop('end_s')
        item['evidence_samples'] = [e['source_index'] for e in ev]
        for key in ['preconditions', 'postconditions']:
            for condition in item[key]:
                if condition['evidence_id'] not in evidence:
                    raise ValueError('Unknown condition evidence')
                condition['evidence_sample'] = evidence[condition.pop('evidence_id')]['source_index']
        proposal['steps'].append(item)
    specs = [{'id': t['id'], 'sample_range': t['source_sample_range']} for t in tracks.values()]
    rows = [{'index': e['source_index'], 'timestamp_s': e['timestamp_s']} for e in evidence.values()]
    validate_proposal(proposal, rows, specs)
    seen_motion = set()
    for pose in document['motion']:
        if pose['track_id'] not in tracks or pose['experiment_input_id'] not in inputs:
            raise ValueError('Motion references unknown track or experiment')
        if not 0 <= pose['timestamp_s'] <= source['duration_s']:
            raise ValueError('Motion outside source interval')
        key = (pose['track_id'], pose['timestamp_s'])
        if key in seen_motion:
            raise ValueError('Duplicate motion observation at one source time')
        seen_motion.add(key)
        matrix = pose['camera_from_native_mesh']
        if pose['status'] not in ALLOWED_POSES and matrix is not None:
            raise ValueError('Unsupported pose must not expose a replay transform')
        if pose['status'] in ALLOWED_POSES and matrix is None:
            raise ValueError('Supported/reference pose lacks transform')
        if matrix is not None:
            import numpy as np
            m = np.asarray(matrix)
            if m.shape != (4, 4) or not np.allclose(m[3], [0, 0, 0, 1]) or not np.allclose(m[:3, :3].T@m[:3, :3], np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(m[:3, :3]), 1., atol=1e-5):
                raise ValueError('Motion transform is not rigid')
        if verify_files:
            pack = read(safe_path(root, inputs[pose['experiment_input_id']]['path']))
            validate_motion_join(pack, pose, list(assets.values()), root)
            if pack['provenance']['source_video_sha256'] != source['sha256'] or pack['object_id'] != pose['track_id']:
                raise ValueError('Motion upstream source/object mismatch')
            upstream = [f for f in pack['frames'] if f['sample_index'] == pose['source_sample_index']]
            if len(upstream) != 1 or upstream[0]['timestamp_s'] != pose['timestamp_s'] or upstream[0]['status'] != pose['status']:
                raise ValueError('Motion does not match upstream evidence')
            if matrix is not None:
                expected = [list(r)+[t] for r, t in zip(upstream[0]['candidate_rotation'], upstream[0]['candidate_translation'])]+[[0, 0, 0, 1]]
                if matrix != expected:
                    raise ValueError('Motion transform differs from model experiment')
    for value in document['links'].values():
        if value:
            safe_path(root, value)
    return True


def text_export(document):
    lines = ['INTERACTION TWIN / EVIDENCE PROCEDURE 0.1', document['summary'], '',
             'This program replays observations, not physical simulation. Unknown state does not persist.',
             f"Reasoning: {document['reasoning']['provider']} / {document['reasoning']['model']} ({document['reasoning']['inference_location']})", '', 'CANDIDATE COMPONENT RELATIONSHIPS']
    for relation in document['relationships']:
        lines.append(f"{relation['subject_track_id']} --{relation['predicate']}--> {relation['object_track_id']} [{relation['status']}; evidence {', '.join(relation['evidence_ids'])}]")
    for step in document['steps']:
        lines += ['', f"{step['id']}: {step['title']} [{step['start_s']:.6f}–{step['end_s']:.6f} source seconds]", step['description'], 'Evidence: '+', '.join(step['evidence_ids'])]
        for key in ['preconditions', 'postconditions']:
            for condition in step[key]:
                result = condition_result(document, condition)
                lines.append(f"  {key}: {condition['variable_id']} == {condition['value']} at {condition['evidence_id']} -> {result['result']} [{result['status']}]")
        lines.extend('  Limitation: '+x for x in step['limitations'])
    lines += ['', 'LIMITATIONS']+[f'- {x}' for x in document['limitations']]
    return '\n'.join(lines)+'\n'
