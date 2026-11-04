"""Omission-only seed refinement from an exact source-bound visual audit.

No box is edited or invented. The result is a new configuration for a separately
requested SAM attempt; it does not rewrite the original segmentation or verdicts.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def derive(config, audit, track_index_sha256):
    if audit.get('status') != 'complete_candidate_visual_audit' or audit.get('source_track_index_sha256') != track_index_sha256:
        raise ValueError('The visual audit does not bind this exact mask-track index')
    verdicts = {}
    for row in audit['result']['observations']:
        key = (row['track_id'], row['sample_index'])
        if key in verdicts:
            raise ValueError('Duplicate audited observation')
        verdicts[key] = row
    result, changes, tracks = deepcopy(config), [], []
    for track in result['tracks']:
        keep = []
        for prompt in track['prompts']:
            verdict = verdicts.get((track['id'], prompt['sample_index']))
            accepted = verdict is not None and verdict['verdict'] == 'plausible_visible_mask'
            changes.append({'track_id': track['id'], 'sample_index': prompt['sample_index'],
                            'operation': 'retain_identical_model_seed' if accepted else 'omit_seed',
                            'audit_verdict': verdict['verdict'] if verdict else 'not_model_audited',
                            'reason': verdict.get('reason') if verdict else 'No visual audit of this seed'})
            if accepted:
                keep.append(prompt)
        if keep:
            track['prompts'] = keep
            tracks.append(track)
    result['tracks'] = tracks
    result['seed_origin'] = 'meta_api_model_proposal_with_visual_audit_omission_only'
    result['seed_review_policy'] = 'Retain exact model boxes only where the previous source-bound model audit judged its mask plausible; this is not human validation.'
    return result, changes


def write_derivative(run, destination):
    run, destination = Path(run).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError('Preserve the existing configuration')
    config_path, track_path = run / 'annotations.json', run / 'tracks.json'
    audit_path = run / 'model-audit-v1/audit.json'
    config, audit = json.loads(config_path.read_text()), json.loads(audit_path.read_text())
    metadata = json.loads((run / 'metadata.json').read_text())
    if metadata.get('config_sha256') != sha(config_path) or metadata.get('source_sha256') != audit.get('source_sha256'):
        raise ValueError('Configuration or source identity differs from the original segmentation receipt')
    if not metadata.get('sam_video_propagation_executed') or metadata.get('manual_seed_boxes_used') is not False:
        raise ValueError('Require actual model-seeded SAM video provenance')
    result, changes = derive(config, audit, sha(track_path))
    result['seed_review'] = {'source_config_sha256': sha(config_path), 'source_tracks_sha256': sha(track_path),
                             'source_audit_sha256': sha(audit_path), 'changes': changes,
                             'manual_seed_boxes_used': False, 'new_boxes_added': False}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + '\n')
    return result
