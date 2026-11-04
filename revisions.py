"""Immutable, evidence-preserving human review of published interaction programs.

Review never changes model artifacts, physical identity, geometry or poses. The
sidecar records authorship separately from the unchanged model provenance.
"""
from __future__ import annotations

import copy
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path

import stage5
from jsonschema.exceptions import ValidationError

CASE_ID = re.compile(r'[A-Za-z0-9_-]{1,180}')
REV_ID = re.compile(r'rev-[0-9]{6}-[0-9a-f]{16}')
SECTIONS = ('steps', 'relationships', 'assertions')
MAX_BODY = 192 * 1024
MAX_REVISIONS = 1000
EDITABLE = {
    'steps': {'title', 'description', 'participant_track_ids', 'evidence_ids', 'limitations'},
    'relationships': {'subject_track_id', 'object_track_id', 'predicate', 'status', 'rationale', 'evidence_ids', 'limitations'},
    'assertions': {'variable_id', 'value', 'status', 'rationale', 'evidence_id'},
}
DISCLOSURE = ('User-reviewed derivative: review decisions and edits are authored by a person, '
              'not new model inference. Acceptance does not verify physical identity, contact, '
              'seating, dimensions or simulation readiness. States apply only at cited samples.')


class Conflict(ValueError):
    """The source or current review head changed since this draft was opened."""


def _read(path):
    return json.loads(Path(path).read_text())


def _confined(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Review path escapes its allowed directory')
    return path


def published_source(root, case_id):
    """Only allow source files explicitly embedded in the published workspace."""
    root = Path(root).resolve()
    if not isinstance(case_id, str) or not CASE_ID.fullmatch(case_id):
        raise ValueError('Invalid published case identifier')
    published = _read(_confined(root / 'workspace/cases.json', root / 'workspace'))
    rows = [row for row in published.get('cases', []) if row.get('id') == case_id]
    if len(rows) != 1:
        raise ValueError('Choose a published workspace case')
    value = rows[0].get('interaction_path')
    if not isinstance(value, str) or Path(value).is_absolute():
        raise ValueError('Published interaction path is invalid')
    path = _confined(root / value, root / 'data')
    if path.name != 'interaction.json' or not path.is_file():
        raise ValueError('Published interaction is unavailable')
    raw = path.read_bytes()
    document = json.loads(raw)
    try:
        stage5.validate_interaction(document, root=root, verify_files=False)
    except ValidationError as exc:
        raise ValueError('Published procedure is invalid: ' + exc.message) from exc
    digest = hashlib.sha256(raw).hexdigest()
    # A rebuilt source must be explicitly republished before reviewing it.
    expected = rows[0].get('original_sha256')
    if expected and expected != digest:
        raise Conflict('The original program changed; rebuild and reload this workspace first')
    return path, document, digest


def _bounded(value):
    if isinstance(value, str) and len(value) > 6000:
        raise ValueError('Review text must be at most 6000 characters per field')
    if isinstance(value, list):
        if len(value) > 512:
            raise ValueError('A review list is too large')
        for child in value:
            _bounded(child)
    if isinstance(value, dict):
        if len(value) > 512:
            raise ValueError('A review object is too large')
        for child in value.values():
            _bounded(child)


def validate_derivative(document, *, root=None, verify_files=False):
    """Adapter for the two human-authorship flags in the frozen model schema.

    Old model outputs cite the exact compiler/schema hashes. Keep those files
    unchanged: validate every ordinary contract on a copy with only the two
    automatic-output flags normalized, preserving true flags in the derivative.
    """
    candidate = copy.deepcopy(document)
    for key in ['human_reviewed', 'manual_semantic_story_used']:
        if type(candidate['reasoning'].get(key)) is not bool:
            raise ValueError('Human review provenance flags must be booleans')
        candidate['reasoning'][key] = False
    try:
        return stage5.validate_interaction(candidate, root=root, verify_files=verify_files)
    except ValidationError as exc:
        raise ValueError('The revised procedure does not satisfy the evidence contract: ' + exc.message) from exc


def apply_edits(original, edits, *, root=None, verify_files=False):
    """Apply a complete cumulative review set to the immutable original.

    Rejected assertions are absent, making their state unknown. A changed
    assertion also updates existing boundary conditions at its exact sample.
    No assertion is carried forward to intervening video frames.
    """
    if not isinstance(edits, dict) or set(edits) != set(SECTIONS):
        raise ValueError('Review must contain steps, relationships and assertions maps')
    _bounded(edits)
    document = copy.deepcopy(original)
    evidence = {row['id']: row for row in original['evidence']}
    decisions = []
    for section in SECTIONS:
        changes = edits[section]
        if not isinstance(changes, dict):
            raise ValueError('Each review section must be a map keyed by original IDs')
        rows = {row['id']: row for row in original[section]}
        if set(changes) - set(rows):
            raise ValueError('Review references an unknown original item')
        revised = []
        for original_row in original[section]:
            row = copy.deepcopy(original_row)
            review = changes.get(row['id'])
            if review is None:
                revised.append(row)
                continue
            if not isinstance(review, dict) or set(review) != {'decision', 'note', 'patch'}:
                raise ValueError('Review needs a decision, note and patch')
            decision, note, patch = review['decision'], review['note'], review['patch']
            if decision not in {'accepted', 'rejected', 'edited'} or not isinstance(note, str) or not isinstance(patch, dict):
                raise ValueError('Invalid review decision, note or patch')
            if decision != 'edited' and patch:
                raise ValueError('Only an edited item may contain a patch')
            if set(patch) - EDITABLE[section]:
                raise ValueError('Review cannot change geometry, source, IDs or runtime policy')
            if 'status' in patch and patch['status'] not in {'inferred', 'unknown'}:
                raise ValueError('A human edit must remain inferred or unknown')
            if decision == 'edited' and not patch:
                raise ValueError('An edited item needs a change')
            decisions.append({'section': section, 'id': row['id'], **copy.deepcopy(review)})
            if decision == 'rejected':
                continue
            row.update(copy.deepcopy(patch))
            if decision == 'edited' and section in {'relationships', 'assertions'}:
                # A human edit cannot be represented as a newly produced model proposal.
                if section == 'assertions' and patch.get('status') == 'unknown':
                    if 'value' in patch and patch['value'] != 'unknown':
                        raise ValueError('Unknown status cannot assert a known state value')
                    row['value'] = 'unknown'
                row['status'] = ('unknown' if row.get('value') == 'unknown' else 'inferred') if section == 'assertions' else ('unknown' if row['status'] == 'unknown' else 'inferred')
                if section == 'assertions' and row['status'] == 'unknown':
                    row['value'] = 'unknown'
            if section == 'steps':
                ids = row['evidence_ids']
                if not isinstance(ids, list) or not ids or any(i not in evidence for i in ids):
                    raise ValueError('A step must cite existing source evidence')
                row['start_s'] = min(evidence[i]['timestamp_s'] for i in ids)
                row['end_s'] = max(evidence[i]['timestamp_s'] for i in ids)
                # Boundary conditions outside the edited window no longer apply.
                for key, time in [('preconditions', row['start_s']), ('postconditions', row['end_s'])]:
                    row[key] = [c for c in row[key] if c['evidence_id'] in ids and evidence[c['evidence_id']]['timestamp_s'] == time
                                and next(v for v in original['variables'] if v['id'] == c['variable_id'])['subject_track_id'] in row['participant_track_ids']]
            revised.append(row)
        document[section] = revised
    # Keep checks consistent with explicitly edited point assertions; rejecting a
    # point assertion leaves its expected check unresolved, never successful.
    changed_assertions = {a['id'] for a in decisions if a['section'] == 'assertions' and a['decision'] == 'edited'}
    values = {(a['variable_id'], a['evidence_id']): a['value'] for a in document['assertions'] if a['id'] in changed_assertions}
    for step in document['steps']:
        for key in ['preconditions', 'postconditions']:
            for condition in step[key]:
                pair = (condition['variable_id'], condition['evidence_id'])
                if pair in values:
                    condition['value'] = values[pair]
    document['reasoning']['human_reviewed'] = bool(decisions)
    document['reasoning']['manual_semantic_story_used'] = any(d['decision'] != 'accepted' for d in decisions)
    if decisions:
        document['limitations'].append(DISCLOSURE)
    validate_derivative(document, root=root, verify_files=verify_files)
    return document, decisions


def readable_guide(document, decisions, base_sha256, revision_id=None):
    heading = ['INTERACTION TWIN / USER REVIEW', DISCLOSURE,
               'Original program SHA256: ' + base_sha256,
               'Revision: ' + (revision_id or 'UNSAVED DRAFT'), '', 'REVIEW DECISIONS']
    for row in decisions:
        heading.append(f"{row['section']}/{row['id']}: {row['decision']}" + (f" — {row['note']}" if row['note'] else ''))
    heading += ['', 'POINT-SAMPLED VISUAL STATES']
    for row in document['assertions']:
        heading.append(f"{row['variable_id']} = {row['value']} [{row['status']}; evidence {row['evidence_id']}] — {row['rationale']}")
    return '\n'.join(heading) + '\n\n' + stage5.text_export(document)


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def _directory(self, case_id, create=False):
        if not isinstance(case_id, str) or not CASE_ID.fullmatch(case_id):
            raise ValueError('Invalid published case identifier')
        location = self.root / 'data/user-revisions' / case_id
        if location.is_symlink() or location.parent.is_symlink():
            raise ValueError('Review directories cannot be symlinks')
        directory = _confined(location, self.root / 'data')
        # No external symlinks through either the case or store directory.
        _confined(directory, self.root / 'data/user-revisions')
        if create:
            directory.mkdir(parents=True, exist_ok=True, mode=0o750)
        return directory

    def _records(self, case_id):
        directory = self._directory(case_id)
        paths = sorted(directory.glob('rev-*.json')) if directory.is_dir() else []
        if len(paths) > MAX_REVISIONS:
            raise ValueError('This case reached its review history limit')
        records = []
        for path in paths:
            if not REV_ID.fullmatch(path.stem):
                continue
            path = _confined(path, directory)
            value = _read(path)
            if value.get('revision_id') != path.stem or value.get('case_id') != case_id:
                raise ValueError('Review history is invalid')
            records.append(value)
        return records

    def list(self, case_id):
        _, _, digest = published_source(self.root, case_id)
        rows = self._records(case_id)
        return {'case_id': case_id, 'original_sha256': digest,
                'head_revision_id': rows[-1]['revision_id'] if rows else None,
                'revisions': [{key: row[key] for key in ['revision_id', 'parent_revision_id', 'created_at', 'base_sha256', 'review_counts']} for row in reversed(rows)]}

    def get(self, case_id, revision_id):
        if not isinstance(revision_id, str) or not REV_ID.fullmatch(revision_id):
            raise ValueError('Invalid revision identifier')
        _, original, digest = published_source(self.root, case_id)
        rows = self._records(case_id)
        row = next((r for r in rows if r['revision_id'] == revision_id), None)
        if row is None:
            raise ValueError('That review revision does not exist')
        if row['base_sha256'] != digest:
            raise Conflict('This revision refers to an older original program')
        document, decisions = apply_edits(original, row['edits'], root=self.root, verify_files=True)
        if row.get('document') != document or row.get('decisions') != decisions:
            raise ValueError('Saved revision does not match its recorded edits')
        expected_replay = stage5.replay(document)
        expected_guide = readable_guide(document, decisions, digest, revision_id)
        if row.get('replay') != expected_replay or row.get('guide') != expected_guide:
            raise ValueError('Saved revision exports do not match its validated document')
        return row

    def save(self, case_id, request):
        if not isinstance(request, dict) or set(request) != {'base_sha256', 'parent_revision_id', 'edits'}:
            raise ValueError('Save requires the original hash, current parent and cumulative edits')
        stage5._finite(request)
        _bounded(request)
        if len(json.dumps(request, allow_nan=False).encode()) > MAX_BODY:
            raise ValueError('Review request exceeds the bounded size')
        directory = self._directory(case_id, create=True)
        lock_path = _confined(directory / '.lock', directory)
        with lock_path.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            _, original, digest = published_source(self.root, case_id)
            if request['base_sha256'] != digest:
                raise Conflict('The source program changed; reload before saving your review')
            rows = self._records(case_id)
            head = rows[-1]['revision_id'] if rows else None
            if request['parent_revision_id'] != head:
                raise Conflict('Another review was saved; load the latest revision before saving')
            if len(rows) >= MAX_REVISIONS:
                raise ValueError('This case reached its review history limit')
            document, decisions = apply_edits(original, request['edits'], root=self.root, verify_files=True)
            if not decisions:
                raise ValueError('Make at least one review decision before saving')
            revision_id = f'rev-{len(rows) + 1:06d}-{os.urandom(8).hex()}'
            row = {'format': 'interaction-twin-user-review', 'version': '0.1', 'case_id': case_id,
                   'revision_id': revision_id, 'parent_revision_id': head, 'base_sha256': digest,
                   'created_at': dt.datetime.now(dt.timezone.utc).isoformat(),
                   'authorship': 'user_review_via_local_workspace', 'disclosure': DISCLOSURE,
                   'edits': copy.deepcopy(request['edits']), 'decisions': decisions,
                   'review_counts': {d: sum(r['decision'] == d for r in decisions) for d in ['accepted', 'rejected', 'edited']},
                   'document': document, 'replay': stage5.replay(document),
                   'guide': readable_guide(document, decisions, digest, revision_id)}
            target = directory / (revision_id + '.json')
            temporary = directory / ('.pending-' + os.urandom(8).hex())
            try:
                with temporary.open('x') as stream:
                    json.dump(row, stream, ensure_ascii=False, allow_nan=False, indent=2)
                    stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
                temporary.chmod(0o444)
                os.link(temporary, target)  # Exclusive publication; never overwrite history.
            finally:
                temporary.unlink(missing_ok=True)
            return row
