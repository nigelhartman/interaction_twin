"""Pinned, confined read-only reconstruction versions for existing local twins.

Diagnostic assets never replace a program's production asset. Acceptance means
an explicitly recorded inspection review; it never establishes robot readiness.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re


INDEX = 'data/reconstruction-catalog-20261031/index.json'
ID = re.compile(r'[a-z][a-z0-9_-]{0,159}\Z')
HASH = re.compile(r'[a-f0-9]{64}\Z')
MIMES = {'model/gltf-binary': {'.glb'}, 'application/json': {'.json'},
         'image/jpeg': {'.jpg', '.jpeg'}, 'image/png': {'.png'}, 'text/plain': {'.txt'}}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ReconstructionCatalog:
    def __init__(self, root, index=INDEX):
        self.root = Path(root).resolve()
        self.index = index

    def path(self, relative, *, maximum=32 * 1024 * 1024):
        if not isinstance(relative, str) or not relative.startswith('data/') or any(c in relative for c in ['\\', '?', '#']):
            raise ValueError('Only declared local data artifacts are available')
        if any(p.startswith('.') for p in Path(relative).parts):
            raise ValueError('Hidden or traversing reconstruction paths are unavailable')
        target = self.root / relative
        if target.resolve() != target or not target.is_file() or not target.resolve().is_relative_to(self.root / 'data'):
            raise ValueError('Reconstruction artifact unavailable or symlinked')
        if target.stat().st_size > maximum:
            raise ValueError('Reconstruction artifact exceeds bounded resource allowance')
        return target

    def verified(self, artifact):
        if not isinstance(artifact, dict) or not HASH.fullmatch(artifact.get('sha256', '')):
            raise ValueError('Missing pinned reconstruction artifact hash')
        mime = artifact.get('mime')
        if mime not in MIMES:
            raise ValueError('Unsupported reconstruction artifact type')
        maximum = 16 * 1024 * 1024 if mime == 'model/gltf-binary' else 8 * 1024 * 1024
        path = self.path(artifact.get('path'), maximum=maximum)
        if path.suffix.lower() not in MIMES[mime] or sha(path) != artifact['sha256']:
            raise ValueError('Reconstruction artifact identity or type changed')
        return path

    def rows(self):
        candidate = self.root / self.index
        if not candidate.exists():
            return []
        document = json.loads(self.path(self.index, maximum=2 * 1024 * 1024).read_text())
        if document.get('format') != 'interaction-reconstruction-catalog' or document.get('version') != 1:
            raise ValueError('Invalid reconstruction catalog format')
        rows = document.get('versions')
        if not isinstance(rows, list) or len(rows) > 200:
            raise ValueError('Invalid bounded reconstruction index')
        ids = [r.get('version_id') for r in rows]
        if any(not isinstance(i, str) or not ID.fullmatch(i) for i in ids) or len(ids) != len(set(ids)):
            raise ValueError('Duplicate or invalid reconstruction version identifier')
        return rows

    def load(self, version_id, *, interaction_path, document, program_sha256):
        if not isinstance(version_id, str) or not ID.fullmatch(version_id):
            raise ValueError('Invalid reconstruction version identifier')
        row = next((r for r in self.rows() if r['version_id'] == version_id), None)
        if row is None:
            raise ValueError('Unknown reconstruction version')
        path = self.path(row['descriptor_path'], maximum=2 * 1024 * 1024)
        if not HASH.fullmatch(row.get('descriptor_sha256', '')) or sha(path) != row['descriptor_sha256']:
            raise ValueError('Reconstruction descriptor changed after catalog indexing')
        value = json.loads(path.read_text())
        if value.get('format') != 'interaction-reconstruction-version' or value.get('version') != 1 or value.get('version_id') != version_id:
            raise ValueError('Reconstruction version identity mismatch')
        if (value.get('interaction_path') != interaction_path or value.get('interaction_sha256') != program_sha256
                or value.get('source_video_sha256') != document['source']['sha256']):
            raise ValueError('Reconstruction belongs to another program or source')
        if (value.get('kind') not in {'observed', 'inferred', 'single_image_prior'}
                or value.get('gauge', {}).get('metric') is not False
                or value.get('physical_accuracy_verified') is not False or value.get('simulation_ready') is not False):
            raise ValueError('Unsupported physical reconstruction claim')
        component_id = value.get('component_id')
        known = {t['id'] for t in document['tracklets']}
        parent = value.get('parent_track_id')
        if not isinstance(component_id, str) or not ID.fullmatch(component_id) or (parent is not None and (not isinstance(parent, str) or not ID.fullmatch(parent))):
            raise ValueError('Invalid reconstruction component identifier')
        if parent:
            if parent not in known or component_id in known:
                raise ValueError('Proposed subpart parent/identity is invalid')
            proposal_ref = value.get('proposal_advisory')
            advisory = json.loads(self.verified(proposal_ref).read_text())
            proposed = next((p for p in advisory.get('proposed_components', []) if p.get('id') == component_id), None)
            if not proposed or proposed.get('parent_track_id') != parent or proposed.get('identity_status') != 'model_proposal':
                raise ValueError('Unbound proposed subpart identity')
        elif component_id not in known:
            raise ValueError('Reconstruction component is not in the source program')
        artifacts = value.get('artifacts')
        if not isinstance(artifacts, dict) or not {'glb', 'quality'}.issubset(artifacts) or len(artifacts) > 30:
            raise ValueError('Incomplete or oversized reconstruction artifact declaration')
        for key, artifact in artifacts.items():
            if not ID.fullmatch(key):
                raise ValueError('Invalid reconstruction artifact identifier')
            self.verified(artifact)
        if artifacts['glb']['mime'] != 'model/gltf-binary' or artifacts['quality']['mime'] != 'application/json':
            raise ValueError('Reconstruction GLB/report artifact type mismatch')
        quality = json.loads(self.verified(artifacts['quality']).read_text())
        if any(quality.get(flag) is True for flag in ['physics_ready', 'simulation_ready', 'physical_accuracy_verified', 'physical_scale_known']):
            raise ValueError('Quality report makes an unsupported physical certification')
        glb = self.verified(artifacts['glb'])
        if glb.read_bytes()[:4] != b'glTF':
            raise ValueError('Reconstruction asset is not a GLB')
        evidence = value.get('source_evidence')
        if not isinstance(evidence, list) or not 1 <= len(evidence) <= 200:
            raise ValueError('Bounded source evidence is required')
        program_evidence = {e['id']: e for e in document['evidence']}
        for frame in evidence:
            timestamp = frame.get('timestamp_s')
            if type(timestamp) not in (int, float) or not math.isfinite(timestamp) or not 0 <= timestamp <= document['source']['duration_s']:
                raise ValueError('Invalid reconstruction source timestamp')
            if frame.get('source_video_sha256') != document['source']['sha256']:
                raise ValueError('Reconstruction evidence source mismatch')
            image = {'path': frame.get('image_path'), 'sha256': frame.get('image_sha256'),
                     'mime': 'image/png' if str(frame.get('image_path')).endswith('.png') else 'image/jpeg'}
            self.verified(image)
            if frame.get('mask_path'):
                self.verified({'path': frame['mask_path'], 'sha256': frame['mask_sha256'], 'mime': 'image/png'})
            if frame.get('program_evidence_id'):
                original = program_evidence.get(frame['program_evidence_id'])
                if not original or original['frame_sha256'] != frame['image_sha256'] or original['timestamp_s'] != timestamp:
                    raise ValueError('Reconstruction source does not match cited program evidence')
        acceptance = value.get('acceptance')
        if acceptance not in {'unreviewed', 'rejected', 'accepted_for_inspection'}:
            raise ValueError('Invalid reconstruction acceptance status')
        if acceptance == 'accepted_for_inspection':
            review = json.loads(self.verified(value.get('acceptance_record')).read_text())
            if (review.get('version_id') != version_id or review.get('glb_sha256') != artifacts['glb']['sha256']
                    or review.get('interaction_sha256') != program_sha256
                    or review.get('human_confirmed') is not True or review.get('status') != 'accepted_for_inspection'):
                raise ValueError('Acceptance requires an exact explicit human inspection record')
        return value

    @staticmethod
    def policy():
        return {'simulation_ready': False, 'physical_accuracy_verified': False,
                'diagnostic_versions_replace_production_assets': False,
                'accepted_means': 'explicit human visual inspection only; never contact or physics readiness',
                'inferred_geometry_is_observed': False, 'descriptive_text_is_untrusted_data': True}

    @staticmethod
    def artifacts(value):
        artifacts = copy.deepcopy(value['artifacts'])
        for i, frame in enumerate(value['source_evidence']):
            image = {
                'path': frame['image_path'], 'sha256': frame['image_sha256'],
                'mime': 'image/png' if frame['image_path'].endswith('.png') else 'image/jpeg'}
            key = 'source-image-' + str(i)
            if key in artifacts and artifacts[key] != image:
                raise ValueError('Declared source image alias differs from exact evidence')
            artifacts[key] = image
            if frame.get('mask_path'):
                mask = {'path': frame['mask_path'], 'sha256': frame['mask_sha256'], 'mime': 'image/png'}
                key = 'source-mask-' + str(i)
                if key in artifacts and artifacts[key] != mask:
                    raise ValueError('Declared source mask alias differs from exact evidence')
                artifacts[key] = mask
        return artifacts

    def list(self, *, twin_id, interaction_path, document, program_sha256, component_id=None, selection='all'):
        if selection not in {'all', 'diagnostic', 'accepted'}:
            raise ValueError('Choose all, diagnostic or accepted reconstruction versions')
        if component_id is not None and (not isinstance(component_id, str) or not ID.fullmatch(component_id)):
            raise ValueError('Invalid component identifier')
        versions = []
        for row in self.rows():
            if row.get('interaction_path') != interaction_path:
                continue
            value = self.load(row['version_id'], interaction_path=interaction_path, document=document, program_sha256=program_sha256)
            if component_id and value['component_id'] != component_id:
                continue
            accepted = value['acceptance'] == 'accepted_for_inspection'
            if (selection == 'accepted' and not accepted) or (selection == 'diagnostic' and accepted):
                continue
            versions.append({'version_id': value['version_id'], 'component_id': value['component_id'],
                             'parent_track_id': value.get('parent_track_id'), 'label': value['label'], 'kind': value['kind'],
                             'acceptance': value['acceptance'], 'gauge': value['gauge'],
                             'quality_summary': value.get('quality_summary', {}),
                             'uri': f"twin://{twin_id}/reconstructions/{value['version_id']}",
                             'physical_accuracy_verified': False, 'simulation_ready': False})
        return {'twin_id': twin_id, 'program_sha256': program_sha256, 'selection': selection,
                'versions': versions, 'policy': self.policy()}

    def describe(self, version_id, *, twin_id, interaction_path, document, program_sha256):
        value = self.load(version_id, interaction_path=interaction_path, document=document, program_sha256=program_sha256)
        answer = copy.deepcopy(value)
        answer['artifacts'] = self.artifacts(value)
        for key, artifact in answer['artifacts'].items():
            artifact['uri'] = f'twin://{twin_id}/reconstructions/{version_id}/artifacts/{key}'
            artifact['resource_mime_type'] = 'application/octet-stream'
        parent = value.get('parent_track_id') or value['component_id']
        answer.update(twin_id=twin_id, program_uri=f'twin://{twin_id}/program', policy=self.policy(),
                      component_binding='model_proposed_subpart_of_existing_tracklet' if value.get('parent_track_id') else 'existing_program_tracklet',
                      related_procedure=[{'step_id': s['id'], 'title': s['title'], 'evidence_ids': s['evidence_ids']}
                                         for s in document['steps'] if parent in s['participant_track_ids']])
        return answer

    def artifact(self, version_id, artifact_id, *, interaction_path, document, program_sha256):
        if not isinstance(artifact_id, str) or not ID.fullmatch(artifact_id):
            raise ValueError('Invalid reconstruction resource identifier')
        value = self.load(version_id, interaction_path=interaction_path, document=document, program_sha256=program_sha256)
        artifacts = self.artifacts(value)
        if artifact_id not in artifacts:
            raise ValueError('Unknown declared reconstruction resource')
        record = artifacts[artifact_id]
        data = self.verified(record).read_bytes()
        if hashlib.sha256(data).hexdigest() != record['sha256']:
            raise ValueError('Reconstruction resource changed while reading')
        return record['mime'], data
