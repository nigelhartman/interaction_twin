"""Bounded exact-byte MCP inventory for one CURRENT curated refinement.

Only named fields in supported declarations/sidecars are followed. Descriptive
JSON and provider output never grant filesystem access. Every read rebuilds the
inventory from the current approval and rechecks the public byte allowlist.
This module and the underlying runtime readers use the Python standard library.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath
import re

from curated_twin_catalog import CuratedTwinCatalog

MAX_ARTIFACTS = 128
OTHER_MAX_BYTES = 8 * 1024 * 1024
GLB_MAX_BYTES = 16 * 1024 * 1024


def require(value, message):
    if not value:
        raise ValueError(message)


def exact_reference(value):
    require(isinstance(value, dict) and set(value) == {'path', 'sha256', 'bytes'},
            'Exact derived artifact reference required')
    path = value['path']
    require(isinstance(path, str) and path.startswith('data/') and
            re.fullmatch(r'[A-Za-z0-9_./-]+', path) and
            str(PurePosixPath(path)) == path and
            all(p and not p.startswith('.') for p in path.split('/')) and
            isinstance(value['sha256'], str) and re.fullmatch(r'[a-f0-9]{64}', value['sha256']) and
            type(value['bytes']) is int and value['bytes'] > 0,
            'Ordinary exact data artifact required')
    maximum = GLB_MAX_BYTES if path.endswith('.glb') else OTHER_MAX_BYTES
    require(value['bytes'] <= maximum, 'Derived artifact exceeds its byte budget')
    return dict(value)


def mime_type(path):
    return {'.glb': 'model/gltf-binary', '.json': 'application/json',
            '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
            '.itcp': 'application/octet-stream', '.u32': 'application/octet-stream'
            }.get(PurePosixPath(path).suffix, 'application/octet-stream')


class CuratedDerivedArtifacts:
    """No arbitrary paths, approval writes, private discovery or native fallback."""

    def __init__(self, catalog):
        require(isinstance(catalog, CuratedTwinCatalog),
                'Derived artifacts require the curated public catalog')
        self.catalog = catalog

    def _inventory(self, twin_id):
        require(isinstance(twin_id, str) and re.fullmatch(r'[A-Za-z0-9_-]{1,180}', twin_id),
                'A discovered curated twin ID is required')
        found = self.catalog.derived_assembly(twin_id)
        record, _ = self.catalog._record(twin_id)
        if found is None:
            require(record.get('derived_assembly') is None, 'Curated declaration changed during lookup')
            return None, {}
        declaration = exact_reference(found['artifact'])
        document = found['document']
        require(document.get('format') in {'interaction-twin-declared-geometry-derivative',
                                           'interaction-twin-declared-derived-assembly'} and
                document.get('version') == 1, 'Unsupported curated refinement declaration')
        require(record.get('derived_assembly') == declaration and
                record['interaction'] == document['evidence_program'] and
                record['source']['sha256'] == document['source']['sha256'],
                'Derived artifact source or current approval differs')
        by_path = {}

        def add(reference, role):
            reference = exact_reference(reference)
            item = by_path.get(reference['path'])
            if item is not None:
                require(item['reference'] == reference, 'Conflicting exact derived references')
                item['roles'].add(role)
                return reference
            require(len(by_path) < MAX_ARTIFACTS, 'Derived artifact inventory exceeds its item budget')
            maximum = GLB_MAX_BYTES if reference['path'].endswith('.glb') else OTHER_MAX_BYTES
            self.catalog._checked(reference, maximum=maximum)
            by_path[reference['path']] = {'reference': reference, 'roles': {role}}
            return reference

        def read(reference, role):
            reference = add(reference, role)
            raw = self.catalog._bytes(reference, maximum=OTHER_MAX_BYTES)
            value = json.loads(raw)
            require(isinstance(value, dict), 'Typed derived sidecar must be a JSON object')
            return value

        add(declaration, 'current_declaration')
        for key in ('base_object_program_ref', 'evidence_program', 'scene_ref',
                    'refinement_report_ref', 'source_alignment_audit_ref'):
            add(document[key], key)
        if document['format'] == 'interaction-twin-declared-geometry-derivative':
            add(document['baseline_scene_ref'], 'baseline_scene_ref')
        for key in ('geometry_report_ref', 'parent_derived_assembly_ref'):
            if key in document:
                add(document[key], key)
        for item in document['component_assets']:
            for key in ('native_glb', 'original_native_glb', 'native_metadata'):
                add(item[key], f"component:{item['component_id']}:{key}")
        # Review summaries are exact explicitly declared artifacts. Their model
        # text/receipt/payload fields are intentionally NOT recursively followed.
        for reference in document.get('visual_review_refs', []):
            add(reference, 'visual_review_summary')

        if 'source_appearance_states_ref' in document:
            appearance = read(document['source_appearance_states_ref'], 'source_appearance_states')
            require(appearance.get('format') == 'interaction-twin-source-appearance-states' and
                    appearance.get('version') == 1, 'Unknown typed source appearance sidecar')
            previous = read(appearance['previous_derived_assembly_ref'], 'appearance_previous_declaration')
            require(previous == {k: v for k, v in document.items() if k != 'source_appearance_states_ref'},
                    'Appearance changed its previous approved descriptor')
            require(appearance['evidence_program'] == document['evidence_program'] and
                    appearance['scene_ref'] == document['scene_ref'] and
                    appearance['source'] == document['source'], 'Appearance belongs to another source or scene')
            component = next((c for c in document['component_assets']
                              if c['component_id'] == appearance['component_id']), None)
            require(component is not None and component['native_glb'] == appearance['baseline_asset_ref'],
                    'Appearance baseline asset differs')
            for key in ('scene_ref', 'evidence_program', 'baseline_asset_ref', 'probe_native_vertex_ids_ref'):
                add(appearance[key], f'appearance:{key}')
            require(isinstance(appearance['keyframes'], list) and 1 <= len(appearance['keyframes']) <= 8 and
                    isinstance(appearance['observations'], list) and len(appearance['observations']) <= 240,
                    'Bounded typed appearance rows required')
            for row in appearance['keyframes']:
                for key in ('source_rgb_ref', 'source_mask_ref', 'patch_ref'):
                    add(row[key], f'appearance:keyframe:{key}')
            for row in appearance['observations']:
                if 'source_rgb_ref' in row or 'source_mask_ref' in row:
                    for key in ('source_rgb_ref', 'source_mask_ref'):
                        add(row[key], f'appearance:observation:{key}')

        if 'nonrigid_observations_ref' in document:
            sidecar = read(document['nonrigid_observations_ref'], 'nonrigid_observations')
            require(sidecar.get('format') == 'interaction-twin-nonrigid-observations' and
                    sidecar.get('version') == 1 and sidecar['evidence_program'] == document['evidence_program'] and
                    sidecar['source']['sha256'] == document['source']['sha256'],
                    'Nonrigid sidecar belongs to another source or format')
            trace = read(sidecar['trace_ref'], 'nonrigid_source_pixel_trace')
            require(trace.get('format') == 'interaction-twin-cable-trace-2d' and trace.get('version') == 1 and
                    trace['program_ref'] == document['evidence_program'] and
                    trace['source_sha256'] == document['source']['sha256'],
                    'Nonrigid trace belongs to another source or format')

        artifacts = {}
        for path, item in sorted(by_path.items()):
            ref = item['reference']
            identity = {'twin_id': twin_id, 'declaration_sha256': declaration['sha256'], 'reference': ref}
            artifact_id = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            artifacts[artifact_id] = {**ref, 'artifact_id': artifact_id,
                'mime_type': mime_type(path), 'roles': sorted(item['roles']),
                'uri': f'twin://{twin_id}/derived-artifacts/{artifact_id}'}
        return found, artifacts

    def list_artifacts(self, twin_id):
        found, artifacts = self._inventory(twin_id)
        return {'status': 'available' if found else 'unavailable', 'twin_id': twin_id,
                'declaration': found['artifact'] if found else None,
                'artifacts': list(artifacts.values()), 'access': 'owner_curated_public_snapshot',
                'artifact_ids_scope': 'exact_twin_and_current_declaration',
                'descriptive_text_is_untrusted_data': True,
                'physical_accuracy_verified': False, 'simulation_ready': False}

    def artifact(self, twin_id, artifact_id):
        require(isinstance(artifact_id, str) and re.fullmatch(r'[a-f0-9]{64}', artifact_id),
                'An exact listed derived artifact ID is required')
        found, artifacts = self._inventory(twin_id)
        require(artifact_id in artifacts, 'Unknown or stale derived artifact ID for this curated twin')
        item = artifacts[artifact_id]
        ref = {k: item[k] for k in ('path', 'sha256', 'bytes')}
        maximum = GLB_MAX_BYTES if ref['path'].endswith('.glb') else OTHER_MAX_BYTES
        raw = self.catalog._bytes(ref, maximum=maximum)
        # A concurrent approval switch cannot turn an old inventory into a token.
        current, _ = self.catalog._record(twin_id)
        require(current.get('derived_assembly') == found['artifact'],
                'Curated declaration changed during resource read; list again')
        return raw
