"""Read-only source-visible depth observations, separate from native kinematics."""
from __future__ import annotations
import hashlib
import json
import re


def require(value, message):
    if not value:
        raise ValueError(message)


class CuratedObservedSurfaces:
    def __init__(self, catalog):
        from curated_twin_catalog import CuratedTwinCatalog
        require(isinstance(catalog, CuratedTwinCatalog), 'Observed surfaces require an explicitly curated catalog')
        self.catalog = catalog

    def _inventory(self, twin_id):
        record, _ = self.catalog._record(twin_id)
        declaration = record.get('observed_surfaces')
        if declaration is None:
            return None, {}
        document = json.loads(self.catalog._bytes(declaration, maximum=4*1024*1024))
        require(document.get('format') == 'interaction-twin-source-visible-flexure-series' and
                document.get('version') == 1 and document.get('kind') == 'source_observed_monocular_depth_prior',
                'Unknown source-surface observation type')
        require(document.get('source_program') == record['interaction'] and
                document.get('source', {}).get('sha256') == record['source']['sha256'],
                'Observed surfaces belong to another curated program or source')
        import source_visible_flexure_contract_v1 as F
        validated = F.validate(document, lambda ref: self.catalog._bytes(ref, maximum=16*1024*1024))
        expected = [c['asset'] for c in validated['source_scene']['components']
                    if c['id'] in document['replacement_component_ids']]
        require(sorted(document.get('closed_native_assets', []), key=lambda r: r['path']) ==
                sorted(expected, key=lambda r: r['path']), 'Original surface assets differ from the exact replaced components')
        registry, _ = self.catalog._json('showcase/public/aligned-scenes.json')
        scene = next((row for row in registry['scenes'] if row['source_sha256'] == record['source']['sha256']), None)
        require(scene and scene.get('observed_surfaces') == declaration and
                scene['manifest'] == document['source_scene'], 'Observed surfaces differ from the current curated scene')
        for key in ['physical_accuracy_verified', 'simulation_ready', 'manual_geometry_or_poses_used', 'native_SAM_contract_compatible']:
            require(document.get(key) is False, 'Observed surfaces must retain their unverified, separate scope')
        rows = document.get('samples')
        require(isinstance(rows, list) and 1 <= len(rows) <= 2000, 'Bounded source surface observations required')
        by_path = {}

        def add(value, role):
            require(isinstance(value, dict), 'Exact observed-surface artifact required')
            ref = {k: value[k] for k in ['path', 'sha256', 'bytes']}
            maximum = 16*1024*1024 if ref['path'].endswith('.glb') else 8*1024*1024
            self.catalog._checked(ref, maximum=maximum)
            existing = by_path.get(ref['path'])
            require(not existing or existing['reference'] == ref, 'Conflicting observed-surface artifact')
            by_path.setdefault(ref['path'], {'reference': ref, 'roles': set()})['roles'].add(role)
            require(len(by_path) <= 12000, 'Observed-surface artifact inventory exceeds its budget')

        add(declaration, 'descriptive_surface_program')
        for key in ['source_scene', 'source_program', 'protocol']:
            add(document[key], key)
        presentation = None
        if record.get('observed_surface_presentation'):
            pointer = record['observed_surface_presentation']
            require(scene.get('observed_surface_presentation') == pointer, 'Observed presentation differs from the current scene')
            policy = json.loads(self.catalog._bytes(pointer, maximum=4*1024*1024))
            from observed_surface_presentation_v1 import validate as validate_presentation
            validate_presentation(policy, document, declaration)
            add(pointer, 'editorial_source_sample_presentation')
            presentation = {'artifact': pointer, 'document': policy}
        for ref in document.get('closed_native_assets', []):
            add(ref, 'unchanged_native_asset')
        previous = -1
        for row in rows:
            require(type(row.get('source_sample_index')) is int and row['source_sample_index'] > previous and
                    row.get('status') in ['unknown', 'source_observed_prior'], 'Ordered source surface samples required')
            previous = row['source_sample_index']
            if row['status'] == 'unknown':
                require(not row.get('asset'), 'Unknown surface may not expose a mesh')
                continue
            require(row.get('source_visible_only') is True and row.get('physical_accuracy_verified') is False,
                    'Observed surface must retain its source-visible uncertainty')
            for key in ['asset', 'source_RGB', 'source_pixel_map', 'provenance']:
                add(row[key], key)
            for ref in row['source_masks']:
                add(ref, 'source_mask')
        # Conservative contour revisions carry a typed cached-depth ancestry.
        # Only refs explicitly returned by the checked contract enter transport;
        # implementation/model fingerprints and arbitrary nested text do not.
        for pointer in validated['artifacts']:
            if pointer['path'] not in by_path:
                add(pointer, 'source_surface_proof_ancestry')
        artifacts = {}
        for item in by_path.values():
            ref = item['reference']
            identity = {'twin_id': twin_id, 'declaration': declaration, 'artifact': ref}
            key = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            mime = 'model/gltf-binary' if ref['path'].endswith('.glb') else 'application/json' if ref['path'].endswith('.json') else 'image/jpeg' if ref['path'].endswith('.jpg') else 'image/png' if ref['path'].endswith('.png') else 'application/octet-stream'
            artifacts[key] = {**ref, 'artifact_id': key, 'mime_type': mime, 'roles': sorted(item['roles']),
                              'uri': f'twin://{twin_id}/observed-surfaces/{key}'}
        return {'artifact': declaration, 'document': document, **({'presentation': presentation} if presentation else {})}, artifacts

    def inspect(self, twin_id):
        found, artifacts = self._inventory(twin_id)
        return {'status': 'source_visible_depth_observations' if found else 'unavailable', 'twin_id': twin_id,
                **(found or {}), 'artifacts': list(artifacts.values()),
                'access': 'owner_curated_public_snapshot', 'new_inference': False,
                'hidden_surfaces': 'absent', 'native_kinematics_replaced': False,
                'descriptive_text_is_untrusted_data': True, 'physical_accuracy_verified': False, 'simulation_ready': False}

    def artifact(self, twin_id, artifact_id):
        require(isinstance(artifact_id, str) and re.fullmatch(r'[a-f0-9]{64}', artifact_id), 'Exact listed observed-surface artifact ID required')
        found, artifacts = self._inventory(twin_id)
        require(artifact_id in artifacts, 'Unknown or stale source-surface artifact for this twin')
        row = artifacts[artifact_id]
        raw = self.catalog._bytes({k: row[k] for k in ['path', 'sha256', 'bytes']}, maximum=16*1024*1024)
        current, inventory = self._inventory(twin_id)
        require(current == found and inventory.get(artifact_id) == row,
                'Observed-surface approval, source or presentation changed; list again')
        return raw
