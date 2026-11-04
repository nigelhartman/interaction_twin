"""Discover source-bound SAM 3D previews without promoting their motion or physics.

Only validated saved native imports and their unchanged GLBs are exposed. This
index is derived from footage artifacts, never a hand-maintained list of objects.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import re

SHA = re.compile(r'^[a-f0-9]{64}$')
ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,180}$')


def _file(root, relative, *, maximum=128 * 1024 * 1024):
    if not isinstance(relative, str) or not relative.startswith('data/'):
        raise ValueError('Native artifacts must remain in data')
    if not re.fullmatch(r'[A-Za-z0-9_./-]+', relative):
        raise ValueError('Invalid native artifact path')
    path = root
    for part in relative.split('/'):
        if not part or part.startswith('.'):
            raise ValueError('Hidden or traversing native path')
        path = path / part
        if path.is_symlink():
            raise ValueError('Symlink native artifact')
    if not path.is_file() or not 0 < path.stat().st_size <= maximum:
        raise ValueError('Native artifact absent or outside budget')
    return path


def _ref(root, relative, expected=None, maximum=128 * 1024 * 1024):
    path = _file(root, relative, maximum=maximum)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    sha = h.hexdigest()
    if expected is not None and sha != expected:
        raise ValueError('Native artifact digest differs')
    return {'path': relative, 'sha256': sha, 'bytes': path.stat().st_size}


def _json(path):
    value = json.loads(path.read_text(), parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
    if not isinstance(value, dict):
        raise ValueError('Native metadata must be an object')
    return value


def discover_reviews(root):
    """Latest valid display transport per exact video/component/model identity."""
    root = Path(root).resolve()
    found = {}
    directories = sorted((root / 'data').glob('sam3d*/*/review-manifest.json'))
    for path in directories:
        try:
            relative = path.relative_to(root).as_posix()
            manifest = _json(_file(root, relative, maximum=2 * 1024 * 1024))
            if manifest.get('version') not in (1, 2) or manifest.get('fixture') is not False:
                continue
            kind = manifest.get('format')
            if kind == 'sam3d-initial-native-review':
                if manifest.get('native_only') is not True:
                    continue
                source, component = manifest['source'], manifest['component_id']
                expected_glb = manifest['native_glb_sha256']
            elif kind == 'sam3d-native-review':
                source = manifest['binding']
                component = source['track_id']
                expected_glb = manifest['candidate']['asset_sha256']
            else:
                continue
            video = source['video_sha256']
            if not SHA.fullmatch(video) or not ID.fullmatch(component):
                continue
            scope = manifest['scope']
            if any(scope.get(k) is not False for k in ['physical_accuracy_verified', 'metric_scale_verified', 'assembly_verified', 'contact_verified']):
                continue
            if any(scope.get(k) is True for k in ['selected', 'catalog_changed', 'preferred_model_changed', 'source_video_placement_established', 'source_video_alignment_established']):
                continue
            receipt_ref = manifest['import_receipt']
            receipt_file = _ref(root, receipt_ref['path'], receipt_ref['sha256'], maximum=2 * 1024 * 1024)
            receipt = _json(root / receipt_file['path'])
            if receipt.get('format') not in {'sam3d-native-milestone-readiness', 'sam3d-initial-native-readiness'} or receipt.get('physical_accuracy_verified') is not False:
                continue
            imported = {f['path']: f for f in receipt['import_files']}
            model_entry = imported['metadata.json']
            model_path = str(Path(receipt_file['path']).parent / 'metadata.json')
            model_ref = _ref(root, model_path, model_entry['sha256'], maximum=2 * 1024 * 1024)
            model = _json(root / model_ref['path'])
            if (model.get('learned_inference_executed') is not True or model.get('object_id') != component
                    or model.get('source', {}).get('video_sha256') != video
                    or model.get('physical_accuracy_verified') not in (None, False)):
                continue
            if (model.get('backend') != 'Meta SAM 3D Objects' or model.get('sam3d_output') is not True
                    or model.get('geometry_reconstructed_by_model') is not True
                    or model.get('manual_geometry_fallback') is not False or model.get('manual_pose_used') is not False
                    or model.get('fixture') not in (None, False)):
                continue
            if (type(source.get('sample_index')) is not int or source['sample_index'] < 1
                    or type(source.get('timestamp_s')) not in (int, float) or not math.isfinite(source['timestamp_s'])
                    or source['timestamp_s'] < 0 or source.get('time_domain') != 'source_video_presentation_seconds'
                    or any(source.get(k) != model['source'].get(k) for k in ['sample_index', 'timestamp_s', 'time_domain'])):
                continue
            files = {f['path']: f for f in manifest['files']}
            parent = path.parent.relative_to(root)
            def declared(name):
                entry = files[name]
                ref = _ref(root, (parent / name).as_posix(), entry['sha256'])
                if ref['bytes'] != entry['bytes']:
                    raise ValueError('Native declared length differs')
                return ref
            glb, image, mask = declared('candidate.glb'), declared('source-rgb.png'), declared('source-mask.png')
            if glb['sha256'] != expected_glb:
                continue
            if imported['asset.glb']['sha256'] != glb['sha256'] or model['source']['image_sha256'] != image['sha256'] or model['source']['mask_sha256'] != mask['sha256']:
                continue
            original = _ref(root, str(Path(receipt_file['path']).parent / 'asset.glb'), glb['sha256'], maximum=16 * 1024 * 1024)
            if original['bytes'] != glb['bytes'] or imported['asset.glb']['bytes'] != glb['bytes']:
                continue
            row = {'component_id': component, 'label': manifest.get('title', component.replace('_', ' ').capitalize()),
                   'source_sha256': video, 'backend': 'Meta SAM 3D Objects',
                   'review_page': _ref(root, (parent / 'index.html').as_posix(), maximum=1024 * 1024),
                   'source_image': image, 'source_mask': mask, 'native_glb': glb,
                   'review_manifest': _ref(root, relative, maximum=2 * 1024 * 1024),
                   'source_timestamp_s': source['timestamp_s'], 'source_sample_index': source['sample_index'],
                   'imported_at_utc': receipt.get('created_at_utc', receipt.get('created_at')),
                   'display_transport_version': manifest['version'],
                   'selected': False, 'physical_accuracy_verified': False,
                   'source_video_alignment_verified': False, 'simulation_ready': False}
            # V2 display transports supersede older pages for the same model.
            key = (video, component, glb['sha256'])
            found[key] = row
        except (OSError, ValueError, KeyError, TypeError):
            # Invalid optional previews cannot hide the valid semantic program.
            continue
    return list(found.values())


def match_reviews(reviews, record):
    """Prefer the exact program asset, then the newest verified import date."""
    ids = {c['id'] for c in record.get('components', [])}
    video = record.get('source', {}).get('sha256')
    candidates = [r for r in reviews if r['source_sha256'] == video and r['component_id'] in ids]
    assets = {c['id']: c.get('asset', {}).get('sha256') for c in record.get('components', []) if c.get('asset')}
    assets.update({a['track_id']: a['sha256'] for a in record.get('assets', [])})
    def priority(row):
        try:
            date = datetime.fromisoformat(row['imported_at_utc'].replace('Z', '+00:00'))
            timestamp = date.timestamp() if date.tzinfo is not None else float('-inf')
        except (KeyError, TypeError, ValueError, AttributeError):
            timestamp = float('-inf')
        return (assets.get(row['component_id']) == row['native_glb']['sha256'], timestamp,
                row.get('display_transport_version', 0), row['review_page']['path'])
    latest = {}
    for row in sorted(candidates, key=priority):
        latest[row['component_id']] = row
    return sorted(latest.values(), key=lambda r: r['component_id'])


def reviews_for_record(root, record):
    return match_reviews(discover_reviews(root), record)
