"""Immutable, exact-input component inventory for a compiled interaction.

This companion does not add geometry to an old program or discover a convenient
model from another review. Native inputs must be explicitly compiled inputs.
Missing components stay pending and the historical evidence stays unchanged.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
import uuid
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
FORMAT = 'interaction-twin-part-ledger'
VERSION = 1
SHA = re.compile(r'[a-f0-9]{64}\Z')
ID = re.compile(r'[a-z][a-z0-9_]{0,79}\Z')
MAX_JSON = 24 * 1024 * 1024


def require(ok, message):
    if not ok:
        raise ValueError(message)


def local(path, root=ROOT, *, exists=True, data_only=True):
    root = Path(root).resolve()
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    require('..' not in path.parts and path.is_relative_to(root), 'Part ledger path escapes project')
    if data_only:
        require(path.is_relative_to(root / 'data'), 'Part ledger evidence must stay in data')
    for item in [path, *path.parents]:
        if item == root:
            break
        require(not item.is_symlink(), 'Part ledger symlink refused')
    require(path.resolve().is_relative_to(root), 'Part ledger resolved path escapes project')
    if exists:
        require(path.is_file(), 'Part ledger input missing')
    return path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def ref(path, root=ROOT):
    root = Path(root).resolve()
    path = local(path, root)
    return {'path': path.relative_to(root).as_posix(), 'sha256': sha(path), 'bytes': path.stat().st_size}


def checked(value, root=ROOT):
    require(isinstance(value, dict) and set(value) == {'path', 'sha256', 'bytes'}, 'Exact part reference required')
    require(isinstance(value['sha256'], str) and SHA.fullmatch(value['sha256']) and
            type(value['bytes']) is int and value['bytes'] > 0, 'Invalid part reference')
    path = local(value['path'], root)
    require(path.stat().st_size == value['bytes'] and sha(path) == value['sha256'], 'Part ledger input bytes changed')
    return path


def read(path):
    path = Path(path)
    require(path.stat().st_size <= MAX_JSON, 'Part ledger JSON budget exceeded')
    result = json.loads(path.read_text())
    require(isinstance(result, dict), 'Part ledger JSON object required')
    return result


def _input_refs(program, root):
    result = {}
    for row in program.get('inputs', []):
        if row.get('id') in {'mask_tracks', 'mask_quality', 'annotations'} or row.get('kind') == 'learned_asset_metadata':
            require(row['id'] not in result, 'Repeated compiled part input')
            path = local(row['path'], root, data_only=row.get('id') != 'annotations')
            require(sha(path) == row['sha256'], 'Compiled part input changed')
            result[row['id']] = ref(path, root) if path.is_relative_to(root / 'data') else {
                'path': path.relative_to(root).as_posix(), 'sha256': sha(path), 'bytes': path.stat().st_size}
    require('mask_tracks' in result and 'mask_quality' in result, 'Source-bound SAM track/quality inputs required')
    tracks, quality = [local(result[k]['path'], root) for k in ['mask_tracks', 'mask_quality']]
    require(tracks.name == 'tracks.json' and quality.name == 'frames.json' and tracks.parent == quality.parent,
            'Part ledger SAM inputs belong to different attempts')
    return result


def _native(metadata_ref, program, root):
    metadata_path = checked(metadata_ref, root)
    model = read(metadata_path)
    source = model.get('source', {})
    require(model.get('backend') == 'Meta SAM 3D Objects' and model.get('sam3d_output') is True and
            model.get('learned_inference_executed') is True and model.get('manual_geometry_fallback') is False and
            model.get('manual_pose_used') is False and source.get('video_sha256') == program['source']['sha256'],
            'Part ledger requires genuine source-matched native geometry')
    tid = model.get('object_id')
    require(isinstance(tid, str) and ID.fullmatch(tid), 'Invalid native component identity')
    asset = metadata_path.parent / 'asset.glb'
    rows = [r for r in model.get('artifact_files', []) if r.get('file') == 'asset.glb']
    require(len(rows) == 1, 'Exact original native asset record required')
    asset_ref = ref(asset, root)
    require(asset_ref['sha256'] == rows[0]['sha256'] and asset_ref['bytes'] == rows[0]['bytes'], 'Native part bytes changed')
    matches = [a for a in program.get('assets', []) if a['track_id'] == tid]
    require(len(matches) == 1 and matches[0]['path'] == asset_ref['path'] and matches[0]['sha256'] == asset_ref['sha256'],
            'Native part is not the exact compiled program asset')
    source_image = local(metadata_path.parent / source['image_path'], root)
    source_mask = local(metadata_path.parent / source['mask_path'], root)
    require(sha(source_image) == source['image_sha256'] and sha(source_mask) == source['mask_sha256'], 'Native source pixels changed')
    return tid, {'metadata': metadata_ref, 'asset': asset_ref, 'program_asset_id': matches[0]['id'],
                 'source_image': ref(source_image, root), 'source_mask': ref(source_mask, root),
                 'source_sample_index': source.get('source_sample_index', source.get('sample_index')),
                 'source_timestamp_s': source['timestamp_s'], 'manual_geometry_used': False, 'manual_pose_used': False}


def document(program_path, *, selection_receipt=None, reconstruction_receipt=None, root=ROOT):
    root = Path(root).resolve()
    path = local(program_path, root)
    program = read(path)
    import stage5
    stage5.validate_interaction(program, root=root)
    inputs = _input_refs(program, root)
    tracks = read(local(inputs['mask_tracks']['path'], root))
    require(isinstance(tracks.get('tracks'), list) and len(tracks['tracks']) <= 64, 'Bounded SAM part inventory required')
    rows = {r['track_id']: r for r in tracks['tracks']}
    require(len(rows) == len(tracks['tracks']), 'Duplicate SAM component ID')
    identities = [t['id'] for t in program['tracklets']]
    require(len(set(identities)) == len(identities) and set(rows) == set(identities), 'Compiler and SAM component IDs disagree')
    models = {}
    for row in program['inputs']:
        if row.get('kind') != 'learned_asset_metadata':
            continue
        tid, native = _native(inputs[row['id']], program, root)
        require(tid in identities and tid not in models, 'Unknown or duplicate compiled native part')
        models[tid] = native
    # All native program assets must be represented, including pending/no-view
    # parts. No convenient model can enter through a saved review directory.
    native_ids = {a['track_id'] for a in program['assets'] if a.get('backend') == 'Meta SAM 3D Objects'}
    require(native_ids == set(models), 'Compiled native assets and metadata inventory disagree')
    selection, reconstruction, extra_inputs = {}, {}, []
    if selection_receipt:
        selection_path = local(selection_receipt, root); raw = read(selection_path)
        require(isinstance(raw.get('decisions'), list), 'Component selection decisions required')
        selection = {r['track_id']: r for r in raw['decisions']}
        require(len(selection) == len(raw['decisions']), 'Duplicate selection component ID')
        require(set(selection) == set(identities), 'Selection ledger and compiler IDs disagree')
        extra_inputs.append(ref(selection_path, root))
    if reconstruction_receipt:
        reconstruction_path = local(reconstruction_receipt, root); raw = read(reconstruction_path)
        reconstruction = {r['track_id']: r for r in raw['components']}
        require(len(reconstruction) == len(raw['components']), 'Duplicate reconstruction component ID')
        require(set(reconstruction) <= set(identities), 'Reconstruction ledger contains another component')
        extra_inputs.append(ref(reconstruction_path, root))
    parts = []
    for track in program['tracklets']:
        tid = track['id']; native = models.get(tid); selected = selection.get(tid)
        relations = [r['id'] for r in program.get('relationships', []) if tid in {r.get('subject_track_id'), r.get('object_track_id'), r.get('parent_id'), r.get('child_id')}]
        statuses = Counter(f.get('status', 'unknown') for f in rows[tid].get('frames', []))
        parts.append({'id': tid, 'label': track['label'], 'physical_identity_verified': False,
                      'mask_track_id': tid, 'mask_status_counts': dict(sorted(statuses.items())),
                      'selection_status': selected.get('status') if selected else 'not_supplied',
                      'reconstruction_status': reconstruction.get(tid, {}).get('status', 'not_supplied'),
                      'geometry_status': 'native_generated' if native else 'pending', 'native': native,
                      'pending_reason': None if native else (selected.get('status', 'no_compiled_native_geometry') if selected else 'no_compiled_native_geometry'),
                      'relationship_ids': relations, 'physical_scale_known': False})
    return {'format': FORMAT, 'version': VERSION, 'evidence_program': ref(path, root),
            'source_sha256': program['source']['sha256'], 'original_source_sha256': program['source']['original_sha256'],
            'segmentation': {'tracks': inputs['mask_tracks'], 'quality': inputs['mask_quality']},
            'parts': parts, 'inputs': extra_inputs,
            'summary': {'components': len(parts), 'native_generated': len(models), 'pending': len(parts)-len(models),
                        'all_native_assets_compiled': True, 'physical_identity_merged': False},
            'implementation': {'path': 'part_ledger.py', 'sha256': sha(__file__)},
            'physical_accuracy_verified': False, 'simulation_ready': False, 'training_performed': False}


def validate(value, *, program_path=None, root=ROOT):
    root = Path(root).resolve()
    require(value.get('format') == FORMAT and value.get('version') == VERSION, 'Unsupported part ledger')
    original = checked(value['evidence_program'], root)
    if program_path is not None:
        require(original == local(program_path, root), 'Part ledger belongs to a different program')
    refs = value.get('inputs', [])
    require(isinstance(refs, list) and len(refs) <= 2, 'Invalid ledger receipt count')
    selection = reconstruction = None
    for item in refs:
        candidate = checked(item, root); doc = read(candidate)
        if 'decisions' in doc:
            require(selection is None, 'Repeated selection receipt'); selection = candidate
        elif 'components' in doc:
            require(reconstruction is None, 'Repeated reconstruction receipt'); reconstruction = candidate
        else:
            raise ValueError('Unknown part ledger receipt')
    expected = document(original, selection_receipt=selection, reconstruction_receipt=reconstruction, root=root)
    require(value == expected, 'Part ledger content or implementation changed')
    return value


def build(program_path, output=None, *, selection_receipt=None, reconstruction_receipt=None, root=ROOT):
    root = Path(root).resolve()
    value = document(program_path, selection_receipt=selection_receipt, reconstruction_receipt=reconstruction_receipt, root=root)
    identity = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    path = local(output or root/'data/part-ledgers'/identity/'part-ledger.json', root, exists=False)
    raw = (json.dumps(value, indent=2, allow_nan=False)+'\n').encode()
    if path.exists():
        require(path.read_bytes() == raw, 'Immutable part ledger already differs')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.parent / ('.part-ledger-'+uuid.uuid4().hex+'.tmp')
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(descriptor, 'wb') as stream:
                stream.write(raw); stream.flush(); os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                require(not path.is_symlink() and path.read_bytes() == raw, 'Immutable part ledger already differs')
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)
    validate(read(path), program_path=program_path, root=root)
    return ref(path, root)


def metadata_paths(ledger_path, *, program_path=None, root=ROOT):
    value = validate(read(local(ledger_path, root)), program_path=program_path, root=root)
    return [checked(p['native']['metadata'], root) for p in value['parts'] if p['native'] is not None]
