"""Portable, evidence-bound exports of an observed open surface.

This copies saved observations. It performs no reconstruction, smoothing,
completion, rescaling, inference or physics preparation. glTF uses float32
positions and top-left texture coordinates; those representation conversions
are reported, while topology and original JPEG bytes remain exact.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import struct
import zipfile

import numpy as np
from PIL import Image

FORMAT = 'observed-surface-exports'
VERSION = 1
MAX_FILE = 64 * 1024 * 1024
MAX_TOTAL = 200 * 1024 * 1024
MAX_FILES = 256
ID = re.compile(r'[a-z][a-z0-9-]{0,95}\Z')
GLTF_SPEC = 'https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()


def safe_name(name):
    if not isinstance(name, str) or not name or not re.fullmatch(r'[A-Za-z0-9_./-]+', name):
        raise ValueError('Unsafe portable member name')
    parts = name.split('/')
    if any(not p or p in ('.', '..') or p.startswith('.') for p in parts):
        raise ValueError('Unsafe portable member name')
    if str(PurePosixPath(name)) != name:
        raise ValueError('Unsafe portable member name')
    return name


def local_path(root, path_or_url):
    """Only explicit project data paths, with neither traversal nor symlinks."""
    value = str(path_or_url)
    if value.startswith('/files/'):
        value = value[7:]
    name = safe_name(value)
    if name.split('/')[0] != 'data':
        raise ValueError('Only project data evidence may be exported')
    root = Path(root).resolve()
    path = root / name
    if not path.resolve().is_relative_to(root):
        raise ValueError('Evidence path escapes project root')
    current = root
    for part in PurePosixPath(name).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError('Symlink evidence is unsupported')
    return path


def read_checked(path, expected=None):
    if path.stat().st_size > MAX_FILE:
        raise ValueError('Evidence exceeds file budget')
    raw = path.read_bytes()
    if expected is not None and digest(raw) != expected:
        raise ValueError(f'Evidence hash mismatch: {path.name}')
    return raw


def make_glb(vertices, triangles, display_uv, texture, *, name, extras):
    """Encode a single one-sided, unlit open patch. Never recompute normals."""
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles)
    uv = np.asarray(display_uv, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError('Expected finite nonempty Nx3 geometry')
    if uv.shape != (len(vertices), 2) or not np.isfinite(uv).all() or np.any((uv < 0) | (uv > 1)):
        raise ValueError('Expected finite normalized source UVs')
    if triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles) or triangles.dtype.kind not in 'iu':
        raise ValueError('Expected nonempty integer triangle indices')
    if triangles.min() < 0 or triangles.max() >= len(vertices):
        raise ValueError('Triangle index outside observed vertices')
    with Image.open(io.BytesIO(texture)) as image:
        if image.format != 'JPEG' or image.width * image.height > 16_777_216:
            raise ValueError('Expected bounded original JPEG texture')
        image.verify()
    positions = vertices.astype('<f4')
    gltf_uv = np.c_[uv[:, 0], 1 - uv[:, 1]].astype('<f4')
    if not np.isfinite(positions).all():
        raise ValueError('Position cannot be represented as float32')
    index_type = 5123 if len(vertices) <= 65535 else 5125
    indices = triangles.astype('<u2' if index_type == 5123 else '<u4')
    binary, views = bytearray(), []

    def append(raw, target=None):
        while len(binary) % 4:
            binary.append(0)
        item = {'buffer': 0, 'byteOffset': len(binary), 'byteLength': len(raw)}
        if target is not None:
            item['target'] = target
        views.append(item)
        binary.extend(raw)
        return len(views) - 1

    position_view = append(positions.tobytes(), 34962)
    uv_view = append(gltf_uv.tobytes(), 34962)
    index_view = append(indices.tobytes(), 34963)
    image_view = append(texture)
    document = {
        'asset': {'version': '2.0', 'generator': 'Interaction Twin observed-patch exporter v1', 'extras': extras},
        'scene': 0, 'scenes': [{'nodes': [0]}], 'nodes': [{'name': name, 'mesh': 0}],
        'buffers': [{'byteLength': len(binary)}], 'bufferViews': views,
        'accessors': [
            {'bufferView': position_view, 'componentType': 5126, 'count': len(vertices), 'type': 'VEC3',
             'min': positions.min(axis=0).tolist(), 'max': positions.max(axis=0).tolist()},
            {'bufferView': uv_view, 'componentType': 5126, 'count': len(vertices), 'type': 'VEC2'},
            {'bufferView': index_view, 'componentType': index_type, 'count': int(indices.size), 'type': 'SCALAR'}],
        'meshes': [{'name': name, 'primitives': [{'attributes': {'POSITION': 0, 'TEXCOORD_0': 1},
                    'indices': 2, 'material': 0, 'mode': 4}]}],
        'images': [{'name': 'Exact original source frame', 'bufferView': image_view, 'mimeType': 'image/jpeg'}],
        'samplers': [{'magFilter': 9729, 'minFilter': 9729, 'wrapS': 33071, 'wrapT': 33071}],
        'textures': [{'source': 0, 'sampler': 0}],
        'materials': [{'name': 'Observed source appearance, unlit', 'doubleSided': False, 'alphaMode': 'OPAQUE',
            'pbrMetallicRoughness': {'baseColorTexture': {'index': 0, 'texCoord': 0},
                'baseColorFactor': [1, 1, 1, 1], 'metallicFactor': 0, 'roughnessFactor': 1},
            'extensions': {'KHR_materials_unlit': {}}}],
        'extensionsUsed': ['KHR_materials_unlit'], 'extensionsRequired': ['KHR_materials_unlit']}
    encoded = json.dumps(document, separators=(',', ':'), allow_nan=False).encode()
    encoded += b' ' * (-len(encoded) % 4)
    binary.extend(b'\x00' * (-len(binary) % 4))
    raw = (struct.pack('<III', 0x46546c67, 2, 12 + 8 + len(encoded) + 8 + len(binary)) +
           struct.pack('<II', len(encoded), 0x4e4f534a) + encoded +
           struct.pack('<II', len(binary), 0x004e4942) + binary)
    return raw, {'position_max_float32_error': float(np.max(np.abs(vertices - positions))),
                 'uv_max_float32_error': float(np.max(np.abs(np.c_[uv[:, 0], 1 - uv[:, 1]] - gltf_uv))),
                 'indices_exact': True, 'original_texture_bytes_exact': True,
                 'position_transform': 'none', 'uv_transform': 'gltf_v = 1 - display_v',
                 'normals_added': False, 'colour_attributes_added': False, 'smoothing': False}


def _load_evidence(root, source_run):
    run = local_path(root, source_run)
    evidence = {}

    def record(path, expected=None):
        path = Path(path)
        name = path.relative_to(Path(root).resolve()).as_posix()
        raw = read_checked(path, expected)
        evidence['evidence/' + safe_name(name)] = raw
        return raw

    provenance = json.loads(record(run / 'display-provenance.json'))
    viewer_raw = record(run / 'viewer-display.json', provenance['display_viewer_sha256'])
    viewer = json.loads(viewer_raw)
    raw_viewer = json.loads(record(run / 'viewer.json', provenance['source_viewer_sha256']))
    protocol_raw = record(run / 'protocol.json', provenance['source_protocol_sha256'])
    protocol = json.loads(protocol_raw)
    record(run / 'diagnostic-disparity.npz', provenance['source_disparity_sha256'])
    record(run / 'display-implementation.py', provenance['implementation_sha256'])
    record(run / 'implementation.py', protocol['implementation_sha256'])
    for name in ['visible-patch.json', 'surface.json', 'points.ply', 'disparity-preview.png']:
        record(run / name)
    if viewer.get('kind') != 'observed_surface' or viewer.get('id') != run.name:
        raise ValueError('Expected declared observed-surface result identity')
    surface = viewer['surface']
    if surface.get('hole_filling') is not False or surface.get('watertight') is not False or surface.get('thickness_known') is not False:
        raise ValueError('Only explicitly incomplete open observed patches may be exported')
    for key in ['vertices', 'triangles', 'colors']:
        if surface[key] != raw_viewer['surface'][key]:
            raise ValueError('Display derivative changed original geometry or colours')
    if viewer['points'] != raw_viewer['points']:
        raise ValueError('Display derivative changed saved point evidence')
    if viewer['display_derivation'] != {k: v for k, v in provenance.items() if k != 'display_viewer_sha256'}:
        raise ValueError('Display derivative provenance disagrees')
    for item in protocol['input_views']:
        if item['split'] != 'train':
            raise ValueError('Surface evidence unexpectedly contains nontraining inputs')
        record(local_path(root, item['image_url']), item['image_sha256'])
        record(local_path(root, item['mask_url']), item['mask_sha256'])
    sparse = local_path(root, protocol['sparse_input'])
    for name, expected in protocol['sparse_files'].items():
        record(sparse / safe_name(name), expected)
    sparse_protocol = json.loads((sparse / 'protocol.json').read_text())
    record(sparse / 'implementation.py', sparse_protocol['implementation_sha256'])
    if sparse_protocol.get('correspondence_provider'):
        provider = sparse_protocol['correspondence_provider']
        record(sparse / 'correspondence-implementation.py', provider['provider_implementation_sha256'])
        record(sparse / 'cotracker-predictor.py', provider['predictor_implementation_sha256'])
        audit = json.loads(record(sparse / 'correspondence-audit.json'))
        record(sparse / 'tracking.json', audit['tracking_json_sha256'])
        record(sparse / 'cotracker-model-provenance.json')
        for path in sorted(sparse.glob('*.npz')):
            record(path)
        if (sparse / 'automatic-queries.json').is_file():
            record(sparse / 'automatic-queries.json')
    texture_path = local_path(root, surface['texture_url'])
    texture = record(texture_path, surface['texture_sha256'])
    return viewer, protocol, evidence, texture


def _verify_surface_projection(viewer, texture):
    surface = viewer['surface']
    vertices = np.asarray(surface['vertices'], float)
    uv = np.asarray(surface['uv'], float)
    triangles = np.asarray(surface['triangles'], int)
    points = viewer['points']
    if len(points) != len(vertices) or not np.array_equal(vertices, [p['xyz'] for p in points]):
        raise ValueError('Surface vertex order differs from saved observations')
    frame = next(f for f in viewer['frames'] if f['id'] == surface['source_frame_id'])
    camera = next(c for c in viewer['cameras'] if c['frame_id'] == frame['id'])
    R, t, K = (np.asarray(camera[k], float) for k in ['R', 't', 'K'])
    cam = vertices @ R.T + t
    if np.any(cam[:, 2] <= 0):
        raise ValueError('Observed surface is behind its source camera')
    projected = cam @ K.T
    pixels = projected[:, :2] / projected[:, 2:]
    expected = np.c_[(pixels[:, 0] + .5) / frame['width'], 1 - (pixels[:, 1] + .5) / frame['height']]
    uv_error = float(np.max(np.abs(expected - uv)))
    if uv_error > 1e-10:
        raise ValueError('Source UV does not match exact saved camera projection')
    with Image.open(io.BytesIO(texture)) as image:
        if image.size != (frame['width'], frame['height']):
            raise ValueError('Source texture dimensions disagree with camera frame')
    tri = vertices[triangles]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    camera_centre = -R.T @ t
    facing = np.sum(normal * (camera_centre - tri.mean(axis=1)), axis=1)
    if np.any(facing <= 0):
        raise ValueError('Saved triangle winding is not one-sided towards source camera')
    if not all(p.get('validated') is False and p.get('heldout_scored_projection_support') == [] for p in points):
        raise ValueError('Unexpected independent validation claim in observed surface')
    return {'uv_projection_max_normalized_error': uv_error, 'triangles_source_facing': int(len(triangles)),
            'source_frame_id': frame['id'], 'source_timestamp_s': frame['timestamp_s'],
            'source_camera': camera, 'source_dimensions': [frame['width'], frame['height']]}


def validate_package(raw):
    """Validate a bounded portable ZIP without extracting or executing anything."""
    if len(raw) > MAX_TOTAL:
        raise ValueError('Package exceeds archive budget')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        names = [safe_name(info.filename) for info in entries]
        if len(entries) > MAX_FILES or len(names) != len(set(names)):
            raise ValueError('Duplicate or excessive package members')
        if sum(i.file_size for i in entries) > MAX_TOTAL or any(i.file_size > MAX_FILE for i in entries):
            raise ValueError('Package exceeds decompression budget')
        if any(i.flag_bits & 1 or i.is_dir() or ((i.external_attr >> 16) & 0o170000) == 0o120000 for i in entries):
            raise ValueError('Unsupported encrypted, directory or symlink package entry')
        manifest = json.loads(archive.read('package-manifest.json'))
        if manifest.get('format') != 'observed-surface-portable' or manifest.get('version') != VERSION:
            raise ValueError('Unsupported portable observed-surface package')
        declared = manifest['files']
        if not {'observed-patch.glb', 'metadata.json', 'README.txt', 'export-implementation.py'} <= set(declared):
            raise ValueError('Required portable members are absent')
        if set(declared) != set(names) - {'package-manifest.json'}:
            raise ValueError('Package members differ from manifest')
        for name, entry in declared.items():
            safe_name(name)
            value = archive.read(name)
            if len(value) != entry['bytes'] or digest(value) != entry['sha256']:
                raise ValueError('Package member hash mismatch')
        return {'files': len(names), 'uncompressed_bytes': sum(i.file_size for i in entries),
                'all_member_hashes_verified': True, 'archive_sha256': digest(raw)}


def export_run(root, source_run, output):
    root, output = Path(root).resolve(), Path(output)
    output = output if output.is_absolute() else root / output
    if not output.resolve().is_relative_to(root / 'data'):
        raise ValueError('Exports must remain inside project data/')
    current = output
    while current != root:
        if current.is_symlink():
            raise ValueError('Symlink export destinations are unsupported')
        current = current.parent
    if output.exists():
        raise ValueError('Export output already exists; preserve prior exports')
    viewer, protocol, evidence, texture = _load_evidence(root, source_run)
    identifier = viewer['id']
    if not ID.fullmatch(identifier):
        raise ValueError('Invalid export identity')
    projection = _verify_surface_projection(viewer, texture)
    surface = viewer['surface']
    confidence = {'shape_accuracy': 'unmeasured', 'camera_intrinsics': 'ambiguous_estimate' if 'refined' in identifier else 'uncalibrated_assumption',
                  'physical_scale': 'unknown', 'completeness': 'observed_open_partial_patch',
                  'mask_quality': 'model_propagated_unaudited', 'support': 'training_photometric_consistency_only'}
    extra = {'source_run_id': identifier, 'status': viewer['status'], 'units': 'arbitrary reconstruction gauge; not measured metres',
             'physical_scale_known': False, 'complete_object': False, 'physics_ready': False,
             'confidence': confidence, 'source_texture_sha256': digest(texture)}
    glb, conversion = make_glb(surface['vertices'], surface['triangles'], surface['uv'], texture, name=identifier, extras=extra)
    limitations = list(viewer['limitations']) + [
        'This is an open diagnostic patch, not a complete object, collision mesh or simulation-ready asset.',
        'Position values retain an arbitrary reconstruction gauge. A glTF viewer may label units as metres; no physical measurement is supplied.',
        'No surfaces, holes, backs, thickness, cavities or missing lens geometry have been invented or completed.',
        'Observed appearance includes illumination from the recording and is not a measured intrinsic material.',
        'The package contains selected source RGB frames and masks; the full original video is not included.',
        'Source evidence paths in saved JSON describe original provenance; portable copies are under evidence/.']
    metadata = {'format': 'observed-surface-export', 'version': VERSION, 'id': identifier,
        'label': viewer['title'], 'status': viewer['status'], 'source_run': str(source_run),
        'source_viewer_sha256': digest(evidence['evidence/' + str(source_run) + '/viewer-display.json']),
        'glb_sha256': digest(glb), 'vertices': len(surface['vertices']), 'triangles': len(surface['triangles']),
        'texture_sha256': digest(texture), 'texture_original_bytes': True, 'source_texture': True,
        'physical_scale_known': False, 'camera_calibrated': False, 'complete_object': False, 'physics_ready': False,
        'watertight': False, 'hole_filling': False, 'thickness_known': False, 'confidence': confidence,
        'conversion': conversion, 'projection_verification': projection,
        'limitations': limitations, 'new_api_calls': 0, 'new_inference_calls': 0,
        'implementation_sha256': digest(Path(__file__).read_bytes()),
        'format_reference': GLTF_SPEC,
        'portable_layout': {'asset': 'observed-patch.glb', 'evidence_root': 'evidence/', 'manifest': 'package-manifest.json'}}
    readme = ('OBSERVED SURFACE — PORTABLE DIAGNOSTIC PATCH\n\n' + viewer['title'] + '\n\n' +
        f"{metadata['vertices']:,} observed vertices; {metadata['triangles']:,} original local triangles.\n" +
        'Open observed-patch.glb in a glTF 2.0 viewer supporting KHR_materials_unlit.\n' +
        'The original JPEG is embedded. No network or inference is needed to inspect it.\n' +
        'The material is one-sided; looking from behind reveals the missing surface.\n\n' +
        'READ BEFORE USING\n' + ''.join('- ' + text + '\n' for text in limitations) +
        '\nPROVENANCE\nmetadata.json records conversion, source camera and confidence categories.\n' +
        'package-manifest.json records integrity checksums of every other member by SHA-256.\n' +
        'evidence/ preserves exact source frames, SAM masks, raw/display results, stereo diagnostics,\n' +
        'sparse geometry inputs and frozen implementations. No code runs automatically.\n' +
        'Raw evidence JSON contains historical project paths; use its matching evidence/ copy.\n\n' +
        'COORDINATES AND COLOUR\nNo geometry transform or physical rescaling was applied.\n' +
        'Positions and UVs are encoded as glTF float32; the maximum conversion error is in metadata.\n' +
        'Original triangle indices/winding are unchanged. No normals, smoothing or vertex tint were added.\n' +
        'Display V is inverted for glTF top-left image UVs. JPEG base colour is sRGB;\n' +
        'unlit white base factor avoids lighting or double tint. Linear sampling and clamp are explicit.\n').encode()
    files = dict(evidence)
    files.update({'observed-patch.glb': glb, 'metadata.json': json_bytes(metadata), 'README.txt': readme,
                  'export-implementation.py': Path(__file__).read_bytes()})
    package_manifest = {'format': 'observed-surface-portable', 'version': VERSION, 'source_run_id': identifier,
        'files': {name: {'sha256': digest(raw), 'bytes': len(raw)} for name, raw in sorted(files.items())}}
    files['package-manifest.json'] = json_bytes(package_manifest)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, raw in sorted(files.items()):
            safe_name(name)
            info = zipfile.ZipInfo(name, (2026, 10, 31, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, raw)
    zip_bytes = buffer.getvalue()
    package_check = validate_package(zip_bytes)
    output.mkdir(parents=True)
    for name, raw in [('observed-patch.glb', glb), ('portable.zip', zip_bytes),
                      ('metadata.json', json_bytes(metadata)), ('README.txt', readme),
                      ('package-validation.json', json_bytes(package_check))]:
        (output / name).write_bytes(raw)
    base = '/files/' + output.resolve().relative_to(root).as_posix()
    return {'id': identifier, 'source_run_id': identifier, 'label': viewer['title'], 'status': viewer['status'],
        'glb_url': base + '/observed-patch.glb', 'zip_url': base + '/portable.zip',
        'metadata_url': base + '/metadata.json', 'readme_url': base + '/README.txt',
        'vertices': metadata['vertices'], 'triangles': metadata['triangles'], 'source_texture': True,
        'physical_scale_known': False, 'complete_object': False,
        'glb_sha256': digest(glb), 'zip_sha256': digest(zip_bytes), 'metadata_sha256': digest(json_bytes(metadata)),
        'glb_bytes': len(glb), 'zip_bytes': len(zip_bytes), 'package_validation': package_check}


def export_all(root, output_root, run_ids):
    root, output_root = Path(root).resolve(), Path(output_root)
    output_root = output_root if output_root.is_absolute() else root / output_root
    if not run_ids or not output_root.resolve().is_relative_to(root / 'data'):
        raise ValueError('Expected source runs and project data export destination')
    if output_root.exists():
        raise ValueError('Export root already exists; preserve prior exports')
    exports = []
    for identifier in run_ids:
        if not ID.fullmatch(identifier):
            raise ValueError('Invalid source run identity')
        source = 'data/observed-surface-experiment-20261031/' + identifier
        exports.append(export_run(root, source, output_root / identifier))
    manifest = {'format': FORMAT, 'version': VERSION, 'exports': exports,
                'new_api_calls': 0, 'new_inference_calls': 0, 'production_catalog_changed': False}
    (output_root / 'manifest.json').write_bytes(json_bytes(manifest))
    return manifest
