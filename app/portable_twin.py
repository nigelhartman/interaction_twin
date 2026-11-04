"""Self-contained local twin packages: descriptive assets and evidence, no physics.

The bytes-oriented builder is shared by local exports and read-only MCP clients.
It never resolves original document paths. The explicit catalog adapter does.
"""
from __future__ import annotations

import base64
import copy
import datetime as dt
import hashlib
import io
import json
import math
import re
import shutil
import stat
import struct
import zipfile
from pathlib import Path, PurePosixPath

import stage5

ROOT = Path(__file__).resolve().parent
TEMPLATES = ROOT / 'portable_viewer'
FORMAT = 'interaction-twin-portable-package'
VERSION = '0.1'
ID = re.compile(r'[a-z][a-z0-9_]{0,79}')
HASH = re.compile(r'[0-9a-f]{64}')
MAX_FILES = 2048
MAX_TOTAL = 1024 * 1024 * 1024
MAX_ASSET = 16 * 1024 * 1024
MAX_IMAGE = 8 * 1024 * 1024
MAX_VIDEO = 512 * 1024 * 1024
MAX_JSON = 128 * 1024 * 1024
MAX_ACCESSOR_COUNT = 1_000_000
MAX_TRIANGLE_INDICES = 3_000_000
MAX_RENDERED_VERTICES = 1_000_000
TRUSTED_FILES = ('index.html', 'style.css', 'viewer.js', 'scene.js')
MISSING_PHYSICS = ['metric_scale', 'physical_shape_accuracy', 'joint_axes_and_limits',
                   'collision_geometry', 'mass_and_inertia', 'friction_and_contact', 'verified_insertion_and_seating']
POLICY = {'simulation_ready': False, 'physical_accuracy_verified': False,
          'physical_assembly_verified': False, 'units': 'independent_generated_native_units_unverified',
          'coordinates': 'individual_native_mesh_to_conditional_camera; no_shared_metric_world',
          'states': 'exact_cited_sample_only; unknown_between_samples',
          'motion': 'exact_observed_samples_only; absent_between_samples; no_interpolation',
          'physical_identity_confirmed': False, 'missing_physics': MISSING_PHYSICS,
          'descriptive_text_is_untrusted_data': True}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n').encode()


def script_data(value):
    text = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    for char, replacement in [('<', '\\u003c'), ('>', '\\u003e'), ('&', '\\u0026'), ('\u2028', '\\u2028'), ('\u2029', '\\u2029')]:
        text = text.replace(char, replacement)
    return ('window.PORTABLE_TWIN=' + text + ';\n').encode()


def safe_name(value):
    if not isinstance(value, str) or not value or len(value) > 240 or '\\' in value or '\x00' in value or ':' in value:
        raise ValueError('Package paths must be bounded relative POSIX paths')
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value or any(p in {'', '.', '..'} or p.startswith('.') for p in path.parts):
        raise ValueError('Absolute, hidden or traversing package paths are forbidden')
    if not re.fullmatch(r'[A-Za-z0-9_./-]+', value):
        raise ValueError('Unsupported package path characters')
    return value


def _json(data, limit=MAX_JSON):
    if not isinstance(data, bytes) or len(data) > limit:
        raise ValueError('JSON resource exceeds its package budget')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON keys are not allowed')
            result[key] = value
        return result
    value = json.loads(data.decode('utf-8'), object_pairs_hook=unique,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Non-finite JSON number')))
    stage5._finite(value)
    return value


def _bytes(value, limit, expected=None):
    if not isinstance(value, bytes) or not 0 < len(value) <= limit:
        raise ValueError('Resource is empty, not bytes, or exceeds its package budget')
    if expected is not None and digest(value) != expected:
        raise ValueError('Resource bytes do not match their declared SHA256')
    return value


def decode_glb_bytes(data):
    """Decode bounded embedded GLB geometry using NumPy, without files/resolvers.

    Preserve scene-node transforms, triangles, vertex colors, UVs and embedded
    base-color images. Unsupported sparse/morph/skin or external URI data fails
    explicitly; there is no network, checkpoint or filesystem lookup.
    """
    import numpy as np
    _bytes(data, MAX_ASSET)
    if len(data) < 20:
        raise ValueError('Truncated GLB header')
    magic, version, length = struct.unpack_from('<III', data)
    if magic != 0x46546C67 or version != 2 or length != len(data):
        raise ValueError('Expected a complete GLB 2.0 resource')
    chunks, offset = {}, 12
    while offset < len(data):
        if offset + 8 > len(data):
            raise ValueError('Truncated GLB chunk')
        count, kind = struct.unpack_from('<II', data, offset); offset += 8
        if kind in chunks or count > len(data) - offset or count % 4:
            raise ValueError('Duplicate, unaligned or overflowing GLB chunk')
        chunks[kind] = data[offset:offset + count]; offset += count
    if set(chunks) - {0x4E4F534A, 0x004E4942} or 0x4E4F534A not in chunks:
        raise ValueError('Unsupported GLB chunks')
    gltf = _json(chunks[0x4E4F534A].rstrip(b' \x00'), 4 * 1024 * 1024)
    binary = chunks.get(0x004E4942, b'')
    if gltf.get('asset', {}).get('version') != '2.0' or gltf.get('skins') or gltf.get('animations'):
        raise ValueError('Only static GLB 2.0 generated assets are supported')
    buffers = gltf.get('buffers', [])
    if len(buffers) != 1 or 'uri' in buffers[0] or not 0 <= buffers[0].get('byteLength', -1) <= len(binary):
        raise ValueError('Only one embedded GLB buffer is supported')
    views, accessors = gltf.get('bufferViews', []), gltf.get('accessors', [])
    if len(views) > 4096 or len(accessors) > 4096:
        raise ValueError('GLB accessor budget exceeded')
    def view(index):
        if type(index) is not int or not 0 <= index < len(views):
            raise ValueError('Invalid GLB buffer view')
        v = views[index]; start, count = v.get('byteOffset', 0), v.get('byteLength', -1)
        if v.get('buffer', 0) != 0 or type(start) is not int or type(count) is not int or start < 0 or count < 0 or start + count > len(binary):
            raise ValueError('GLB buffer view escapes embedded data')
        return v, binary[start:start + count]
    types = {5120: ('i1', 1), 5121: ('u1', 1), 5122: ('<i2', 2), 5123: ('<u2', 2), 5125: ('<u4', 4), 5126: ('<f4', 4)}
    dimensions = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}
    def accessor(index, *, triangle_indices=False):
        if type(index) is not int or not 0 <= index < len(accessors):
            raise ValueError('Invalid GLB accessor')
        a = accessors[index]; count = a.get('count', -1)
        limit = MAX_TRIANGLE_INDICES if triangle_indices else MAX_ACCESSOR_COUNT
        if triangle_indices and (a.get('type') != 'SCALAR' or a.get('componentType') not in {5121, 5123, 5125} or a.get('normalized', False) is not False):
            raise ValueError('Triangle indices require a nonnormalized unsigned SCALAR accessor')
        if 'sparse' in a or a.get('componentType') not in types or a.get('type') not in dimensions or type(count) is not int or not 0 < count <= limit:
            raise ValueError('Unsupported or unbounded GLB accessor')
        dtype, size = types[a['componentType']]; dims = dimensions[a['type']]
        v, raw = view(a.get('bufferView')); start = a.get('byteOffset', 0); stride = v.get('byteStride', dims * size)
        if type(start) is not int or type(stride) is not int or start < 0 or stride < dims * size or stride > 256 or start + (count - 1) * stride + dims * size > len(raw):
            raise ValueError('GLB accessor escapes its buffer view')
        out = np.ndarray((count, dims), dtype=dtype, buffer=raw, offset=start, strides=(stride, size)).copy()
        if a.get('normalized'):
            if a['componentType'] == 5126:
                raise ValueError('Floating accessor cannot be normalized')
            info = np.iinfo(out.dtype); out = np.maximum(-1, out.astype(float) / info.max)
        if not np.isfinite(out).all():
            raise ValueError('Non-finite GLB data')
        return out
    def texture(material):
        info = material.get('pbrMetallicRoughness', {}).get('baseColorTexture')
        if info is None:
            return None
        if info.get('texCoord', 0) != 0:
            raise ValueError('Only base-color UV set zero is supported')
        textures, images = gltf.get('textures', []), gltf.get('images', [])
        ti = info.get('index'); image_index = textures[ti].get('source') if type(ti) is int and 0 <= ti < len(textures) else None
        if type(image_index) is not int or not 0 <= image_index < len(images):
            raise ValueError('Invalid embedded GLB texture')
        image = images[image_index]
        if 'uri' in image:
            match = re.fullmatch(r'data:(image/(?:png|jpeg));base64,([A-Za-z0-9+/=]+)', image['uri'])
            if not match:
                raise ValueError('External GLB image URIs are forbidden')
            mime, pixels = match[1], base64.b64decode(match[2], validate=True)
        else:
            mime, pixels = image.get('mimeType'), view(image.get('bufferView'))[1]
        if mime not in {'image/png', 'image/jpeg'} or not 0 < len(pixels) <= MAX_IMAGE:
            raise ValueError('Only bounded embedded PNG/JPEG textures are supported')
        if not (pixels.startswith(b'\x89PNG\r\n\x1a\n') if mime == 'image/png' else pixels.startswith(b'\xff\xd8')):
            raise ValueError('Texture bytes do not match their image type')
        return f'data:{mime};base64,' + base64.b64encode(pixels).decode()
    # Reject external references even if an unused material would hide them.
    for image in gltf.get('images', []):
        if 'uri' in image and not re.fullmatch(r'data:image/(?:png|jpeg);base64,[A-Za-z0-9+/=]+', image['uri']):
            raise ValueError('External GLB image URIs are forbidden')
    nodes, meshes, materials = gltf.get('nodes', []), gltf.get('meshes', []), gltf.get('materials', [])
    if len(nodes) > 512 or len(meshes) > 256 or len(materials) > 256:
        raise ValueError('GLB scene budget exceeded')
    def transform(node):
        if 'matrix' in node:
            matrix = np.asarray(node['matrix'], float)
            if matrix.size != 16:
                raise ValueError('Invalid GLB node matrix')
            return matrix.reshape(4, 4).T
        t, s, q = np.asarray(node.get('translation', [0, 0, 0]), float), np.asarray(node.get('scale', [1, 1, 1]), float), np.asarray(node.get('rotation', [0, 0, 0, 1]), float)
        if t.shape != (3,) or s.shape != (3,) or q.shape != (4,) or not np.isclose(np.linalg.norm(q), 1, atol=1e-4):
            raise ValueError('Invalid GLB TRS transform')
        x, y, z, w = q
        r = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        matrix = np.eye(4); matrix[:3, :3] = r @ np.diag(s); matrix[:3, 3] = t
        return matrix
    primitives, visiting, visited_count, emitted_indices, emitted_vertices = [], set(), 0, 0, 0
    def walk(index, parent, depth=0):
        nonlocal visited_count, emitted_indices, emitted_vertices
        visited_count += 1
        if visited_count > 2048:
            raise ValueError('GLB scene traversal budget exceeded')
        if type(index) is not int or not 0 <= index < len(nodes) or index in visiting or depth > 64:
            raise ValueError('Invalid or cyclic GLB node graph')
        visiting.add(index); node = nodes[index]
        if 'skin' in node:
            raise ValueError('Skinned GLB assets are unsupported')
        matrix = parent @ transform(node)
        if not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]):
            raise ValueError('Invalid GLB affine node transform')
        if 'mesh' in node:
            mi = node['mesh']
            if type(mi) is not int or not 0 <= mi < len(meshes):
                raise ValueError('Unknown GLB mesh')
            for primitive in meshes[mi].get('primitives', []):
                if primitive.get('mode', 4) != 4 or primitive.get('targets') or len(primitives) >= 256:
                    raise ValueError('Only bounded static triangle primitives are supported')
                attributes = primitive.get('attributes', {})
                vertices = accessor(attributes.get('POSITION'))
                if vertices.shape[1] != 3 or len(vertices) > MAX_RENDERED_VERTICES:
                    raise ValueError('Unsupported GLB positions')
                vertices = vertices @ matrix[:3, :3].T + matrix[:3, 3]
                indices = accessor(primitive['indices'], triangle_indices=True).reshape(-1) if 'indices' in primitive else np.arange(len(vertices))
                if not np.issubdtype(indices.dtype, np.integer) or len(indices) % 3 or len(indices) > MAX_TRIANGLE_INDICES or indices.min(initial=0) < 0 or indices.max(initial=0) >= len(vertices):
                    raise ValueError('Invalid or unbounded GLB triangle indices')
                emitted_indices += len(indices)
                emitted_vertices += len(vertices)
                if emitted_indices > MAX_TRIANGLE_INDICES or emitted_vertices > MAX_RENDERED_VERTICES:
                    raise ValueError('GLB expanded scene geometry exceeds the viewer budget')
                colors = accessor(attributes['COLOR_0']) if 'COLOR_0' in attributes else None
                uv = accessor(attributes['TEXCOORD_0']) if 'TEXCOORD_0' in attributes else None
                if colors is not None:
                    if colors.shape not in {(len(vertices), 3), (len(vertices), 4)}:
                        raise ValueError('GLB colors do not match vertices')
                    if colors.shape[1] == 3:
                        colors = np.c_[colors, np.ones(len(colors))]
                if uv is not None and uv.shape != (len(vertices), 2):
                    raise ValueError('GLB UVs do not match vertices')
                mati = primitive.get('material')
                if mati is not None and (type(mati) is not int or not 0 <= mati < len(materials)):
                    raise ValueError('Unknown GLB material')
                material = materials[mati] if mati is not None else {}
                factor = material.get('pbrMetallicRoughness', {}).get('baseColorFactor', [1, 1, 1, 1])
                if len(factor) != 4 or not all(isinstance(x, (int, float)) and math.isfinite(x) and 0 <= x <= 1 for x in factor):
                    raise ValueError('Invalid GLB material color')
                tex = texture(material)
                if tex and uv is None:
                    raise ValueError('Textured GLB lacks UV coordinates')
                primitives.append({'node': node.get('name', str(index)), 'geometry': meshes[mi].get('name', str(mi)), 'material_index': mati,
                                   'vertices': vertices.tolist(), 'faces': indices.reshape(-1, 3).tolist(),
                                   'colors': colors.tolist() if colors is not None else None, 'uv': uv.tolist() if uv is not None else None,
                                   'texture': tex, 'base_color_factor': factor,
                                   'appearance_mode': 'uv_texture' if tex else 'vertex_rgb' if colors is not None else 'material_color'})
        for child in node.get('children', []):
            walk(child, matrix, depth + 1)
        visiting.remove(index)
    scenes = gltf.get('scenes', [])
    scene_index = gltf.get('scene', 0)
    if scenes and (type(scene_index) is not int or not 0 <= scene_index < len(scenes)):
        raise ValueError('Invalid GLB default scene')
    roots = scenes[scene_index].get('nodes', []) if scenes else [i for i in range(len(nodes)) if not any(i in n.get('children', []) for n in nodes)]
    for index in roots:
        walk(index, np.eye(4))
    if not primitives:
        raise ValueError('GLB contains no supported geometry')
    points = np.concatenate([np.asarray(p['vertices']) for p in primitives]); lo, hi = points.min(0), points.max(0); span = float(np.linalg.norm(hi - lo))
    if not math.isfinite(span) or span < 1e-10 or len(points) > MAX_RENDERED_VERTICES:
        raise ValueError('GLB geometry is degenerate or exceeds the viewer budget')
    return {'source_glb_sha256': digest(data), 'primitives': primitives,
            'mesh_stats': {'bounds': [lo.tolist(), hi.tolist()], 'primitive_count': len(primitives), 'rendered_vertices': len(points),
                           'faces': sum(len(p['faces']) for p in primitives), 'appearance_modes': sorted({p['appearance_mode'] for p in primitives}),
                           'rendering_policy': 'unlit base color; embedded images/UVs/node transforms preserved; PBR physics not simulated'},
            'native_scene': {'target': ((lo + hi) / 2).tolist(), 'span': span, 'initial_distance': span * 1.8}}


def _media(data):
    if data.startswith(b'\xff\xd8'):
        return '.jpg', 'image/jpeg'
    if data.startswith(b'\x89PNG\r\n\x1a\n'):
        return '.png', 'image/png'
    raise ValueError('Evidence must be JPEG or PNG bytes')


def _scene(manifest, meshes):
    import numpy as np
    objects = []
    for asset in manifest['assets']:
        tid = asset['track_id']; mesh = meshes[tid]
        rows, transformed = [], []
        vertices = np.concatenate([np.asarray(p['vertices']) for p in mesh['primitives']])
        for row in sorted([m for m in manifest['motion'] if m['track_id'] == tid], key=lambda r: r['timestamp_s']):
            allowed = row['status'] in stage5.ALLOWED_POSES and row['camera_from_native_mesh'] is not None
            rows.append({'timestamp_s': row['timestamp_s'], 'sample_index': row['source_sample_index'], 'status': row['status'],
                         'visible': allowed, 'reference': row['status'].startswith('reference_'), 'reasons': row['reasons'],
                         'camera_from_native_mesh': row['camera_from_native_mesh'] if allowed else None})
            if allowed:
                matrix = np.asarray(row['camera_from_native_mesh']); transformed.append(vertices @ matrix[:3, :3].T + matrix[:3, 3])
        layout = mesh['native_scene']
        if transformed:
            points = np.concatenate(transformed); lo, hi = points.min(0), points.max(0); span = float(np.linalg.norm(hi-lo))
            layout = {'target': ((lo+hi)/2).tolist(), 'span': span, 'initial_distance': span*1.8}
        objects.append({'object_id': tid, 'label': next(t['label'] for t in manifest['components'] if t['id'] == tid),
                        'primitives': mesh['primitives'], 'mesh_stats': mesh['mesh_stats'], 'native_scene': mesh['native_scene'],
                        'motion_scene': layout, 'source_projection_rows': None, 'frames': rows, 'max_hold_s': 0,
                        'provenance': {'mesh_sha256': asset['sha256'], 'package_path': asset['path']}})
    return {'objects': objects, 'scene': objects[0]['native_scene'] if objects else None, 'native_only': True,
            'physical_assembly_verified': False, 'source_projection_rows': None, 'image_size': [1, 1],
            'source_duration_s': manifest['source']['duration_s'], 'initial_source_time_s': manifest['evidence'][0]['timestamp_s']}


def viewer_payload(manifest, meshes):
    keys = ['title', 'twin_id', 'source', 'policy', 'components', 'assets', 'relationships', 'variables', 'assertions', 'steps', 'motion', 'evidence', 'limitations', 'program']
    return {key: copy.deepcopy(manifest[key]) for key in keys} | {'scene': _scene(manifest, meshes)}


def build_package(output_dir, *, document, program_bytes, asset_bytes, evidence_bytes,
                  mask_bytes=None, mesh_primitives=None, source_video_bytes=None,
                  source_program_sha256=None, program_bytes_kind='original_file_bytes',
                  twin_id=None, title=None, provenance=None, zip_output=None):
    """Build a new directory+ZIP only from supplied data, not workspace artifacts."""
    stage5.validate_interaction(document, verify_files=False)
    _bytes(program_bytes, MAX_JSON)
    if _json(program_bytes) != document:
        raise ValueError('Program bytes differ from the supplied interaction document')
    if program_bytes_kind not in {'original_file_bytes', 'mcp_json_serialization'}:
        raise ValueError('Declare original bytes or explicit MCP JSON serialization')
    actual_program_hash = digest(program_bytes)
    source_program_sha256 = source_program_sha256 or actual_program_hash
    if not isinstance(source_program_sha256, str) or not HASH.fullmatch(source_program_sha256):
        raise ValueError('Original program citation needs a SHA256')
    if program_bytes_kind == 'original_file_bytes' and source_program_sha256 != actual_program_hash:
        raise ValueError('Exact original bytes must match their original program hash')
    if not isinstance(asset_bytes, dict) or not isinstance(evidence_bytes, dict):
        raise ValueError('Assets and evidence must be supplied as byte maps')
    assets = {a['track_id']: a for a in document['assets']}; evidence = {e['id']: e for e in document['evidence']}
    if len(assets) != len(document['assets']) or set(asset_bytes) != set(assets) or set(evidence_bytes) - set(evidence):
        raise ValueError('Provide all declared assets and only declared evidence IDs')
    mask_bytes = mask_bytes or {}; mesh_primitives = mesh_primitives or {}
    if set(mesh_primitives) - set(assets):
        raise ValueError('Mesh primitives reference an unknown generated component')
    files, meshes, packaged_assets, packaged_evidence = {}, {}, [], []
    program_path = 'program/original-program.json' if program_bytes_kind == 'original_file_bytes' else 'program/mcp-resource-program.json'
    files[program_path] = program_bytes
    for tid, asset in assets.items():
        if not ID.fullmatch(tid):
            raise ValueError('Invalid asset track identifier')
        data = _bytes(asset_bytes[tid], MAX_ASSET, asset['sha256'])
        decoded = decode_glb_bytes(data)
        if tid in mesh_primitives and mesh_primitives[tid] != decoded:
            raise ValueError('Precomputed mesh primitives differ from the actual GLB bytes')
        meshes[tid] = decoded; path = 'assets/' + tid + '.glb'; files[path] = data
        packaged_assets.append({'track_id': tid, 'asset_id': asset['id'], 'path': path, 'sha256': asset['sha256'],
                                'original_workspace_path': asset['path'], 'backend': asset['backend'], 'units': POLICY['units'],
                                'physical_shape_verified': False, 'appearance': 'model_generated_unverified'})
    known_masks = set()
    for eid, original in evidence.items():
        row = {'id': eid, 'source_index': original['source_index'], 'timestamp_s': original['timestamp_s'],
               'original_workspace_path': original['frame_path'], 'sha256': original['frame_sha256'],
               'included': eid in evidence_bytes, 'path': None, 'masks': []}
        if eid in evidence_bytes:
            data = _bytes(evidence_bytes[eid], MAX_IMAGE, original['frame_sha256']); extension, _ = _media(data)
            row['path'] = 'evidence/' + eid + extension; files[row['path']] = data
        for obs in original['observations']:
            key = (eid, obs['track_id']); known_masks.add(key)
            mask = {'track_id': obs['track_id'], 'sha256': obs['mask_sha256'], 'included': key in mask_bytes,
                    'path': None, 'gate_status': obs['gate_status'], 'audit_status': obs['audit_status']}
            if key in mask_bytes:
                data = _bytes(mask_bytes[key], MAX_IMAGE, obs['mask_sha256']); extension, _ = _media(data)
                mask['path'] = 'masks/' + eid + '-' + obs['track_id'] + extension; files[mask['path']] = data
            row['masks'].append(mask)
        packaged_evidence.append(row)
    if set(mask_bytes) - known_masks:
        raise ValueError('Mask bytes reference an undeclared observation')
    video = None
    if source_video_bytes is not None:
        data = _bytes(source_video_bytes, MAX_VIDEO, document['source']['sha256'])
        extension = Path(document['source']['path']).suffix.lower()
        if extension not in {'.mp4', '.mov', '.webm', '.m4v'}:
            raise ValueError('Unsupported source video extension')
        video = 'source/video' + extension; files[video] = data
    manifest = {'format': FORMAT, 'version': VERSION, 'created_at': dt.datetime.now(dt.timezone.utc).isoformat(),
                'twin_id': twin_id or document['case_id'], 'title': title or document['case_id'],
                'program': {'path': program_path, 'sha256': actual_program_hash, 'serialization': program_bytes_kind,
                            'original_program_sha256': source_program_sha256, 'original_bytes_included': program_bytes_kind == 'original_file_bytes'},
                'source': {'sha256': document['source']['sha256'], 'original_sha256': document['source']['original_sha256'],
                           'duration_s': document['source']['duration_s'], 'time_domain': document['source']['time_domain'],
                           'original_workspace_path': document['source']['path'], 'video_included': video is not None, 'video_path': video},
                'policy': copy.deepcopy(POLICY), 'components': copy.deepcopy(document['tracklets']), 'assets': packaged_assets,
                'relationships': copy.deepcopy(document['relationships']), 'variables': copy.deepcopy(document['variables']),
                'assertions': copy.deepcopy(document['assertions']), 'steps': copy.deepcopy(document['steps']),
                'motion': copy.deepcopy(document['motion']), 'evidence': packaged_evidence, 'reasoning': copy.deepcopy(document['reasoning']),
                'upstream_inputs_not_bundled': copy.deepcopy(document['inputs']), 'limitations': copy.deepcopy(document['limitations']),
                'provenance': copy.deepcopy(provenance or {}), 'viewer_version': VERSION, 'files': []}
    payload = viewer_payload(manifest, meshes)
    files['viewer-data.json'] = encoded(payload); files['viewer-data.js'] = script_data(payload)
    for name in TRUSTED_FILES:
        files[name] = (TEMPLATES / name).read_bytes()
    files['README.txt'] = ('INTERACTION TWIN / PORTABLE LOCAL PACKAGE\nOpen index.html in a browser. No server or external project folder is required.\n'
                           'Run inspect_twin_package.py before opening an untrusted package. Hashes provide integrity, not publisher authenticity.\n'
                           'Private recording evidence is included where listed. Source video is opt-in. Nothing is published.\n'
                           'Native components and exact conditional samples are descriptive, not a shared metric simulator.\n'
                           'Missing physics: ' + ', '.join(MISSING_PHYSICS) + '\n').encode()
    if len(files) + 1 > MAX_FILES or sum(len(value) for value in files.values()) > MAX_TOTAL:
        raise ValueError('Portable package exceeds its file/byte budget')
    manifest['files'] = [{'path': safe_name(name), 'sha256': digest(value), 'bytes': len(value)} for name, value in sorted(files.items())]
    files['manifest.json'] = encoded(manifest)
    # Validate the generated representation before exposing its index.
    _validate_files(files, verify_meshes=False)
    out = Path(output_dir).absolute(); archive = Path(zip_output).absolute() if zip_output is not None else Path(str(out) + '.zip')
    if out.exists() or archive.exists() or out.is_symlink() or archive.is_symlink():
        raise ValueError('Package output directory and ZIP must both be new')
    if not out.parent.is_dir() or not archive.parent.is_dir():
        raise ValueError('Create the output parent directory first')
    out.mkdir()
    try:
        for name, value in files.items():
            target = out / safe_name(name); target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(value)
        with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
            for name in sorted(files):
                bundle.write(out / name, name)
    except Exception:
        # Preserve partial output for diagnosis. The caller must choose another
        # fresh destination; no partial directory is advertised as a valid asset.
        raise
    return {'directory': str(out), 'index': str(out / 'index.html'), 'manifest': str(out / 'manifest.json'), 'zip': str(archive),
            'manifest_sha256': digest(files['manifest.json']), 'counts': {'assets': len(packaged_assets),
            'included_evidence': len(evidence_bytes), 'included_masks': len(mask_bytes), 'steps': len(document['steps'])},
            'original_bytes_included': manifest['program']['original_bytes_included'], 'video_included': video is not None,
            'simulation_ready': False}


def _validate_files(files, *, verify_meshes=True):
    if not isinstance(files, dict) or len(files) > MAX_FILES or sum(len(v) for v in files.values()) > MAX_TOTAL:
        raise ValueError('Package exceeds its file/byte budget')
    for name in files:
        safe_name(name)
    if 'manifest.json' not in files:
        raise ValueError('Package has no manifest.json')
    manifest = _json(files['manifest.json'], 8 * 1024 * 1024)
    if manifest.get('format') != FORMAT or manifest.get('version') != VERSION or manifest.get('viewer_version') != VERSION or manifest.get('policy') != POLICY:
        raise ValueError('Unsupported package format/version or unsafe physics policy')
    records = manifest.get('files')
    if not isinstance(records, list) or len(records) > MAX_FILES:
        raise ValueError('Manifest file index is invalid')
    declared = {}
    for row in records:
        if not isinstance(row, dict) or set(row) != {'path', 'bytes', 'sha256'}:
            raise ValueError('Invalid manifest file receipt')
        name = safe_name(row['path'])
        if name == 'manifest.json' or name in declared or type(row['bytes']) is not int or not 0 < row['bytes'] <= MAX_VIDEO or not isinstance(row['sha256'], str) or not HASH.fullmatch(row['sha256']):
            raise ValueError('Duplicate or unbounded manifest file receipt')
        declared[name] = row
    if set(files) != set(declared) | {'manifest.json'}:
        raise ValueError('Package has undeclared or missing files')
    for name, row in declared.items():
        if len(files[name]) != row['bytes'] or digest(files[name]) != row['sha256']:
            raise ValueError('Package file hash/size mismatch: ' + name)
    for name in TRUSTED_FILES:
        if files.get(name) != (TEMPLATES / name).read_bytes():
            raise ValueError('Package viewer code differs from the trusted local version')
    program = manifest['program']; name = safe_name(program['path'])
    if digest(files[name]) != program['sha256']:
        raise ValueError('Packaged program hash mismatch')
    document = _json(files[name]); stage5.validate_interaction(document, verify_files=False)
    if program.get('serialization') not in {'original_file_bytes', 'mcp_json_serialization'} or not isinstance(program.get('original_program_sha256'), str) or not HASH.fullmatch(program['original_program_sha256']):
        raise ValueError('Invalid program serialization provenance')
    original = program['serialization'] == 'original_file_bytes'
    if program.get('original_bytes_included') is not original or (original and program['sha256'] != program['original_program_sha256']):
        raise ValueError('Original and reserialized program identities must stay distinct')
    for key, source_key in [('components', 'tracklets'), ('relationships', 'relationships'), ('variables', 'variables'), ('assertions', 'assertions'), ('steps', 'steps'), ('motion', 'motion'), ('reasoning', 'reasoning'), ('limitations', 'limitations'), ('upstream_inputs_not_bundled', 'inputs')]:
        if manifest[key] != document[source_key]:
            raise ValueError('Portable semantics differ from the preserved program: ' + key)
    source = manifest['source']
    if source.get('original_workspace_path') != document['source']['path']:
        raise ValueError('Portable source path citation differs from its program')
    for key in ['sha256', 'original_sha256', 'duration_s', 'time_domain']:
        if source[key] != document['source'][key]:
            raise ValueError('Portable source provenance differs from its program')
    if source.get('video_included') is not bool(source.get('video_path')):
        raise ValueError('Video inclusion policy is inconsistent')
    if source.get('video_path'):
        if digest(_bytes(files[safe_name(source['video_path'])], MAX_VIDEO)) != source['sha256']:
            raise ValueError('Bundled source video differs from its original hash')
    originals = {a['track_id']: a for a in document['assets']}; meshes = {}
    if len(manifest['assets']) != len(originals) or {a['track_id'] for a in manifest['assets']} != set(originals):
        raise ValueError('Portable component mapping is incomplete')
    payload = _json(files['viewer-data.json'])
    for asset in manifest['assets']:
        original_asset = originals[asset['track_id']]; data = _bytes(files[safe_name(asset['path'])], MAX_ASSET, original_asset['sha256'])
        if (asset['sha256'] != original_asset['sha256'] or asset['asset_id'] != original_asset['id']
                or asset.get('physical_shape_verified') is not False or asset.get('units') != POLICY['units']
                or asset.get('backend') != original_asset['backend']
                or asset.get('original_workspace_path') != original_asset['path']
                or asset.get('appearance') != 'model_generated_unverified'):
            raise ValueError('Portable asset identity/units differ from the original')
        if verify_meshes:
            meshes[asset['track_id']] = decode_glb_bytes(data)
        else:
            # Builder already decoded and bound primitives to the exact bytes.
            obj = next(o for o in payload['scene']['objects'] if o['object_id'] == asset['track_id'])
            meshes[asset['track_id']] = {k: obj[k] for k in ['primitives', 'mesh_stats', 'native_scene']}
    frames = {e['id']: e for e in document['evidence']}
    if len(manifest['evidence']) != len(frames) or {e['id'] for e in manifest['evidence']} != set(frames):
        raise ValueError('Portable evidence index is incomplete')
    for row in manifest['evidence']:
        frame = frames[row['id']]
        if row.get('original_workspace_path') != frame['frame_path']:
            raise ValueError('Portable frame path citation differs from its source observation')
        if any(row[key] != frame[source_key] for key, source_key in [('timestamp_s', 'timestamp_s'), ('source_index', 'source_index'), ('sha256', 'frame_sha256')]):
            raise ValueError('Portable evidence differs from its source observation')
        if row['included'] is not bool(row['path']):
            raise ValueError('Evidence inclusion flag/path mismatch')
        if row['path']:
            _media(_bytes(files[safe_name(row['path'])], MAX_IMAGE, frame['frame_sha256']))
        observations = {o['track_id']: o for o in frame['observations']}
        if len(row['masks']) != len(frame['observations']) or {m['track_id'] for m in row['masks']} != set(observations):
            raise ValueError('Portable masks do not map original observations')
        for mask in row['masks']:
            obs = observations[mask['track_id']]
            if any(mask[k] != obs[source_key] for k, source_key in [('sha256', 'mask_sha256'), ('gate_status', 'gate_status'), ('audit_status', 'audit_status')]) or mask['included'] is not bool(mask['path']):
                raise ValueError('Portable mask provenance differs from its source')
            if mask['path']:
                _media(_bytes(files[safe_name(mask['path'])], MAX_IMAGE, obs['mask_sha256']))
    expected_payload = viewer_payload(manifest, meshes)
    if payload != expected_payload or files.get('viewer-data.js') != script_data(expected_payload):
        raise ValueError('Viewer data differs from validated assets/procedure; executable data is forbidden')
    return manifest


def _read_package(source):
    source = Path(source)
    if source.is_symlink():
        raise ValueError('Package root cannot be a symlink')
    files = {}
    if source.is_dir():
        for path in source.rglob('*'):
            if path.is_symlink():
                raise ValueError('Package symlinks are forbidden')
            if not path.is_file():
                continue
            name = safe_name(path.relative_to(source).as_posix())
            if len(files) >= MAX_FILES or path.stat().st_size > MAX_VIDEO:
                raise ValueError('Package file budget exceeded')
            files[name] = path.read_bytes()
            if sum(len(v) for v in files.values()) > MAX_TOTAL:
                raise ValueError('Package byte budget exceeded')
    else:
        if not source.is_file() or source.stat().st_size > MAX_TOTAL:
            raise ValueError('Package ZIP is unavailable or too large')
        with zipfile.ZipFile(source) as archive:
            entries = archive.infolist()
            if len(entries) > MAX_FILES or sum(i.file_size for i in entries) > MAX_TOTAL:
                raise ValueError('ZIP expanded size/file budget exceeded')
            seen = set()
            for item in entries:
                name = safe_name(item.filename[:-1] if item.is_dir() else item.filename)
                if name in seen:
                    raise ValueError('Duplicate ZIP entries are forbidden')
                seen.add(name)
                mode = item.external_attr >> 16
                if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in {0, stat.S_IFREG, stat.S_IFDIR}) or item.flag_bits & 1:
                    raise ValueError('ZIP symlinks, special files and encryption are forbidden')
                if item.file_size > MAX_VIDEO or item.file_size > max(1, item.compress_size) * 1000:
                    raise ValueError('ZIP member expansion budget exceeded')
                if not item.is_dir():
                    files[name] = archive.read(item)
    return files


def inspect_package(source, *, extract_to=None):
    """Validate without executing package code, optionally import to a new folder."""
    files = _read_package(source); manifest = _validate_files(files)
    out = None
    if extract_to is not None:
        out = Path(extract_to).absolute()
        if out.exists() or out.is_symlink() or not out.parent.is_dir():
            raise ValueError('Import destination must be a new directory under an existing parent')
        out.mkdir()
        try:
            for name, value in files.items():
                target = out / safe_name(name); target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as stream:
                    stream.write(value)
        except Exception:
            shutil.rmtree(out)
            raise
    return {'status': 'validated', 'format': FORMAT, 'version': VERSION, 'title': manifest['title'],
            'twin_id': manifest['twin_id'], 'manifest_sha256': digest(files['manifest.json']),
            'files': len(files), 'assets': len(manifest['assets']),
            'included_evidence': sum(e['included'] for e in manifest['evidence']),
            'included_masks': sum(m['included'] for e in manifest['evidence'] for m in e['masks']),
            'video_included': manifest['source']['video_included'],
            'original_bytes_included': manifest['program']['original_bytes_included'],
            'simulation_ready': False, 'directory': str(out) if out else (str(Path(source).absolute()) if Path(source).is_dir() else None),
            'index': str(out / 'index.html') if out else (str(Path(source).absolute() / 'index.html') if Path(source).is_dir() else None)}


def export_case(twin_id, output_dir, *, root=ROOT, include_video=False, include_masks=True, zip_output=None):
    """Explicit local adapter: catalog-validate then gather only declared bytes."""
    from twin_catalog import TwinCatalog
    catalog = TwinCatalog(root); row, doc, source_hash = catalog.load(twin_id)
    program = catalog.data_path(row['interaction_path']).read_bytes()
    assets = {a['track_id']: catalog.data_path(a['path']).read_bytes() for a in doc['assets']}
    evidence = {e['id']: catalog.data_path(e['frame_path']).read_bytes() for e in doc['evidence']}
    masks = {(e['id'], o['track_id']): catalog.data_path(o['mask_path']).read_bytes() for e in doc['evidence'] for o in e['observations']} if include_masks else {}
    video = catalog.data_path(doc['source']['path']).read_bytes() if include_video else None
    return build_package(output_dir, document=doc, program_bytes=program, asset_bytes=assets, evidence_bytes=evidence,
                         mask_bytes=masks, source_video_bytes=video, source_program_sha256=source_hash,
                         twin_id=twin_id, title=row['title'], zip_output=zip_output,
                         provenance={'transport': 'explicit_local_catalog_export', 'original_program_bytes_preserved': True,
                                     'model_inference_performed': False, 'external_upload_performed': False})
