#!/usr/bin/env python3
"""A bounded, lossless binary transport for one real initial SAM3D review.

No inference, geometry edits, pose reuse, catalog selection or cap increases.
The prior initial-review builder and all existing outputs remain unchanged.
"""
import argparse
import base64
import copy
import importlib.util
import io
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import sam3d_milestone as M
import upgrade_bundles as U
from portable_twin import MAX_RENDERED_VERTICES, MAX_TRIANGLE_INDICES

spec = importlib.util.spec_from_file_location('initial_review_v1_readonly', ROOT / 'scripts/build_initial_sam3d_review.py')
V1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V1)
SHARED = V1.SHARED
MAX_SCENE_BYTES = 64 * 1024 * 1024
MAX_NATIVE_GLB_BYTES = 16 * 1024 * 1024
FIELDS = {'vertices': (3, '<f8'), 'colors': (4, '<f8'), 'uv': (2, '<f8'), 'faces': (3, '<u4')}


def encode_scene(value):
    """Encode exact decoded float64 values and uint32 indices, without rounding.

    This is a display transport, not a new model representation. Offsets are
    contiguous; no hidden padding, aliased ranges or trailing payload is emitted.
    The browser validates the same contract before giving nested arrays to the
    existing isolated renderer.
    """
    scene = value['scene']
    objects = scene.get('objects')
    if (scene.get('native_only') is not True or scene.get('physical_assembly_verified') is not False
            or scene.get('timeline') != [] or scene.get('source_projection_rows') is not None
            or not isinstance(objects, list) or len(objects) != 1):
        raise ValueError('Only one independent native scene is supported')
    original = objects[0]
    if (original.get('frames') != [] or original.get('max_hold_s') != 0
            or original.get('source_projection_rows') is not None or original.get('native_up') != 'Y'):
        raise ValueError('Native inspection cannot contain source poses')
    primitives = original.get('primitives')
    if not isinstance(primitives, list) or not 1 <= len(primitives) <= 256:
        raise ValueError('Invalid native primitive count')
    blocks = []; descriptor_primitives = []; offset = 0; vertices = 0; indices = 0
    for primitive in primitives:
        target = {key: value for key, value in primitive.items() if key not in FIELDS}
        vertex_count = len(primitive['vertices'])
        if not 0 < vertex_count <= MAX_RENDERED_VERTICES:
            raise ValueError('Native vertex budget exceeded')
        vertices += vertex_count
        for field, (columns, dtype) in FIELDS.items():
            rows = primitive.get(field)
            if rows is None and field in ('colors', 'uv'):
                target[field] = None
                continue
            if (not isinstance(rows, list) or not rows
                    or any(not isinstance(row, list) or len(row) != columns for row in rows)
                    or (field in ('colors', 'uv') and len(rows) != vertex_count)):
                raise ValueError('Native geometry shape differs: ' + field)
            if field == 'faces':
                if any(type(x) is not int or x < 0 or x >= vertex_count for row in rows for x in row):
                    raise ValueError('Native triangle index differs')
                indices += len(rows) * columns
                if indices > MAX_TRIANGLE_INDICES:
                    raise ValueError('Native triangle index budget exceeded')
            elif any(type(x) not in (int, float) or not math.isfinite(x) for row in rows for x in row):
                raise ValueError('Native geometry must contain finite numbers')
            expected_bytes = len(rows) * columns * (4 if field == 'faces' else 8)
            if offset + expected_bytes > MAX_SCENE_BYTES:
                raise ValueError('Combined native scene exceeds the unchanged browser budget')
            payload = np.asarray(rows, dtype=dtype).tobytes(order='C')
            target[field] = {'binary': {'offset': offset, 'bytes': len(payload), 'rows': len(rows),
                                       'columns': columns, 'component_type': 'uint32_le' if field == 'faces' else 'float64_le'}}
            blocks.append(payload); offset += len(payload)
            if offset > MAX_SCENE_BYTES:
                raise ValueError('Combined native scene exceeds the unchanged browser budget')
        descriptor_primitives.append(target)
    if vertices > MAX_RENDERED_VERTICES:
        raise ValueError('Native aggregate vertex budget exceeded')
    stats = original['mesh_stats']
    if (stats['rendered_vertices'] != vertices or stats['faces'] * 3 != indices
            or stats['primitive_count'] != len(primitives)):
        raise ValueError('Native mesh statistics differ')
    binary = b''.join(blocks)
    obj = {**original, 'primitives': descriptor_primitives}
    descriptor = {'format': 'sam3d-native-binary-scene', 'version': 1,
                  'source_glb_sha256': value['source_glb_sha256'],
                  'geometry_sha256': U.digest(binary), 'geometry_bytes': len(binary),
                  'scene': {**scene, 'objects': [obj]}}
    raw = (json.dumps(descriptor, separators=(',', ':'), allow_nan=False) + '\n').encode()
    if len(raw) + len(binary) > MAX_SCENE_BYTES:
        raise ValueError('Combined native scene exceeds the unchanged browser budget')
    return raw, binary


def build(import_directory, output, *, root=ROOT):
    root = Path(root).resolve(); output = M._directory(output, root, exists=False)
    if output.exists():
        raise ValueError('Choose a fresh binary initial review version')
    initial_directory = M._directory(import_directory, root)
    initial_receipt_bytes = (initial_directory / 'import.json').read_bytes()
    directory, receipt, source, native, model = V1.verified_initial_import(initial_directory, root=root)
    if U._json_bytes(initial_receipt_bytes) != receipt or (directory / 'import.json').read_bytes() != initial_receipt_bytes:
        raise ValueError('Initial receipt changed during validation')
    if output.is_relative_to(directory) or output.is_relative_to(source['directory']):
        raise ValueError('Review must be separate from frozen evidence')
    record = source['input_record']; track = receipt['component_id']; raw = native['files']
    if len(raw['object.glb']) > MAX_NATIVE_GLB_BYTES:
        raise ValueError('Native GLB exceeds the unchanged browser budget')
    scene = SHARED.native_scene(native['decoded'], track, 'Y', [record['width'], record['height']])
    descriptor, geometry = encode_scene(scene)
    mask = np.asarray(Image.open(io.BytesIO(raw['inputs/mask.png']))) > 0
    rgba = np.zeros((*mask.shape, 4), np.uint8); rgba[mask] = [42, 202, 165, 112]
    buffer = io.BytesIO(); Image.fromarray(rgba).save(buffer, format='PNG')
    # The already-verified metadata is serialized from the in-memory snapshot.
    # No source file is reread after validation to create a TOCTOU substitution.
    metadata_raw = (directory / 'metadata.json').read_bytes()
    metadata_record = next(row for row in receipt['import_files'] if row['path'] == 'metadata.json')
    if (U._json_bytes(metadata_raw) != model or len(metadata_raw) != metadata_record['bytes']
            or U.digest(metadata_raw) != metadata_record['sha256']):
        raise ValueError('Initial metadata changed after validation')
    imported_receipt = (directory / 'import.json').read_bytes()
    if imported_receipt != initial_receipt_bytes:
        raise ValueError('Initial receipt changed after validation')
    payloads = {'source-rgb.png': source['files']['image.png'], 'source-mask.png': source['files']['mask.png'],
                'mask-overlay.png': buffer.getvalue(), 'candidate.glb': raw['object.glb'],
                'candidate-scene.json': descriptor, 'candidate-geometry.bin': geometry,
                'model-metadata.json': metadata_raw}
    shared_renderer = (root / 'workspace/scene.js').read_text()
    renderer = SHARED.camera_only_renderer(shared_renderer, 'neutral').encode()
    manifest = {'format': 'sam3d-initial-native-review', 'version': 2, 'fixture': False, 'native_only': True,
                'component_id': track, 'title': track.replace('_', ' ').capitalize(),
                'source': copy.deepcopy(source['binding']['source']), 'image_size': [record['width'], record['height']],
                'native_glb_sha256': U.digest(raw['object.glb']), 'native_up': 'Y', 'mesh_stats': native['decoded']['mesh_stats'],
                'appearance': model['appearance'], 'files': M._records(payloads), 'initial_handoff': receipt['initial_handoff'],
                'import_receipt': {'path': (directory / 'import.json').relative_to(root).as_posix(), 'sha256': U.digest(imported_receipt)},
                'scope': {'baseline_required': False, 'original_geometry_modified': False, 'old_pose_reused': False,
                          'source_video_placement_established': False, 'physical_accuracy_verified': False, 'metric_scale_verified': False,
                          'assembly_verified': False, 'contact_verified': False, 'simulation_ready': False, 'selected': False, 'catalog_changed': False},
                'renderer': {'shared_source_sha256': U.digest(shared_renderer.encode()), 'review_copy_sha256': U.digest(renderer),
                             'display_background': 'neutral', 'display_clear_rgba': [.91, .93, .90, 1.], 'native_up': 'Y',
                             'geometry_or_material_changed': False, 'pbr_rendering': False, 'splat_rendering': False},
                'geometry_transport': {'format': 'lossless-native-geometry', 'version': 1, 'byte_order': 'little',
                                       'float_encoding': 'ieee754-float64', 'index_encoding': 'uint32',
                                       'descriptor_bytes': len(descriptor), 'geometry_bytes': len(geometry),
                                       'combined_scene_budget_bytes': MAX_SCENE_BYTES, 'geometry_sha256': U.digest(geometry)},
                'execution_scope': 'Genuine-result source bytes revalidated; standalone native candidate only. No physical validity or motion acceptance.'}
    manifest_raw = U.encoded(manifest); templates = root / 'initial_native_review_v2'
    client = (templates / 'review.js').read_bytes(); codec = (templates / 'geometry.js').read_bytes()
    style = (templates / 'style.css').read_bytes()
    page = (templates / 'index.html').read_text().replace('%%MANIFEST_SHA%%', U.digest(manifest_raw))
    for role, content in [('SCENE', renderer), ('CLIENT', client), ('GEOMETRY', codec), ('STYLE', style)]:
        page = page.replace('%%' + role + '_INTEGRITY%%', 'sha256-' + base64.b64encode(bytes.fromhex(U.digest(content))).decode())
    final = {**payloads, 'review-manifest.json': manifest_raw, 'scene.js': renderer, 'review.js': client,
             'geometry.js': codec, 'style.css': style, 'index.html': page.encode()}
    output.mkdir(parents=True, exist_ok=False)
    for name, content in final.items():
        (output / name).write_bytes(content)
    U.write(output / 'review-build.json', {'format': 'sam3d-initial-native-review-build', 'version': 2, 'files': M._records(final),
            'builder_sha256': U.sha(Path(__file__)), 'original_import_sha256': U.digest(imported_receipt),
            'verified_import_builder_sha256': U.sha(Path(V1.__file__)), 'shared_review_builder_sha256': U.sha(Path(SHARED.__file__)),
            'new_models_or_api_calls': 0, 'native_geometry_modified': False, 'catalog_changed': False, 'physical_accuracy_verified': False})
    return {'status': 'binary_initial_native_review_built_unselected', 'review': (output / 'index.html').relative_to(root).as_posix(),
            'component_id': track, 'mesh_stats': native['decoded']['mesh_stats'], 'descriptor_bytes': len(descriptor),
            'geometry_bytes': len(geometry), 'combined_scene_bytes': len(descriptor) + len(geometry)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--import-directory', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = build(args.import_directory, args.output)
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps({'status': 'binary_initial_native_review_unavailable', 'error': str(error)})); raise SystemExit(2)
    print(json.dumps(result, indent=2))
