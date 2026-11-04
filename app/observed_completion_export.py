"""Export separate observed and inferred surface layers without altering either.

Only saved, separately generated arrays are packaged. No interpolation, model
inference, fitting, smoothing, hole filling or physical rescaling occurs here.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import struct

import numpy as np

from observed_surface_export import digest, json_bytes, local_path, make_glb, read_checked

OBSERVED_ID = 'camera-body-stereo-v1'
CASES = [
    'data/observed-depth-completion-20261031/camera-body-moge-prior-v1',
    'data/observed-surface-completion-20261031/camera-body-interior-v1',
]


def unpack(raw):
    if len(raw) < 28 or struct.unpack_from('<III', raw) != (0x46546c67, 2, len(raw)):
        raise ValueError('Invalid GLB envelope')
    chunks, offset = {}, 12
    while offset < len(raw):
        length, kind = struct.unpack_from('<II', raw, offset)
        if length % 4 or offset + 8 + length > len(raw) or kind in chunks:
            raise ValueError('Invalid GLB chunk')
        chunks[kind] = raw[offset + 8:offset + 8 + length]
        offset += 8 + length
    document = json.loads(chunks[0x4e4f534a])
    if len(document['buffers']) != 1 or 'uri' in document['buffers'][0]:
        raise ValueError('Only one embedded input buffer is supported')
    return document, chunks[0x004e4942][:document['buffers'][0]['byteLength']]


def combine_layers(observed_glb, inferred_glb, provenances):
    """Keep each input primitive in its own named node, mesh and material."""
    document = {'asset': {'version': '2.0', 'generator': 'Interaction Twin separate-layer exporter v1',
        'extras': {'units': 'arbitrary reconstruction gauge; not measured metres',
                   'physical_scale_known': False, 'physics_ready': False,
                   'observed_geometry_unchanged': True, 'inferred_layer_optional': True}},
        'scene': 0, 'scenes': [{'nodes': [0, 1]}], 'nodes': [], 'meshes': [],
        'accessors': [], 'bufferViews': [], 'materials': [], 'textures': [], 'images': [], 'samplers': [],
        'extensionsUsed': ['KHR_materials_unlit'], 'extensionsRequired': ['KHR_materials_unlit']}
    binary = bytearray()
    for raw, provenance in zip([observed_glb, inferred_glb], provenances):
        source, source_binary = unpack(raw)
        if len(source['nodes']) != 1 or len(source['meshes']) != 1 or len(source['meshes'][0]['primitives']) != 1:
            raise ValueError('Only a single static primitive per input layer is supported')
        if set(source['nodes'][0]) - {'name', 'mesh', 'extras'}:
            raise ValueError('No hidden transform or animation may enter layer export')
        while len(binary) % 4:
            binary.append(0)
        binary_offset = len(binary)
        view_offset = len(document['bufferViews'])
        accessor_offset = len(document['accessors'])
        material_offset = len(document['materials'])
        image_offset = len(document['images'])
        texture_offset = len(document['textures'])
        sampler_offset = len(document['samplers'])
        for view in source['bufferViews']:
            value = copy.deepcopy(view)
            value['byteOffset'] = value.get('byteOffset', 0) + binary_offset
            document['bufferViews'].append(value)
        for accessor in source['accessors']:
            value = copy.deepcopy(accessor)
            value['bufferView'] += view_offset
            document['accessors'].append(value)
        for image in source['images']:
            if 'uri' in image:
                raise ValueError('External image references are unsupported')
            value = copy.deepcopy(image)
            value['bufferView'] += view_offset
            document['images'].append(value)
        document['samplers'].extend(copy.deepcopy(source['samplers']))
        for texture in source['textures']:
            value = copy.deepcopy(texture)
            value['source'] += image_offset
            value['sampler'] += sampler_offset
            document['textures'].append(value)
        for material in source['materials']:
            value = copy.deepcopy(material)
            value['name'] = provenance['layer_name'] + ' — original RGB appearance'
            value['pbrMetallicRoughness']['baseColorTexture']['index'] += texture_offset
            value['extras'] = copy.deepcopy(provenance)
            document['materials'].append(value)
        mesh = copy.deepcopy(source['meshes'][0])
        mesh['name'] = provenance['layer_name']
        mesh['extras'] = copy.deepcopy(provenance)
        primitive = mesh['primitives'][0]
        primitive['attributes'] = {key: value + accessor_offset for key, value in primitive['attributes'].items()}
        primitive['indices'] += accessor_offset
        primitive['material'] += material_offset
        primitive['extras'] = copy.deepcopy(provenance)
        mesh_index = len(document['meshes'])
        document['meshes'].append(mesh)
        document['nodes'].append({'name': provenance['layer_name'], 'mesh': mesh_index, 'extras': copy.deepcopy(provenance)})
        binary.extend(source_binary)
    document['buffers'] = [{'byteLength': len(binary)}]
    encoded = json.dumps(document, separators=(',', ':'), allow_nan=False).encode()
    encoded += b' ' * (-len(encoded) % 4)
    binary.extend(b'\0' * (-len(binary) % 4))
    return (struct.pack('<III', 0x46546c67, 2, 28 + len(encoded) + len(binary)) +
            struct.pack('<II', len(encoded), 0x4e4f534a) + encoded +
            struct.pack('<II', len(binary), 0x004e4942) + binary)


def annotate_single_layer(raw, provenance):
    """Attach explicit layer provenance without changing any binary attribute."""
    document, binary = unpack(raw)
    if len(document['nodes']) != 1 or len(document['meshes']) != 1 or len(document['meshes'][0]['primitives']) != 1:
        raise ValueError('Only one static inferred primitive is supported')
    if set(document['nodes'][0]) - {'name', 'mesh', 'extras'}:
        raise ValueError('No hidden transform may enter an inferred layer')
    for item in [document['nodes'][0], document['meshes'][0], document['meshes'][0]['primitives'][0], *document['materials']]:
        item['extras'] = copy.deepcopy(provenance)
    encoded = json.dumps(document, separators=(',', ':'), allow_nan=False).encode()
    encoded += b' ' * (-len(encoded) % 4)
    binary += b'\0' * (-len(binary) % 4)
    return (struct.pack('<III', 0x46546c67, 2, 28 + len(encoded) + len(binary)) +
            struct.pack('<II', len(encoded), 0x4e4f534a) + encoded +
            struct.pack('<II', len(binary), 0x004e4942) + binary)


def export_completion(root, source, output, *, observed_source=None, observed_export_path=None):
    root = Path(root).resolve()
    output = Path(output)
    if output.exists() or not output.resolve().is_relative_to(root / 'data'):
        raise ValueError('Use a fresh data export directory')
    source_folder = local_path(root, source)
    viewer_raw = read_checked(source_folder / 'viewer.json')
    viewer = json.loads(viewer_raw)
    protocol_raw = read_checked(source_folder / 'protocol.json')
    protocol = json.loads(protocol_raw)
    integrity = json.loads(read_checked(source_folder / 'input-integrity.json'))
    if integrity.get('unchanged') is not True or integrity['input_sha256'] != protocol['input_sha256']:
        raise ValueError('Completion source integrity is not established')
    checked_data_inputs = {}
    recorded_model_inputs = {}
    for path, expected in protocol['input_sha256'].items():
        if path.startswith('data/'):
            read_checked(local_path(root, path), expected)
            checked_data_inputs[path] = expected
        elif path.startswith('models/'):
            recorded_model_inputs[path] = expected
        else:
            raise ValueError('Unexpected completion input scope')
    implementation = read_checked(source_folder / 'implementation.py', protocol['implementation_sha256'])
    observed_folder = (local_path(root, observed_source) if observed_source is not None else
        root / 'data/observed-surface-experiment-20261031' / OBSERVED_ID)
    observed_id = observed_folder.name
    observed = json.loads(read_checked(observed_folder / 'viewer-display.json'))
    observed_export = (local_path(root, observed_export_path) if observed_export_path is not None else
        root / 'data/observed-surface-exports-20261031' / OBSERVED_ID)
    old_metadata_raw = read_checked(observed_export / 'metadata.json')
    old_metadata = json.loads(old_metadata_raw)
    observed_glb = read_checked(observed_export / 'observed-patch.glb', old_metadata['glb_sha256'])
    inferred = viewer['inferred_surface']
    if inferred['depth_observed'] is not False or inferred['observed_face_count'] != 0:
        raise ValueError('Added faces must remain explicitly inferred')
    if inferred['inferred_face_count'] != len(inferred['triangles']) or len(inferred['face_confidence']) != len(inferred['triangles']):
        raise ValueError('Inferred face provenance is incomplete')
    if set(inferred['face_confidence']) != {'unknown'}:
        raise ValueError('Inferred physical confidence must remain unknown')
    if inferred.get('watertight') is not False or inferred.get('thickness_known') is not False or inferred.get('hidden_surface_completion') is not False:
        raise ValueError('Unexpected completion or physical claims')
    if inferred['texture_url'] != observed['surface']['texture_url'] or inferred['texture_sha256'] != observed['surface']['texture_sha256']:
        raise ValueError('Observed and inferred layers must preserve the same original RGB frame')
    texture = read_checked(local_path(root, inferred['texture_url']), inferred['texture_sha256'])
    frame = next(f for f in observed['frames'] if f['id'] == inferred['source_frame_id'])
    camera = next(c for c in observed['cameras'] if c['frame_id'] == frame['id'])
    vertices, triangles = np.asarray(inferred['vertices'], float), np.asarray(inferred['triangles'], int)
    R, t, K = (np.asarray(camera[key], float) for key in ['R', 't', 'K'])
    xyz = vertices @ R.T + t
    projected = xyz @ K.T
    pixels = projected[:, :2] / projected[:, 2:]
    uv = np.c_[(pixels[:, 0] + .5) / frame['width'], 1 - (pixels[:, 1] + .5) / frame['height']]
    max_uv_error = float(np.max(np.abs(uv - np.asarray(inferred['uv']))))
    max_source_pixel_error = float(np.max(np.abs(pixels - np.asarray(inferred['vertex_source_pixels']))))
    if np.any(xyz[:, 2] <= 0) or max_uv_error > 1e-10 or max_source_pixel_error > 1e-8:
        raise ValueError('Saved inferred geometry/source UV association disagrees')
    tri = vertices[triangles]
    facing = np.sum(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]) * (-R.T @ t - tri.mean(axis=1)), axis=1)
    if np.any(facing <= 0):
        raise ValueError('Inferred winding must remain one-sided towards its source image')
    learned = viewer['kind'] == 'model_prior_depth_completion'
    provider = protocol['protocol']['model'] if learned else 'deterministic_harmonic_inverse_depth'
    expected_face_source = 'model_prior_aligned_inverse_depth' if learned else 'inferred_harmonic_inverse_depth'
    if len(inferred['face_provenance']) != len(triangles) or set(inferred['face_provenance']) != {expected_face_source}:
        raise ValueError('Per-face inferred source does not match declared provider')
    common = {'component_id': viewer['component_id'], 'source_frame_id': frame['id'],
        'source_timestamp_s': frame['timestamp_s'], 'appearance_source': 'exact_original_source_RGB_image',
        'texture_sha256': digest(texture), 'physical_confidence': 'unknown',
        'camera_calibrated': False, 'physical_scale_known': False, 'physics_ready': False,
        'coordinate_units': 'independent arbitrary observed reconstruction gauge; not measured metres'}
    observed_provenance = dict(common, layer_name='Observed surface', provider='OpenCV multi-view stereo',
        shape_source='observed_training_multiview_stereo', depth_observed=True,
        triangle_count=len(observed['surface']['triangles']), source_run_id=observed_id,
        independent_shape_accuracy_measured=False, optional_inferred_layer=False)
    inferred_provenance = dict(common, layer_name='Inferred depth prior' if learned else 'Inferred smooth gaps',
        provider=provider, shape_source=expected_face_source, depth_observed=False,
        triangle_count=len(triangles), completion_id=viewer['id'],
        all_faces_inferred=True, every_face_confidence='unknown', optional_inferred_layer=True,
        source_viewer_sha256=digest(viewer_raw), source_protocol_sha256=digest(protocol_raw))
    if learned:
        inferred_provenance['model_provenance'] = protocol['model_provenance']
    inferred_glb, conversion = make_glb(vertices, triangles, inferred['uv'], texture,
        name=inferred_provenance['layer_name'], extras=inferred_provenance)
    inferred_glb = annotate_single_layer(inferred_glb, inferred_provenance)
    combined = combine_layers(observed_glb, inferred_glb, [observed_provenance, inferred_provenance])
    metadata = {'format': 'observed-and-inferred-surface-export', 'version': 1,
        'id': viewer['id'], 'source_run_id': observed_id, 'completion_id': viewer['id'],
        'label': viewer['title'], 'status': viewer['status'], 'physical_confidence': 'unknown',
        'physical_scale_known': False, 'camera_calibrated': False, 'physics_ready': False,
        'complete_object': False, 'watertight': False, 'observed_geometry_changed': False,
        'observed_glb_sha256': digest(observed_glb), 'glb_sha256': digest(combined),
        'inferred_only_glb_sha256': digest(inferred_glb),
        'layers': [observed_provenance, inferred_provenance],
        'observed_triangles': len(observed['surface']['triangles']), 'inferred_triangles': len(triangles),
        'observed_vertices': len(observed['surface']['vertices']), 'inferred_vertices': len(vertices),
        'source_viewer_path': source + '/viewer.json', 'source_viewer_sha256': digest(viewer_raw),
        'source_protocol_sha256': digest(protocol_raw), 'source_implementation_sha256': digest(implementation),
        'source_integrity_sha256': digest((source_folder / 'input-integrity.json').read_bytes()),
        'checked_data_inputs': checked_data_inputs,
        'model_inputs_attested_by_original_inference_protocol_not_reloaded': recorded_model_inputs,
        'representation_conversion': conversion,
        'inferred_source_uv_max_error': max_uv_error, 'inferred_source_pixel_max_error': max_source_pixel_error,
        'source_camera': camera, 'source_dimensions': [frame['width'], frame['height']],
        'limitations': viewer['limitations'] + [
            'Both layers are enabled in this explicitly optional export. Hide or remove the inferred named node to retain the observed patch.',
            'Every added triangle has unknown physical confidence. Original RGB texture does not establish depth or contact accuracy.',
            'No shape, normals, smoothing, rescaling, completion or model inference was added by the exporter.',
            'A glTF viewer may assume metres; these coordinates have an arbitrary unmeasured gauge.'],
        'new_api_calls': 0, 'new_model_inference_calls': 0,
        'export_implementation_sha256': digest(Path(__file__).read_bytes()),
        'geometry_encoder_sha256': digest(Path(__file__).with_name('observed_surface_export.py').read_bytes())}
    readme = ('OBSERVED + INFERRED SURFACE — OPTIONAL TWO-LAYER EXPORT\n\n' +
        viewer['title'] + '\n\n' +
        'Observed surface: unchanged original source-textured stereo patch.\n' +
        inferred_provenance['layer_name'] + ': ' + provider + '; EVERY face is inferred and physically unverified.\n\n' +
        'The two layers are separate named glTF nodes, meshes, primitives and materials.\n' +
        'Each node/primitive/material carries provider, shape source and unknown physical confidence in extras.\n' +
        'Remove or hide the inferred node to inspect only original observed geometry.\n' +
        'The original JPEG is embedded without tint or lighting. Source appearance is not evidence that inferred depth is accurate.\n\n' +
        '\n'.join('- ' + text for text in metadata['limitations']) + '\n').encode()
    output.mkdir(parents=True)
    (output / 'observed-and-inferred.glb').write_bytes(combined)
    (output / 'inferred-only.glb').write_bytes(inferred_glb)
    (output / 'metadata.json').write_bytes(json_bytes(metadata))
    (output / 'README.txt').write_bytes(readme)
    (output / 'export-implementation.py').write_bytes(Path(__file__).read_bytes())
    base = '/files/' + output.resolve().relative_to(root).as_posix()
    return {'id': viewer['id'], 'source_run_id': observed_id, 'completion_id': viewer['id'],
        'label': viewer['title'], 'status': viewer['status'],
        'glb_url': base + '/observed-and-inferred.glb', 'metadata_url': base + '/metadata.json',
        'inferred_only_url': base + '/inferred-only.glb', 'inferred_only_sha256': digest(inferred_glb),
        'readme_url': base + '/README.txt', 'inferred_provider': provider,
        'glb_sha256': digest(combined), 'glb_bytes': len(combined),
        'observed_triangles': metadata['observed_triangles'], 'inferred_triangles': metadata['inferred_triangles'],
        'observed_vertices': metadata['observed_vertices'], 'inferred_vertices': metadata['inferred_vertices'],
        'physical_confidence': 'unknown', 'physical_scale_known': False, 'physics_ready': False,
        'observed_geometry_changed': False, 'inferred_layer_optional': True}


def export_completions(root):
    root = Path(root).resolve()
    output = root / 'data/observed-surface-exports-20261031'
    manifest_path = output / 'completion-exports.json'
    if manifest_path.exists():
        raise ValueError('Preserve existing completion exports')
    entries = [export_completion(root, source, output / Path(source).name) for source in CASES]
    manifest = {'format': 'observed-completion-exports', 'version': 1, 'exports': entries,
        'default_observed_exports_unchanged': True, 'new_api_calls': 0, 'new_model_inference_calls': 0}
    manifest_path.write_bytes(json_bytes(manifest))
    return manifest
