#!/usr/bin/env python3
"""Read-only audit of the bounded reconstruction workflow delivery.

No inference runs. Only explicitly named saved workflow jobs are read. Native
GLB import is numeric inspection, not a rendered or physical-accuracy check.
"""
import hashlib
import io
import json
from pathlib import Path
import struct

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data/observed-app-runs-20261031'
JOBS = ['observed-f50c947783fcc2b10ffde72e', 'observed-4f94939ac54ba1978a75eaf8',
        'observed-0086b178fca527569210ee12', 'observed-9297b9f81b72d593fd8cf41b',
        'observed-112663a646eba4fb38ec3893', 'observed-89f7033aa6a0da0611263a9d',
        'observed-f3273b126bc7a6c81166feb2']


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(value):
    if value.startswith('/files/'):
        value = value[len('/files/'):]
    path = (ROOT / value).resolve()
    assert value.startswith('data/') and path.is_relative_to(ROOT / 'data')
    return path


def decode(raw):
    assert struct.unpack_from('<III', raw) == (0x46546c67, 2, len(raw))
    n, kind = struct.unpack_from('<II', raw, 12)
    assert kind == 0x4e4f534a
    document = json.loads(raw[20:20 + n])
    size, kind = struct.unpack_from('<II', raw, 20 + n)
    assert kind == 0x004e4942
    return document, raw[28 + n:28 + n + size]


def accessor(document, binary, index):
    item = document['accessors'][index]
    view = document['bufferViews'][item['bufferView']]
    size = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3}[item['type']]
    dtype = {5126: '<f4', 5123: '<u2', 5125: '<u4'}[item['componentType']]
    return np.frombuffer(binary, dtype=dtype, count=item['count'] * size,
        offset=view.get('byteOffset', 0) + item.get('byteOffset', 0)).reshape(item['count'], size)


def main():
    checks, rows = 0, []
    def require(condition):
        nonlocal checks
        checks += 1
        assert condition
    for identifier in JOBS:
        folder = BASE / (identifier + '-workflow')
        result = read(folder / 'result.json')
        contract = folder / 'contract.json'
        require(sha(folder / 'implementation.py') == read(contract)['implementation_sha256'])
        require(result['new_api_calls'] == 0 and result['production_assets_changed'] is False)
        request = result['request']
        require(sha(artifact(request['interaction_path'])) == request['interaction_sha256'])
        pixel_files = set()
        first = result['stages'][0].get('result', {})
        if first.get('viewer_url'):
            protocol = read(artifact(first['viewer_url']).parent / 'protocol.json')
            require(all(protocol[flag] is False for flag in ['manual_coordinates',
                'generated_meshes_read', 'depth_maps_read', 'old_poses_read']))
            require(sha(artifact(protocol['source']['path'])) == protocol['source']['sha256'])
            for frame in protocol['input_views']:
                require(np.isfinite(frame['timestamp_s']))
                for url, expected in [('image_url', 'image_sha256'), ('mask_url', 'mask_sha256')]:
                    source = artifact(frame[url])
                    require(sha(source) == frame[expected]); pixel_files.add(str(source))
                for other in frame.get('subtracted_components', []):
                    source = artifact(other['mask_path'])
                    require(sha(source) == other['mask_sha256']); pixel_files.add(str(source))
        prior, artifacts, chained_stages = {}, 0, 0
        for stage in result['stages']:
            start = read(folder / 'stages' / stage['stage'] / 'start.json')
            saved = folder / 'stages' / stage['stage'] / 'result.json'
            require(read(saved) == stage)
            if 'upstream_results_sha256' in start:
                require(start['upstream_results_sha256'] == prior)
                require(start['workflow_contract_sha256'] == sha(contract))
                chained_stages += 1
            for path, expected in stage.get('artifact_sha256', {}).items():
                require(sha(artifact(path)) == expected); artifacts += 1
            prior[stage['stage']] = sha(saved)
        exports, native = result.get('exports', {}), []
        for key in ['observed', 'inferred']:
            if key not in exports:
                continue
            entry = exports[key]
            path = artifact(entry['glb_url'])
            require(sha(path) == entry['glb_sha256'])
            raw = path.read_bytes(); doc, binary = decode(raw)
            scene = trimesh.load(io.BytesIO(raw), file_type='glb', force='scene', process=False)
            require(len(scene.geometry) == len(doc['meshes']))
            expected_vertices, expected_faces = 0, 0
            for mesh in doc['meshes']:
                primitive = mesh['primitives'][0]
                vertices = accessor(doc, binary, primitive['attributes']['POSITION'])
                faces = accessor(doc, binary, primitive['indices']).reshape(-1, 3)
                require(np.isfinite(vertices).all() and faces.max() < len(vertices))
                expected_vertices += len(vertices); expected_faces += len(faces)
            require(sum(len(g.vertices) for g in scene.geometry.values()) == expected_vertices)
            require(sum(len(g.faces) for g in scene.geometry.values()) == expected_faces)
            native.append({'kind': key, 'meshes': len(scene.geometry),
                'vertices': expected_vertices, 'triangles': expected_faces,
                'inspection': 'independent_Trimesh_numeric_import_no_render_or_physical_validation'})
            if key == 'inferred':
                require(len(doc['nodes']) == 2)
                require(doc['nodes'][0]['extras']['depth_observed'] is True)
                require(doc['nodes'][1]['extras']['all_faces_inferred'] is True)
                require(doc['nodes'][1]['extras']['every_face_confidence'] == 'unknown')
                inferred_only = artifact(entry['inferred_only_url'])
                require(sha(inferred_only) == entry['inferred_only_sha256'])
        rows.append({'id': identifier, 'status': result['status'],
            'stage_statuses': {s['stage']: s['status'] for s in result['stages']},
            'artifact_hashes_checked': artifacts, 'native_imports': native,
            'hash_chained_stages': chained_stages,
            'legacy_pre_chain_contract': chained_stages != len(result['stages']),
            'source_pixel_files_verified': len(pixel_files),
            'immutable_result_sha256': sha(folder / 'result.json')})
    report = {'format': 'observed-local-workflow-delivery-audit', 'version': 1,
        'passed': True, 'checks': checks, 'jobs': rows,
        'audit_implementation_sha256': sha(Path(__file__)),
        'test_logs': ['output/geometry-app-20261031/workflow-focused-tests.log',
                      'output/geometry-app-20261031/workflow-follow-on-tests.log'],
        'new_api_calls': 0, 'cluster_contacted': False, 'downloads': False,
        'audit_model_inference_calls': 0,
        'actual_new_MoGe_predictions_in_tested_jobs': 1,
        'actual_new_CoTracker_predictions_in_tested_jobs': 0,
        'caveats': ['CoTracker cache source used genuine separately audited learned 2D tracking before these jobs.',
            'Camera CoTracker downstream surface has zero eligible dense heldout comparisons; optional completion is blocked in the latest workflow.',
            'Jabra lid fails correspondence; no surface or assembly was invented.',
            'MPS-unavailable failed attempt is preserved separately from the later explicitly created successful run.',
            'Arbitrary scale, uncertain intrinsics, incomplete visible patches and physical accuracy remain unresolved.',
            'No native rendered appearance or physical simulator inspection is claimed by this audit.']}
    destination = ROOT / 'output/geometry-app-20261031/workflow-delivery-v2.json'
    with destination.open('x') as stream:
        json.dump(report, stream, indent=2); stream.write('\n')
    print(json.dumps({'passed': True, 'checks': checks, 'jobs': len(rows), 'report': str(destination)}))


if __name__ == '__main__':
    main()
