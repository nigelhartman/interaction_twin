#!/usr/bin/env python3
"""Exact, Host/Origin-fenced private viewer for one CPU appearance research fit."""
import argparse
import json
from http.server import ThreadingHTTPServer
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from scripts import serve_laptop_hinge_review_20261103_v1 as B
from scripts import serve_laptop_hinge_review_20261103_v2 as H
import diagnose_laptop_orientation_20261103_v1 as F
import native_orientation_diagnostic_v1 as D
import object_program as O


def preparation():
    folder = ROOT / F.OUTPUT / 'appearance-refit-v1'
    result_path = folder / 'result.json'; result = json.loads(result_path.read_text())
    verification_path = folder / 'verification-v1/verification.json'; verification = json.loads(verification_path.read_text())
    D.require(result.get('joint_native_anchors_recomputed') is True and result['fit_result']['joints'] == 1 and result['fit_result']['unresolved_joints'] == 0, 'An actual complete new native-anchor hinge fit is required')
    D.require(verification['result_ref'] == F.ref(result_path) and verification['scene_ref'] == result['fit_result']['manifest'], 'Exact private verification lineage required')
    D.require(verification['dense_fk']['maximum_matrix_error'] < 1e-8 and verification['dense_fk']['maximum_native_anchor_gap'] < 1e-8 and verification['dense_fk']['exact_native_vertex_anchors'] is True, 'Native-anchor FK verification required')
    scene_path = B.checked(result['fit_result']['manifest'], ROOT); scene = json.loads(scene_path.read_text())
    program_path = B.checked(scene['original_program'], ROOT); program = json.loads(program_path.read_text())
    object_path = B.checked(result['descriptive_export']['object_program'], ROOT); O.validate(json.loads(object_path.read_text()), root=ROOT)
    baseline = json.loads(B.checked(result['baseline_scene_ref'], ROOT).read_text())
    D.require({c['id']: c['asset'] for c in scene['components']} == {c['id']: c['asset'] for c in baseline['components']}, 'Unchanged exact native meshes required')
    video = ROOT / program['source']['path']; D.require(F.ref(video)['sha256'] == scene['source']['sha256'], 'Exact normalized source video required')
    joint = scene['mechanics']['joints'][0]; supported = [row for row in joint['samples'] if row['q_rad'] is not None]
    D.require(supported, 'No supported hinge interval')
    preview = min(supported, key=lambda row: abs(row['source_sample_index'] - F.PRIMARY))['timestamp_s']
    manifest = {'format': 'interaction-twin-private-laptop-hinge-review', 'version': 1,
                'scene_ref': F.ref(scene_path), 'program_ref': F.ref(program_path), 'object_program_ref': F.ref(object_path), 'source_video_ref': F.ref(video),
                'assets': [{'id': c['id'], 'asset_ref': c['asset']} for c in scene['components']],
                'supported_sample_count': len(supported), 'unknown_sample_count': len(joint['samples']) - len(supported), 'preview_time_s': preview,
                'qa_ref': F.ref(verification_path), 'scope': 'Automatic appearance-initialized hinge research; exact native geometry unchanged. Joint configuration is not a verified opening angle. Unknown intervals are withheld.',
                'physical_accuracy_verified': False, 'simulation_ready': False, 'published': False}
    manifest_path = folder / 'browser-manifest.json'
    if manifest_path.exists(): D.require(json.loads(manifest_path.read_text()) == manifest, 'Preserve exact viewer manifest')
    else: F.write_new(manifest_path, manifest)
    files = [F.ref(ROOT / name) for name in B.RUNTIME] + [F.ref(manifest_path), manifest['scene_ref'], manifest['program_ref'], manifest['object_program_ref'], manifest['source_video_ref'], manifest['qa_ref'], result['diagnostic_ref'], *[a['asset_ref'] for a in manifest['assets']]]
    routes = {'/' + name: item for name, item in zip(B.RUNTIME, files)}
    for item in files[len(B.RUNTIME):]: routes['/files/' + item['path']] = item
    routes['/manifest.json'] = F.ref(manifest_path); routes['/'] = routes['/showcase/laptop-hinge-review-v1.html']
    for item in routes.values(): B.checked(item, ROOT)
    return manifest, routes


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--port', type=int, default=0); parser.add_argument('--server-id', default='appearance1'); args = parser.parse_args()
    D.require(0 <= args.port <= 65535 and re.fullmatch(r'[a-z0-9-]{1,32}', args.server_id), 'Bounded local server identity required')
    manifest, routes = preparation(); server = ThreadingHTTPServer(('127.0.0.1', args.port), H.handler(routes, root=ROOT))
    receipt = {'format': 'interaction-twin-private-laptop-appearance-server', 'version': 1, 'origin': f'http://127.0.0.1:{server.server_port}', 'private_manifest': manifest, 'routes': routes,
               'loopback_host_and_origin_guard': True, 'exact_hash_allowlist': True, 'read_only': True,
               'implementation': F.ref(Path(__file__)), 'base_implementation': F.ref(Path(B.__file__)), 'host_guard_implementation': F.ref(Path(H.__file__)),
               'new_api_calls': 0, 'new_gpu_calls': 0, 'published': False}
    F.write_new(ROOT / 'output/new-captures-20261103' / ('laptop-hinge-server-' + args.server_id + '.json'), receipt)
    print(receipt['origin'], flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


if __name__ == '__main__': main()
