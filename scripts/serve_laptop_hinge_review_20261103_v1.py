#!/usr/bin/env python3
"""Read-only loopback allowlist for one private, unapproved laptop hinge review."""
import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import laptop_hinge_relocalization_v2 as L
import laptop_hinge_omission_recovery_v1 as R
import object_program as O

RUNTIME = [
    'showcase/laptop-hinge-review-v1.html', 'showcase/laptop-hinge-review-v1.js',
    'showcase/brand/v2/mark.svg',
    'showcase/aligned-scene.html', 'showcase/aligned-scene.js', 'showcase/aligned-scene-contract.js', 'showcase/aligned-scene-client.js',
    'workspace/robot-vendor/three.module.js', 'workspace/robot-vendor/three.core.js', 'workspace/robot-vendor/GLTFLoader.js',
    'workspace/robot-vendor/OrbitControls.js', 'workspace/robot-vendor/BufferGeometryUtils.js',
]


def checked(reference, root):
    R.require(isinstance(reference, dict) and set(reference) == {'path', 'sha256', 'bytes'}, 'Exact private review reference required')
    path = root / reference['path']
    R.require(path.resolve() == path and path.is_relative_to(root) and not any(p.startswith('.') for p in Path(reference['path']).parts)
              and not any(p.is_symlink() for p in [path, *path.parents]) and path.is_file()
              and type(reference['bytes']) is int and 0 < reference['bytes'] <= 128 * 1024 * 1024,
              'Ordinary bounded review artifact required')
    R.require(R.ref(path, root) == reference, 'Private review artifact changed')
    return path


def preparation(*, root=ROOT):
    root = Path(root).resolve()
    folder = root / L.OUTPUT / 'local-fit-v1'
    result = R.read(folder / 'result.json')
    scene_path = checked(result['scene']['manifest'], root)
    scene = R.read(scene_path)
    program_path = checked(scene['original_program'], root)
    program = R.read(program_path)
    object_path = checked(result['descriptive_export']['object_program'], root)
    O.validate(R.read(object_path), root=root)
    video = checked(R.ref(root / program['source']['path'], root), root)
    R.require(R.M.sha(video) == scene['source']['sha256'] and len(scene['components']) == 2, 'Exact normalized source/two native parts required')
    joint = scene['mechanics']['joints'][0]
    manifest = {'format': 'interaction-twin-private-laptop-hinge-review', 'version': 1,
        'scene_ref': R.ref(scene_path, root), 'program_ref': R.ref(program_path, root),
        'object_program_ref': R.ref(object_path, root), 'source_video_ref': R.ref(video, root),
        'assets': [{'id': c['id'], 'asset_ref': c['asset']} for c in scene['components']],
        'supported_sample_count': sum(s['q_rad'] is not None for s in joint['samples']),
        'unknown_sample_count': sum(s['q_rad'] is None for s in joint['samples']),
        'preview_time_s': 20.160089, 'qa_ref': R.ref(folder / 'dense-fk-audit.json', root),
        'scope': 'Estimated native hinge; configuration values are not verified opening angles. Missing intervals are withheld.',
        'physical_accuracy_verified': False, 'simulation_ready': False, 'published': False}
    path = folder / 'browser-manifest.json'
    if path.exists():
        R.require(R.read(path) == manifest, 'Preserved browser review manifest changed')
    else:
        R.write_new(path, manifest)
    files = [R.ref(root / p, root) for p in RUNTIME] + [R.ref(path, root), manifest['scene_ref'], manifest['program_ref'],
        manifest['object_program_ref'], manifest['source_video_ref'], manifest['qa_ref'], *[a['asset_ref'] for a in manifest['assets']]]
    routes = {'/' + name: item for name, item in zip(RUNTIME, files)}
    for item in files[len(RUNTIME):]:
        routes['/files/' + item['path']] = item
    routes['/manifest.json'] = R.ref(path, root)
    routes['/'] = routes['/showcase/laptop-hinge-review-v1.html']
    for item in routes.values():
        checked(item, root)
    return manifest, routes


def handler(routes, *, root=ROOT):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.respond(False)

        def do_HEAD(self):
            self.respond(True)

        def respond(self, head):
            parts = urlsplit(self.path)
            if len(self.path) > 4096 or '%' in parts.path or '\\' in parts.path:
                self.send_error(400)
                return
            item = routes.get(parts.path)
            if item is None:
                self.send_error(404)
                return
            try:
                path = checked(item, root)
            except (ValueError, OSError):
                self.send_error(409)
                return
            length = item['bytes']
            start, end, code = 0, length - 1, 200
            requested = self.headers.get('Range')
            if requested:
                match = re.fullmatch(r'bytes=(\d+)-(\d*)', requested)
                if not match or int(match[1]) >= length:
                    self.send_error(416)
                    return
                start = int(match[1]); end = min(int(match[2]) if match[2] else end, end)
                if end < start:
                    self.send_error(416)
                    return
                code = 206
            mime = 'text/javascript' if path.suffix == '.js' else mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
            self.send_response(code)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(end - start + 1))
            self.send_header('Accept-Ranges', 'bytes')
            if code == 206:
                self.send_header('Content-Range', f'bytes {start}-{end}/{length}')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.end_headers()
            if not head:
                try:
                    with path.open('rb') as stream:
                        stream.seek(start)
                        remaining = end - start + 1
                        while remaining:
                            raw = stream.read(min(1024 * 1024, remaining))
                            if not raw:
                                break
                            self.wfile.write(raw)
                            remaining -= len(raw)
                except (BrokenPipeError, ConnectionResetError):
                    pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--server-id', default='preview1')
    args = parser.parse_args()
    R.require(0 <= args.port <= 65535 and re.fullmatch(r'[a-z0-9-]{1,32}', args.server_id), 'Bounded loopback server identity required')
    manifest, routes = preparation()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler(routes))
    receipt = {'format': 'interaction-twin-private-laptop-review-server', 'version': 1,
               'origin': f'http://127.0.0.1:{server.server_port}', 'routes': routes,
               'private_manifest': manifest, 'read_only': True, 'new_api_calls': 0, 'new_gpu_calls': 0, 'published': False}
    destination = ROOT / 'output/new-captures-20261103' / ('laptop-hinge-server-' + args.server_id + '.json')
    R.write_new(destination, receipt)
    print(receipt['origin'], flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
