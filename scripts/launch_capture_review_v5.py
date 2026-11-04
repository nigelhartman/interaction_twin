#!/usr/bin/env python3
"""Reopen three frozen, private capture reviews; no inference or app worker."""
from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import socket
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
import uuid
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
CONFIG = 'scripts/capture-review-viewers-20261104-v5.json'
CONFIG_SHA = '096446c164fb614f7a316757a11bd6f7b7fc4eeed406a5d2a5bd7fa66cf1834b'
SELF_PIN = 'scripts/capture-review-launcher-20261104-v5.sha256'
RECEIPTS = 'output/new-captures-20261104/capture-review-launcher-v5'
MAX_FILE = 128 * 1024 * 1024
PRODUCT = 'InteractionTwinCaptureReview/5'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def checked(reference, root=ROOT):
    """One bounded read is both verified and returned; no stat-only trust cache."""
    require(isinstance(reference, dict) and set(reference) == {'path', 'sha256', 'bytes'}, 'Exact frozen file reference required')
    name, size, digest = reference['path'], reference['bytes'], reference['sha256']
    require(isinstance(name, str) and '\\' not in name and len(name) < 2048, 'Ordinary local path required')
    parts = PurePosixPath(name).parts
    require(parts and str(PurePosixPath(name)) == name and not name.startswith('/')
            and all(p not in {'.', '..'} and not p.startswith('.') for p in parts), 'Private or escaping path rejected')
    require(parts[0] in {'data', 'output', 'scripts', 'showcase', 'workspace'}, 'File outside frozen review scope')
    require(type(size) is int and 0 < size <= MAX_FILE and isinstance(digest, str) and re.fullmatch('[a-f0-9]{64}', digest), 'Bounded SHA-256 reference required')
    root = Path(root).resolve(); path = root / name
    require(all(not p.is_symlink() for p in [path, *path.parents]) and path.resolve() == path and path.is_file(), 'Ordinary nonsymlink file required')
    require(path.stat().st_size == size, 'Frozen review file length changed: ' + name)
    with path.open('rb') as stream:
        raw = stream.read(size + 1)
    require(len(raw) == size and sha(raw) == digest, 'Frozen review file hash changed: ' + name)
    return raw


def reference(path, root=ROOT):
    root = Path(root).resolve(); path = Path(path).resolve()
    require(path.is_relative_to(root), 'Receipt file must remain in this project')
    raw = path.read_bytes()
    return {'path': str(path.relative_to(root)), 'sha256': sha(raw), 'bytes': len(raw)}


def load_configuration(root=ROOT):
    root = Path(root).resolve()
    require(sha((root / CONFIG).read_bytes()) == CONFIG_SHA, 'Frozen viewer configuration changed')
    expected = (root / SELF_PIN).read_text().strip()
    require(re.fullmatch('[a-f0-9]{64}', expected) and sha(Path(__file__).read_bytes()) == expected, 'Launcher code changed; review its new version before launch')
    config = json.loads((root / CONFIG).read_bytes())
    require(config.get('format') == 'interaction-twin-frozen-capture-review-launcher' and config.get('version') == 5
            and config.get('public_promotion') is False and config.get('read_only') is True and config.get('model_calls') == 0,
            'Frozen read-only configuration required')
    require(len(config['viewers']) == 3 and len({s['port'] for s in config['viewers']}) == 3, 'Exact three viewer ports required')
    return config


def source_binding(snapshot, spec, root):
    """Recheck the saved source/program identities without importing any compiler."""
    if spec['kind'] == 'laptop':
        m = snapshot['private_manifest']
        require(m.get('format') == 'interaction-twin-private-laptop-hinge-review' and m.get('published') is False
                and m.get('physical_accuracy_verified') is False and m.get('simulation_ready') is False, 'Private unverified laptop manifest required')
        scene = json.loads(checked(m['scene_ref'], root)); program = json.loads(checked(m['program_ref'], root))
        obj = json.loads(checked(m['object_program_ref'], root)); checked(m['source_video_ref'], root); checked(m['qa_ref'], root)
        require(scene['original_program'] == m['program_ref'] and scene['source']['sha256'] == m['source_video_ref']['sha256']
                and program['source']['path'] == m['source_video_ref']['path'] and program['source']['sha256'] == m['source_video_ref']['sha256'], 'Laptop source/scene binding changed')
        require(obj['evidence_program'] == m['program_ref'] and obj['constrained_scene'] == m['scene_ref'], 'Descriptive laptop program binding changed')
        actual = {c['id']: c['asset'] for c in scene['components']}
        require(actual == {a['id']: a['asset_ref'] for a in m['assets']}, 'Laptop native asset inventory changed')
        require(snapshot['routes']['/manifest.json'] == spec['browser_manifest_ref'], 'Laptop browser manifest changed')
        require(json.loads(checked(snapshot['routes']['/manifest.json'], root)) == m, 'Laptop HTTP manifest differs from frozen selection')
    elif spec['id'] == 'capture-review-v6':
        require([c['capture_id'] for c in snapshot['cases']] == ['charger', 'phone', 'laptop'], 'Exact three selected captures required')
        require(sum(c['native_asset_count'] for c in snapshot['cases']) == 9, 'Exact nine native models required')
        for case in snapshot['cases']:
            program = json.loads(checked(case['source_program_ref'], root)); checked(case['source_ref'], root)
            checked(case['normalized_source_ref'], root)
            semantic = json.loads(checked(case['semantic_companion_ref'], root))
            source = program['source']
            require(source['path'] == case['normalized_source_ref']['path'] and source['sha256'] == case['normalized_source_ref']['sha256']
                    and source['original_sha256'] == case['source_ref']['sha256'], 'Capture normalized/original source binding changed')
            require(semantic['program_ref'] == case['source_program_ref'] and semantic['source_sha256'] == source['sha256'], 'Semantic companion source binding changed')
            require(snapshot['routes']['c/' + case['capture_id'] + '/video.mp4'] == case['normalized_source_ref'], 'Served source video differs')
            for part in case['parts']:
                checked(part['asset'], root)
        require(snapshot['supplemental_hinge_experiment']['url'] == 'http://127.0.0.1:52154/'
                and snapshot['separate_glass_diagnostic']['url'] == 'http://127.0.0.1:65139/3ze7m4pGSDGkBdaGpzsM05qaHbLEmVRr/', 'Frozen linked viewer changed')
    else:
        require(spec['id'] == 'glass-diagnostic-v4' and len(snapshot['cases']) == 1
                and snapshot['cases'][0]['semantic_status'] == 'old_bottle_run_not_approved', 'Glass diagnostic must stay separate and unapproved')


def prepare_spec(spec, *, root=ROOT, check_source=True):
    require(re.fullmatch('[a-z0-9-]{1,40}', spec['id']) and type(spec['port']) is int and 1024 <= spec['port'] <= 65535, 'Fixed loopback viewer identity required')
    require(spec['prefix'] == '/' or re.fullmatch('/[A-Za-z0-9_-]{24,64}/', spec['prefix']), 'Exact frozen viewer prefix required')
    for item in spec['code_refs']:
        checked(item, root)
    snapshot = json.loads(checked(spec['snapshot_ref'], root))
    if spec['kind'] == 'batch':
        require(snapshot['format'] == 'interaction-twin-private-native-batch-snapshot' and snapshot['public_promotion'] is False, 'Private frozen batch required')
        old = json.loads(checked(spec['legacy_launch_ref'], root))
        require(old['snapshot'] == spec['snapshot_ref'] and old['url'] == url(spec), 'Frozen launch/snapshot mismatch')
        routes = {}
        for name, item in snapshot['routes'].items():
            require(isinstance(name, str) and name and not name.startswith('/') and all(p not in {'.', '..'} for p in name.split('/'))
                    and not any(c in name for c in ['%', '\\', '?', '#']), 'Unsafe frozen route')
            routes[spec['prefix'] + name] = item
        routes[spec['prefix']] = snapshot['routes']['index.html']
    else:
        require(spec['kind'] == 'laptop' and snapshot['format'] == 'interaction-twin-private-laptop-appearance-server'
                and snapshot['version'] == 1 and snapshot['read_only'] is True and snapshot['published'] is False
                and snapshot['origin'] + '/' == url(spec), 'Guarded private laptop snapshot required')
        routes = snapshot['routes']
        require('/' in routes and '/manifest.json' in routes, 'Laptop readiness routes missing')
    for name, item in routes.items():
        require(name.startswith('/') and not any(c in name for c in ['%', '\\', '?', '#'])
                and all(p not in {'.', '..'} for p in name.split('/')), 'Invalid exact route')
        checked(item, root)
    if check_source:
        source_binding(snapshot, spec, Path(root))
    return {**spec, 'routes': routes}


def url(spec):
    return f"http://127.0.0.1:{spec['port']}{spec['prefix']}"


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, file, code, message, headers, newurl):
        return None


def http_ready(spec, *, timeout=3):
    """Verify every exact served byte, with no redirects, proxies, cookies or auth."""
    opener = build_opener(ProxyHandler({}), NoRedirect())
    try:
        for route, item in spec['routes'].items():
            with opener.open(Request(f"http://127.0.0.1:{spec['port']}" + route), timeout=timeout) as response:
                if response.status != 200 or response.headers.get('Content-Length') != str(item['bytes']):
                    return False
                raw = response.read(item['bytes'] + 1)
                if len(raw) != item['bytes'] or sha(raw) != item['sha256']:
                    return False
        for path in [spec['prefix'] + '__not_allowlisted__', spec['prefix'] + '.secrets', spec['prefix'] + 'admin/']:
            try:
                with opener.open(Request(f"http://127.0.0.1:{spec['port']}" + path), timeout=timeout):
                    return False
            except HTTPError as exc:
                status = exc.code; exc.close()
                if status != 404:
                    return False
        return True
    except (OSError, ValueError):
        return False


def port_free(port):
    try:
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', port))
        return True
    except OSError:
        return False


def listener_identity(port):
    """Query only this fixed listening port and its PID; never enumerate processes."""
    result = subprocess.run(['/usr/sbin/lsof', '-n', '-P', '-t', '-iTCP:' + str(port), '-sTCP:LISTEN'], capture_output=True, text=True, timeout=5)
    pids = sorted(set(result.stdout.split()))
    require(result.returncode == 0 and len(pids) == 1 and pids[0].isdigit(), 'Occupied port has no unique verifiable owner')
    pid = int(pids[0])
    result = subprocess.run(['/bin/ps', '-p', str(pid), '-o', 'uid=', '-o', 'command='], capture_output=True, text=True, timeout=5)
    parts = result.stdout.strip().split(None, 1)
    require(result.returncode == 0 and len(parts) == 2 and parts[0].isdigit(), 'Cannot verify listener owner')
    return {'pid': pid, 'uid': int(parts[0]), 'argv': shlex.split(parts[1])}


def owned_listener(spec, identity, *, root=ROOT, uid=None):
    if identity.get('uid') != (os.getuid() if uid is None else uid):
        return False
    argv = identity.get('argv', [])
    if len(argv) < 2:
        return False
    script = str((Path(root) / 'scripts/launch_capture_review_v5.py').resolve())
    if argv[1:] == [script, '--serve', spec['id']]:
        return True
    if identity.get('pid') != spec['legacy_pid']:
        return False
    # Frozen historical launches used project-relative script and snapshot paths.
    expected = spec['legacy_command_script']
    if argv[1] not in {expected, str(Path(root) / expected)}:
        return False
    if spec['kind'] == 'batch':
        return argv[2:] == ['serve', '--snapshot', spec['legacy_snapshot'], '--port', '0']
    return argv[2:] == ['--port', '0', '--server-id', 'appearance1']


def preflight(spec, *, root=ROOT):
    if port_free(spec['port']):
        return {'status': 'free'}
    identity = listener_identity(spec['port'])
    require(owned_listener(spec, identity, root=root), 'Port ' + str(spec['port']) + ' is occupied by an unrelated process; nothing was stopped')
    require(http_ready(spec), 'Existing viewer failed exact HTTP readiness; nothing was stopped')
    return {'status': 'reuse', 'pid': identity['pid'], 'uid': identity['uid'], 'http_route_count': len(spec['routes'])}


def handler(spec, *, root=ROOT):
    class Handler(BaseHTTPRequestHandler):
        server_version = PRODUCT
        sys_version = ''

        def log_message(self, *args):
            pass

        def do_GET(self):
            self.respond(False)

        def do_HEAD(self):
            self.respond(True)

        def do_POST(self):
            self.send_error(405, 'Read-only review')

        def respond(self, head):
            hosts = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
            if self.headers.get('Host') not in hosts or self.headers.get('Origin') not in {None, *{'http://' + h for h in hosts}}:
                self.send_error(403); return
            parts = urlsplit(self.path)
            if len(self.path) > 4096 or parts.query or parts.fragment or any(c in parts.path for c in ['%', '\\']):
                self.send_error(404); return
            item = spec['routes'].get(parts.path)
            if item is None:
                self.send_error(404); return
            try:
                raw = checked(item, root)
            except (OSError, ValueError):
                self.send_error(409, 'Frozen artifact changed'); return
            start, end, code = 0, len(raw) - 1, 200
            requested = self.headers.get('Range')
            if requested:
                match = re.fullmatch(r'bytes=(\d+)-(\d*)', requested)
                if not match or int(match[1]) >= len(raw):
                    self.send_error(416); return
                start = int(match[1]); end = min(int(match[2]) if match[2] else end, end)
                if end < start:
                    self.send_error(416); return
                code = 206
            mime = 'text/javascript' if item['path'].endswith('.js') else mimetypes.guess_type(item['path'])[0] or 'application/octet-stream'
            self.send_response(code)
            self.send_header('Content-Type', mime); self.send_header('Content-Length', str(end - start + 1))
            self.send_header('Cache-Control', 'no-store'); self.send_header('Accept-Ranges', 'bytes')
            self.send_header('X-Content-Type-Options', 'nosniff'); self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('X-Frame-Options', 'SAMEORIGIN')
            self.send_header('X-Interaction-Twin-Review', spec['id'] + ':' + CONFIG_SHA)
            if code == 206:
                self.send_header('Content-Range', f'bytes {start}-{end}/{len(raw)}')
            self.end_headers()
            if not head:
                try:
                    self.wfile.write(raw[start:end + 1])
                except (BrokenPipeError, ConnectionResetError):
                    pass
    return Handler


def write_receipt(root, value, name):
    destination = Path(root) / RECEIPTS / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True); stream.write('\n')
    return destination


def serve(spec):
    server = ThreadingHTTPServer(('127.0.0.1', spec['port']), handler(spec))
    receipt = {'format': 'interaction-twin-capture-review-restarted-server', 'version': 1, 'viewer_id': spec['id'], 'url': url(spec),
               'pid': os.getpid(), 'uid': os.getuid(), 'configuration': reference(ROOT / CONFIG), 'implementation': reference(Path(__file__)),
               'frozen_snapshot': spec['snapshot_ref'], 'route_count': len(spec['routes']), 'read_only': True, 'new_api_calls': 0, 'new_gpu_calls': 0}
    write_receipt(ROOT, receipt, 'server-' + spec['id'] + '-' + uuid.uuid4().hex + '.json')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def reopen(specs, *, root=ROOT, no_open=False, identity_probe=preflight, spawn=None, browser=webbrowser.open, deadline_s=15):
    """Inspect every port before starting anything; clean up only our failed children."""
    states = [identity_probe(spec, root=root) for spec in specs]
    children = []
    try:
        for spec, state in zip(specs, states):
            if state['status'] == 'reuse':
                continue
            require(state['status'] == 'free', 'Unknown listener state')
            log = Path(root) / RECEIPTS / ('server-' + spec['id'] + '-' + uuid.uuid4().hex + '.log')
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open('xb') as stream:
                child = (spawn or subprocess.Popen)([sys.executable, str(Path(root) / 'scripts/launch_capture_review_v5.py'), '--serve', spec['id']],
                    cwd=root, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            children.append(child)
            deadline = time.monotonic() + deadline_s
            while time.monotonic() < deadline and child.poll() is None:
                if not port_free(spec['port']):
                    identity = listener_identity(spec['port'])
                    require(identity['pid'] == child.pid and owned_listener(spec, identity, root=root), 'Startup port was claimed by another process')
                    require(http_ready(spec), 'New viewer failed exact HTTP readiness')
                    state.update(status='started', pid=child.pid, uid=identity['uid'], http_route_count=len(spec['routes']))
                    break
                time.sleep(.1)
            else:
                raise ValueError('Viewer did not become ready: ' + spec['id'])
        receipt = {'format': 'interaction-twin-capture-review-launch', 'version': 1, 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                   'read_only': True, 'new_api_calls': 0, 'new_gpu_calls': 0, 'main_app_or_worker_started': False,
                   'configuration': reference(Path(root) / CONFIG, root), 'implementation': reference(Path(__file__), ROOT),
                   'viewers': [{'id': spec['id'], 'url': url(spec), 'snapshot_ref': spec['snapshot_ref'], **state} for spec, state in zip(specs, states)],
                   'browser_open_requested': not no_open}
        saved = write_receipt(root, receipt, 'launch-' + uuid.uuid4().hex + '.json')
        print('Ready: ' + url(specs[0]), flush=True)
        print('Verified private review, glass diagnostic and partial laptop hinge. No inference or app worker was started.', flush=True)
        print('Launch receipt: ' + str(saved), flush=True)
        if not no_open:
            browser(url(specs[0]))
        return receipt
    except BaseException:
        for child in children:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    # Do not escalate to killing any process; retain the exact child PID for inspection.
                    pass
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-open', action='store_true', help='Verify/reopen review servers without opening a browser')
    parser.add_argument('--serve', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        config = load_configuration()
        if args.serve:
            choices = [s for s in config['viewers'] if s['id'] == args.serve]
            require(len(choices) == 1, 'Unknown frozen viewer')
            serve(prepare_spec(choices[0])); return 0
        specs = [prepare_spec(s) for s in config['viewers']]
        reopen(specs, no_open=args.no_open)
        return 0
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print('Capture review was not opened: ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
