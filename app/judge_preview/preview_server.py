#!/usr/bin/env python3
"""Serve one frozen curated preview using Python's standard library only.

No repository/owner routes, uploads, inference, MCP runtime, environment loading,
external network, installation, credentials or write endpoints exist here.
"""
from __future__ import annotations
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
import sys
from urllib.parse import parse_qs, unquote, urlsplit

ROOT = Path(__file__).resolve().parent
# Support an isolated ``python3 -I -S`` launch without site packages or an
# inherited PYTHONPATH, and do not write import caches beside frozen files.
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
from library_query import FIELDS, query_records

FORMAT = 'interaction-twin-frozen-judge-preview'
MAX_FILE = 128 * 1024 * 1024
SHA = re.compile(r'[a-f0-9]{64}\Z')
SHOWCASE_PAGES = {'index.html', 'library.html', 'robotics-lab.html', 'agents.html',
                  'agent-setup.html', 'native-model.html', 'aligned-scene.html', 'film.html', 'walkthrough.html', 'robot-workbench.html','physics-bench.html',
                  'trailers.html',
                  'demos/charge-phone-v1/index.html'}
VENDOR = {'three.module.js', 'three.core.js', 'GLTFLoader.js', 'OrbitControls.js',
          'BufferGeometryUtils.js', 'LICENSE', 'manifest.json'}
PHYSICS_STARTER='showcase/media/interaction-twin-newton-starter-20261104-v1.zip'
# These six published editorial transcripts are inert text downloads. Do not
# permit arbitrary showcase text, implementation provenance or owner files.
TRAILER_TRANSCRIPTS = {
    'showcase/media/trailer-collection-20261104-v1/' + stem + '-transcript.txt'
    for stem in ['01-origin', '02-blueprint', '03-mission', '04-proof',
                 '05-pocket-worlds', '06-agent']
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ordinary(root, relative):
    require(isinstance(relative, str) and relative and '\\' not in relative and '%' not in relative
            and not relative.startswith('/') and not any(part.startswith('.') for part in relative.split('/')),
            'Ordinary manifest path required')
    path = root / relative
    require(path.resolve() == path and path.is_relative_to(root) and path.is_file()
            and not any(parent.is_symlink() for parent in [path, *path.parents]), 'File is outside the frozen preview')
    return path


def checked_bytes(root, row):
    require(isinstance(row, dict) and SHA.fullmatch(row.get('sha256', '')) is not None
            and type(row.get('bytes')) is int and 0 < row['bytes'] <= MAX_FILE, 'Bounded exact file reference required')
    path = ordinary(root, row['path'])
    require(path.stat().st_size == row['bytes'], 'Frozen file length differs')
    raw = path.read_bytes()
    require(len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256'], 'Frozen file hash differs')
    return raw


def static_scope(relative):
    if relative == 'preview.html':
        return True
    if relative.startswith('workspace/'):
        return relative in {'workspace/case-data.js', 'workspace/scene.js'} or (
            relative.startswith('workspace/robot-vendor/') and relative.removeprefix('workspace/robot-vendor/') in VENDOR)
    if relative.startswith('showcase/'):
        if relative==PHYSICS_STARTER:return True
        if relative in TRAILER_TRANSCRIPTS:return True
        name = Path(relative).name
        if relative.startswith('showcase/public/'):
            return Path(relative).suffix in {'.json', '.js'}
        if Path(relative).suffix == '.html':
            return relative.removeprefix('showcase/') in SHOWCASE_PAGES
        if relative.startswith('showcase/demos/'):
            return (relative in {'showcase/demos/charge-phone-v1/' + name for name in
                    ['demo.css', 'demo.js', 'task.json', 'trace.json', 'summary.json', 'muse-review.json']} or
                    relative in {'showcase/demos/robotics/index.json', 'showcase/demos/robotics/robot-kit.json'} or
                    re.fullmatch(r'showcase/demos/robotics/(jabra|wallet)/(task\.json|starter\.zip)', relative) is not None)
        return Path(relative).suffix in {'.js', '.css', '.svg', '.png', '.jpg', '.jpeg', '.webp', '.ico',
                                        '.mp4', '.webm', '.mp3', '.wav', '.vtt', '.srt', '.woff', '.woff2'}
    return relative.startswith('data/')


class Snapshot:
    def __init__(self, root):
        self.root = Path(root).resolve()
        manifest_path = ordinary(self.root, 'bundle-manifest.json')
        require(manifest_path.stat().st_size <= 2 * 1024 * 1024, 'Bounded bundle manifest required')
        self.manifest = json.loads(manifest_path.read_bytes())
        require(self.manifest.get('format') == FORMAT and self.manifest.get('version') == 1
                and self.manifest.get('read_only') is True and self.manifest.get('inference_available') is False
                and self.manifest.get('mcp_runtime_included') is False, 'Unsupported preview manifest')
        rows = self.manifest.get('files')
        require(isinstance(rows, list) and 1 <= len(rows) <= 4000, 'Bounded frozen preview inventory required')
        require(sum(row.get('bytes',0)for row in rows)<=768*1024*1024,'Frozen preview exceeds its reviewed total byte budget')
        self.rows = {}
        for row in rows:
            require(row['path'] not in self.rows and type(row.get('http')) is bool, 'Duplicate preview file or route flag')
            checked_bytes(self.root, row)
            self.rows[row['path']] = row
        self.served = {path: row for path, row in self.rows.items() if row['http']}
        require(all(static_scope(path) for path in self.served), 'Owner/private/runtime source cannot be a preview route')
        self.assets = self.json('showcase/public/public-assets.json')['assets']
        approved = {row['path']: row for row in self.assets}
        require(len(approved) == len(self.assets), 'Duplicate curated asset reference')
        for path, row in self.served.items():
            if path.startswith('data/'):
                require(path in approved and all(row[k] == approved[path][k] for k in ['path', 'sha256', 'bytes']),
                        'Only exact current-curated data bytes may be served')
        index = self.json('showcase/public/library-index.json')
        pointer = index['catalog']
        require(re.fullmatch(r'library-catalog-[a-f0-9]{64}\.json', pointer['path']) is not None
                and pointer['path'] == 'library-catalog-' + pointer['sha256'] + '.json', 'Exact catalog pointer required')
        path = 'showcase/public/' + pointer['path']
        require(path in self.rows and all(self.rows[path][k] == pointer[k] for k in ['sha256', 'bytes']), 'Curated catalog pointer changed')
        self.catalog = self.json(path)
        cases = self.json('showcase/public/cases-index.json')
        require(cases.get('curated') is True and {r['id'] for r in cases['cases']} == {r['id'] for r in self.catalog['records']},
                'Curated cases/catalog differ')
        self.robotics_index = None
        index_path = 'showcase/demos/robotics/index.json'
        if index_path in self.rows:
            value = self.json(index_path)
            require(value.get('format') == 'interaction-twin-frozen-robotics-index' and value.get('version') == 1
                    and value.get('scope') == 'authored_kinematic_snapshot' and value.get('inference_available') is False
                    and value.get('physical_accuracy_verified') is False and value.get('simulation_ready') is False,
                    'Frozen robotics scope must retain its unverified physical status')
            tasks = value.get('tasks')
            require(isinstance(tasks, list) and 1 <= len(tasks) <= 4 and len({t['twin_id'] for t in tasks}) == len(tasks),
                    'Bounded distinct supported robotics tasks required')
            self._frozen_ref(value['kit'])
            require(self.json(value['kit']['path']).get('id') == value['kit_id'], 'Frozen robot kit identity differs')
            for task in tasks:
                require(task['twin_id'] in {r['id'] for r in self.catalog['records']} and
                        isinstance(task['instruction'], str) and 1 <= len(task['instruction']) <= 2000,
                        'A frozen robot task must belong to the exact curated library')
                self._frozen_ref(task['task']); self._frozen_ref(task['starter'])
                document = self.json(task['task']['path'])
                claims = document.get('claims', {})
                require(document.get('status') == 'ready' and document.get('twin_id') == task['twin_id']
                        and document.get('kit_id') == value['kit_id'] and claims.get('authored_rehearsal') is True
                        and claims.get('physical_accuracy_verified') is False and claims.get('simulation_ready') is False,
                        'Frozen task cannot claim physical readiness or substitute another twin')
                refs = [document['source_binding']['program_ref'], document['source_binding'].get('object_program_ref')]
                refs += [obj['asset_ref'] for obj in document['objects']]
                for reference in refs:
                    if reference is None: continue
                    require(reference['path'] in approved and
                            all(reference[k] == approved[reference['path']][k] for k in ['path', 'sha256', 'bytes']),
                            'Frozen robot task references bytes outside current approved assets')
                    self._frozen_ref(reference)
            self.robotics_index = value
        self.physics_snapshot=None
        physics=self.manifest.get('frozen_physics_snapshot')
        if physics is not None:
            require(physics.get('scope')=='saved_actual_newton_cpu_states_not_browser_physics_or_robot_grasp_validation'and physics['starter_ref']['path']==PHYSICS_STARTER,'Unknown frozen physics scope')
            for pointer in [physics['source_ref'],physics['replay_ref'],physics['starter_ref'],*physics['asset_refs']]:self._frozen_ref(pointer)
            replay=self.json(physics['replay_ref']['path']);claims=replay.get('claims',{})
            require(replay.get('format')=='interaction-twin-newton-contact-experiment'and replay.get('version')==1 and len(replay.get('frames',[]))==241 and len(replay.get('objects',[]))==3,'Incomplete frozen Newton experiment')
            require(physics['claims']==claims and claims.get('real_physics_engine_execution')is True and all(claims.get(k)is False for k in ['metric_scale_measured','material_parameters_measured','physical_accuracy_verified','robot_grasp_validated','simulation_ready']),'Physics assumptions/accuracy were upgraded')
            require([o['asset_ref']for o in replay['objects']]==physics['asset_refs'],'Physics objects differ from the exact declared native assets')
            for pointer in physics['asset_refs']:
                require(pointer['path'].endswith('.glb')and 0<pointer['bytes']<=32*1024*1024 and pointer['path']in approved and all(pointer[k]==approved[pointer['path']][k]for k in ['path','sha256','bytes']),'Physics model is outside approved curated data')
            self.physics_snapshot=physics

    def _frozen_ref(self, reference):
        require(reference['path'] in self.rows and all(reference[k] == self.rows[reference['path']][k]
                    for k in ['path', 'sha256', 'bytes']), 'Frozen robotics reference changed')
        return checked_bytes(self.root, self.rows[reference['path']])

    def robotics_response(self, path, query):
        """Saved baseline retrieval only; local JS compiles subsequent instructions."""
        value = self.robotics_index
        require(value is not None, 'No frozen robotics rehearsal is available')
        require(isinstance(query, dict) and all(isinstance(v, list) and len(v) == 1 for v in query.values()),
                'One value per robotics snapshot parameter required')
        if path == '/api/robotics/kits':
            require(not query, 'Kit list has no query parameters')
            kit = self.json(value['kit']['path'])
            return {'kits': [{'id': value['kit_id'], 'title': kit.get('title', 'Authored rehearsal arm')}],
                    'snapshot_mode': True, 'read_only': True, 'physical_robot_execution_available': False}, None
        if path == '/api/robotics/kit':
            require(set(query) <= {'kit_id'} and query.get('kit_id', [value['kit_id']])[0] == value['kit_id'],
                    'Choose the included frozen robot kit')
            return {'kit': self.json(value['kit']['path']), 'snapshot_mode': True,
                    'instruction_compilation': 'bounded_local_browser_compiler', 'new_api_calls': 0,
                    'read_only': True, 'robot_execution_available': False}, None
        require(path in {'/api/robotics/plan', '/api/robotics/bundle'} and
                set(query) == {'twin_id', 'kit_id', 'instruction'} and query['kit_id'][0] == value['kit_id'],
                'Only exact included baseline task requests are available')
        task = next((t for t in value['tasks'] if t['twin_id'] == query['twin_id'][0] and t['instruction'] == query['instruction'][0]), None)
        require(task is not None, 'That task has no included baseline; no live planner exists')
        if path.endswith('/plan'): return self.json(task['task']['path']), None
        return None, task['starter']['path']

    def json(self, path):
        require(path in self.rows, 'Frozen preview JSON is not included')
        return json.loads(checked_bytes(self.root, self.rows[path]))

    def route(self, raw_path):
        parsed = urlsplit(raw_path)
        require(not parsed.scheme and not parsed.netloc and raw_path.startswith('/') and not raw_path.startswith('//'), 'Local relative route required')
        path = unquote(parsed.path, errors='strict')
        require('\\' not in path and '\x00' not in path and '%' not in path
                and not any(part.startswith('.') for part in path.split('/') if part), 'Hidden or traversing route refused')
        if path in {'/', '/preview.html'}:
            relative = 'preview.html'
        elif path in {'/showcase', '/showcase/'}:
            relative = 'showcase/index.html'
        elif path.startswith('/files/data/'):
            relative = path.removeprefix('/files/')
        elif path.startswith(('/showcase/', '/workspace/')):
            relative = path.removeprefix('/')
        else:
            raise ValueError('No owner, runtime or arbitrary filesystem route exists')
        require(relative in self.served, 'Only a declared frozen preview file is available')
        return relative, checked_bytes(self.root, self.served[relative])

    def query(self, query, *, facets=False):
        require(len(query) <= 200, 'Bounded query required')
        allowed = {'query', 'sort', 'direction', 'offset', 'limit', 'group_by', *FIELDS}
        require(not set(query) - allowed, 'Unknown library filter')
        arguments = {}
        for key, values in query.items():
            require(isinstance(values, list) and len(values) <= 25, 'Bounded filter values required')
            if key in FIELDS:
                arguments[key] = values
            else:
                require(len(values) == 1, 'One scalar filter value required')
                arguments[key] = int(values[0]) if key in {'offset', 'limit'} else values[0]
        result = query_records(self.catalog['records'], **arguments)
        if facets:
            return {'facets': result['facets'], 'query_contract': result['query_contract'],
                    'recordings': result['total'], 'curated_only': True}
        result['curated_only'] = True
        return result


def byte_range(value, total):
    if value is None:
        return 0, total - 1, False
    match = re.fullmatch(r'bytes=([0-9]*)-([0-9]*)', value)
    require(match is not None and any(match.groups()), 'One byte range required')
    left, right = match.groups()
    if not left:
        count = int(right)
        require(count > 0, 'Positive suffix range required')
        return max(0, total - count), total - 1, True
    start = int(left)
    end = min(int(right), total - 1) if right else total - 1
    require(0 <= start <= end < total, 'Byte range outside file')
    return start, end, True


class Handler(BaseHTTPRequestHandler):
    server_version = 'InteractionTwinPreview/1'

    def log_message(self, *args):
        pass

    def common_headers(self):
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('X-Interaction-Twin-Preview', 'curated-read-only')

    def json(self, status, value, head=False):
        raw = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.common_headers()
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        if not head:
            self.wfile.write(raw)

    def do_GET(self):
        self.read(False)

    def do_HEAD(self):
        self.read(True)

    def read(self, head):
        try:
            host = urlsplit('http://' + (self.headers.get('Host') or ''))
            if host.hostname not in {'127.0.0.1', 'localhost'} or host.username or host.password or host.port != self.server.server_port:
                return self.json(403, {'error': 'A matching loopback Host is required'}, head)
            parsed = urlsplit(self.path)
            if parsed.path == '/health':
                return self.json(200, {'status': 'preview_only', 'read_only': True,
                    'inference_available': False, 'uploads_available': False, 'mcp_runtime_included': False}, head)
            if parsed.path in {'/api/library', '/api/library/facets'}:
                query = parse_qs(parsed.query, keep_blank_values=True, max_num_fields=200)
                return self.json(200, self.server.snapshot.query(query, facets=parsed.path.endswith('/facets')), head)
            if parsed.path.startswith('/api/robotics/'):
                value, relative = self.server.snapshot.robotics_response(parsed.path,
                    parse_qs(parsed.query, keep_blank_values=True, max_num_fields=10))
                if relative is None: return self.json(200, value, head)
                raw = checked_bytes(self.server.snapshot.root, self.server.snapshot.served[relative])
            else:
                relative, raw = self.server.snapshot.route(self.path)
            try:
                start, end, partial = byte_range(self.headers.get('Range'), len(raw))
            except ValueError:
                self.send_response(416)
                self.common_headers()
                self.send_header('Content-Range', f'bytes */{len(raw)}')
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
            mime = mimetypes.guess_type(relative)[0] or 'application/octet-stream'
            if relative.endswith('.js'):
                mime = 'text/javascript'
            self.send_response(206 if partial else 200)
            self.common_headers()
            self.send_header('Content-Type', mime)
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Content-Length', str(end - start + 1))
            self.send_header('ETag', '"' + self.server.snapshot.served[relative]['sha256'] + '"')
            if partial:
                self.send_header('Content-Range', f'bytes {start}-{end}/{len(raw)}')
            if relative.startswith('data/') and relative.endswith('.html'):
                self.send_header('Content-Disposition', 'attachment; filename="' + Path(relative).name + '"')
                self.send_header('Content-Security-Policy', "sandbox; default-src 'none'; base-uri 'none'")
            else:
                self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; media-src 'self' blob:; connect-src 'self'; frame-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'self'")
            self.end_headers()
            if not head:
                self.wfile.write(raw[start:end + 1])
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (ValueError, KeyError, OSError, UnicodeError):
            self.json(404, {'error': 'Not an available curated preview resource'}, head)

    def mutation(self):
        self.close_connection = True
        self.json(405, {'error': 'This preview is read-only; no uploads, admin, inference or execution endpoints exist'})

    do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = mutation


def create_server(root=ROOT, port=0):
    require(type(port) is int and 0 <= port <= 65535, 'Valid loopback port required')
    snapshot = Snapshot(root)
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.daemon_threads = True
    server.snapshot = snapshot
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8767, help='Loopback port; use0 to choose a free port')
    args = parser.parse_args()
    server = create_server(port=args.port)
    print('Frozen Interaction Twin preview: http://127.0.0.1:' + str(server.server_port) + '/', flush=True)
    print('Read-only curated examples. No admin, upload, model inference or MCP runtime. Ctrl-C to stop.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
