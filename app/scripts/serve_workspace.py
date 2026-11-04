#!/usr/bin/env python3
"""Local-only Stage 6 workspace and bounded single-worker model API.

Loopback only. Optional source-bound SAM 3D work uses an explicitly configured
existing SSH/GPU worker; no credential discovery or implicit downloads.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import json
import mimetypes
import os
import re
import signal
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse, parse_qs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import Runner, SLUG, confined, list_cases, list_runs, load_profile, public_url, read, provider_disclosure
import package_exports
import revisions
import upgrade_api
import observed_jobs
import interval_jobs
import calibration_jobs
import admin_auth

MAX_UPLOAD = 512 * 1024 * 1024
PUBLIC_ROOT_FILES = {'generated_motion_observer.js', 'jabra_assembly_observer.js', 'reconstruction_observer.js'}
PUBLIC_WORKSPACE_FILES = {'case-data.js','scene.js'}
PUBLIC_STATIC_EXTENSIONS={'.html','.js','.css','.svg','.png','.jpg','.jpeg','.webp','.ico',
                          '.mp4','.webm','.mp3','.wav','.vtt','.woff','.woff2'}
PUBLIC_DEMO_JSON={'showcase/demos/charge-phone-v1/'+name for name in
                  ('task.json','trace.json','summary.json','muse-review.json')}
PUBLIC_PHYSICS_DOWNLOADS={'showcase/media/interaction-twin-newton-starter-20261104-v1.zip'}


def sam3d_runtime_status(root=ROOT):
    """Configuration only: GET never authenticates, contacts GPUs or runs models."""
    from sam3d_cluster_bridge import configuration
    try:
        value = configuration(root)
        return {'provider': 'sam3d_native', 'configured': value is not None,
                'maximum_parallel_workers': value['maximum_parallel_workers'] if value else 0,
                'availability': 'checked_when_processing' if value else 'awaiting_gpu_worker',
                'fallback_backend': None, 'segmentation': 'SAM 3.1', 'reasoning': 'Muse Spark'}
    except (OSError, ValueError, TypeError):
        return {'provider': 'sam3d_native', 'configured': False,
                'availability': 'configuration_unavailable', 'fallback_backend': None}


def library_request(query):
    """Shared read-only MCP discovery contract, with bounded query parameters."""
    from library_query import FIELDS
    allowed = {'query', 'sort', 'direction', 'offset', 'limit', 'group_by', *FIELDS}
    if set(query) - allowed:
        raise ValueError('Unknown library query fields')
    result = {}
    for key, values in query.items():
        if key in FIELDS:
            result[key] = values
        else:
            if len(values) != 1:
                raise ValueError('One value required for ' + key)
            result[key] = int(values[0]) if key in {'offset', 'limit'} else values[0]
    return result


def meta_budget_status(root=ROOT):
    """Local estimated spend only; never authenticate or create a ledger on GET."""
    path = Path(root) / '.secrets/meta-budget.sqlite3'
    common = {'currency': 'USD', 'scope': 'local_ledger_only_not_account_balance',
              'actual_meta_credit_balance_verified': False}
    if not path.is_file():
        return dict(common, status='not_started', cap_usd='100.000000', settled_usd='0.000000',
                    reserved_usd='0.000000', remaining_estimated_usd='100.000000', request_count=0,
                    blocked=False, unresolved_blockers=0)
    try:
        from meta_budget import read_budget_status
        result = read_budget_status(path)
        fields = {'currency', 'cap_usd', 'per_request_cap_usd', 'settled_usd', 'reserved_usd',
                  'remaining_estimated_usd', 'over_cap_usd', 'blocked', 'unresolved_blockers',
                  'request_count', 'scope', 'unknown_usage', 'unknown_usage_count', 'bounded_unknown_count', 'bounded_unknown_usd'}
        return dict(common, status='available', **{k: v for k, v in result.items() if k in fields and k not in common})
    except Exception:
        # A billing database problem must not expose paths or its raw contents.
        return dict(common, status='unavailable', blocked=True,
                    message='Estimated spend is unavailable; paid requests must use the budget guard.')


def static_path(url_path, root=ROOT):
    """Allow only product assets and evidence, including nested symlink checks."""
    path = unquote(url_path)
    if '\x00' in path or '\\' in path:
        raise ValueError('Invalid static path')
    if any(part.startswith('.') for part in path.split('/') if part):
        raise ValueError('Hidden or traversing static paths are unavailable')
    if path in {'/admin','/admin/'}:
        path='/admin/index.html'
    if path.startswith('/admin/'):
        target=confined(root/path.lstrip('/'),root/'admin')
        if target.suffix not in {'.html','.js','.css','.svg','.png','.ico'}:
            raise ValueError('Not an admin UI asset')
        return target
    if path in {'/', '/showcase', '/showcase/'}:
        path = '/showcase/index.html'
    if path in {'/workspace', '/workspace/'}:
        path = '/workspace/index.html'
    if path.startswith('/workspace/'):
        return confined(root / path.lstrip('/'), root / 'workspace')
    if path.startswith('/showcase/'):
        return confined(root / path.lstrip('/'), root / 'showcase')
    if not path.startswith('/files/'):
        raise ValueError('Not a public route')
    relative = path[len('/files/'):]
    if relative.startswith('data/'):
        return confined(root / relative, root / 'data')
    if relative in PUBLIC_ROOT_FILES:
        return confined(root / relative, root)
    raise ValueError('Only showcase, workspace and evidence files are public')


def valid_origin(origin, host):
    if origin is None:
        return True  # Command-line clients may omit Origin; browser JSON writes require same origin.
    parsed = urlparse(origin)
    return parsed.scheme == 'http' and parsed.netloc == host and parsed.hostname in {'127.0.0.1', 'localhost'}


def public_static(path, *, root=ROOT, cache=None):
    """Anonymous access is exact published artifacts, never broad data prefixes."""
    import hashlib
    root=Path(root).resolve();path=Path(path).resolve()
    if not path.is_relative_to(root):return False
    relative=path.relative_to(root).as_posix()
    if relative.startswith('admin/'):
        return path.suffix in {'.html','.js','.css','.svg','.png','.ico'}
    if relative.startswith('workspace/robot-vendor/'):
        return True
    if relative.startswith('workspace/'):
        return relative.removeprefix('workspace/') in PUBLIC_WORKSPACE_FILES
    if relative in PUBLIC_ROOT_FILES:return True
    if relative.startswith('showcase/public/scenes/'):
        scene_relative=relative.removeprefix('showcase/public/')
        if not re.fullmatch(r'scenes/[a-f0-9]{64}\.json',scene_relative):return False
        index=root/'showcase/public/cases-index.json'
        try:
            if not index.is_file() or index.is_symlink() or index.stat().st_size>8*1024*1024:return False
            payload=json.loads(index.read_text());cases=payload.get('cases')
            if payload.get('curated') is not True or not isinstance(cases,list) or len(cases)>1000:return False
            for case in cases:
                item=case.get('scene_ref')
                if not item or item.get('path')!=scene_relative:continue
                if (not re.fullmatch(r'[a-f0-9]{64}',item.get('sha256','')) or path.name!=item['sha256']+'.json' or
                    type(item.get('bytes')) is not int or not 0<item['bytes']<=64*1024*1024 or
                    not path.is_file() or path.stat().st_size!=item['bytes']):return False
                info=path.stat();key=(str(path),info.st_dev,info.st_ino,info.st_mtime_ns,info.st_ctime_ns,info.st_size,item['sha256'])
                if cache is not None and cache.get(key):return True
                actual=hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
                if actual and cache is not None:
                    if len(cache)>=4096:cache.clear()
                    cache[key]=True
                return actual
        except (OSError,ValueError,KeyError,TypeError):return False
        return False
    if relative.startswith('showcase/public/'):
        return path.suffix in PUBLIC_STATIC_EXTENSIONS|{'.json'}
    if relative in PUBLIC_DEMO_JSON:
        return path.is_file() and 0<path.stat().st_size<=2*1024*1024
    if relative in PUBLIC_PHYSICS_DOWNLOADS:
        return path.is_file() and 0<path.stat().st_size<=32*1024*1024
    if relative.startswith('showcase/'):
        name=path.name
        if name.startswith(('library-catalog','library-index','cases','aligned-scenes')):return False
        return path.suffix in PUBLIC_STATIC_EXTENSIONS and (not name.startswith('sam3d-models') or name=='sam3d-models.js')
    if not relative.startswith('data/') or any(p.startswith('.') for p in Path(relative).parts):return False
    manifest=root/'showcase/public/public-assets.json'
    if not manifest.is_file() or manifest.is_symlink() or manifest.resolve()!=manifest or manifest.stat().st_size>8*1024*1024:return False
    try:
        saved=json.loads(manifest.read_text())
        if saved.get('format')!='interaction-twin-public-assets' or saved.get('version')!=1 or not isinstance(saved.get('assets'),list) or len(saved['assets'])>20000:return False
        for item in saved['assets']:
            if (not isinstance(item,dict) or not isinstance(item.get('path'),str) or not item['path'].startswith('data/') or
                any(p.startswith('.') for p in Path(item['path']).parts) or not re.fullmatch(r'[a-f0-9]{64}',item.get('sha256','')) or
                type(item.get('bytes')) is not int or item['bytes']<=0):return False
            if item['path']!=relative:continue
            if not path.is_file() or path.stat().st_size!=item['bytes']:return False
            info=path.stat();key=(str(path),info.st_dev,info.st_ino,info.st_mtime_ns,info.st_ctime_ns,info.st_size,item['sha256'])
            if cache is not None and cache.get(key):return True
            actual=hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(8*1024*1024),b''):actual.update(block)
            valid=actual.hexdigest()==item['sha256']
            if valid and cache is not None:
                if len(cache)>=4096:cache.clear()
                cache[key]=True
            return valid
    except (OSError,ValueError,KeyError,TypeError):return False
    return False


def upload_name(name):
    if not name or Path(name).name != name or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_. -]{0,119}', name):
        raise ValueError('Upload filename must be a simple bounded basename')
    if Path(name).suffix.lower() not in {'.mp4', '.mov', '.m4v', '.webm'}:
        raise ValueError('Only MP4, MOV, M4V and WebM videos are supported')
    return name


class WorkspaceServer(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, address, handler):
        if address[0] != '127.0.0.1':
            raise ValueError('This server binds to 127.0.0.1 only')
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix='twin-model-worker')
        self.active = None
        self.lock = threading.Lock()
        self.admin_auth=admin_auth.AdminAuth(ROOT)
        if (ROOT/'data/quality-learning-20261104-v1/rules.json').exists():
            from admin_quality_learning_v1 import activate_quality_learning
            self.quality_learning=activate_quality_learning(root=ROOT)
        else:
            # A fresh checkout or isolated test has no learned case catalog.
            # A present but invalid catalog still fails closed in activation.
            self.quality_learning={'status':'unavailable_no_method_catalog',
                                   'new_model_requests':0,'training_performed':False}
        self.public_asset_cache={}
        super().__init__(address, handler)

    def server_close(self):
        if self.active:
            self.active.cancel()
        self.executor.shutdown(wait=True, cancel_futures=True)
        if hasattr(self,'marketing_executor'):
            self.marketing_executor.shutdown(wait=True,cancel_futures=True)
        super().server_close()


class Handler(BaseHTTPRequestHandler):
    server_version = 'InteractionTwinLocal/0.1'

    def json(self, code, value):
        data = json.dumps(value, allow_nan=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy','same-origin')
        for key,value in getattr(self,'extra_response_headers',[]):self.send_header(key,value)
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.json(403, {'error': 'Cross-origin requests are not enabled'})

    def do_GET(self):
        self.extra_response_headers=[]
        self._read(False)

    def do_HEAD(self):
        self.extra_response_headers=[]
        self._read(True)

    def _read(self, head):
        route = urlparse(self.path).path
        try:
            if urlparse('http://' + (self.headers.get('Host') or '')).hostname not in {'127.0.0.1', 'localhost'}:
                return self.json(403, {'error': 'A local Host header is required'})
            if route in {'/', '/showcase', '/showcase/'}:
                self.send_response(302)
                self.send_header('Location', '/showcase/index.html')
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
            if route=='/admin':
                self.send_response(302);self.send_header('Location','/admin/');self.send_header('Content-Length','0');self.end_headers();return
            if route == '/health':
                return self.json(200, {'status': 'ready', 'host': '127.0.0.1', 'single_worker': True, 'cluster_connected': False})
            auth=getattr(getattr(self,'server',None),'admin_auth',None)
            authenticated=bool(auth and auth.authenticated(self.headers.get('Cookie')))
            if authenticated:self.admin_actor='nigel'
            if route=='/api/admin/auth':
                if auth is None:return self.json(503,{'error':'Administrator session service unavailable'})
                value,token=auth.status(self.headers.get('Cookie'))
                if token:self.extra_response_headers=[('Set-Cookie',admin_auth.cookie_header(token))]
                return self.json(200,value)
            if route in {'/api/library','/api/library/facets'}:
                import curated_publication
                query=library_request(parse_qs(urlparse(self.path).query,keep_blank_values=True))
                return self.json(200,curated_publication.query(query,facets=route.endswith('/facets'),root=ROOT))
            if route.startswith('/api/robotics/'):
                # Only this adapter's exact read-only routes are anonymous. It
                # uses the current curated snapshot, never the private catalog,
                # and cannot schedule inference, write files or execute plans.
                import public_robotics_api
                if len(self.path)>8192:
                    return self.json(400,{'error':'Robot-task query exceeds its bounded size'})
                query=parse_qs(urlparse(self.path).query,keep_blank_values=True)
                response=public_robotics_api.response(route,query,root=ROOT)
                if response is not None:
                    code,data,mime,headers=response
                    self.send_response(code)
                    self.send_header('Content-Type',mime)
                    self.send_header('Content-Length',str(len(data)))
                    self.send_header('Cache-Control','no-store')
                    self.send_header('X-Content-Type-Options','nosniff')
                    self.send_header('Referrer-Policy','same-origin')
                    for key,value in headers.items():self.send_header(key,value)
                    self.end_headers()
                    if not head:self.wfile.write(data)
                    return
            if route.startswith('/api/') and not authenticated:
                return self.json(401,{'error':'Administrator authentication required'})
            if route.startswith('/api/admin/'):
                return self.admin_read(route)
            if route == '/api/config':
                return self.json(200, {'profiles': [load_profile(p.stem) for p in sorted((ROOT / 'profiles').glob('*.json'))],
                                       'default_profile': 'meta_sam3d_default',
                                       'sam3d_runtime': sam3d_runtime_status(),
                                       'max_upload_bytes': MAX_UPLOAD, 'hosted_reasoning_requires_run_consent': True,
                                       'meta_budget_url': '/api/meta/budget',
                                       'model_runtime': 'prepared .venv-sam; selected hosted provider; no silent installation or fallback'})
            if route == '/api/meta/budget':
                return self.json(200, meta_budget_status())
            if route == '/api/intelligence':
                import intelligence_api
                query = parse_qs(urlparse(self.path).query)
                if set(query) != {'case'} or len(query['case']) != 1:
                    raise ValueError('Choose one published project for its task analysis')
                with self.server.lock:
                    active = self.server.active
                    active_id = active.id if isinstance(active, intelligence_api.IntelligenceRunner) else None
                    saved = intelligence_api.record(query['case'][0], root=ROOT, active_job_id=active_id)
                return self.json(200, saved)
            if route == '/api/cases':
                return self.json(200, {'cases': list_cases()})
            if route == '/api/calibration/jobs':
                active = self.server.active
                active_id = active.id if isinstance(active, calibration_jobs.CalibrationRunner) else None
                return self.json(200, calibration_jobs.list_jobs(root=ROOT, active_job_id=active_id))
            calibration_job = re.fullmatch(r'/api/calibration/jobs/(calibration-[a-f0-9]{24})', route)
            if calibration_job:
                active = self.server.active
                active_id = active.id if isinstance(active, calibration_jobs.CalibrationRunner) else None
                return self.json(200, calibration_jobs.record(calibration_job[1], root=ROOT, active_job_id=active_id))
            if route == '/api/motion/catalog':
                return self.json(200, interval_jobs.catalog(root=ROOT))
            if route == '/api/motion/jobs':
                active = self.server.active
                active_id = active.id if isinstance(active, interval_jobs.MotionRunner) else None
                return self.json(200, interval_jobs.list_jobs(root=ROOT, active_job_id=active_id))
            motion_job = re.fullmatch(r'/api/motion/jobs/(interval-[a-f0-9]{24})', route)
            if motion_job:
                active = self.server.active
                active_id = active.id if isinstance(active, interval_jobs.MotionRunner) else None
                return self.json(200, interval_jobs.record(motion_job[1], root=ROOT, active_job_id=active_id))
            if route == '/api/observed/catalog':
                return self.json(200, observed_jobs.catalog(root=ROOT))
            if route == '/api/observed/jobs':
                active = self.server.active
                active_id = observed_jobs.active_identifier(active)
                return self.json(200, observed_jobs.list_jobs(root=ROOT, active_job_id=active_id))
            observed_job = re.fullmatch(r'/api/observed/jobs/(observed-[a-f0-9]{24})', route)
            if observed_job:
                active = self.server.active
                active_id = observed_jobs.active_identifier(active)
                return self.json(200, observed_jobs.record(observed_job[1], root=ROOT, active_job_id=active_id))
            if route == '/api/runs/summary':
                from run_summaries import compact_run_summary
                return self.json(200, {'runs': [compact_run_summary(row) for row in list_runs()],
                                       'scope': 'polling_summaries_full_records_unchanged'})
            if route == '/api/runs':
                return self.json(200, {'runs': list_runs()})
            if route == '/api/upgrades/catalog':
                return self.json(200, upgrade_api.catalog(root=ROOT))
            if route == '/api/upgrades/scene':
                query=parse_qs(urlparse(self.path).query)
                if set(query)!={'case'} or len(query['case'])!=1:
                    raise ValueError('Choose one published project for its scene')
                return self.json(200,upgrade_api.scene(query['case'][0],root=ROOT))
            if route == '/api/upgrades/jobs':
                import upgrade_bundles
                return self.json(200, {'jobs': [upgrade_api.job_view(r,root=ROOT) for r in upgrade_bundles.list_jobs(root=ROOT)],
                                       'imports': [upgrade_api.import_view(r,root=ROOT) for r in upgrade_bundles.list_imports(root=ROOT)]})
            if route == '/api/upgrades/runs':
                return self.json(200, upgrade_api.runs(root=ROOT))
            upgrade = re.fullmatch(r'/api/upgrades/runs/(upgrade-[a-z0-9-]{1,90})', route)
            if upgrade:
                return self.json(200, upgrade_api.run_record(upgrade[1],root=ROOT))
            review = re.fullmatch(r'/api/cases/([A-Za-z0-9_-]{1,180})/revisions(?:/(rev-[0-9]{6}-[0-9a-f]{16}))?', route)
            if review:
                store = revisions.Store(ROOT)
                return self.json(200, store.get(review[1], review[2]) if review[2] else store.list(review[1]))
            match = re.fullmatch(r'/api/runs/([a-z0-9-]+)', route)
            if match:
                path = confined(ROOT / 'data/flow-runs' / match[1] / 'run.json', ROOT / 'data/flow-runs')
                return self.json(200, read(path))
            target=static_path(route)
            cache=getattr(getattr(self,'server',None),'public_asset_cache',None)
            if not authenticated and not public_static(target,root=ROOT,cache=cache):
                if route.startswith('/workspace/') and target.suffix=='.html':
                    self.send_response(302);self.send_header('Location','/admin/');self.send_header('Content-Length','0');self.end_headers();return
                return self.json(404,{'error':'File not found'})
            self.serve_file(target, head)
        except (BrokenPipeError, ConnectionResetError):
            # Browsers cancel old video ranges when seeking. Headers may have
            # been sent already, so never attempt a second error response.
            return
        except admin_auth.AuthError as exc:
            self.json(exc.status,{'error':str(exc)})
        except revisions.Conflict as exc:
            self.json(409, {'error': str(exc)})
        except (OSError, ValueError, KeyError) as exc:
            try:
                self.json(404, {'error': str(exc) if route.startswith('/api/') else 'File not found'})
            except (BrokenPipeError, ConnectionResetError):
                return

    def serve_file(self, path, head):
        if not path.is_file():
            raise FileNotFoundError('File not found')
        size = path.stat().st_size
        start, end, code = 0, size - 1, 200
        requested = self.headers.get('Range')
        if requested:
            match = re.fullmatch(r'bytes=(\d*)-(\d*)', requested)
            if not match or not any(match.groups()):
                return self.json(416, {'error': 'Unsupported byte range'})
            if match[1]:
                start = int(match[1])
                end = min(size - 1, int(match[2])) if match[2] else size - 1
            else:
                start = max(0, size - int(match[2]))
            if start >= size or start > end:
                self.send_response(416)
                self.send_header('Content-Range', f'bytes */{size}')
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
            code = 206
        self.send_response(code)
        self.send_header('Content-Type', mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Length', str(max(0, end - start + 1)))
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Cache-Control', 'no-store' if path.suffix in {'.json', '.js', '.html'} or self.path.startswith('/api/admin/') else 'private, max-age=60')
        for key,value in getattr(self,'extra_response_headers',[]):self.send_header(key,value)
        if code == 206:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.end_headers()
        if not head:
            with path.open('rb') as stream:
                stream.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    block = stream.read(min(1024 * 1024, remaining))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)

    def do_POST(self):
        self.extra_response_headers=[]
        try:
            if urlparse('http://' + (self.headers.get('Host') or '')).hostname not in {'127.0.0.1', 'localhost'}:
                return self.json(403, {'error': 'A local Host header is required'})
            if self.headers.get('Origin') is None or not valid_origin(self.headers.get('Origin'), self.headers.get('Host')):
                return self.json(403, {'error': 'Writes require the same local origin'})
            route=urlparse(self.path).path
            auth=self.server.admin_auth
            preauth=route in {'/api/admin/login','/api/admin/setup'}
            self.admin_actor=auth.require(self.headers.get('Cookie'),self.headers.get('X-CSRF-Token'),mutation=True,owner=not preauth)
            length = int(self.headers.get('Content-Length', '-1'))
            if length < 0:
                return self.json(411, {'error': 'A bounded Content-Length is required'})
            if route=='/api/admin/marketing/music/upload':
                import admin_studio_http
                return admin_studio_http.upload(self,length,root=ROOT)
            if route == '/api/upload':
                return self.upload(length)
            returned = re.fullmatch(r'/api/upgrades/jobs/(job-[a-f0-9]{32})/result', route)
            if returned:
                return self.upload_upgrade_result(returned[1], length)
            review = re.fullmatch(r'/api/cases/([A-Za-z0-9_-]{1,180})/revisions', route)
            if self.headers.get_content_type() != 'application/json' or length > (revisions.MAX_BODY if review else 16384):
                return self.json(400, {'error': 'A small application/json request is required'})
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError('Expected a JSON object')
            if route.startswith('/api/admin/'):
                return self.admin_post(route,body)
            if review:
                return self.json(201, revisions.Store(ROOT).save(review[1], body))
            package = re.fullmatch(r'/api/cases/([A-Za-z0-9_-]{1,180})/portable-package', route)
            if package:
                return self.json(201, package_exports.create_export(package[1], body, root=ROOT))
            if route == '/api/upgrades/plan':
                return self.json(200, upgrade_api.plan(body,root=ROOT))
            if route == '/api/upgrades/jobs':
                import upgrade_bundles
                return self.json(201, upgrade_api.job_view(upgrade_bundles.prepare_job(body,root=ROOT),root=ROOT))
            if route == '/api/upgrades/runs':
                return self.start_upgrade(body)
            if route == '/api/observed/jobs':
                return self.start_observed(body)
            if route == '/api/motion/jobs':
                return self.start_motion(body)
            if route == '/api/calibration/jobs':
                return self.start_calibration(body)
            calibration_cancel = re.fullmatch(r'/api/calibration/jobs/(calibration-[a-f0-9]{24})/cancel', route)
            if calibration_cancel:
                if body:
                    raise ValueError('Cancellation accepts an empty JSON object')
                with self.server.lock:
                    active = self.server.active
                    if isinstance(active, calibration_jobs.CalibrationRunner) and active.id == calibration_cancel[1] and active.state['status'] in calibration_jobs.ACTIVE:
                        active.cancel()
                        return self.json(202, {'id': active.id, 'status': 'cancellation_requested'})
                return self.json(409, {'error': 'That calibration job is not active'})
            motion_cancel = re.fullmatch(r'/api/motion/jobs/(interval-[a-f0-9]{24})/cancel', route)
            if motion_cancel:
                if body:
                    raise ValueError('Cancellation accepts an empty JSON object')
                with self.server.lock:
                    active = self.server.active
                    if isinstance(active, interval_jobs.MotionRunner) and active.id == motion_cancel[1] and active.state['status'] in interval_jobs.ACTIVE:
                        active.cancel()
                        return self.json(202, {'id': active.id, 'status': 'cancellation_requested'})
                return self.json(409, {'error': 'That local motion job is not active'})
            observed_cancel = re.fullmatch(r'/api/observed/jobs/(observed-[a-f0-9]{24})/cancel', route)
            if observed_cancel:
                if body:
                    raise ValueError('Cancellation accepts an empty JSON object')
                with self.server.lock:
                    active = self.server.active
                    if isinstance(active, observed_jobs.ObservedRunner) and active.id == observed_cancel[1] and active.state['status'] in observed_jobs.ACTIVE:
                        active.cancel()
                        return self.json(202, {'id': active.id, 'status': 'cancellation_requested'})
                return self.json(409, {'error': 'That local geometry job is not active'})
            if route == '/api/intelligence/run':
                return self.start_intelligence(body)
            upgrade_cancel = re.fullmatch(r'/api/upgrades/runs/(upgrade-[a-z0-9-]{1,90})/cancel', route)
            if upgrade_cancel:
                active=self.server.active
                if active and active.id==upgrade_cancel[1] and active.state['status'] in {'created','queued','running'}:
                    active.cancel()
                    return self.json(202,{'id':active.id,'status':'cancellation_requested'})
                return self.json(409,{'error':'That upgrade run is not active'})
            match = re.fullmatch(r'/api/runs/([a-z0-9-]+)/cancel', route)
            if match:
                active = self.server.active
                if active and active.id == match[1]:
                    active.cancel()
                    return self.json(202, {'id': active.id, 'status': 'cancellation_requested'})
                return self.json(409, {'error': 'That run is not active'})
            if route != '/api/run':
                return self.json(404, {'error': 'Unknown API route'})
            return self.start_run(body)
        except admin_auth.AuthError as exc:
            self.json(exc.status,{'error':str(exc)})
        except (revisions.Conflict, package_exports.ExportConflict) as exc:
            self.json(409, {'error': str(exc)})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.json(400, {'error': str(exc)})

    def admin_read(self,route):
        """Owner-only workflow routing; no inference is performed by auth."""
        import admin_studio_http
        if admin_studio_http.dispatch(self,route,'GET',root=ROOT):return
        try:
            import admin_http
        except ImportError:
            return self.json(503,{'error':'Private curation workflow is not ready'})
        if not admin_http.dispatch(self,route,'GET',None):
            return self.json(404,{'error':'Unknown administrator route'})

    def admin_post(self,route,body):
        auth=self.server.admin_auth
        if route in {'/api/admin/login','/api/admin/setup'}:
            expected={'password'} if route.endswith('/login') else {'password','setup_code'}
            if set(body)!=expected:raise ValueError('Unexpected sign-in fields')
            if route.endswith('/login'):
                value,token=auth.login(body['password'],self.headers.get('Cookie'),self.headers.get('X-CSRF-Token'))
            else:
                value,token=auth.setup(body['setup_code'],body['password'],self.headers.get('Cookie'),self.headers.get('X-CSRF-Token'))
            self.extra_response_headers=[('Set-Cookie',admin_auth.cookie_header(token))]
            return self.json(200,value)
        if route=='/api/admin/logout':
            if body:raise ValueError('Logout accepts an empty object')
            auth.logout(self.headers.get('Cookie'),self.headers.get('X-CSRF-Token'))
            self.extra_response_headers=[('Set-Cookie',admin_auth.cookie_header('',clear=True))]
            return self.json(200,{'authenticated':False})
        import admin_studio_http
        if admin_studio_http.dispatch(self,route,'POST',body,root=ROOT):return
        try:
            import admin_http
        except ImportError:
            return self.json(503,{'error':'Private curation workflow is not ready'})
        if not admin_http.dispatch(self,route,'POST',body):
            return self.json(404,{'error':'Unknown administrator route'})

    def upload_upgrade_result(self, job_id, length):
        """Receive a bounded local ZIP; the importer validates every member."""
        if self.headers.get_content_type() not in {'application/zip','application/octet-stream'}:
            return self.json(400,{'error':'Choose a result ZIP archive'})
        if not 0 < length <= MAX_UPLOAD:
            return self.json(413,{'error':'Result ZIP must be between one byte and 512 MiB'})
        import upgrade_bundles
        # Validate the job before accepting a large request body.
        known=upgrade_bundles.list_jobs(root=ROOT)
        if not any(row.get('job_id',row.get('id'))==job_id for row in known):
            raise ValueError('Choose a prepared upgrade job before importing its result')
        directory=confined(ROOT/'data/upgrade-uploads'/('upload-'+os.urandom(12).hex()),ROOT/'data')
        directory.mkdir(parents=True,exist_ok=False);path=directory/'result.zip';temporary=directory/'result.partial'
        self.connection.settimeout(60)
        try:
            with temporary.open('xb') as stream:
                remaining=length
                while remaining:
                    block=self.rfile.read(min(1024*1024,remaining))
                    if not block:raise ValueError('Result upload ended before its declared length')
                    stream.write(block);remaining-=len(block)
            temporary.replace(path)
            result=upgrade_bundles.import_result(job_id,path,root=ROOT)
            self.json(201,upgrade_api.import_view(result,root=ROOT))
        except Exception as exc:
            # Failed/partial uploads remain auditable; no learned result is fabricated.
            (directory/'failure.json').write_text(json.dumps({'status':'failed','job_id':job_id,'error':str(exc)},indent=2)+'\n')
            raise

    def start_observed(self, body):
        request = observed_jobs.canonical_request(body, root=ROOT)
        with self.server.lock:
            active = self.server.active
            if active and active.state['status'] in {'created', 'queued', 'running'}:
                return self.json(409, {'error': 'Another local worker is active; wait for it to finish', 'id': active.id})
            runner = observed_jobs.ObservedRunner(request, root=ROOT)
            runner.state['status'] = 'queued'; runner.save()
            self.server.active = runner
            self.server.executor.submit(runner.execute)
        return self.json(202, {'id': runner.id, 'status': 'queued', 'status_url': '/api/observed/jobs/' + runner.id})

    def start_calibration(self, body):
        request = calibration_jobs.canonical_request(body, root=ROOT)
        with self.server.lock:
            active = self.server.active
            if active and active.state['status'] in {'created', 'queued', 'running'}:
                return self.json(409, {'error': 'Another local operation is active; wait for it to finish', 'id': active.id})
            runner = calibration_jobs.CalibrationRunner(request, root=ROOT)
            runner.state['status'] = 'queued'; runner.save()
            self.server.active = runner
            self.server.executor.submit(runner.execute)
        return self.json(202, {'id': runner.id, 'status': 'queued', 'status_url': '/api/calibration/jobs/' + runner.id})

    def start_motion(self, body):
        request = interval_jobs.canonical_request(body, root=ROOT)
        with self.server.lock:
            active = self.server.active
            if active and active.state['status'] in {'created', 'queued', 'running'}:
                return self.json(409, {'error': 'Another local worker is active; wait for it to finish', 'id': active.id})
            runner = interval_jobs.MotionRunner(request, root=ROOT)
            runner.state['status'] = 'queued'; runner.save()
            self.server.active = runner
            self.server.executor.submit(runner.execute)
        return self.json(202, {'id': runner.id, 'status': 'queued', 'status_url': '/api/motion/jobs/' + runner.id})

    def start_upgrade(self, body):
        from upgrade_runner import UpgradeRunner
        resume=body.get('resume_run_id')
        if resume is not None and (not isinstance(resume,str) or not upgrade_api.RUN_ID.fullmatch(resume)):
            raise ValueError('Invalid upgrade resume ID')
        request=upgrade_api.canonical_request({k:v for k,v in body.items() if k!='resume_run_id'})
        prepared=upgrade_api.plan(request,root=ROOT)
        if prepared.get('can_run') is False:
            raise ValueError('This upgrade is not ready: '+'; '.join(prepared.get('blockers',[])))
        with self.server.lock:
            if self.server.active and self.server.active.state['status'] in {'created','queued','running'}:
                return self.json(409,{'error':'Another local model run is active; wait for it to finish','id':self.server.active.id})
            runner=UpgradeRunner(request,root=ROOT,run_id=resume)
            if runner.state.get('request') and runner.state['request']!=request:
                raise ValueError('Resume requires the identical saved source, component and mode')
            runner.state['status']='queued';runner.save();self.server.active=runner
            self.server.executor.submit(runner.execute,request)
        return self.json(202,{'id':runner.id,'status':'queued','status_url':'/api/upgrades/runs/'+runner.id})

    def start_intelligence(self, body):
        import intelligence_api
        request = intelligence_api.canonical_request(body, root=ROOT)
        with self.server.lock:
            # Cache inspection precedes both worker creation and paid admission.
            if intelligence_api.cached_match(request, root=ROOT):
                return self.json(200, {'status': 'cached', 'case_id': request['case_id'], 'new_api_call': False})
            active = self.server.active
            if active and active.state['status'] in {'created', 'queued', 'running'}:
                if isinstance(active, intelligence_api.IntelligenceRunner) and active.request == request:
                    return self.json(202, {'id': active.id, 'status': active.state['status'], 'case_id': request['case_id']})
                return self.json(409, {'error': 'Another model operation is active. Wait for it to finish.', 'id': active.id})
            budget = meta_budget_status()
            if budget.get('blocked') or budget.get('status') == 'unavailable':
                return self.json(409, {'error': 'Paid calls are blocked by the shared estimated-cost guard. Inspect Pipeline runs for details.'})
            runner = intelligence_api.IntelligenceRunner(request, root=ROOT)
            runner.state['status'] = 'queued'; runner.save()
            self.server.active = runner
            self.server.executor.submit(runner.execute)
        return self.json(202, {'id': runner.id, 'status': 'queued', 'case_id': request['case_id']})

    def upload(self, length):
        if length <= 0 or length > MAX_UPLOAD:
            return self.json(413, {'error': 'Video upload must be between one byte and 512 MiB'})
        name = upload_name(self.headers.get('X-Filename'))
        directory = ROOT / 'data/uploads' / ('upload-' + os.urandom(8).hex())
        directory.mkdir(parents=True)
        path = directory / name
        temporary = directory / (name + '.partial')
        self.connection.settimeout(60)
        try:
            with temporary.open('xb') as stream:
                remaining = length
                while remaining:
                    block = self.rfile.read(min(1024 * 1024, remaining))
                    if not block:
                        raise ValueError('Upload ended before its declared length')
                    stream.write(block)
                    remaining -= len(block)
            temporary.replace(path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        self.json(201, {'input_path': str(path.relative_to(ROOT)), 'bytes': length, 'url': public_url(path)})

    def start_run(self, body):
        allowed = {'profile', 'input_path', 'case_id', 'hosted_reasoning_consent', 'resume_run_id'}
        if set(body) - allowed:
            raise ValueError('Unknown run fields: ' + ', '.join(sorted(set(body) - allowed)))
        for field in ['profile', 'input_path', 'case_id', 'resume_run_id']:
            if field in body and body[field] is not None and not isinstance(body[field], str):
                raise ValueError(field + ' must be a string')
        profile = load_profile(body.get('profile', 'meta_sam3d_default'))
        if profile['mode'] == 'local_inference' and body.get('hosted_reasoning_consent') is not True:
            raise ValueError(provider_disclosure(profile))
        input_path = body.get('input_path')
        if input_path:
            input_path = confined(ROOT / input_path, ROOT / 'data')
            if not input_path.is_file():
                raise ValueError('Uploaded/local data video does not exist')
        case_id = body.get('case_id')
        if case_id and profile['mode'] == 'local_inference':
            selected = next((row for row in list_cases() if row['id'] == case_id), None)
            case_id = selected['case_id'] if selected else case_id
            if not SLUG.fullmatch(case_id):
                raise ValueError('Invalid source case identifier')
        if input_path and case_id:
            raise ValueError('Choose an input video or saved case, not both')
        if profile['mode'] == 'local_inference' and not (input_path or case_id):
            raise ValueError('Choose an input video or saved case')
        with self.server.lock:
            if self.server.active and self.server.active.state['status'] in {'created', 'queued', 'running'}:
                return self.json(409, {'error': 'A model run is already active; local inference is serialized', 'id': self.server.active.id})
            runner = Runner(body.get('resume_run_id'))
            request = {'profile': profile['id'], 'input_path': str(input_path) if input_path else None, 'case_id': case_id}
            if runner.state.get('request') and runner.state['request'] != request:
                raise ValueError('Resume requires the identical profile and input/case request')
            runner.state['status'] = 'queued'
            runner.save()
            self.server.active = runner
            self.server.executor.submit(runner.execute, profile['id'], input_path=input_path, case_id=case_id)
        self.json(202, {'id': runner.id, 'status': 'queued', 'status_url': '/api/runs/' + runner.id})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', choices=['127.0.0.1'], default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error('Use an unprivileged port from 1024 to 65535')
    server = WorkspaceServer((args.host, args.port), Handler)
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    print(f'Interaction Twin: http://127.0.0.1:{server.server_port}/showcase/index.html', flush=True)
    print('Single local worker; no cluster, implicit model downloads or CORS. Ctrl-C terminates running model subprocesses.', flush=True)
    try:
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
