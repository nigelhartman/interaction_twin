"""Local geometry jobs for registered workspace components.

The browser supplies identifiers only. A fixed existing Python/CLI argument
vector reads saved source RGB/SAM masks. The explicit gated workflow may use
prepared local CoTracker/MoGe models; no API, credential, download, learned
mesh, prior pose, arbitrary model path or command is accepted by this boundary.
"""
from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import threading

import revisions

ROOT = Path(__file__).resolve().parent
LOADED_SOURCE = Path(__file__).read_bytes()
LOADED_SOURCE_SHA256 = hashlib.sha256(LOADED_SOURCE).hexdigest()
JOB_ID = re.compile(r'observed-[0-9a-f]{24}\Z')
BASE = 'data/observed-app-runs-20261031'
ACTIVE = {'created', 'queued', 'running'}
MAX_SECONDS = 180
POLICY = {'local_only': True, 'cpu_only': True, 'new_api_calls': 0, 'model_inference': False,
          'downloads': False, 'prior_geometry_or_poses_used': False,
          'max_views': 35, 'max_ba_evaluations': 70, 'focal_sensitivity': False,
          'maximum_runtime_s': MAX_SECONDS, 'automatic_retry': False,
          'physical_scale_known': False, 'physics_ready': False}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    raw = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
    temporary = path.parent / ('.job-' + os.urandom(12).hex() + '.tmp')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def canonical_request(body, *, root=ROOT):
    workflow = isinstance(body, dict) and body.get('workflow') is not None
    if not isinstance(body, dict) or (not workflow and set(body) != {'case_id', 'track_id'}):
        raise ValueError('Choose only a registered workspace case_id and existing track_id')
    path, doc, digest = revisions.published_source(root, body.get('case_id'))
    track_id = body.get('track_id')
    if not isinstance(track_id, str) or not any(t['id'] == track_id for t in doc['tracklets']):
        raise ValueError('Choose an existing component in that registered workspace case')
    count = sum(any(o['track_id'] == track_id for o in frame['observations']) for frame in doc['evidence'])
    if count < 3:
        raise ValueError('This component needs at least three saved source RGB/SAM observations')
    request = {'case_id': body['case_id'], 'track_id': track_id,
            'interaction_path': path.relative_to(Path(root).resolve()).as_posix(),
            'interaction_sha256': digest, 'source_observations': count,
            'label': next(t['label'] for t in doc['tracklets'] if t['id'] == track_id)}
    if workflow:
        from observed_pipeline import workflow_options
        request.update(workflow_options(body, request, root=root))
    return request


def catalog(*, root=ROOT):
    root = Path(root).resolve()
    published = json.loads((root / 'workspace/cases.json').read_text())
    cases, errors = [], []
    for row in published.get('cases', []):
        try:
            _, doc, digest = revisions.published_source(root, row.get('id'))
            components = []
            for track in doc['tracklets']:
                count = sum(any(o['track_id'] == track['id'] for o in frame['observations']) for frame in doc['evidence'])
                components.append({'track_id': track['id'], 'label': track['label'],
                    'source_observations': count, 'eligible': count >= 3,
                    'identity_status': track['identity_status']})
            cases.append({'id': row['id'], 'title': row.get('title') or doc['case_id'],
                'interaction_sha256': digest, 'components': components})
        except (OSError, ValueError, KeyError, TypeError):
            errors.append({'id': row.get('id'), 'error': 'Registered source is unavailable or invalid'})
    available = (root / '.venv-sam/bin/python').is_file() and (root / 'scripts/test_observed_geometry.py').is_file()
    from observed_pipeline import sparse_caches
    caches = sparse_caches(root)
    for case in cases:
        case['sparse_caches'] = [row for row in caches if row['interaction_sha256'] == case['interaction_sha256']]
    return {'cases': cases, 'errors': errors, 'runtime_available': available, 'policy': POLICY,
        'workflows': [{'id': 'sparse-only-v1', 'label': 'Original sparse feasibility test', 'inference': False},
            {'id': 'gated-local-v1', 'label': 'Gated reconstruction, diagnostics and export',
             'feature_providers': ['sift', 'cotracker', 'reuse_verified'], 'completion_modes': ['none', 'smooth_gaps', 'moge'],
             'provider_availability': {'sift': available,
                 'cotracker': available and (root / 'observed_cotracker.py').is_file() and (root / 'models/cotracker3/scaled_offline.pth').is_file(),
                 'reuse_verified': bool(caches)},
             'camera_availability': {'assumed': True, 'moge_prior': available and (root / 'automatic_camera_prior.py').is_file() and (root / 'models/moge-2-vits-normal/model.pt').is_file()},
             'completion_availability': {'none': True, 'smooth_gaps': available,
                 'moge': available and (root / 'models/moge-2-vits-normal/model.pt').is_file()},
             'new_api_calls': 0, 'model_inference': 'Only selected local tracking, camera or completion models; no downloads'}]}


def _path(identifier, root):
    if not isinstance(identifier, str) or not JOB_ID.fullmatch(identifier):
        raise ValueError('Invalid local geometry job identifier')
    base = Path(root).resolve() / BASE
    folder = base / identifier
    if folder.is_symlink() or not folder.resolve().is_relative_to(base):
        raise ValueError('Invalid local geometry job location')
    return folder / 'job.json'


def record(identifier, *, root=ROOT, active_job_id=None):
    path = _path(identifier, root)
    if path.is_symlink() or path.stat().st_size > 1024 * 1024:
        raise ValueError('Invalid local geometry job record')
    value = json.loads(path.read_text())
    if value.get('id') != identifier:
        raise ValueError('Local geometry job identity mismatch')
    if value.get('status') in ACTIVE and identifier != active_job_id:
        value = dict(value, saved_status=value['status'], status='interrupted',
            error='The local worker stopped. Saved outputs are preserved; starting again creates a separate attempt.')
    return value


def list_jobs(*, root=ROOT, active_job_id=None):
    base = Path(root).resolve() / BASE
    jobs = []
    for folder in base.glob('observed-*'):
        if not JOB_ID.fullmatch(folder.name):
            continue
        try:
            jobs.append(record(folder.name, root=root, active_job_id=active_job_id))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return {'jobs': sorted(jobs, key=lambda value: value.get('created_at', ''), reverse=True), 'policy': POLICY}


def active_identifier(active):
    """Resolve a standalone job or the parent's explicitly reported child."""
    if isinstance(active, ObservedRunner):
        return active.id
    state = getattr(active, 'state', {})
    nested = state.get('observed_child_job_id') if isinstance(state, dict) and state.get('status') in ACTIVE else None
    return nested if isinstance(nested, str) and JOB_ID.fullmatch(nested) else None


class ObservedRunner:
    """Compatible with the workspace's existing shared single-worker lock."""
    def __init__(self, request, *, root=ROOT, parent_worker_lock_held=False, state_callback=None):
        self.root = Path(root).resolve()
        self.request = request
        self.id = 'observed-' + os.urandom(12).hex()
        base = self.root / BASE
        if base.is_symlink() or not base.resolve().is_relative_to(self.root / 'data'):
            raise ValueError('Local geometry storage must remain inside project data')
        self.directory = self.root / BASE / self.id
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / 'job-manager-implementation.py').write_bytes(LOADED_SOURCE)
        self.workflow = request.get('workflow') == 'gated-local-v1'
        self.output = self.root / BASE / (self.id + ('-workflow' if self.workflow else '-sparse'))
        self.cancelled = threading.Event()
        self.timed_out = threading.Event()
        self._lock = threading.RLock()
        self._process = None
        # Internal orchestration only; neither option is accepted by HTTP.
        self.parent_worker_lock_held = bool(parent_worker_lock_held)
        self.state_callback = state_callback
        self.state = dict(request, id=self.id, status='created', phase='created',
            created_at=now(), policy=dict(POLICY), automatic_retry=False,
            result_url=None, viewer_url=None, result_kind=None,
            result_status=None, metrics=None, progress=[],
            mask_provider='saved component SAM masks; source-program provider and audit provenance retained',
            output_path=self.output.relative_to(self.root).as_posix(),
            job_manager_source={'path': (self.directory / 'job-manager-implementation.py').relative_to(self.root).as_posix(),
                                'sha256': LOADED_SOURCE_SHA256},
            geometry_engine_sha256=sha(self.root / 'observed_geometry.py'),
            geometry_cli_sha256=sha(self.root / 'scripts/test_observed_geometry.py'))
        if request.get('camera_mode') == 'moge_prior':
            self.state['camera_prior_engine_sha256'] = sha(self.root / 'automatic_camera_prior.py')
        if self.workflow:
            self.state.update(workflow_manifest_url='/files/' + (self.output / 'workflow.json').relative_to(self.root).as_posix(),
                stages=[], capture_diagnostics=[], exports={},
                workflow_code_sha256=sha(self.root / 'observed_pipeline.py'),
                workflow_cli_sha256=sha(self.root / 'scripts/run_observed_workflow.py'))
            self.state['policy'] = dict(POLICY, cpu_only=request['feature_provider'] != 'cotracker' and request['completion'] != 'moge' and request.get('camera_mode') != 'moge_prior',
                model_inference=request['feature_provider'] == 'cotracker' or request['completion'] == 'moge' or request.get('camera_mode') == 'moge_prior',
                maximum_runtime_s=900, workflow='gated-local-v1')
        self.save()

    def save(self):
        with self._lock:
            self.state['updated_at'] = now()
            _write(self.directory / 'job.json', self.state)
            if self.state_callback:
                self.state_callback(json.loads(json.dumps(self.state)))

    def _terminate(self):
        with self._lock:
            process = self._process
        if process and process.poll() is None:
            try:
                process.terminate()
            except ProcessLookupError:
                return
            def hard_stop():
                if process.poll() is None:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
            timer = threading.Timer(3, hard_stop)
            timer.daemon = True
            timer.start()

    def cancel(self):
        self.cancelled.set()
        with self._lock:
            self.state['cancellation_requested'] = True
            self.state['cancellation_requested_at'] = now()
            self.save()
        self._terminate()

    def _timeout(self):
        self.timed_out.set()
        self._terminate()

    def _launch(self, arguments):
        environment = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
        return subprocess.Popen(arguments, cwd=self.root, env=environment, shell=False,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)

    def execute(self):
        if self.cancelled.is_set():
            self.state.update(status='cancelled', phase='cancelled', finished_at=now()); self.save(); return
        self.state.update(status='running', phase='validate_source', started_at=now()); self.save()
        timer = model_lock = None
        try:
            body = {key: self.request[key] for key in ['case_id', 'track_id']}
            if self.workflow:
                body.update({key: self.request[key] for key in ['workflow', 'feature_provider', 'completion']})
                if 'camera_mode' in self.request:
                    body['camera_mode'] = self.request['camera_mode']
                if self.request.get('reuse_sparse_id'):
                    body['reuse_sparse_id'] = self.request['reuse_sparse_id']
            validated = canonical_request(body, root=self.root)
            if validated != self.request:
                raise ValueError('Registered source changed before execution')
            if sha(self.root / 'observed_geometry.py') != self.state['geometry_engine_sha256'] or sha(self.root / 'scripts/test_observed_geometry.py') != self.state['geometry_cli_sha256']:
                raise ValueError('Geometry code changed before execution')
            if self.request.get('camera_mode') == 'moge_prior' and sha(self.root / 'automatic_camera_prior.py') != self.state['camera_prior_engine_sha256']:
                raise ValueError('Camera estimation code changed before execution')
            runtime = self.root / '.venv-sam/bin/python'
            if not runtime.is_file():
                raise ValueError('Prepared local CPU runtime is unavailable; no installation was attempted')
            if not self.parent_worker_lock_held:
                lock_directory = self.root / 'data/flow-runs'
                if lock_directory.is_symlink() or not lock_directory.resolve().is_relative_to(self.root / 'data'):
                    raise ValueError('The fixed local worker lock must remain inside project data')
                lock_directory.mkdir(exist_ok=True)
                descriptor = os.open(lock_directory / '.model-worker.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
                model_lock = os.fdopen(descriptor, 'a+')
                fcntl.flock(model_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            arguments = [str(runtime), str(self.root / 'scripts/test_observed_geometry.py'),
                '--program', self.request['interaction_path'], '--track', self.request['track_id'],
                '--output', self.output.relative_to(self.root).as_posix(), '--max-views', '35',
                '--max-ba-evaluations', '70', '--no-sensitivity']
            if self.workflow:
                if sha(self.root / 'observed_pipeline.py') != self.state['workflow_code_sha256'] or sha(self.root / 'scripts/run_observed_workflow.py') != self.state['workflow_cli_sha256']:
                    raise ValueError('Workflow code changed before execution')
                request_path = self.directory / 'workflow-request.json'
                from observed_pipeline import write_once
                write_once(request_path, self.request)
                arguments = [str(runtime), str(self.root / 'scripts/run_observed_workflow.py'),
                    '--request', request_path.relative_to(self.root).as_posix(),
                    '--output', self.output.relative_to(self.root).as_posix()]
            self.state.update(phase='source_pixel_geometry', command_policy='fixed_existing_python_cli_argument_vector_no_shell')
            self.save()
            with self._lock:
                if self.cancelled.is_set():
                    self.state.update(status='cancelled', phase='cancelled'); self.save(); return
                self._process = self._launch(arguments)
            timer = threading.Timer(900 if self.workflow else MAX_SECONDS, self._timeout)
            timer.daemon = True; timer.start()
            logged = 0
            with (self.directory / 'progress.log').open('w') as log:
                for line in self._process.stdout:
                    if logged < 262144:
                        safe = line[:262144 - logged]; log.write(safe); log.flush(); logged += len(safe)
                    try:
                        event = json.loads(line)
                    except (ValueError, TypeError):
                        continue
                    if isinstance(event, dict) and isinstance(event.get('phase'), str):
                        saved = {'phase': event['phase'][:100], 'at': now()}
                        for key in ['selected_views', 'track_count', 'pair_count', 'points', 'train_views', 'status', 'stage']:
                            if isinstance(event.get(key), (str, int, float)):
                                saved[key] = event[key]
                        self.state['progress'] = (self.state['progress'] + [saved])[-40:]
                        self.state['phase'] = saved.get('stage') or saved['phase']
                        if self.workflow:
                            self._workflow_progress()
                        self.save()
            returncode = self._process.wait()
            self.state['worker_returncode'] = returncode
            viewer_path = self.output / 'viewer.json'
            if self.workflow:
                self._workflow_progress()
                self.state['workflow_status'] = self.state.get('workflow_status', 'failed')
                if self.state['workflow_status'] == 'failed':
                    returncode = 1
            if viewer_path.is_file():
                viewer = json.loads(viewer_path.read_text())
                if viewer.get('id') != self.output.name or viewer.get('new_api_calls') != 0 or viewer.get('prior_geometry_or_depth_used') is not False:
                    raise ValueError('Local geometry output failed identity/policy checks')
                self.state.update(result_url='/files/' + viewer_path.relative_to(self.root).as_posix(),
                    viewer_url='/files/' + viewer_path.relative_to(self.root).as_posix(),
                    result_kind='observed_sparse_geometry', result_status=viewer['status'],
                    metrics=viewer['metrics'], result_reason=viewer['reason'])
            if self.cancelled.is_set():
                self.state.update(status='cancelled', phase='cancelled')
            elif self.timed_out.is_set():
                self.state.update(status='failed', phase='time_limit', error='Bounded local geometry time limit reached. Saved partial outputs are preserved.')
            elif returncode != 0 or (not self.state['result_url'] and not self.workflow):
                self.state.update(status='failed', phase='failed', error='Local geometry did not complete. Saved logs and partial outputs are preserved; no retry ran.')
            else:
                self.state.update(status='completed', phase='complete')
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.state.update(status='cancelled' if self.cancelled.is_set() else 'failed', phase='failed',
                error='The registered local geometry operation could not complete. Its artifacts are preserved; no automatic retry or model fallback ran.',
                failure_type=type(error).__name__)
        finally:
            if timer:
                timer.cancel()
            if model_lock:
                model_lock.close()
            self.state['finished_at'] = now()
            self.state['log_url'] = '/files/' + (self.directory / 'progress.log').relative_to(self.root).as_posix() if (self.directory / 'progress.log').is_file() else None
            self.state['partial_outputs_preserved'] = self.output.exists()
            self.save()

    def _workflow_progress(self):
        path = self.output / 'workflow.json'
        if not path.is_file():
            return
        value = json.loads(path.read_text())
        if value.get('id') != self.output.name or value.get('new_api_calls') != 0 or value.get('production_assets_changed') is not False:
            raise ValueError('Workflow result identity/policy check failed')
        self.state.update(workflow_status=value['status'], stages=value.get('stages', []),
            capture_diagnostics=value.get('capture_diagnostics', []), exports=value.get('exports', {}),
            result_url=value.get('result_url'), viewer_url=value.get('result_url'),
            result_kind=value.get('result_kind'), result_status=value.get('result_status'), metrics=value.get('metrics'))


def run_follow_on_components(case_id, track_ids=None, *, root=ROOT, cancel_event,
                             on_state=None, parent_worker_lock_held=False,
                             feature_provider='cotracker', completion='smooth_gaps', max_components=2, camera_mode=None):
    """Serial follow-on inside an already locked, registered upload pipeline.

    The caller must retain its existing model-worker flock throughout this
    function. The HTTP job boundary cannot select this internal lock bypass.
    Each child remains an independent preserved run, including failures.
    """
    if parent_worker_lock_held is not True:
        raise ValueError('Follow-on requires the parent pipeline to retain its fixed model-worker lock')
    if not isinstance(max_components, int) or isinstance(max_components, bool) or not 1 <= max_components <= 3:
        raise ValueError('A follow-on may attempt one to three components only')
    if camera_mode is not None and camera_mode not in {'assumed', 'moge_prior'}:
        raise ValueError('Unsupported follow-on camera mode')
    root = Path(root).resolve()
    _, program, _ = revisions.published_source(root, case_id)
    available = [row['id'] for row in program['tracklets']]
    counts = {key: sum(any(o['track_id'] == key for o in frame['observations'])
                      for frame in program['evidence']) for key in available}
    if track_ids is None:
        selected = [key for key in available if counts[key] >= 3][:max_components]
    else:
        if not isinstance(track_ids, list) or not track_ids or len(track_ids) > max_components or any(not isinstance(key, str) for key in track_ids) or len(set(track_ids)) != len(track_ids):
            raise ValueError('Choose a unique bounded list of registered components')
        if any(not isinstance(key, str) or key not in available or counts[key] < 3 for key in track_ids):
            raise ValueError('Every selected component needs three registered source observations')
        selected = track_ids
    requests = [canonical_request({'case_id': case_id, 'track_id': key, 'workflow': 'gated-local-v1',
                                  'feature_provider': feature_provider, 'completion': completion,
                                  **({'camera_mode': camera_mode} if camera_mode is not None else {})}, root=root)
                for key in selected]
    rows = []
    for request in requests:
        if cancel_event.is_set():
            break
        runner = ObservedRunner(request, root=root, parent_worker_lock_held=True, state_callback=on_state)
        finished = threading.Event()
        def relay_cancel():
            while not finished.wait(.1):
                if cancel_event.is_set():
                    runner.cancel()
                    return
        relay = threading.Thread(target=relay_cancel, daemon=True)
        relay.start()
        try:
            if cancel_event.is_set():
                runner.cancel()
            runner.execute()
        finally:
            finished.set(); relay.join(1)
        rows.append({'id': runner.id, 'track_id': request['track_id'],
            'status_url': '/api/observed/jobs/' + runner.id,
            'job_url': '/files/' + (runner.directory / 'job.json').relative_to(root).as_posix(),
            **{key: runner.state.get(key) for key in ['status', 'workflow_status', 'result_url', 'result_kind',
                'result_status', 'workflow_manifest_url', 'exports', 'capture_diagnostics', 'log_url', 'job_manager_source']}})
    return {'format': 'registered-local-reconstruction-follow-on', 'version': 1, 'case_id': case_id,
        'components': rows, 'selected_component_ids': selected,
        'skipped_components': [{'track_id': key, 'reason': 'fewer_than_three_saved_observations' if counts[key] < 3
                               else 'bounded_component_count'} for key in available if key not in selected],
        'unstarted_component_ids': selected[len(rows):], 'cancelled': cancel_event.is_set(),
        'maximum_components': max_components, 'maximum_runtime_s_per_component': 900,
        'new_api_calls': 0, 'automatic_retry': False, 'production_assets_changed': False,
        'physics_ready': False}
