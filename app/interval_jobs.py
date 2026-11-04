"""Registered, bounded local motion jobs; browsers supply identifiers only.

This boundary never calls a hosted model, selects new geometry or edits a
published program. The fixed child CLI owns the interval experiment; the parent
retains the same model-worker flock used by uploads and reconstruction jobs.
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
BASE = 'data/interval-motion-20261031/jobs'
JOB_ID = re.compile(r'interval-[0-9a-f]{24}\Z')
ACTIVE = {'created', 'queued', 'running'}
POLICY = {'local_only': True, 'new_api_calls': 0, 'max_episodes': 2,
          'maximum_runtime_s': 900, 'downloads': False, 'automatic_retry': False,
          'original_assets_changed': False, 'physical_accuracy_verified': False,
          'cross_episode_interpolation': False, 'common_assembly_established': False}
LOADED_SOURCE = Path(__file__).read_bytes()


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.parent / ('.interval-' + os.urandom(8).hex() + '.tmp')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def data_file(root, relative, digest=None):
    if not isinstance(relative, str) or Path(relative).is_absolute() or any(p.startswith('.') for p in Path(relative).parts):
        raise ValueError('A declared source artifact must be a visible local data file')
    path = (root / relative).resolve()
    if not path.is_relative_to(root / 'data') or not path.is_file():
        raise ValueError('A declared source artifact is unavailable')
    if digest and sha(path) != digest:
        raise ValueError('A declared source artifact changed')
    return path


def canonical_request(body, *, root=ROOT):
    if not isinstance(body, dict) or set(body) != {'case_id', 'track_id'}:
        raise ValueError('Choose only a registered case_id and component track_id')
    root = Path(root).resolve()
    path, doc, digest = revisions.published_source(root, body['case_id'])
    return _request_from_source(body, root, path, doc, digest)


def _request_from_source(body, root, path, doc, digest):
    """Project a hash-attested source; never itself admits a model worker."""
    tid = body['track_id']
    track = next((t for t in doc['tracklets'] if t['id'] == tid), None)
    if not isinstance(tid, str) or track is None:
        raise ValueError('Choose an existing component')
    assets = [a for a in doc['assets'] if a['track_id'] == tid]
    if len(assets) != 1:
        raise ValueError('This experiment needs one saved native mesh for the component')
    asset = assets[0]
    data_file(root, asset['path'], asset['sha256'])
    refs = {r['experiment_input_id'] for r in doc['motion'] if r['track_id'] == tid}
    baseline = [i for i in doc['inputs'] if i['id'] in refs and i['kind'] == 'model_conditional_motion_experiment']
    if len(baseline) != 1:
        raise ValueError('This component needs one preserved baseline motion experiment')
    data_file(root, baseline[0]['path'], baseline[0]['sha256'])
    count = sum(any(o['track_id'] == tid for o in frame['observations']) for frame in doc['evidence'])
    if count < 3:
        raise ValueError('This component needs three saved image/mask observations')
    result = {'case_id': body['case_id'], 'track_id': tid, 'label': track['label'],
            'interaction_path': path.relative_to(root).as_posix(), 'interaction_sha256': digest,
            'source_video_sha256': doc['source']['sha256'], 'source_observations': count,
            'asset_path': asset['path'], 'asset_sha256': asset['sha256'],
            'baseline_motion_path': baseline[0]['path'], 'baseline_motion_sha256': baseline[0]['sha256'],
            'baseline_supported_samples': sum(r['track_id'] == tid and r['status'] == 'supported_conditional_motion_candidate' for r in doc['motion'])}
    index = root / 'data/interval-motion-20261031/saved-audit-index-v2.json'
    if not index.is_file():
        index = root / 'data/interval-motion-20261031/saved-audit-index-v1.json'
    if index.is_file():
        saved = json.loads(index.read_text())
        if saved.get('format') != 'interval-motion-saved-audit-index' or saved.get('version') not in {1, 2}:
            raise ValueError('Saved interval audit index is invalid')
        matches = [row for row in saved.get('entries', []) if row.get('interaction_sha256') == digest
                   and row.get('source_video_sha256') == doc['source']['sha256'] and row.get('track_id') == tid]
        if len(matches) > 1:
            raise ValueError('Saved interval plan is ambiguous')
        if matches:
            row = matches[0]
            for key in ['plan', 'audit']:
                candidate = data_file(root, row[key + '_path'], row[key + '_sha256'])
                if not candidate.is_relative_to(root / 'data/interval-motion-20261031'):
                    raise ValueError('Saved interval audit is outside its versioned directory')
                result[key + '_path'] = row[key + '_path']
                result[key + '_sha256'] = row[key + '_sha256']
            audit = json.loads((root / row['audit_path']).read_text())
            if audit.get('plan_sha256') != row['plan_sha256'] or audit.get('interaction_sha256') != digest or audit.get('source_video_sha256') != doc['source']['sha256']:
                raise ValueError('Saved interval audit source binding is invalid')
    return result


def catalog(*, root=ROOT):
    root = Path(root).resolve()
    published = json.loads((root / 'workspace/cases.json').read_text())
    cases, errors = [], []
    for row in published.get('cases', []):
        try:
            # A catalog is a read-only display projection. Load the published
            # snapshot once rather than parsing its embedded scene payload for
            # every component. Admission/execution still fully validate through
            # canonical_request -> revisions.published_source.
            if not isinstance(row.get('id'), str) or not revisions.CASE_ID.fullmatch(row['id']):
                raise ValueError('Invalid published case identifier')
            path = data_file(root, row.get('interaction_path'), row.get('original_sha256'))
            if path.name != 'interaction.json':
                raise ValueError('Not a published interaction')
            digest = sha(path)
            doc = json.loads(path.read_text())
            components = []
            for track in doc['tracklets']:
                try:
                    item = _request_from_source({'case_id': row['id'], 'track_id': track['id']}, root, path, doc, digest)
                    item = {key: item[key] for key in ['track_id', 'label', 'source_observations', 'asset_path', 'asset_sha256', 'baseline_supported_samples']}
                    item['eligible'] = True
                except (ValueError, KeyError, OSError) as exc:
                    item = {'track_id': track['id'], 'label': track['label'], 'eligible': False, 'reason': str(exc)}
                components.append(item)
            cases.append({'id': row['id'], 'title': row.get('title') or doc['case_id'],
                          'interaction_sha256': digest, 'components': components})
        except (ValueError, KeyError, OSError, TypeError):
            errors.append({'id': row.get('id'), 'error': 'Registered source unavailable'})
    runtime = all((root / p).is_file() for p in ['.venv-sam/bin/python', 'scripts/stage4_interval_motion.py', 'interval_motion.py', 'interval_motion_search.py', 'motion_lab.py', 'models/cotracker3/scaled_offline.pth'])
    return {'cases': cases, 'errors': errors, 'runtime_available': runtime, 'policy': POLICY}


def record(job_id, *, root=ROOT, active_job_id=None):
    if not isinstance(job_id, str) or not JOB_ID.fullmatch(job_id):
        raise ValueError('Invalid local motion job identifier')
    root = Path(root).resolve()
    path = root / BASE / job_id / 'job.json'
    if path.is_symlink() or path.parent.is_symlink() or not path.resolve().is_relative_to(root / BASE) or not path.is_file() or path.stat().st_size > 128 * 1024:
        raise ValueError('Saved local motion job unavailable')
    value = json.loads(path.read_text())
    if value.get('id') != job_id:
        raise ValueError('Saved local motion job identity mismatch')
    if value['status'] in ACTIVE and job_id != active_job_id:
        value = dict(value, saved_status=value['status'], status='interrupted',
                     error='No active worker for this saved attempt. Partial results remain preserved.', automatic_retry=False)
    return value


def list_jobs(*, root=ROOT, active_job_id=None):
    root = Path(root).resolve()
    rows = []
    for path in (root / BASE).glob('interval-*/job.json'):
        try:
            rows.append(record(path.parent.name, root=root, active_job_id=active_job_id))
        except (ValueError, OSError, KeyError):
            continue
    return {'jobs': sorted(rows, key=lambda r: r['created_at'], reverse=True), 'new_api_calls': 0}


class MotionRunner:
    def __init__(self, request, *, root=ROOT):
        self.root = Path(root).resolve()
        self.request = request
        self.id = 'interval-' + os.urandom(12).hex()
        base = self.root / BASE
        if not base.resolve().is_relative_to(self.root / 'data') or any(p.is_symlink() for p in [self.root / 'data', base.parent, base]):
            raise ValueError('Local motion storage is not confined')
        self.directory = base / self.id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.output = self.directory / 'result'
        (self.directory / 'job-manager-implementation.py').write_bytes(LOADED_SOURCE)
        self.cancelled, self.timed_out = threading.Event(), threading.Event()
        self._lock, self._process = threading.RLock(), None
        code = [self.root / p for p in ['interval_motion.py', 'scripts/stage4_interval_motion.py', 'interval_motion_search.py', 'motion_lab.py']]
        self.state = dict(request, id=self.id, status='created', phase='created', created_at=now(),
                          policy=dict(POLICY), result_url=None, viewer_url=None, events=[],
                          implementation_sha256={p.relative_to(self.root).as_posix(): sha(p) for p in code},
                          log_url='/files/' + (self.directory / 'progress.log').relative_to(self.root).as_posix())
        self.save()

    def save(self):
        with self._lock:
            self.state['updated_at'] = now()
            write(self.directory / 'job.json', self.state)

    def _terminate(self):
        with self._lock:
            process = self._process
        if process and process.poll() is None:
            process.terminate()
            def hard_stop():
                if process.poll() is None:
                    process.kill()
            timer = threading.Timer(3, hard_stop); timer.daemon = True; timer.start()

    def cancel(self):
        self.cancelled.set()
        self.state['cancellation_requested'] = True
        self.save(); self._terminate()

    def _timeout(self):
        self.timed_out.set(); self._terminate()

    def _launch(self, arguments):
        return subprocess.Popen(arguments, cwd=self.root, env=dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2'),
                                shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    def _verify_implementation(self):
        for path, digest in self.state['implementation_sha256'].items():
            if sha(self.root / path) != digest:
                raise ValueError('Motion implementation changed during this attempt; partial artifacts remain preserved')

    def execute(self):
        lock = timer = None
        self.state.update(status='running', phase='validate_source', started_at=now()); self.save()
        try:
            if self.cancelled.is_set():
                raise InterruptedError('Cancelled before local inference')
            current = canonical_request({k: self.request[k] for k in ['case_id', 'track_id']}, root=self.root)
            if current != self.request:
                raise ValueError('Published component changed before execution')
            self._verify_implementation()
            lockdir = self.root / 'data/flow-runs'
            if lockdir.is_symlink() or not lockdir.resolve().is_relative_to(self.root / 'data'):
                raise ValueError('Local model worker lock is not confined')
            lockdir.mkdir(exist_ok=True)
            fd = os.open(lockdir / '.model-worker.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            lock = os.fdopen(fd, 'a+')
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            arguments = [str(self.root / '.venv-sam/bin/python'), str(self.root / 'scripts/stage4_interval_motion.py'),
                         '--interaction', self.request['interaction_path'], '--track', self.request['track_id'],
                         '--out', self.output.relative_to(self.root).as_posix()]
            if self.request.get('audit_path'):
                arguments.extend(['--audit', self.request['audit_path'], '--audit-plan', self.request['plan_path']])
            write(self.directory / 'request.json', {'request': self.request, 'arguments': arguments, 'policy': POLICY})
            timer = threading.Timer(POLICY['maximum_runtime_s'], self._timeout); timer.daemon = True; timer.start()
            self.state.update(phase='local_interval_tracking'); self.save()
            with self._lock:
                self._process = self._launch(arguments)
                if self.cancelled.is_set():
                    self._terminate()
            logged = 0
            with (self.directory / 'progress.log').open('w') as log:
                for line in self._process.stdout:
                    if logged < 2 * 1024 * 1024:
                        log.write(line[:16384]); log.flush(); logged += len(line)
                    try:
                        event = json.loads(line)
                        if isinstance(event, dict) and len(line) <= 16384:
                            self.state['events'] = (self.state['events'] + [event])[-24:]; self.save()
                    except (ValueError, TypeError):
                        pass
                code = self._process.wait()
            if self.cancelled.is_set():
                raise InterruptedError('Cancelled; partial results are preserved')
            if self.timed_out.is_set():
                raise TimeoutError('Bounded local motion runtime exceeded; no retry')
            if code:
                raise ValueError('Local motion attempt failed; inspect its saved log')
            self._verify_implementation()
            report_path = self.output / 'interval-motion.json'
            if not report_path.is_file():
                raise ValueError('Local motion report was not produced')
            report = json.loads(report_path.read_text())
            if report.get('format') != 'short-interval-motion' or report.get('version') != 1:
                raise ValueError('Unsupported local motion report format')
            # The exact result contract is checked independently by the viewer
            # builder as well. Keep browser transport limited to registered IDs.
            if report.get('track_id') != self.request['track_id'] or report.get('interaction_sha256') != self.request['interaction_sha256']:
                raise ValueError('Local motion report source identity mismatch')
            if report.get('mesh_sha256') != self.request['asset_sha256'] or report.get('source_video_sha256') != self.request['source_video_sha256'] or report.get('new_api_calls') != 0:
                raise ValueError('Local motion report asset/source policy mismatch')
            if report.get('physical_pose_accuracy_measured') is not False or report.get('usable_physical_pose_recovered') is not False:
                raise ValueError('Local diagnostic cannot claim verified physical motion')
            import motion_lab
            if motion_lab.LOADED_SOURCE_SHA256 != self.state['implementation_sha256']['motion_lab.py']:
                raise ValueError('The running display adapter is stale; restart the local processor before another attempt')
            viewer = self.output / 'viewer.json'
            normalized = motion_lab.normalize(self.root, report_path)
            self._verify_implementation()
            write(viewer, normalized)
            self.state.update(status='completed', phase='complete', result_status=report.get('status'),
                              result_sha256=sha(report_path), result_url='/files/' + report_path.relative_to(self.root).as_posix(),
                              viewer_data_sha256=sha(viewer), viewer_data_url='/files/' + viewer.relative_to(self.root).as_posix(),
                              viewer_url='/workspace/motion-lab.html?job=' + self.id + '&case=' + self.request['case_id'] + '&track=' + self.request['track_id'])
        except InterruptedError as exc:
            self.state.update(status='cancelled', phase='cancelled', error=str(exc))
        except Exception as exc:
            self.state.update(status='failed', phase='failed', error=str(exc), partial_artifacts_preserved=True)
        finally:
            if timer:
                timer.cancel()
            if lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN); lock.close()
            self.state['finished_at'] = now(); self.save()
