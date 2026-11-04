"""Bounded local checkerboard jobs. No model, API, camera or geometry fallback."""
from __future__ import annotations

import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess
import threading

from pipeline import now, sha, write

ROOT = Path(__file__).resolve().parent
BASE = Path('data/calibration-runs-20261101')
IDENTIFIER = re.compile(r'calibration-[0-9a-f]{24}\Z')
UPLOAD = re.compile(r'data/uploads/upload-[0-9a-f]{16}/[A-Za-z0-9][A-Za-z0-9_. -]{0,119}\Z')
ACTIVE = {'created', 'queued', 'running'}
CODE = ['calibration_core.py', 'scripts/calibrate_camera.py']
POLICY = {'new_api_calls': 0, 'learned_model_calls': 0, 'cpu_only': True,
          'max_frames': 50, 'maximum_runtime_s': 120, 'automatic_retry': False,
          'changes_existing_geometry_or_poses': False, 'metric_object_scale_verified': False,
          'automatically_applied_to_object_capture': False}


def plain(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError('Expected a confined relative local file')
    path = root / relative
    if path.resolve() != path or not path.is_file():
        raise ValueError('Choose an existing local file without symlinks')
    return path


def canonical_request(body, *, root=ROOT):
    root = Path(root).resolve()
    if not isinstance(body, dict) or set(body) != {'input_path', 'inner_rows', 'inner_cols', 'square_size_mm'}:
        raise ValueError('Choose one uploaded calibration video and its checkerboard specification')
    name = body['input_path']
    if not isinstance(name, str) or not UPLOAD.fullmatch(name):
        raise ValueError('Choose a video received by this app upload endpoint')
    source = plain(root, name)
    if source.suffix.lower() not in {'.mp4', '.mov', '.m4v', '.webm'} or not 0 < source.stat().st_size <= 512*1024*1024:
        raise ValueError('Choose a bounded uploaded MP4, MOV or WebM')
    for key in ['inner_rows', 'inner_cols']:
        if type(body[key]) is not int or not 4 <= body[key] <= 16:
            raise ValueError('Inner corner rows and columns must be integers from 4 to 16')
    if body['inner_rows'] * body['inner_cols'] > 240:
        raise ValueError('Use at most 240 inner corners')
    size = body['square_size_mm']
    if size is not None and (type(size) not in {int, float} or not math.isfinite(size) or not .1 <= size <= 500):
        raise ValueError('Measured square size must be between 0.1 and 500 mm, or left unknown')
    code = {name: sha(plain(root, name)) for name in CODE}
    return {**body, 'source_sha256': sha(source), 'code_sha256': code}


def _record_path(identifier, root):
    if not isinstance(identifier, str) or not IDENTIFIER.fullmatch(identifier):
        raise ValueError('Invalid calibration identifier')
    return plain(Path(root).resolve(), BASE / identifier / 'job.json')


def record(identifier, *, root=ROOT, active_job_id=None):
    path = _record_path(identifier, root)
    if path.stat().st_size > 1024*1024:
        raise ValueError('Oversized calibration record')
    value = json.loads(path.read_text())
    if value.get('id') != identifier:
        raise ValueError('Calibration record identity mismatch')
    if value.get('status') in ACTIVE and identifier != active_job_id:
        return dict(value, saved_status=value['status'], status='interrupted',
                    error='The worker stopped. Original outputs remain; another run creates a separate attempt.')
    return value


def list_jobs(*, root=ROOT, active_job_id=None):
    rows = []
    for folder in (Path(root) / BASE).glob('calibration-*'):
        try:
            rows.append(record(folder.name, root=root, active_job_id=active_job_id))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return {'jobs': sorted(rows, key=lambda r: r.get('created_at', ''), reverse=True), 'policy': POLICY}


def validate_camera(camera, report, request, *, root, result):
    """Reject malformed/cross-source records before publishing a download link.

    These are integrity and numerical-shape checks, not calibration acceptance
    or proof of the physical camera. The core owns the frozen view tests.
    """
    if camera.get('format') != 'explicit-checkerboard-camera-record' or camera.get('version') != 1:
        raise ValueError('Unsupported camera record')
    if camera.get('physical_accuracy_verified') is not False or camera.get('object_metric_scale_established') is not False or camera.get('heldout_checks_passed') is not True or camera.get('metric_scale_verified') is not False:
        raise ValueError('Unexpected camera record claims')
    if camera.get('source', {}).get('sha256') != request['source_sha256'] or camera.get('source', {}).get('path') != request['input_path']:
        raise ValueError('Camera record source identity mismatch')
    board = camera.get('board', {})
    if any(board.get(k) != request[k] for k in ['inner_rows', 'inner_cols', 'square_size_mm']):
        raise ValueError('Camera board specification mismatch')
    if board.get('board_pose_output_units') != ('grid_units' if request['square_size_mm'] is None else 'mm') or board.get('metric_scale_verified') is not False:
        raise ValueError('Camera board units mismatch')
    size = camera.get('image_size')
    if not isinstance(size, list) or len(size) != 2 or any(type(v) is not int or v <= 0 for v in size) or size[0]*size[1] > 20_000_000 or size != report.get('image_size'):
        raise ValueError('Invalid native camera dimensions')
    matrix = camera.get('K')
    number = lambda v: type(v) in {int, float} and math.isfinite(v)
    if not isinstance(matrix, list) or len(matrix) != 3 or any(not isinstance(row, list) or len(row) != 3 or not all(number(v) for v in row) for row in matrix):
        raise ValueError('Invalid finite camera matrix')
    if matrix[2] != [0, 0, 1] or matrix[0][0] <= 0 or matrix[1][1] <= 0 or matrix[0][1] != 0 or matrix[1][0] != 0:
        raise ValueError('Invalid pinhole camera matrix')
    distortion = camera.get('distortion_coefficients')
    if camera.get('distortion_model') != 'opencv_brown_k1_k2_p1_p2_fixed_k3_zero' or not isinstance(distortion, list) or len(distortion) != 5 or not all(number(v) for v in distortion) or distortion[4] != 0:
        raise ValueError('Invalid declared camera distortion')
    provenance = camera.get('provenance')
    if not isinstance(provenance, dict) or set(provenance) != {'protocol', 'detections', 'implementation'} or provenance != report.get('provenance'):
        raise ValueError('Camera evidence provenance mismatch')
    for artifact in provenance.values():
        if not isinstance(artifact, dict) or set(artifact) != {'path', 'sha256'}:
            raise ValueError('Malformed camera evidence artifact')
        path = plain(root, artifact['path'])
        if not path.is_relative_to(result) or path.stat().st_size > 10*1024*1024 or sha(path) != artifact['sha256']:
            raise ValueError('Camera evidence identity mismatch')


class CalibrationRunner:
    def __init__(self, request, *, root=ROOT):
        self.root = Path(root).resolve()
        self.request = request
        self.id = 'calibration-' + os.urandom(12).hex()
        self.directory = self.root / BASE / self.id
        if self.directory.resolve() != self.directory:
            raise ValueError('Calibration output must remain in local data')
        self.directory.mkdir(parents=True, exist_ok=False)
        self.result = self.directory / 'result'
        self.cancelled = threading.Event()
        self.timed_out = threading.Event()
        self.process = None
        self.state = {'id': self.id, 'status': 'created', 'phase': 'created', 'created_at': now(),
                      'request': request, 'policy': POLICY, 'result_url': None, 'camera_record_url': None}
        self.save()

    def save(self):
        self.state['updated_at'] = now()
        write(self.directory / 'job.json', self.state)

    def cancel(self):
        self.cancelled.set()
        self._stop()

    def _stop(self):
        p = self.process
        if p and p.poll() is None:
            p.terminate()
            def kill():
                if p.poll() is None:
                    p.kill()
            threading.Timer(3, kill).start()

    def _timeout(self):
        self.timed_out.set()
        self._stop()

    def _launch(self, argv, log):
        return subprocess.Popen(argv, cwd=self.root, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, shell=False,
                                env={'PATH': '/opt/homebrew/bin:/usr/bin:/bin', 'PYTHONDONTWRITEBYTECODE': '1',
                                     'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'})

    def execute(self):
        lock = timer = None
        try:
            if self.cancelled.is_set():
                self.state.update(status='cancelled', phase='cancelled'); return
            fresh = canonical_request({k: self.request[k] for k in ['input_path', 'inner_rows', 'inner_cols', 'square_size_mm']}, root=self.root)
            if fresh != self.request:
                raise ValueError('Calibration input or implementation changed after admission')
            worker = self.root / 'data/flow-runs/.model-worker.lock'
            worker.parent.mkdir(parents=True, exist_ok=True)
            if worker.is_symlink() or worker.resolve() != worker:
                raise ValueError('Invalid shared worker lock')
            lock = worker.open('a+')
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError('Another local operation holds the shared worker lock') from None
            for name in CODE:
                (self.directory / Path(name).name).write_bytes(plain(self.root, name).read_bytes())
            argv = [str(self.root / '.venv-sam/bin/python'), '-B', str(self.root / 'scripts/calibrate_camera.py'),
                    '--input', self.request['input_path'], '--out', self.result.relative_to(self.root).as_posix(),
                    '--inner-rows', str(self.request['inner_rows']), '--inner-cols', str(self.request['inner_cols'])]
            if self.request['square_size_mm'] is not None:
                argv += ['--square-size-mm', str(self.request['square_size_mm'])]
            self.state.update(status='running', phase='detect_and_validate', started_at=now()); self.save()
            if self.cancelled.is_set():
                self.state.update(status='cancelled', phase='cancelled'); return
            with (self.directory / 'worker.log').open('wb') as log:
                self.process = self._launch(argv, log)
                if self.cancelled.is_set(): self._stop()
                timer = threading.Timer(POLICY['maximum_runtime_s'], self._timeout); timer.start()
                code = self.process.wait()
            if self.cancelled.is_set() or self.timed_out.is_set():
                self.state.update(status='cancelled' if self.cancelled.is_set() else 'timed_out', phase='stopped'); return
            if code != 0:
                raise ValueError('Calibration worker failed; original log and outputs are preserved')
            if canonical_request({k: self.request[k] for k in ['input_path', 'inner_rows', 'inner_cols', 'square_size_mm']}, root=self.root) != self.request:
                raise ValueError('Calibration input or implementation changed during execution')
            path = plain(self.root, self.result.relative_to(self.root) / 'report.json')
            if path.stat().st_size > 2*1024*1024:
                raise ValueError('Calibration report exceeds display bound')
            report = json.loads(path.read_text())
            if report.get('format') != 'checkerboard-camera-calibration' or report.get('version') != 1:
                raise ValueError('Unsupported calibration output')
            if report.get('source', {}).get('sha256') != self.request['source_sha256']:
                raise ValueError('Calibration report belongs to another source')
            if report.get('status') not in {'candidate_passed_checks', 'insufficient_evidence', 'rejected_degenerate_views', 'rejected_validation', 'timeout'}:
                raise ValueError('Unknown calibration outcome')
            if report.get('physical_accuracy_verified') is not False or report.get('object_metric_scale_established') is not False or report.get('new_api_calls') != 0:
                raise ValueError('Unexpected calibration claims or paid execution')
            self.state.update(status='completed', phase='completed', outcome=report['status'],
                              result_url='/files/' + path.relative_to(self.root).as_posix(), result_sha256=sha(path))
            camera = report.get('camera_record')
            if report['status'] == 'candidate_passed_checks' and not camera:
                raise ValueError('Passing output has no camera record')
            if camera:
                if report['status'] != 'candidate_passed_checks':
                    raise ValueError('A failed calibration cannot publish a camera record')
                cp = plain(self.root, camera['path'])
                if not cp.is_relative_to(self.result) or sha(cp) != camera['sha256']:
                    raise ValueError('Camera record identity mismatch')
                if cp.stat().st_size > 2*1024*1024:
                    raise ValueError('Camera record exceeds display bound')
                camera_value = json.loads(cp.read_text())
                validate_camera(camera_value, report, self.request, root=self.root, result=self.result)
                self.state['camera_record_url'] = '/files/' + cp.relative_to(self.root).as_posix()
            if self.cancelled.is_set() or self.timed_out.is_set():
                self.state.update(status='cancelled' if self.cancelled.is_set() else 'timed_out', phase='stopped',
                                  result_url=None, camera_record_url=None)
        except Exception as error:
            self.state.update(status='failed', phase='failed', error=str(error)[:500], result_url=None, camera_record_url=None)
        finally:
            if timer: timer.cancel()
            if lock: lock.close()
            self.state['finished_at'] = now(); self.save()
