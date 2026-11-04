"""Source-bound SAM3D upload worker using the existing, optional GPU allocation.

An absent/unreachable worker is a resumable queue state, never a synthetic model
or a TripoSR fallback. One source handoff has one durable model-attempt identity,
even when a pipeline run is resumed under a different output directory.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import time

import initial_sam3d_handoff as H
import initial_sam3d_import as N
import upgrade_bundles as U
import sam3d_cluster_remote as R

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = 'config/sam3d-cluster.json'
REMOTE_HELPER = ROOT / 'sam3d_cluster_remote.py'
REMOTE_PYTHON = str(R.TASK / 'runtime/environment/bin/python')
CONFIG_FIELDS = {'version', 'enabled', 'ssh_control_path', 'maximum_parallel_workers', 'poll_seconds', 'wait_seconds'}
EVIDENCE = {'launch.json', 'supervisor-result.json', 'prepare.json', 'plan.json', 'command-spec.json',
            'execution.json', 'offline-preflight.json', 'offline-run.json', 'offline-manifest.json'}
NATIVE = {'metadata.json', 'object.glb', 'object-splat.ply', 'layout-and-pointmap.npz',
          'inputs/manifest.json', 'inputs/image-source.png', 'inputs/mask-source.png', 'inputs/rgb.png', 'inputs/mask.png'}


class WorkerUnavailable(RuntimeError):
    """No model failure is inferred from an unavailable SSH channel."""
    def __init__(self, message, diagnostic=None):
        super().__init__(message)
        self.diagnostic = diagnostic or {'phase': 'unknown', 'classification': 'worker_unavailable'}


def transport_diagnostic(phase, *, returncode=None, stderr=b'', timeout=False):
    """Classify only fixed error categories; never expose raw remote output."""
    text = stderr.decode('utf-8', errors='replace') if isinstance(stderr, bytes) else str(stderr)
    classification = 'response_unavailable'
    if timeout:
        classification = 'timeout'
    elif 'SyntaxError' in text:
        classification = 'remote_python_syntax_error'
    elif ('python3: command not found' in text or 'python3: not found' in text
          or (REMOTE_PYTHON in text and ('No such file' in text or 'not found' in text))):
        classification = 'remote_python_missing'
    elif 'Permission denied' in text:
        classification = 'ssh_authentication_unavailable'
    elif 'Control socket' in text or 'No such file or directory' in text or 'master is not running' in text:
        classification = 'control_socket_unavailable'
    elif 'Connection timed out' in text or 'Connection refused' in text or 'Connection closed' in text:
        classification = 'ssh_channel_unavailable'
    elif 'Traceback' in text:
        classification = 'remote_helper_exception'
    return {'phase': phase, 'classification': classification, 'returncode': returncode,
            'raw_remote_output_exposed': False}


def now():
    return datetime.now(timezone.utc).isoformat()


def confined(path, root, *, exists=False):
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    H.require(path.is_relative_to(root / 'data') and path.resolve().is_relative_to(root / 'data'), 'Worker output must be inside project data')
    H.require(not any(part.startswith('.') for part in path.relative_to(root / 'data').parts), 'Ordinary output directory required')
    for item in [path, *path.parents]:
        if item == root:
            break
        H.require(not item.is_symlink(), 'Worker path symlink refused')
    if exists:
        H.require(path.is_file(), 'Worker artifact unavailable')
    return path


def reference(path, root):
    return H.reference(path, root)


def save(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.new')
    with temporary.open('w') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')
    temporary.replace(path)


def configuration(root=ROOT):
    path = Path(root) / CONFIG_PATH
    if not path.exists():
        return None
    H.require(not path.parent.is_symlink(), 'Worker configuration directory symlink refused')
    H.require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 4096, 'Bounded worker configuration required')
    value = json.loads(path.read_text())
    H.require(isinstance(value, dict) and set(value) == CONFIG_FIELDS and value.get('version') == 1
              and type(value.get('enabled')) is bool, 'Worker configuration schema differs')
    H.require(type(value['maximum_parallel_workers']) is int and 1 <= value['maximum_parallel_workers'] <= 2,
              'At most two existing GPU workers are permitted')
    H.require(type(value['poll_seconds']) in (int, float) and 1 <= value['poll_seconds'] <= 30
              and type(value['wait_seconds']) in (int, float) and 0 <= value['wait_seconds'] <= 1860,
              'Bounded worker polling required')
    socket = value['ssh_control_path']
    H.require(isinstance(socket, str) and re.fullmatch(r'/private/tmp/[A-Za-z0-9_-]{1,100}/cm-%C', socket),
              'Existing task-scoped SSH control socket required')
    return value if value['enabled'] else None


class SSHTransport:
    """Only a live master connection is used. Never fall back to fresh login."""
    def __init__(self, config):
        self.socket = config['ssh_control_path']
        H.require(REMOTE_HELPER.is_file() and not REMOTE_HELPER.is_symlink(), 'Remote bridge source unavailable')
        self.helper = REMOTE_HELPER.read_bytes()
        self.helper_sha = hashlib.sha256(self.helper).hexdigest()
        self.contacted = False

    def request(self, payload):
        options = ['ssh', '-o', 'BatchMode=yes', '-o', 'NumberOfPasswordPrompts=0', '-o', 'ConnectionAttempts=1',
                   '-o', 'ConnectTimeout=8', '-o', 'ControlMaster=no', '-o', 'ProxyCommand=false', '-S', self.socket]
        # The master check ensures SSH cannot silently establish a new connection.
        try:
            check = subprocess.run(options + ['-O', 'check', 'KISSKI'], stdin=subprocess.DEVNULL,
                                   capture_output=True, timeout=12)
        except (subprocess.TimeoutExpired, OSError) as error:
            raise WorkerUnavailable('The local SSH master check did not respond.',
                transport_diagnostic('master_check', timeout=isinstance(error, subprocess.TimeoutExpired))) from error
        if check.returncode:
            raise WorkerUnavailable('The existing cluster SSH connection is unavailable.',
                transport_diagnostic('master_check', returncode=check.returncode, stderr=check.stderr))
        data = dict(payload)
        if data['action'] == 'dispatch':
            data.update(helper_base64=base64.b64encode(self.helper).decode(), helper_sha256=self.helper_sha)
        # Login nodes may expose an older system Python. The existing frozen
        # runtime has the supported interpreter used by the GPU job scripts.
        # This path is fixed code, never browser/config supplied or a fallback.
        command = shlex.join([REMOTE_PYTHON, '-I', '-c', self.helper.decode()])
        try:
            self.contacted = True
            p = subprocess.run(options + ['KISSKI', command], input=json.dumps(data).encode(), capture_output=True,
                               timeout=90 if data['action'] in {'dispatch', 'chunk'} else 45)
        except (subprocess.TimeoutExpired, OSError) as error:
            raise WorkerUnavailable('The GPU worker did not respond; a dispatched attempt is preserved.',
                transport_diagnostic('remote_helper', timeout=isinstance(error, subprocess.TimeoutExpired))) from error
        try:
            # Login banners may precede our single JSON response, but raw SSH output is never surfaced.
            lines = p.stdout.splitlines()
            result = json.loads(next(line for line in reversed(lines) if line.startswith(b'{')))
        except (ValueError, StopIteration) as error:
            raise WorkerUnavailable('The cluster helper did not return a usable response.',
                transport_diagnostic('remote_helper_response', returncode=p.returncode, stderr=p.stderr)) from error
        H.require(isinstance(result, dict), 'Worker JSON response required')
        if result.get('status') == 'worker_unavailable':
            raise WorkerUnavailable('The existing GPU allocation is not currently available.',
                {'phase': 'allocation_probe', 'classification': 'allocation_unavailable', 'raw_remote_output_exposed': False})
        if result.get('status') == 'bridge_rejected':
            raise ValueError('GPU worker rejected the source/runtime contract: ' + str(result.get('reason', 'unknown'))[:240])
        if p.returncode:
            raise WorkerUnavailable('The existing cluster channel failed; the queued attempt is preserved.',
                transport_diagnostic('remote_helper_response', returncode=p.returncode, stderr=p.stderr))
        return result


@contextmanager
def worker_slot(directory, maximum):
    directory.mkdir(parents=True, exist_ok=True)
    stream = None
    try:
        for index in range(maximum):
            candidate = (directory / ('slot-' + str(index) + '.lock')).open('a')
            try:
                fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                stream = candidate
                break
            except BlockingIOError:
                candidate.close()
        yield stream is not None
    finally:
        if stream is not None:
            fcntl.flock(stream, fcntl.LOCK_UN)
            stream.close()


def payloads(handoff):
    rows = []
    for name in sorted(H.FILES | {'handoff.json'}):
        path = handoff.parent / name
        raw = path.read_bytes()
        H.require(0 < len(raw) <= 16 * 2**20, 'Portable input file budget exceeded')
        rows.append({'path': name, 'sha256': U.digest(raw), 'bytes': len(raw), 'base64': base64.b64encode(raw).decode()})
    H.require(sum(row['bytes'] for row in rows) <= 64 * 2**20, 'Portable handoff total budget exceeded')
    return rows


def verify_execution(directory, identifier, handoff_sha):
    """Independent launcher/offline receipts complement the native adapter checks."""
    evidence = directory / 'returned/evidence'
    launch = U.read(evidence / 'launch.json')
    complete = U.read(evidence / 'supervisor-result.json')
    execution = U.read(evidence / 'execution.json')
    spec = U.read(evidence / 'command-spec.json')
    prepared = U.read(evidence / 'prepare.json')
    plan = U.read(evidence / 'plan.json')
    offline = U.read(evidence / 'offline-run.json')
    preflight = U.read(evidence / 'offline-preflight.json')
    H.require(launch.get('format') == 'sam3d-upload-bridge-launch' and launch.get('job_id') == identifier
              and launch.get('handoff_sha256') == handoff_sha and launch.get('maximum_model_commands') == 1
              and launch.get('automatic_retry') is False and launch.get('parent_allocation_modified') is False,
              'Returned launch identity/policy differs')
    H.require(launch.get('helper_sha256') == U.sha(REMOTE_HELPER), 'Returned supervisor source differs')
    expected = ['srun', '--jobid=16889584', '--job-name=object-reconstruction', '--exact', '--nodes=1', '--ntasks=1',
                '--gres=gpu:A100:1', '--cpus-per-task=16', '--mem=100G', '--time=00:30:00', '--immediate=60']
    H.require(launch.get('argv') == expected + ['bash', str(R.TASK / 'run_component_job_v1.sh'), identifier,
                                              R.BASE, R.READY, handoff_sha], 'Returned GPU resources/runtime differ')
    H.require(complete.get('job_id') == identifier and complete.get('returncode') == 0 and complete.get('automatic_retry') is False,
              'GPU supervisor did not complete')
    H.require(execution.get('format') == 'sam3d-component-offline-execution' and execution.get('job_id') == identifier
              and execution.get('status') == 'native_output_ready_for_independent_import_review'
              and execution.get('model_commands_attempted') == 1 and execution.get('model_inference_executed') is True
              and execution.get('parent_allocation_modified') is False and execution.get('automatic_retry') is False,
              'One successful GPU execution is required')
    H.require(execution.get('executor_sha256') == R.SCRIPTS['execute_component_job_v1.py']
              and execution.get('ready_sha256') == R.READY and spec.get('resources') == {
                  'existing_allocation': '16889584', 'nodes': 1, 'tasks': 1, 'gpus': 1, 'cpu_threads': 16,
                  'memory_gib': 100, 'time_seconds': 1800}, 'Returned executor/runtime/resources differ')
    H.require(execution.get('spec_sha256') == U.sha(evidence / 'command-spec.json')
              and execution.get('output_metadata_sha256') == U.sha(directory / 'returned/native/metadata.json')
              and spec.get('job_id') == identifier and spec.get('maximum_model_commands') == 1
              and spec.get('automatic_retry') is False and spec.get('manifest', {}).get('sha256') == U.sha(evidence / 'offline-manifest.json'),
              'Returned execution bytes differ')
    H.require(prepared.get('status') == 'prepared_offline_preflight_passed_not_submitted'
              and prepared.get('command_spec', {}).get('sha256') == U.sha(evidence / 'command-spec.json')
              and prepared.get('plan_sha256') == U.sha(evidence / 'plan.json')
              and plan.get('handoff', {}).get('sha256') == handoff_sha and plan.get('job_id') == identifier
              and plan.get('maximum_model_commands') == 1 and plan.get('automatic_retry') is False,
              'Returned source-bound preparation differs')
    H.require(offline.get('adapter_completed') is True and offline.get('input_integrity_unchanged') is True
              and offline.get('status') == 'adapter_completed_pending_output_review'
              and offline.get('wrapper_sha256') == R.SCRIPTS['offline_run.py']
              and offline.get('manifest_sha256') == spec['manifest']['sha256']
              and preflight.get('manifest_sha256') == spec['manifest']['sha256']
              and preflight.get('status') == 'local_manifest_valid_runtime_not_tested', 'Offline completion differs')
    H.require(len(execution.get('commands', [])) == 2 and all(row.get('returncode') == 0 for row in execution['commands']),
              'Preflight/model command completion required')
    return {'status': 'execution_and_source_contract_verified', 'job_id': identifier,
            'execution_sha256': U.sha(evidence / 'execution.json'), 'parent_allocation_modified': False,
            'model_commands_attempted': 1, 'automatic_retry': False, 'physical_accuracy_verified': False}


def fetch(transport, directory, identifier):
    response = transport.request({'action': 'inventory', 'job_id': identifier})
    rows = response.get('files')
    expected = {'native/' + name for name in NATIVE} | {'evidence/' + name for name in EVIDENCE}
    H.require(response.get('status') == 'native_output_ready' and isinstance(rows, list)
              and len(rows) == len(expected) and {row.get('path') for row in rows} == expected,
              'Exact native/evidence return inventory required')
    for row in rows:
        H.require(set(row) == {'path', 'bytes', 'sha256'} and type(row['bytes']) is int and 0 < row['bytes'] <= 256 * 2**20
                  and isinstance(row['sha256'], str) and re.fullmatch(r'[0-9a-f]{64}', row['sha256']), 'Return file budget/hash differs')
    H.require(sum(row['bytes'] for row in rows) <= 512 * 2**20, 'Native return total budget exceeded')
    returned = directory / 'returned'
    returned.mkdir(exist_ok=True)
    for row in rows:
        target = returned / row['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        H.require(not target.is_symlink(), 'Returned path symlink refused')
        if target.exists():
            H.require(target.stat().st_size == row['bytes'] and U.sha(target) == row['sha256'], 'Previously fetched file differs')
            continue
        temporary = target.with_name(target.name + '.partial')
        # A partial transfer can restart; this never repeats model inference.
        with temporary.open('wb') as stream:
            offset = 0
            while offset < row['bytes']:
                chunk = transport.request({'action': 'chunk', 'job_id': identifier, 'path': row['path'], 'offset': offset})
                H.require(chunk.get('path') == row['path'] and chunk.get('offset') == offset, 'Chunk source/offset differs')
                raw = base64.b64decode(chunk['base64'], validate=True)
                H.require(0 < len(raw) <= min(4 * 2**20, row['bytes'] - offset), 'Return chunk budget differs')
                stream.write(raw)
                offset += len(raw)
        H.require(temporary.stat().st_size == row['bytes'] and U.sha(temporary) == row['sha256'], 'Returned native bytes differ')
        temporary.replace(target)
    save(directory / 'return-inventory.json', {'files': rows, 'all_sha256_verified': True})


def result(record, receipt, root):
    response = {key: record[key] for key in ('status', 'message', 'job_id', 'handoff', 'has_gaps') if key in record}
    response['receipt'] = receipt.relative_to(root).as_posix()
    response.update(backend='sam3d', automatic_retry=False, cluster_contacted=record.get('cluster_contacted', False),
                    remote_job_submitted=record.get('remote_job_submitted', False), physical_accuracy_verified=False,
                    source_video_pose_validated=False, simulation_ready=False)
    if record['status'] == 'native_candidate_ready':
        candidate = receipt.parent / 'candidate'
        metadata = U.read(candidate / 'metadata.json')
        H.require(metadata.get('backend') == 'Meta SAM 3D Objects' and metadata.get('selected') is False
                  and metadata.get('simulation_ready') is False, 'Cached native candidate scope differs')
        imported = U.read(candidate / 'import.json')
        for name in ('metadata.json', 'asset.glb'):
            artifact = candidate / name
            row = next(row for row in imported['import_files'] if row['path'] == name)
            H.require(artifact.stat().st_size == row['bytes'] and U.sha(artifact) == row['sha256'], 'Cached native candidate bytes differ')
        response.update(model=(candidate / 'metadata.json').relative_to(root).as_posix(),
                        native_asset=(candidate / 'asset.glb').relative_to(root).as_posix())
    return response


def reconstruct(handoff, destination, *, cancel_event=None, root=ROOT, transport_factory=SSHTransport):
    """Dispatch/reconcile one immutable handoff; keep missing GPU work resumable.

    destination is recorded for lineage only. Actual returned/imported bytes live
    under data/sam3d-upload-jobs/<full handoff SHA>, independent of retrying a
    pipeline phase. Cancellation stops waiting locally and leaves GPU jobs alone.
    """
    root = Path(root).resolve()
    handoff = U._local(handoff, root)
    H.inspect(handoff, root=root)
    requested = confined(destination, root)
    bound = reference(handoff, root)
    identifier = 'upload-' + bound['sha256'][:40]
    directory = confined(root / 'data/sam3d-upload-jobs' / bound['sha256'], root)
    directory.mkdir(parents=True, exist_ok=True)
    receipt = directory / 'bridge.json'
    with (directory / 'reconcile.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {'status': 'awaiting_gpu_worker', 'message': 'This source component is already being processed.',
                    'handoff': bound, 'receipt': receipt.relative_to(root).as_posix(), 'has_gaps': True, 'backend': 'sam3d',
                    'cluster_contacted': False, 'remote_job_submitted': None}
        if receipt.exists():
            record = U.read(receipt)
            H.require(record.get('handoff') == bound and record.get('job_id') == identifier, 'Cached source/worker identity differs')
            if record['status'] in {'native_candidate_ready', 'failed_no_retry'}:
                response = result(record, receipt, root)
                response['cluster_contacted'] = False
                return response
        else:
            record = {'format': 'sam3d-upload-cluster-bridge', 'version': 1, 'job_id': identifier,
                      'handoff': bound, 'requested_destination': requested.relative_to(root).as_posix(), 'created_utc': now(),
                      'status': 'awaiting_gpu_worker', 'has_gaps': True, 'dispatch_requested': False,
                      'maximum_model_commands': 1, 'automatic_retry': False, 'parent_allocation_modified': False,
                      'old_poses_read': False, 'preferred_assets_changed': False, 'paid_api_calls': 0}
        record['cluster_contacted'] = False

        def update(status, message):
            record.update(status=status, message=message, updated_utc=now())
            save(receipt, record)
            return result(record, receipt, root)

        config = configuration(root)
        if config is None:
            return update('awaiting_gpu_worker', 'SAM 3D is queued; the GPU worker is not enabled. Tracking and descriptive output are retained.')
        if cancel_event is not None and cancel_event.is_set():
            return update('awaiting_gpu_worker', 'Local waiting stopped; any existing GPU attempt remains intact.')
        with worker_slot(root / '.state/sam3d-cluster', config['maximum_parallel_workers']) as acquired:
            if not acquired:
                return update('awaiting_gpu_worker', 'Both GPU worker slots are in use; this source component remains queued.')
            try:
                transport = transport_factory(config)
                available = transport.request({'action': 'probe'})
                record['cluster_contacted'] = True
                H.require(available.get('status') == 'ready' and available.get('parent_allocation_modified') is False,
                          'Verified worker readiness required')
                H.inspect(handoff, root=root)
                H.require(reference(handoff, root) == bound, 'Source handoff changed before dispatch')
                # Persist intent BEFORE network submission; ambiguous transport failures
                # resume the same id, whose remote launch is exclusive and idempotent.
                record.update(dispatch_requested=True)
                update('awaiting_gpu_worker', 'Dispatching or reconciling one source-bound GPU attempt.')
                remote = transport.request({'action': 'dispatch', 'job_id': identifier,
                                            'handoff_sha256': bound['sha256'], 'files': payloads(handoff)})
                if remote.get('status') in {'running', 'native_output_ready', 'failed_no_retry'}:
                    record['remote_job_submitted'] = True
                deadline = time.monotonic() + config['wait_seconds']
                while remote.get('status') in {'running', 'waiting_for_worker_slot'}:
                    record['remote_status'] = remote['status']
                    if remote['status'] == 'waiting_for_worker_slot':
                        return update('awaiting_gpu_worker', 'Both remote GPU worker slots are in use; no new model attempt started.')
                    if time.monotonic() >= deadline or (cancel_event is not None and cancel_event.is_set()):
                        return update('awaiting_gpu_worker', 'The source-bound GPU job is running; resume to import its result. The allocation remains open.')
                    if cancel_event is not None:
                        cancel_event.wait(config['poll_seconds'])
                    else:
                        time.sleep(config['poll_seconds'])
                    remote = transport.request({'action': 'poll', 'job_id': identifier})
                if remote.get('status') == 'failed_no_retry':
                    return update('failed_no_retry', 'This GPU attempt failed; evidence is preserved and it will not be rerun automatically.')
                H.require(remote.get('status') == 'native_output_ready', 'Unexpected GPU worker status')
                fetch(transport, directory, identifier)
                execution = verify_execution(directory, identifier, bound['sha256'])
                save(directory / 'execution-review.json', execution)
                H.inspect(handoff, root=root)
                candidate = directory / 'candidate'
                if not candidate.exists():
                    N.import_output(handoff, directory / 'returned/native', candidate, root=root)
                record['has_gaps'] = True
                return update('native_candidate_ready', 'SAM 3D geometry is ready. Fresh video alignment and physical assembly remain unverified.')
            except (WorkerUnavailable, subprocess.TimeoutExpired, OSError) as error:
                record['cluster_contacted'] = bool(record['cluster_contacted'] or getattr(locals().get('transport'), 'contacted', False))
                record['worker_diagnostic'] = getattr(error, 'diagnostic', {'phase': 'transport', 'classification': type(error).__name__})
                return update('awaiting_gpu_worker', 'The existing GPU connection is unavailable. The same source-bound job can resume without repeating inference.')
            except (ValueError, KeyError, TypeError, StopIteration) as error:
                record['error_type'] = type(error).__name__
                return update('failed_no_retry', 'Source, runtime or returned-result verification failed; no automatic model retry or fallback was performed.')
