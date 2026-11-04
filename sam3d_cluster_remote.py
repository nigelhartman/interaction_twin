"""Frozen, credential-free upload bridge for the already allocated SAM3D runtime.

Executed over an existing SSH master. This helper never authenticates, downloads,
installs software, modifies the allocation, or retries a model command.
"""
from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone

TASK = Path('/mnt/vast-kisski/home/nigel.hartman01/u29467/.local/state/ai-training/sam3d-20261101-v1')
PARENT = '16889584'
READY = '68d810635473d555a8c0175d6169d188d3b26079391070342b880600253c1ead'
BASE = '7e1d128da1b78bb79a7d8edee639f5bf4edafd32ee2132329236d3950600ce36'
SCRIPTS = {'prepare_component_offline_job_v1.py': '15dacd600865872a47c438861368969627c8f801c595154b4682662423061a97',
           'execute_component_job_v1.py': 'd9c0fc526b39fd44d4ebf41f0b3473cef275ab6c1b275b699a965fa0b5f5f901',
           'run_component_job_v1.sh': '900c037a1d319a3d975208a1c5b2241f0486392bc3287294896034c2709e8645',
           'offline_run.py': 'f6725eedd67bde4b1da03133b1dea5476addb5eb308873235c5e0ffa32fea74b'}
OUTPUTS = {'metadata.json', 'object.glb', 'object-splat.ply', 'layout-and-pointmap.npz',
           'inputs/manifest.json', 'inputs/image-source.png', 'inputs/mask-source.png', 'inputs/rgb.png', 'inputs/mask.png'}
HANDOFF = {'input.json', 'image.png', 'mask.png', 'provenance/selection.json', 'provenance/case-manifest.json',
           'provenance/frame-index.json', 'provenance/source-frame.jpg', 'provenance/source-mask.png',
           'adapter/stage3_sam3d.py', 'implementation.py', 'source-binding.json', 'run-plan.json', 'README.txt', 'handoff.json'}
MAX_FILE = 256 * 2**20
MAX_TOTAL = 512 * 2**20


class AllocationUnavailable(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 4 * 2**20, 'BoundedRegularJSONRequired')
    value = json.loads(path.read_text())
    require(isinstance(value, dict), 'JSONObjectRequired')
    return value


def write(path, record):
    temporary = path.with_suffix(path.suffix + '.new')
    with temporary.open('w') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')
    temporary.replace(path)


def job_id(value):
    require(isinstance(value, str) and re.fullmatch(r'upload-[a-z0-9-]{1,64}', value), 'UploadJobIdRequired')
    return value


def parent():
    try:
        p = subprocess.run(['scontrol', 'show', 'job', PARENT, '-o'], capture_output=True, text=True, timeout=15, check=True)
    except (OSError, subprocess.SubprocessError) as error:
        raise AllocationUnavailable('The existing GPU allocation cannot currently be inspected.') from error
    fields = dict(part.split('=', 1) for part in p.stdout.split() if '=' in part)
    selected = {key: fields.get(key) for key in ('JobId', 'JobState', 'NodeList', 'EndTime', 'AllocTRES')}
    if not (selected['JobId'] == PARENT and selected['JobState'] == 'RUNNING' and selected['NodeList'] == 'ggpu182'
            and 'gres/gpu=2' in (selected['AllocTRES'] or '')):
        raise AllocationUnavailable('The existing two-GPU allocation is not running and available.')
    return selected


def probe():
    require(TASK.is_dir() and TASK.resolve() == TASK and not any(p.is_symlink() for p in [TASK, *TASK.parents]), 'OrdinaryTaskRootRequired')
    before = parent()
    for name, digest in {**SCRIPTS, 'offline-manifest-v5.json': BASE, 'runtime/READY-v8.json': READY}.items():
        p = TASK / name
        require(p.is_file() and not p.is_symlink() and sha(p) == digest, 'FrozenRuntimeDiffers:' + name)
    ready = read(TASK / 'runtime/READY-v8.json')
    require(ready['status'] == 'ready_for_one_offline_model_attempt' and ready['pip_check_passed'] is True, 'ActualRuntimeReadinessRequired')
    return {'status': 'ready', 'parent': before, 'runtime_ready_sha256': READY, 'maximum_parallel_workers': 2,
            'parent_allocation_modified': False, 'automatic_retry': False}


def folder(identifier):
    return TASK / 'upload-bridge-jobs' / job_id(identifier)


def poll(identifier):
    directory = folder(identifier)
    launch = read(directory / 'launch.json')
    require(launch['job_id'] == identifier, 'LaunchIdentityDiffers')
    complete = directory / 'supervisor-result.json'
    execution = TASK / 'component-jobs' / identifier / 'execution.json'
    if complete.exists():
        result = read(complete)
        if result['returncode'] != 0:
            return {'status': 'failed_no_retry', 'job_id': identifier, 'model_attempted': execution.exists(),
                    'reason': 'GPU job failed; evidence preserved and no model retry scheduled.'}
        row = read(execution)
        require(row.get('status') == 'native_output_ready_for_independent_import_review' and row.get('model_commands_attempted') == 1
                and row.get('model_inference_executed') is True and row.get('automatic_retry') is False, 'CompletedExecutionContractDiffers')
        return {'status': 'native_output_ready', 'job_id': identifier}
    return {'status': 'running', 'job_id': identifier, 'model_attempted': execution.exists()}


def dispatch(request):
    verified = probe()
    identifier = job_id(request['job_id'])
    directory = folder(identifier)
    bridge_root = directory.parent
    bridge_root.mkdir(mode=0o700, exist_ok=True)
    with (bridge_root / 'dispatch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if directory.exists():
            launch = read(directory / 'launch.json')
            require(launch['handoff_sha256'] == request['handoff_sha256'] and launch['helper_sha256'] == request['helper_sha256'], 'ExistingDispatchBindingDiffers')
            return poll(identifier)
        occupied = 0
        for sibling in bridge_root.iterdir():
            if sibling.is_dir() and not (sibling / 'supervisor-result.json').exists():
                occupied += 1
        if occupied >= 2:
            return {'status': 'waiting_for_worker_slot', 'maximum_parallel_workers': 2}
        incoming = request['files']
        require(isinstance(incoming, list) and len(incoming) == len(HANDOFF) and {r['path'] for r in incoming} == HANDOFF, 'ExactHandoffInventoryRequired')
        payloads = {}
        for row in incoming:
            require(set(row) == {'path', 'sha256', 'bytes', 'base64'}, 'PayloadSchemaDiffers')
            require(type(row['bytes']) is int and 0 < row['bytes'] <= 16 * 2**20, 'HandoffFileBudgetExceeded')
            raw = base64.b64decode(row['base64'], validate=True)
            require(len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256'], 'HandoffPayloadDiffers')
            payloads[row['path']] = raw
        require(sum(map(len, payloads.values())) <= 64 * 2**20, 'HandoffTotalBudgetExceeded')
        require(hashlib.sha256(payloads['handoff.json']).hexdigest() == request['handoff_sha256'], 'HandoffBindingDiffers')
        staged = TASK / 'inputs/initial' / identifier
        require(not staged.exists() and not (TASK / 'component-jobs' / identifier).exists()
                and not (TASK / 'outputs' / identifier).exists(), 'FreshRemoteJobPathsRequired')
        directory.mkdir(mode=0o700)
        staged.mkdir(parents=True, mode=0o700)
        for name, raw in payloads.items():
            path = staged / name
            path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
            with path.open('xb') as stream:
                stream.write(raw)
        # Only the reviewed bytes supplied as our own immutable helper may supervise.
        helper = base64.b64decode(request['helper_base64'], validate=True)
        require(hashlib.sha256(helper).hexdigest() == request['helper_sha256'], 'SupervisorHelperDiffers')
        helper_path = directory / 'supervisor.py'
        helper_path.write_bytes(helper)
        argv = ['srun', '--jobid=' + PARENT, '--job-name=object-reconstruction', '--exact', '--nodes=1', '--ntasks=1',
                '--gres=gpu:A100:1', '--cpus-per-task=16', '--mem=100G', '--time=00:30:00', '--immediate=60',
                'bash', str(TASK / 'run_component_job_v1.sh'), identifier, BASE, READY, request['handoff_sha256']]
        launch = {'format': 'sam3d-upload-bridge-launch', 'version': 1, 'job_id': identifier, 'argv': argv,
                  'handoff_sha256': request['handoff_sha256'], 'helper_sha256': request['helper_sha256'], 'created_utc': now(),
                  'parent_before': verified['parent'], 'parent_allocation_modified': False, 'maximum_model_commands': 1,
                  'automatic_retry': False, 'credentials_read': False, 'paid_api_calls': 0}
        write(directory / 'launch.json', launch)
        with (directory / 'supervisor.log').open('xb') as log:
            process = subprocess.Popen([sys.executable, '-I', str(helper_path), 'supervise', identifier],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        launch['supervisor_pid'] = process.pid
        write(directory / 'launch.json', launch)
        after = parent()
        require(after == verified['parent'], 'ParentAllocationChangedDuringDispatch')
        return {'status': 'running', 'job_id': identifier, 'maximum_parallel_workers': 2}


def supervise(identifier):
    directory = folder(identifier)
    launch = read(directory / 'launch.json')
    require(sha(Path(__file__)) == launch['helper_sha256'], 'SupervisorBytesChanged')
    result = {'job_id': identifier, 'started_utc': now(), 'returncode': -1, 'automatic_retry': False}
    try:
        with (directory / 'srun.log').open('xb') as log:
            p = subprocess.run(launch['argv'], stdout=log, stderr=subprocess.STDOUT, timeout=1860)
        result['returncode'] = p.returncode
    except BaseException as error:
        result['error_type'] = type(error).__name__
    finally:
        result['finished_utc'] = now()
        write(directory / 'supervisor-result.json', result)


def inventory(identifier):
    require(poll(identifier)['status'] == 'native_output_ready', 'NativeOutputNotReady')
    directory = folder(identifier)
    output = TASK / 'outputs' / identifier
    files = {}
    for p in output.rglob('*'):
        require(not p.is_symlink(), 'OutputSymlinkRefused')
        if p.is_file():
            name = p.relative_to(output).as_posix()
            require(name in OUTPUTS, 'UnexpectedAdapterOutput')
            files['native/' + name] = p
    require(set(files) == {'native/' + name for name in OUTPUTS}, 'IncompleteAdapterOutput')
    for name in ['launch.json', 'supervisor-result.json']:
        files['evidence/' + name] = directory / name
    for name in ['prepare.json', 'plan.json', 'command-spec.json', 'execution.json', 'offline-preflight.json', 'offline-run.json']:
        files['evidence/' + name] = TASK / 'component-jobs' / identifier / name
    files['evidence/offline-manifest.json'] = TASK / ('offline-job-' + identifier + '.json')
    rows = []
    for name, p in sorted(files.items()):
        require(p.is_file() and not p.is_symlink() and 0 < p.stat().st_size <= MAX_FILE, 'ReturnedFileBudgetExceeded')
        rows.append({'path': name, 'bytes': p.stat().st_size, 'sha256': sha(p)})
    require(sum(r['bytes'] for r in rows) <= MAX_TOTAL, 'ReturnedTotalBudgetExceeded')
    return files, rows


def serve(request):
    action = request['action']
    if action == 'probe':
        return probe()
    if action == 'dispatch':
        return dispatch(request)
    if action == 'poll':
        return poll(job_id(request['job_id']))
    files, rows = inventory(job_id(request['job_id']))
    if action == 'inventory':
        return {'status': 'native_output_ready', 'files': rows}
    require(action == 'chunk', 'UnknownBridgeAction')
    name, offset = request['path'], request['offset']
    require(name in files and type(offset) is int and 0 <= offset < files[name].stat().st_size, 'BoundedOutputChunkRequired')
    with files[name].open('rb') as stream:
        stream.seek(offset)
        raw = stream.read(4 * 2**20)
    return {'path': name, 'offset': offset, 'base64': base64.b64encode(raw).decode()}


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == 'supervise':
        supervise(job_id(sys.argv[2]))
    else:
        try:
            raw = sys.stdin.buffer.read(96 * 2**20 + 1)
            require(len(raw) <= 96 * 2**20, 'RequestBudgetExceeded')
            result = serve(json.loads(raw))
            print(json.dumps(result, allow_nan=False))
        except AllocationUnavailable as error:
            print(json.dumps({'status': 'worker_unavailable', 'reason': str(error)}))
        except Exception as error:
            print(json.dumps({'status': 'bridge_rejected', 'error_type': type(error).__name__, 'reason': str(error)[:240]}))
            sys.exit(1)
