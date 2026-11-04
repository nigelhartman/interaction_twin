#!/usr/bin/env python3
"""Retrieve only two already-submitted bottle jobs, even after allocation expiry.

``prepare`` is entirely local. ``fetch`` needs a working, pre-authenticated SSH
master, but NEVER probes the GPU allocation, dispatches, retries inference or
changes original bridge/run/selection records. Downloaded bytes have their own
source-bound recovery receipt; an optional native import keeps existing caps.
"""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sam3d_cluster_bridge as B
import initial_sam3d_handoff as H
import initial_sam3d_import as N
import upgrade_bundles as U

FORMAT = 'interaction-twin-submitted-bottle-fetch-only'
INTENT_PATH = 'output/new-captures-20261103/bottle-fetch-only-v1/intent.json'
RUN_ID = 'run-capture-bottle-consistency-20261103-v2'
CASE = 'flow-capture-bottle-consistency-20261103-v2'
ORIGINAL_SHA = '8e2ea4d25f09bd85c4ee6bfcb2c8cbfbeaf059c9e13410689896178a98792978'
NORMALIZED_SHA = '5cbb4911529582aaf6875911cea249f43918ba8a3a84d37e3d7550c2ede3579d'
JOBS = {
    'bottle_early': '68a622c2258bc0d7749b6a7fe664714fb47cdfb8156de2136f353b7bf967a023',
    'glass_1': 'c192e312cf3123424a617d4cf0e28cc3422612c6920f049787da5239b504e8ec',
}
CODE = ['scripts/fetch_submitted_bottle_outputs_20261103_v1.py', 'sam3d_cluster_bridge.py',
        'sam3d_cluster_remote.py', 'initial_sam3d_handoff.py', 'initial_sam3d_import.py',
        'sam3d_milestone.py', 'upgrade_bundles.py']


def local(root, relative, *, exists=True):
    if not isinstance(relative, str) or not relative.startswith(('data/', 'output/', 'scripts/')):
        raise ValueError('Declared ordinary recovery path required')
    parts = Path(relative).parts
    if any(p.startswith('.') or '\\' in p for p in parts):
        raise ValueError('Hidden/traversing recovery path refused')
    path = root
    for part in parts:
        path = path / part
        if path.is_symlink():
            raise ValueError('Recovery symlinks refused')
    if exists and not path.is_file():
        raise ValueError('Recovery evidence missing')
    return path


def ref(path, root):
    return {'path': path.relative_to(root).as_posix(), 'sha256': U.sha(path), 'bytes': path.stat().st_size}


def checked(value, root):
    if not isinstance(value, dict) or set(value) != {'path', 'sha256', 'bytes'}:
        raise ValueError('Exact file reference required')
    path = local(root, value['path'])
    if ref(path, root) != value:
        raise ValueError('Frozen recovery evidence changed')
    return path


def immutable(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def copy_once(source, destination):
    raw = source.read_bytes()
    if destination.exists():
        if destination.is_symlink() or destination.read_bytes() != raw:
            raise ValueError('Preserved preparation snapshot differs')
        return
    with destination.open('xb') as stream:
        stream.write(raw)


def code_refs(root):
    return [ref(root / name, root) for name in CODE]


def bridge_binding(record, component, digest, *, require_submitted):
    identifier = 'upload-' + digest[:40]
    if (record.get('format') != 'sam3d-upload-cluster-bridge' or record.get('version') != 1
            or record.get('job_id') != identifier or record.get('handoff', {}).get('sha256') != digest
            or record.get('maximum_model_commands') != 1 or record.get('automatic_retry') is not False
            or record.get('parent_allocation_modified') is not False
            or record.get('preferred_assets_changed') is not False
            or record.get('requested_destination') != f'data/{CASE}/flow-model-1-{1 if component == "bottle_early" else 2}'):
        raise ValueError('Exact previously submitted bottle job binding required')
    if require_submitted and (record.get('remote_job_submitted') is not True or record.get('dispatch_requested') is not True):
        raise ValueError('No retrieval for an unsubmitted job')
    return identifier


def prepare(*, root=ROOT):
    root = Path(root).resolve()
    target = local(root, INTENT_PATH, exists=False)
    if target.exists():
        return load_intent(root=root)
    run_path = local(root, f'data/flow-runs/{RUN_ID}/run.json')
    state = U.read(run_path)
    if state.get('id') != RUN_ID or state.get('source', {}).get('sha256') != ORIGINAL_SHA:
        raise ValueError('Corrected bottle source/run required')
    recon = next((p for p in state.get('phases', []) if p.get('name') == 'reconstruct'), None)
    if not recon or recon.get('status') not in {'complete_with_gaps', 'complete', 'reused_verified', 'reused_verified_with_gaps'}:
        raise ValueError('Durable reconstructed handoff phase required')
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for component, digest in JOBS.items():
        bridge_path = local(root, f'data/sam3d-upload-jobs/{digest}/bridge.json')
        record = U.read(bridge_path)
        identifier = bridge_binding(record, component, digest, require_submitted=True)
        handoff = checked(record['handoff'], root)
        H.inspect(handoff, root=root)
        binding = U.read(handoff.parent / 'source-binding.json')
        if (binding.get('component_id') != component or binding.get('source', {}).get('video_sha256') != NORMALIZED_SHA
                or binding.get('source', {}).get('source_case') != CASE):
            raise ValueError('Handoff component/source differs from the corrected bottle')
        selected = next((c for c in recon['result'].get('components', []) if c.get('track_id') == component), {})
        if selected.get('output', {}).get('handoff') != record['handoff'] or selected.get('output', {}).get('job_id') != identifier:
            raise ValueError('Run-to-submitted-handoff join differs')
        snapshot = target.parent / (component + '-bridge-before.json')
        copy_once(bridge_path, snapshot)
        rows.append({'component_id': component, 'job_id': identifier, 'handoff': record['handoff'],
                     'original_bridge': ref(bridge_path, root), 'submitted_bridge_snapshot': ref(snapshot, root),
                     'source_binding': ref(handoff.parent / 'source-binding.json', root),
                     'recovery_directory': 'data/sam3d-fetch-only-recoveries/' + digest})
    run_snapshot = target.parent / 'run-before.json'
    copy_once(run_path, run_snapshot)
    doc = {'format': FORMAT, 'version': 1, 'created_utc': B.now(), 'run_id': RUN_ID,
           'original_source_sha256': ORIGINAL_SHA, 'normalized_source_sha256': NORMALIZED_SHA,
           'run_snapshot': ref(run_snapshot, root), 'jobs': rows, 'implementation': code_refs(root),
           'allowed_remote_actions': ['poll', 'inventory', 'chunk'], 'maximum_new_model_commands': 0,
           'allocation_probe_permitted': False, 'dispatch_permitted': False, 'automatic_retry': False,
           'original_bridge_or_selection_mutation_permitted': False, 'physical_accuracy_verified': False}
    immutable(target, doc)
    return doc


def load_intent(*, root=ROOT):
    root = Path(root).resolve()
    doc = U.read(local(root, INTENT_PATH))
    if (doc.get('format') != FORMAT or doc.get('version') != 1 or doc.get('run_id') != RUN_ID
            or doc.get('original_source_sha256') != ORIGINAL_SHA or doc.get('normalized_source_sha256') != NORMALIZED_SHA
            or doc.get('maximum_new_model_commands') != 0 or doc.get('dispatch_permitted') is not False
            or doc.get('allocation_probe_permitted') is not False or doc.get('implementation') != code_refs(root)
            or doc.get('allowed_remote_actions') != ['poll', 'inventory', 'chunk']
            or len(doc.get('jobs', [])) != 2):
        raise ValueError('Fixed fetch-only recovery policy/code changed')
    checked(doc['run_snapshot'], root)
    if [j.get('component_id') for j in doc['jobs']] != list(JOBS):
        raise ValueError('Only the two already-submitted components may be retrieved')
    for row in doc['jobs']:
        digest = JOBS[row['component_id']]
        old = U.read(checked(row['submitted_bridge_snapshot'], root))
        if bridge_binding(old, row['component_id'], digest, require_submitted=True) != row['job_id']:
            raise ValueError('Frozen job identity changed')
        if row.get('handoff') != old.get('handoff') or row.get('recovery_directory') != 'data/sam3d-fetch-only-recoveries/' + digest:
            raise ValueError('Recovery destination/handoff changed')
        if row.get('original_bridge', {}).get('path') != f'data/sam3d-upload-jobs/{digest}/bridge.json':
            raise ValueError('Original bridge path changed')
        original = row['original_bridge']
        snapshot = row['submitted_bridge_snapshot']
        if any(original.get(k) != snapshot.get(k) for k in ['sha256', 'bytes']):
            raise ValueError('Original submitted bridge snapshot digest differs')
        handoff = checked(row['handoff'], root)
        H.inspect(handoff, root=root)
        if row.get('source_binding', {}).get('path') != (handoff.parent / 'source-binding.json').relative_to(root).as_posix():
            raise ValueError('Exact handoff source binding required')
        checked(row['source_binding'], root)
    return doc


class FetchOnlyTransport:
    """Enforce retrieval policy before a request can reach the SSH transport."""
    def __init__(self, transport, identifier):
        self.transport = transport
        self.identifier = identifier
        self.actions = []

    def request(self, payload):
        if not isinstance(payload, dict) or payload.get('job_id') != self.identifier:
            raise ValueError('Exact submitted job ID required')
        action = payload.get('action')
        expected_keys = {'action', 'job_id', 'path', 'offset'} if action == 'chunk' else {'action', 'job_id'}
        if action not in {'poll', 'inventory', 'chunk'} or set(payload) != expected_keys:
            raise ValueError('Only poll/inventory/chunk are permitted; no allocation probe or dispatch')
        if action == 'chunk':
            expected_paths = {'native/' + n for n in B.NATIVE} | {'evidence/' + n for n in B.EVIDENCE}
            if payload['path'] not in expected_paths or type(payload['offset']) is not int or not 0 <= payload['offset'] < 256 * 2**20:
                raise ValueError('Exact bounded artifact chunk required')
        self.actions.append(action)
        return self.transport.request(dict(payload))


def fetch_job(row, *, root=ROOT, transport_factory=B.SSHTransport, import_native=False):
    root = Path(root).resolve()
    digest = JOBS[row['component_id']]
    # Current original bridge may now say failed_no_retry. It is never edited:
    # the immutable submitted-before snapshot proves the old dispatch identity.
    bridge = local(root, row['original_bridge']['path'])
    current = U.read(bridge)
    bridge_binding(current, row['component_id'], digest, require_submitted=False)
    original_bytes = bridge.read_bytes()
    config = B.configuration(root)
    if config is None:
        raise ValueError('An enabled existing-SSH-master configuration is required for retrieval')
    transport = FetchOnlyTransport(transport_factory(config), row['job_id'])
    status = transport.request({'action': 'poll', 'job_id': row['job_id']})
    if status.get('job_id') != row['job_id'] or status.get('status') not in {'running', 'native_output_ready', 'failed_no_retry'}:
        raise ValueError('Exact remote job status required')
    result = {'component_id': row['component_id'], 'job_id': row['job_id'], 'remote_status': status['status'],
              'original_bridge_before': ref(bridge, root), 'new_model_commands': 0, 'new_gpu_dispatches': 0,
              'allocation_probed': False, 'original_bridge_modified': False, 'physical_accuracy_verified': False}
    if status['status'] == 'native_output_ready':
        directory = local(root, row['recovery_directory'], exists=False)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / 'fetch-only.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            B.fetch(transport, directory, row['job_id'])
            execution = B.verify_execution(directory, row['job_id'], digest)
            proof = directory / 'execution-review.json'
            if proof.exists():
                if U.read(proof) != execution:
                    raise ValueError('Preserved recovery execution verification differs')
            else:
                immutable(proof, execution)
            result.update(status='raw_native_and_execution_verified', directory=row['recovery_directory'],
                          inventory=ref(directory / 'return-inventory.json', root), execution_review=ref(proof, root))
            if import_native:
                candidate = directory / 'candidate'
                if candidate.exists():
                    raise ValueError('Preserved import candidate must not be overwritten')
                try:
                    N.import_output(checked(row['handoff'], root), directory / 'returned/native', candidate, root=root)
                    result.update(status='native_candidate_imported_unselected', model=ref(candidate / 'metadata.json', root),
                                  import_receipt=ref(candidate / 'import.json', root))
                except (ValueError, KeyError, TypeError) as error:
                    result.update(status='raw_verified_native_import_unavailable', import_error_type=type(error).__name__,
                                  import_error=str(error)[:240], raw_bytes_preserved=True, model_retry_performed=False)
    else:
        result['status'] = 'remote_still_running_no_dispatch' if status['status'] == 'running' else 'remote_failed_no_retry'
    if bridge.read_bytes() != original_bytes:
        raise ValueError('Original bridge changed concurrently; recovery cannot claim an unchanged snapshot')
    result['remote_actions'] = transport.actions
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'fetch'])
    parser.add_argument('--attempt-id', default='retrieve-1')
    parser.add_argument('--import-native', action='store_true')
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,31}', args.attempt_id):
        raise ValueError('Bounded explicit retrieval attempt ID required')
    if args.action == 'prepare':
        doc = prepare()
        print(json.dumps({'status': 'prepared_no_remote_commands', 'intent': ref(ROOT / INTENT_PATH, ROOT),
                          'jobs': [j['job_id'] for j in doc['jobs']]}))
        return 0
    doc = load_intent()
    output = (ROOT / INTENT_PATH).parent / (args.attempt_id + '.json')
    if output.exists():
        print(json.dumps(U.read(output)))
        return 0
    with output.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        results = []
        for row in doc['jobs']:
            try:
                results.append(fetch_job(row, import_native=args.import_native))
            except B.WorkerUnavailable as error:
                results.append({'component_id': row['component_id'], 'job_id': row['job_id'],
                                'status': 'existing_ssh_unavailable', 'diagnostic': error.diagnostic,
                                'new_model_commands': 0, 'new_gpu_dispatches': 0})
            except (ValueError, KeyError, TypeError, OSError) as error:
                results.append({'component_id': row['component_id'], 'job_id': row['job_id'],
                                'status': 'recovery_verification_unavailable', 'error_type': type(error).__name__,
                                'reason': str(error)[:240], 'raw_evidence_preserved': True,
                                'new_model_commands': 0, 'new_gpu_dispatches': 0})
        receipt = {'format': FORMAT + '-receipt', 'version': 1, 'intent': ref(ROOT / INTENT_PATH, ROOT),
                   'attempt_id': args.attempt_id, 'created_utc': B.now(), 'results': results,
                   'new_model_commands': 0, 'allocation_modified': False, 'automatic_publication': False}
        immutable(output, receipt)
        print(json.dumps(receipt), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
