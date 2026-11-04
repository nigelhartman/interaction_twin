#!/usr/bin/env python3
"""Resume only the timed-out local workspace derivative for the fixed laptop run.

Never calls Runner.execute, _infer, a paid stage, SSH or a model. Existing failed
topology and unavailable alignment remain unchanged; genuine local completion
therefore remains complete_with_gaps. Original run/timeout evidence is preserved.
"""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_capture_gpu_priority_20261103 as V1
P = V1.P
RUN_ID = 'run-capture-laptop-20261103-v1'
CASE = 'flow-capture-laptop-20261103-v1'
OUTPUT = ROOT / 'output/new-captures-20261103/laptop-workspace-local-v1'
LOCAL_TIMEOUT_S = 600


def workspace_dependencies(state):
    return [ROOT / v for k, v in state['artifacts'].items()
            if k in {'interaction', 'viewer', 'constrained_scene', 'object_program'}] + [ROOT / 'scripts/build_workspace.py']


def verify_predecessor(state, row):
    """Require the identical recorded request and every prior phase unchanged."""
    expected = {'profile': V1.PROFILE, 'input_path': str(ROOT / row['input_path']),
                'case_id': None, 'overnight_task_budget_plan': V1.BUDGET, 'publication_mode': 'admin_review'}
    if (state.get('id') != RUN_ID or state.get('request') != expected
            or state.get('source', {}).get('sha256') != row['source_ref']['sha256']
            or state.get('source', {}).get('bytes') != row['source_ref']['bytes']
            or state.get('profile', {}).get('publication_mode') != 'admin_review'):
        raise ValueError('Exact fixed private laptop request/source required')
    if state.get('status') != 'failed' or state.get('phase') != 'workspace':
        raise ValueError('Only the preserved terminal workspace timeout may resume')
    phases = state.get('phases', [])
    if [p.get('name') for p in phases] != P.PHASES:
        raise ValueError('All preceding pipeline phases must already have durable outcomes')
    for phase in phases:
        inputs = phase.get('inputs')
        if not P.verify_receipts(inputs):
            raise ValueError('Prior phase inputs changed: ' + phase['name'])
        identity = P.fingerprint({'profile': state['profile'], 'source': state['source'],
                                  'inputs': inputs, 'runner_code': P.LOADED_SOURCE_SHA256})
        if phase.get('signature') != identity:
            raise ValueError('Prior phase code/profile signature changed: ' + phase['name'])
        if phase['name'] == 'workspace':
            if phase.get('status') != 'failed' or phase.get('error') != 'Phase command exceeded 90 seconds':
                raise ValueError('Known local library timeout required')
            if P.file_receipts(workspace_dependencies(state)) != inputs:
                raise ValueError('Workspace dependency basis changed')
        elif phase.get('status') in V1.SUCCESS:
            if not P.verify_receipts(phase.get('files')):
                raise ValueError('Prior phase artifacts changed: ' + phase['name'])
        elif phase.get('status') != 'failed':
            raise ValueError('Running/cancelled/ambiguous prior phase cannot resume')
    recon = next(p for p in phases if p['name'] == 'reconstruct')
    if recon.get('result', {}).get('pending_components'):
        raise ValueError('Pending GPU work cannot be touched by a CPU-only finisher')


class LocalWorkspaceRunner(P.Runner):
    def command(self, args, timeout):
        values = [str(v) for v in args]
        allowed = [
            [str(ROOT / '.venv-sam/bin/python'), str(ROOT / 'scripts/build_workspace.py'), '--no-replays'],
            [str(ROOT / '.venv-sam/bin/python'), str(ROOT / 'scripts/build_robotics_library.py')],
        ]
        if values not in allowed or timeout != LOCAL_TIMEOUT_S:
            raise ValueError('Only the two fixed local workspace builders are permitted')
        return super().command(values, timeout)

    def execute(self, *args, **kwargs):
        raise RuntimeError('Full pipeline execution is forbidden by this CPU-only finisher')

    def _infer(self, *args, **kwargs):
        raise RuntimeError('Inference is forbidden by this CPU-only finisher')


def resume():
    doc, manifest_ref = V1.load_manifest(ROOT / V1.MANIFEST)
    row = next(r for r in doc['captures'] if r['id'] == 'laptop')
    state_path = ROOT / 'data/flow-runs' / RUN_ID / 'run.json'
    OUTPUT.mkdir(exist_ok=True)
    final = OUTPUT / 'result.json'
    if final.exists():
        saved = P.read(final)
        if saved.get('manifest') != manifest_ref:
            raise ValueError('Preserved result belongs to another approved manifest')
        return saved
    with (ROOT / '.state/capture-gpu-priority-20261103.lock').open('a') as batch_lock:
        fcntl.flock(batch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (ROOT / 'data/flow-runs/.model-worker.lock').open('a') as worker_lock:
            fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            previous = P.read(state_path)
            verify_predecessor(previous, row)
            with (OUTPUT / 'predecessor-run.json').open('xb') as stream:
                stream.write(state_path.read_bytes())
            implementation = V1.reference(Path(__file__).resolve())
            intent = {'format': 'interaction-twin-fixed-local-workspace-resume', 'version': 1,
                      'manifest': manifest_ref, 'predecessor': V1.reference(OUTPUT / 'predecessor-run.json'),
                      'implementation': implementation, 'created_utc': P.now(),
                      'commands': ['build_workspace.py --no-replays', 'build_robotics_library.py'],
                      'maximum_seconds_per_command': LOCAL_TIMEOUT_S, 'paid_or_gpu_dispatch_permitted': False}
            V1.immutable(OUTPUT / 'intent.json', intent)
            runner = LocalWorkspaceRunner(RUN_ID)
            runner.profile = previous['profile']
            protected = P.fingerprint(previous['phases'][:-1])
            started = time.monotonic()

            def build(attempt):
                python = ROOT / '.venv-sam/bin/python'
                runner.command([python, ROOT / 'scripts/build_workspace.py', '--no-replays'], LOCAL_TIMEOUT_S)
                runner.command([python, ROOT / 'scripts/build_robotics_library.py'], LOCAL_TIMEOUT_S)
                report = runner.directory / f'workspace-artifact-{attempt}.json'
                P.write(report, {'case_id': CASE,
                    'source_sha256': P.read(ROOT / 'data' / CASE / 'manifest.json')['source']['sha256'],
                    'artifacts': runner.state['artifacts'], 'run_id': RUN_ID,
                    'physical_accuracy_verified': False, 'hidden_motion_or_seating_generated': False,
                    'cpu_only_resume': V1.reference(OUTPUT / 'intent.json')})
                return {'manifest': P.rel(report)}, [report, ROOT / 'workspace/cases.js', ROOT / 'workspace/cases.json']

            try:
                runner.phase('workspace', workspace_dependencies(previous), build)
                if P.fingerprint(runner.state['phases'][:-1]) != protected:
                    raise ValueError('Protected earlier phase outcomes unexpectedly changed')
                gaps = runner.state.get('native_alignment_has_gaps') or runner.state.get('observed_geometry_has_gaps') or any(
                    p['status'] in {'failed', 'complete_with_gaps', 'reused_verified_with_gaps'} for p in runner.state['phases'])
                runner.state.update(status='complete_with_gaps' if gaps else 'complete', phase=None,
                                    progress=1, error=None, completed_at=P.now())
            except Exception as error:
                runner.state.update(status='failed', phase='workspace', error=str(error), completed_at=P.now())
            finally:
                runner._terminate()
                runner.save()
            result = {'format': 'interaction-twin-fixed-local-workspace-result', 'version': 1,
                      'manifest': manifest_ref, 'implementation': implementation,
                      'run': V1.reference(state_path), 'status': runner.state['status'],
                      'error': runner.state.get('error'), 'wall_seconds': time.monotonic() - started,
                      'earlier_phase_outcomes_unchanged': P.fingerprint(runner.state['phases'][:-1]) == protected,
                      'new_api_calls': 0, 'new_gpu_calls': 0, 'automatic_publication': False,
                      'completed_utc': P.now()}
            V1.immutable(final, result)
            return result


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    result = resume()
    print(json.dumps(result), flush=True)
    return 0 if result['status'] in {'complete', 'complete_with_gaps'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
