#!/usr/bin/env python3
"""Explicit continuation of the one accounted invalid overnight discovery.

Preserve the failed initial request, make one bounded correction through the
reviewed recovery adapter, then resume the same private owner candidate. This
does not retry an uncertain request, invent boxes or approve a revision.
"""
from __future__ import annotations
import fcntl
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PLAN = '.private/discovery-repairs/overnight-jabra-contract-recovery-20261103-v1'
PLAN_SHA = '94ca2403ce501f632492dae6837f7aff33af1ea1a2a4c34b70183d1209f5985a'
RUN = 'run-overnight-jabra-20261103-v1'
CANDIDATE = 'candidate_33da0ff40be648348c40293969fb3ee2'


def main():
    import pipeline as P
    import discovery_repair
    from admin_pipeline import QueueRunner
    from admin_workflow import AdminWorkflow
    from overnight_budget import budget_status
    from admin_visual_audit import activation
    assert activation(root=ROOT), 'Reviewed visual automation required'
    _, plan, _ = discovery_repair.verified_plan(PLAN, PLAN_SHA, root=ROOT)
    case = ROOT / plan['context']['case']
    runner = P.Runner(RUN)
    assert runner.state['status'] == 'failed'
    assert [(p['name'], p['status']) for p in runner.state['phases']] == [('prepare', 'complete'), ('discover', 'failed')]
    runner.profile = runner.state['profile']
    out = ROOT / 'output/overnight-pipeline-20261102'
    with (out / 'fresh-jabra-before-accounted-discovery-repair.json').open('x') as stream:
        json.dump(runner.state, stream, indent=2); stream.write('\n')
    deps = [ROOT / PLAN / 'plan.json', ROOT / 'discovery_repair.py', ROOT / 'rigid_discovery.py',
            ROOT / 'scripts/discover_rigid_components.py', ROOT / 'schemas/discovery.schema.json',
            case / 'manifest.json', case / 'frames', case / 'reasoning/flow-discovery-1',
            ROOT / 'overnight_budget.py', ROOT / 'output/overnight-pipeline-20261102/task-budget-plan.json']
    def correction(attempt):
        assert attempt == 2, 'Only the explicit second discovery attempt is authorized'
        runner.command([ROOT / '.venv-sam/bin/python', ROOT / 'scripts/discover_rigid_components.py',
            'recover', '--case', case.name, '--run', 'flow-discovery-2', '--prepared-repair', PLAN,
            '--authorized-plan-sha256', PLAN_SHA], 900)
        directory = case / 'reasoning/flow-discovery-2'
        config = ROOT / P.read(directory / 'metadata.json')['tracking_config']
        return {'directory': P.rel(directory), 'discovery': P.rel(directory / 'discovery.json'),
                'annotations': P.rel(config), 'recovery_plan_ref': discovery_repair.ref(ROOT / PLAN / 'plan.json', ROOT),
                'initial_paid_request_resent': False}, [directory, config]
    with (ROOT / 'data/flow-runs/.model-worker.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runner.phase('discover', deps, correction)
    workflow = AdminWorkflow(ROOT)
    resumed = workflow.resume_initial(CANDIDATE, processing_consent=True)
    (out / 'fresh-jabra-contract-resume-submission.json').write_text(json.dumps(resumed, indent=2) + '\n')
    queue = QueueRunner(ROOT); queue.execute()
    result = {'queue': queue.state, 'outcomes': queue.outcomes, 'candidate_id': CANDIDATE,
              'budget': budget_status(root=ROOT), 'initial_paid_request_resent': False,
              'automatic_publication': False, 'parent_gpu_allocation_modified': False}
    (out / 'fresh-jabra-contract-resume-result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'queue': queue.state, 'candidate_id': CANDIDATE, 'jobs_finished': len(queue.outcomes),
                      'budget': result['budget']}), flush=True)


if __name__ == '__main__':
    main()
