#!/usr/bin/env python3
"""One explicitly authorized private Jabra upload through the production queue.

Wait for the existing bridge's two local transfer/inference slots before any
new paid work. Never renew, cancel or replace the parent GPU allocation. This
script has no approval/publication operation and never reads credentials.
"""
from __future__ import annotations
import fcntl
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def slots_idle():
    streams = []
    try:
        for index in range(2):
            stream = (ROOT / f'.state/sam3d-cluster/slot-{index}.lock').open('a')
            streams.append(stream)
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
        return True
    finally:
        for stream in streams:
            stream.close()


def main():
    from admin_pipeline import QueueRunner
    from admin_workflow import AdminWorkflow
    from admin_execution_policy import improvement_policy
    from admin_visual_audit import activation
    from overnight_budget import budget_status
    out = ROOT / 'output/overnight-pipeline-20261102'
    marker = out / 'fresh-jabra-validation-intent.json'
    if marker.exists():
        raise ValueError('This validation was already requested; inspect its durable job, do not submit a duplicate')
    assert activation(root=ROOT), 'Root-reviewed visual controller activation required'
    assert budget_status(root=ROOT)['can_admit'], 'Authorized pass budget unavailable'
    workflow = AdminWorkflow(ROOT)
    with workflow.db() as db:
        assert not db.execute("SELECT 1 FROM jobs WHERE state IN ('queued','running')").fetchone(), 'Existing owner queue needs inspection'
    source = 'data/flow-rigid-parts-jabra-20261031-v1/source/original.mp4'
    with marker.open('x') as stream:
        json.dump({'source': source, 'authorization': 'explicit overnight user approval',
                   'run_id': 'run-overnight-jabra-20261103-v1',
                   'automatic_publication': False, 'parent_gpu_allocation_modified': False}, stream, indent=2)
        stream.write('\n')
    started = time.monotonic()
    print('Waiting for existing GPU output transfers to release their two bridge slots.', flush=True)
    while not slots_idle():
        if time.monotonic() - started > 5400:
            raise TimeoutError('Existing GPU transfers still active; no new paid request sent')
        time.sleep(5)
    print('Existing bridge slots are free; starting the real private production upload.', flush=True)
    submission = workflow.submit_upload(run_id='run-overnight-jabra-20261103-v1', input_path=source,
        processing_consent=True, object_family='hinged_assemblies', improvement_policy=improvement_policy(ROOT))
    (out / 'fresh-jabra-validation-submission.json').write_text(json.dumps(submission, indent=2) + '\n')
    queue = QueueRunner(ROOT)
    queue.execute()
    result = {'queue': queue.state, 'outcomes': queue.outcomes, 'candidate_id': submission['candidate']['id'],
              'budget': budget_status(root=ROOT), 'automatic_publication': False,
              'parent_gpu_allocation_modified': False}
    (out / 'fresh-jabra-validation-result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'queue': queue.state, 'candidate_id': result['candidate_id'],
                      'jobs_finished': len(queue.outcomes), 'budget': result['budget']}), flush=True)
    return 0 if queue.state['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
