#!/usr/bin/env python3
"""One private bounded loop adopting the completed, exact fresh Jabra run.

The unsuccessful original review/round remain immutable. This explicit lineage
uses the omission-only derivative without resending any original request. New
mechanics experiments still pass the normal production queue and budget gates.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main(*,local_parser_recovery_v2=False):
    from admin_workflow import AdminWorkflow
    from admin_pipeline import QueueRunner,state_result
    from admin_execution_policy import improvement_policy
    from admin_visual_audit import activation,prior_critic
    from object_program import ref
    from overnight_budget import budget_status
    from review_agents import verified_ref
    version=2 if local_parser_recovery_v2 else 1
    if local_parser_recovery_v2:
        failure=ROOT/'data/flow-runs/run-admin-repair-1b68159f67384a8b8e9c1844/pipeline.log'
        assert 'Known structured parser controlflow changed' in failure.read_text()
        assert (ROOT/'output/overnight-pipeline-20261102/visual-repair-validation-result-v1.json').is_file()
    run=f'run-overnight-jabra-visual-recovery-20261103-v{version}'
    out=ROOT/'output/overnight-pipeline-20261102'
    marker=out/f'visual-repair-validation-intent-v{version}.json'
    assert not marker.exists(),'Validation already requested; inspect preserved state instead of duplicating'
    assert activation(root=ROOT),'Reviewed V2 visual controller required'
    state=json.loads((ROOT/'data/flow-runs/run-overnight-jabra-20261103-v1/run.json').read_text())
    result=state_result(state,root=ROOT,repair_execution=[{
        'strategy':'source_bound_completed_pipeline_adoption_for_visual_contract_recovery',
        'base_run_id':state['id'],'original_attempt_preserved':True,
        'new_gpu_calls':0,'initial_paid_request_resent':False,
        'human_approval':False,'automatic_publication':False}])
    assert result['interaction_ref']['sha256']=='51c3aba2dbb07141643f9ffc87756383d5425cae868a4ccd1159c8fc9fcebf6b'
    source='data/flow-rigid-parts-jabra-20261031-v1/source/original.mp4'
    source_ref=ref(ROOT/source,ROOT)
    assert source_ref['sha256']=='0de6f8cb659d256ed53657dea6c796facbb108a645fbe0d9e753c38694969622'
    for reference in result['artifacts']:verified_ref(ROOT,reference)
    prior,unknown=prior_critic(dict(result,source_ref=source_ref),root=ROOT)
    assert prior and not unknown and prior[1]['role']=='critic'
    workflow=AdminWorkflow(ROOT)
    with workflow.db()as db:
        assert not db.execute("SELECT 1 FROM jobs WHERE state IN ('queued','running')").fetchone(),'Owner queue must be idle'
    assert budget_status(root=ROOT)['can_admit']
    with marker.open('x')as stream:
        json.dump({'run_id':run,'source_ref':source_ref,'adopted_result':result,
            'original_failed_review_preserved':True,'maximum_repairs':2,
            'authorization':'explicit user-approved overnight pipeline',
            'automatic_publication':False},stream,indent=2);stream.write('\n')
    submission=workflow.submit_upload(run_id=run,input_path=source,processing_consent=True,
        object_family='hinged_assemblies',improvement_policy=improvement_policy(ROOT))
    recipe=ROOT/'.private/admin-imports'/(run+'.json')
    with recipe.open('x')as stream:json.dump({'source_ref':source_ref,'result':result},stream,indent=2);stream.write('\n')
    recipe.chmod(0o600)
    (out/f'visual-repair-validation-submission-v{version}.json').write_text(json.dumps(submission,indent=2)+'\n')
    queue=QueueRunner(ROOT);queue.execute()
    candidate_id=submission['candidate']['id']
    completed={'queue':queue.state,'outcomes':queue.outcomes,'candidate_id':candidate_id,
        'candidate':workflow.detail(candidate_id),'budget':budget_status(root=ROOT),
        'original_failed_review_preserved':True,'automatic_publication':False}
    (out/f'visual-repair-validation-result-v{version}.json').write_text(json.dumps(completed,indent=2)+'\n')
    print(json.dumps({'queue':queue.state,'candidate_id':candidate_id,'jobs_finished':len(queue.outcomes),
        'budget':completed['budget']}),flush=True)

if __name__=='__main__':main()
