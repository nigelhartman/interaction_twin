#!/usr/bin/env python3
"""Explicit pending-GPU reconciliation for the frozen four-video batch.

No failed model or paid perception phase is retried. Existing pending source
handoffs are reconciled through the original idempotent bridge; ready models
are reused. The original GPU-first driver and its receipts stay unchanged.
"""
from __future__ import annotations
import argparse
import fcntl
import json
from pathlib import Path
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_capture_gpu_priority_20261103 as V1
P=V1.P
V1_SHA='cec0ae1e8a056567cec4fafa9fe4c17a00f6e7ffdec7b90a582b8f35052a9402'
EARLY=('prepare','discover','segment','audit','select')


class ReconciliationBlocked(BaseException):pass


class ReconcileRunner(P.Runner):
    def phase(self,name,dependencies,action,*,optional=False):
        if name in EARLY:
            prior=next((r for r in self.state['phases']if r.get('name')==name),None)
            identity=P.fingerprint({'profile':self.profile,'source':self.state.get('source'),
                'inputs':P.file_receipts(dependencies),'runner_code':P.LOADED_SOURCE_SHA256})
            if (not prior or prior.get('status')not in V1.SUCCESS or prior.get('signature')!=identity
                or not P.verify_receipts(prior.get('files'))):
                raise ReconciliationBlocked('Prior '+name+' cannot be reused exactly; no paid request was resent.')
        result=super().phase(name,dependencies,action,optional=optional)
        if name=='reconstruct':raise V1.GPUCheckpoint()
        return result


def pending_state(state):
    if state.get('status')!='paused_after_reconstruct'or not V1.verified_checkpoint(state):
        raise ValueError('Only a verified paused reconstruction can be reconciled')
    prior=next(r for r in state['phases']if r['name']=='reconstruct')
    result=prior['result'];rows=result.get('components',[])
    if not rows or result.get('failures')or any(r.get('status')not in V1.SUCCESS or r.get('output',{}).get('status')not in {'awaiting_gpu_worker','native_candidate_ready'}for r in rows):
        raise ValueError('A failed or unsupported component is not a pending-GPU reconciliation')
    pending=[r['track_id']for r in rows if r['output']['status']=='awaiting_gpu_worker']
    if pending!=result.get('pending_components',[]):raise ValueError('Exact pending component identity differs')
    return pending


def reconcile(row,manifest_ref,attempt_id,*,runner_class=ReconcileRunner):
    state_path=ROOT/'data/flow-runs'/row['run_id']/'run.json'
    if not state_path.is_file():return {'id':row['id'],'status':'blocked_missing_run','new_model_attempt':False}
    previous=P.read(state_path)
    if previous.get('status')in {'complete','complete_with_gaps'}and V1.verified_checkpoint(previous):
        return {'id':row['id'],'status':'already_finished','run':V1.reference(state_path),'gpu':V1.gpu_outcome(previous),'new_pipeline_execution':False}
    try:pending=pending_state(previous)
    except ValueError as error:return {'id':row['id'],'status':'blocked_existing_attempt','reason':str(error),'run':V1.reference(state_path),'new_pipeline_execution':False}
    if not pending:return {'id':row['id'],'status':'already_gpu_ready','run':V1.reference(state_path),'gpu':V1.gpu_outcome(previous),'new_pipeline_execution':False}
    folder=ROOT/'output/new-captures-20261103'
    snapshot=folder/(row['id']+'-reconcile-'+attempt_id+'-predecessor.json')
    with snapshot.open('xb')as stream:stream.write(state_path.read_bytes())
    started=time.monotonic();runner=runner_class(row['run_id']);blocked=None
    try:
        state=runner.execute(V1.PROFILE,input_path=ROOT/row['input_path'],publication_mode='admin_review',overnight_task_budget_plan=V1.BUDGET)
    except V1.GPUCheckpoint:
        state=runner.state
        state.update(status='paused_after_reconstruct'if V1.verified_checkpoint(state)else'gpu_checkpoint_requires_attention',phase=None,error=None)
        state['gpu_reconciliation']={'attempt_id':attempt_id,'manifest':manifest_ref,'predecessor':V1.reference(snapshot),
            'paid_perception_resends':0,'same_run_identity':True,'same_source_handoffs':True,
            'automatic_publication':False,'parent_allocation_modified':False}
        runner.save()
    except ReconciliationBlocked as error:
        blocked=str(error);state=runner.state
        state.update(status='paused_after_reconstruct',phase=None,error=None,reconciliation_blocked=blocked);runner.save()
    return {'id':row['id'],'run_id':row['run_id'],'status':'blocked_exact_reuse'if blocked else state['status'],
        'reason':blocked,'wall_seconds':time.monotonic()-started,'run':V1.reference(runner.path),
        'predecessor':V1.reference(snapshot),'gpu':V1.gpu_outcome(state),'new_pipeline_execution':True,
        'paid_perception_resends':0,'failed_model_retries':0}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=ROOT/V1.MANIFEST)
    parser.add_argument('--reconciliation-id',required=True,help='A fresh explicit local reconciliation label, e.g. reconnect1')
    args=parser.parse_args(argv)
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,31}',args.reconciliation_id):raise ValueError('Bounded reconciliation identity required')
    if P.sha(Path(V1.__file__))!=V1_SHA:raise ValueError('The original GPU-first driver changed')
    doc,manifest_ref=V1.load_manifest(args.manifest);folder=ROOT/'output/new-captures-20261103'
    code_ref=V1.reference(Path(__file__));outcomes=[]
    with (ROOT/'.state/capture-gpu-priority-20261103.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for row in doc['captures']:
            receipt=folder/(row['id']+'-reconcile-'+args.reconciliation_id+'-result.json')
            if receipt.exists():
                saved=P.read(receipt)
                if saved.get('manifest')!=manifest_ref or saved.get('attempt_id')!=args.reconciliation_id:raise ValueError('Saved reconciliation belongs to another request')
                outcome=saved['outcome']
            else:
                print(json.dumps({'status':'reconciling','capture_id':row['id']}),flush=True)
                P.write(folder/'reconciliation-progress.json',{'status':'running','attempt_id':args.reconciliation_id,'capture_id':row['id'],'manifest':manifest_ref,'outcomes':outcomes,'updated_utc':P.now()})
                outcome=reconcile(row,manifest_ref,args.reconciliation_id)
                V1.immutable(receipt,{'format':'interaction-twin-capture-pending-gpu-reconciliation','version':2,
                    'attempt_id':args.reconciliation_id,'manifest':manifest_ref,'driver':code_ref,
                    'original_driver_sha256':V1_SHA,'outcome':outcome,'completed_utc':P.now(),
                    'automatic_publication':False,'parent_allocation_modified':False})
            outcomes.append(outcome);print(json.dumps(outcome),flush=True)
        summary={'format':'interaction-twin-capture-pending-gpu-batch-result','version':2,'attempt_id':args.reconciliation_id,
            'manifest':manifest_ref,'driver':code_ref,'outcomes':outcomes,'completed_utc':P.now(),
            'automatic_publication':False,'parent_allocation_modified':False}
        final=folder/('batch-reconcile-'+args.reconciliation_id+'-result.json')
        if not final.exists():V1.immutable(final,summary)
        P.write(folder/'reconciliation-progress.json',summary)
    return 0 if all(o.get('status')in {'already_finished','already_gpu_ready','paused_after_reconstruct'}and o.get('gpu',{}).get('all_selected_native_models_ready')for o in outcomes)else 1


if __name__=='__main__':raise SystemExit(main())
