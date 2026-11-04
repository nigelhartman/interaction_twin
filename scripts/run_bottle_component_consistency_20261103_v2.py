#!/usr/bin/env python3
"""Reviewed fresh bottle semantic revision; same bounds, private GPU checkpoint.

Only discovery is replaced with the versioned full-frame consistency role.
The earlier paid valid-but-mistaken proposal and all generated outputs remain.
Finish/reconcile modes retain the same custom discovery dependency signature.
"""
from __future__ import annotations
import argparse
import fcntl
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_capture_gpu_priority_20261103 as V1
from scripts.reconcile_capture_gpu_priority_20261103_v2 import ReconciliationBlocked,pending_state
P=V1.P
RUN_ID='run-capture-bottle-consistency-20261103-v2'
CASE='flow-capture-bottle-consistency-20261103-v2'
PRIOR_CASE='flow-capture-bottle-20261103-v1'
PRIOR_RUN='flow-discovery-1'
ADAPTER=ROOT/'scripts/stage2_component_consistency_20261103.py'
PRIOR=ROOT/'data'/PRIOR_CASE/'reasoning'/PRIOR_RUN


class SemanticRunner(P.Runner):
    def __init__(self,run_id,*,mode):super().__init__(run_id);self.semantic_mode=mode

    def command(self,args,timeout):
        values=[str(v)for v in args]
        prefix=[str(ROOT/'.venv-sam/bin/python'),str(ROOT/'scripts/run_overnight_stage.py'),
            '--task-budget-plan',V1.BUDGET,'discover_rigid_components.py','discover']
        if 'discover_rigid_components.py'in values:
            expected=prefix+['--case',CASE,'--run','flow-discovery-1','--timeout-s',str(self.profile['reasoning_timeout_s']),
                '--model',self.profile['reasoning_model'],'--max-output-tokens',str(self.profile['max_output_tokens'])]
            if values!=expected:raise ValueError('Only the reviewed fixed discovery command can be replaced')
            values=[prefix[0],str(ADAPTER),'discover',*values[6:],'--prior-case',PRIOR_CASE,'--prior-run',PRIOR_RUN,'--allow-paid-api']
        return super().command(values,timeout)

    def phase(self,name,dependencies,action,*,optional=False):
        if name=='discover':dependencies=[*dependencies,ADAPTER,PRIOR/'discovery.json',PRIOR/'meta-request.json',ROOT/'meta_localization.py']
        previous=next((r for r in self.state['phases']if r.get('name')==name),None)
        if self.semantic_mode!='gpu-first'and(name in {'prepare','discover','segment','audit','select'}or
            (name=='reconstruct'and self.semantic_mode=='finish')or(name in {'procedure_proposal','mechanism_topology'}and previous)):
            signature=P.fingerprint({'profile':self.profile,'source':self.state.get('source'),
                'inputs':P.file_receipts(dependencies),'runner_code':P.LOADED_SOURCE_SHA256})
            if (not previous or previous.get('status')not in V1.SUCCESS or previous.get('signature')!=signature
                or not P.verify_receipts(previous.get('files'))or(name=='reconstruct'and previous.get('result',{}).get('pending_components'))):
                raise ReconciliationBlocked('Semantic finish/reconcile requires exact cached '+name+'; no prior paid call/model was resent.')
        result=super().phase(name,dependencies,action,optional=optional)
        if name=='reconstruct'and self.semantic_mode!='finish':raise V1.GPUCheckpoint()
        return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=['gpu-first','reconcile-gpu','finish'],default='gpu-first')
    parser.add_argument('--attempt-id',default='initial');parser.add_argument('--allow-paid-api',action='store_true');a=parser.parse_args(argv)
    import re
    if not a.allow_paid_api or not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,31}',a.attempt_id):raise ValueError('Explicit bounded reviewed semantic action required')
    doc,manifest_ref=V1.load_manifest(ROOT/V1.MANIFEST);row=next(r for r in doc['captures']if r['id']=='bottle')
    folder=ROOT/'output/new-captures-20261103';run_path=ROOT/'data/flow-runs'/RUN_ID/'run.json'
    prior=P.read(PRIOR/'discovery.json');paid=P.read(PRIOR/'meta-request.json')
    if prior.get('status')!='complete_candidate_proposals'or paid.get('status')!='complete_accounted':raise ValueError('Preserved known-accounted prior proposal required')
    with(ROOT/'.state/capture-gpu-priority-20261103.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        receipt=folder/f'bottle-consistency-v2-{a.mode}-{a.attempt_id}.json'
        if receipt.exists():print(json.dumps(P.read(receipt)));return 0
        if a.mode=='gpu-first':
            if run_path.exists():raise ValueError('Existing semantic revision cannot be restarted under another attempt label')
            intent=folder/'bottle-consistency-v2-intent.json'
            V1.immutable(intent,{'format':'interaction-twin-full-frame-semantic-revision-intent','version':1,
                'manifest':manifest_ref,'run_id':RUN_ID,'prior_discovery':V1.reference(PRIOR/'discovery.json'),
                'prior_accounted_request':V1.reference(PRIOR/'meta-request.json'),'adapter':V1.reference(ADAPTER),
                'driver':V1.reference(Path(__file__)),'created_utc':P.now(),'maximum_semantic_requests':2,
                'manual_boxes_or_masks':False,'automatic_publication':False})
        else:
            previous=P.read(run_path)
            if previous.get('status')!='paused_after_reconstruct'or not V1.verified_checkpoint(previous):raise ValueError('Only a verified semantic GPU checkpoint may continue')
            pending=pending_state(previous)
            if a.mode=='finish'and pending:raise ValueError('Pending GPUs must reconcile explicitly before finish')
            if a.mode=='reconcile-gpu'and not pending:
                print(json.dumps({'status':'already_gpu_ready','run_id':RUN_ID,'gpu':V1.gpu_outcome(previous)}));return 0
            snapshot=folder/f'bottle-consistency-v2-{a.mode}-{a.attempt_id}-predecessor.json'
            with snapshot.open('xb')as f:f.write(run_path.read_bytes())
        started=time.monotonic();runner=SemanticRunner(RUN_ID,mode=a.mode)
        try:state=runner.execute(V1.PROFILE,input_path=ROOT/row['input_path'],publication_mode='admin_review',overnight_task_budget_plan=V1.BUDGET)
        except V1.GPUCheckpoint:
            state=runner.state;state.update(status='paused_after_reconstruct'if V1.verified_checkpoint(state)else'gpu_checkpoint_requires_attention',phase=None,error=None,
                semantic_revision={'prior_discovery':V1.reference(PRIOR/'discovery.json'),'full_frame_consistency':True,
                    'prior_outputs_replaced':False,'automatic_publication':False});runner.save()
        except ReconciliationBlocked as error:
            state=runner.state;state.update(status='semantic_revision_requires_attention',phase=None,error=str(error));runner.save()
        result={'format':'interaction-twin-full-frame-semantic-revision-result','version':1,'mode':a.mode,'attempt_id':a.attempt_id,
            'manifest':manifest_ref,'driver':V1.reference(Path(__file__)),'adapter':V1.reference(ADAPTER),'run':V1.reference(runner.path),
            'status':state['status'],'error':state.get('error'),'gpu':V1.gpu_outcome(state),'wall_seconds':time.monotonic()-started,
            'automatic_publication':False,'parent_allocation_modified':False,'old_bottle_outputs_promoted':False}
        V1.immutable(receipt,result);P.write(folder/'bottle-consistency-v2-progress.json',result);print(json.dumps(result),flush=True)
    return 0 if state['status']in {'paused_after_reconstruct','complete','complete_with_gaps'}else 1


if __name__=='__main__':raise SystemExit(main())
