#!/usr/bin/env python3
"""Explicit post-GPU finish of terminal partial results, without model retries.

At least one real imported asset is required. Missing/failed components remain
gaps. Normal, compact-SAM and corrected-bottle dependency contracts are retained.
This never finishes or promotes the mistaken original bottle decomposition.
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
from scripts import finish_capture_gpu_priority_20261103_v2 as FINISH
from scripts import run_bottle_component_consistency_20261103_v2 as SEMANTIC
from scripts.reconcile_capture_gpu_priority_20261103_v2 import ReconciliationBlocked
P=V1.P
TERMINAL={'complete','complete_with_gaps','failed'}


def admitted_partial(state,row,run_id):
    expected_request={'profile':V1.PROFILE,'input_path':str((ROOT/row['input_path']).resolve()),'case_id':None,
        'overnight_task_budget_plan':V1.BUDGET,'publication_mode':'admin_review'}
    if state.get('id')!=run_id or state.get('request')!=expected_request:
        raise ValueError('Partial finish must preserve the exact private source/profile/request identity')
    if state.get('source',{}).get('sha256')!=row['source_ref']['sha256']or state.get('source',{}).get('bytes')!=row['source_ref']['bytes']:
        raise ValueError('Partial finish source differs from the approved recording')
    if state.get('status')not in {'paused_after_reconstruct','complete','complete_with_gaps'}or not V1.verified_checkpoint(state):
        raise ValueError('A verified completed reconstruction phase is required')
    phase=next(p for p in state['phases']if p['name']=='reconstruct');result=phase['result'];components=result.get('components',[])
    if result.get('pending_components')or not components or any(r.get('status')not in TERMINAL for r in components):
        raise ValueError('Pending/running/queued components cannot enter settled-partial finish')
    if any(r.get('output',{}).get('status')=='awaiting_gpu_worker'for r in components):
        raise ValueError('A pending GPU output cannot be treated as a terminal failure')
    imported=[r for r in components if r.get('status')in {'complete','complete_with_gaps'}and
        r.get('output',{}).get('status')=='native_candidate_ready'and r['output'].get('model')in result.get('models',[])
        and r['output'].get('native_asset')]
    if not imported:raise ValueError('At least one actual imported native model is required')
    return {'imported_component_ids':[r['track_id']for r in imported],
        'terminal_failed_component_ids':[r['track_id']for r in components if r['status']=='failed'],
        'unavailable_component_ids':[r['track_id']for r in components if r not in imported],
        'pending_components':[],'reconstruction_action_permitted':False,
        'failure_causes':'Preserved original diagnostics; no inference/OOM/import interpretation is invented.',
        'physical_accuracy_verified':False}


class PartialNormalRunner(FINISH.FinishRunner):
    def phase(self,name,dependencies,action,*,optional=False):
        if name=='reconstruct':
            def action(_):raise ReconciliationBlocked('Settled-partial finisher forbids reconstruction action')
        return super().phase(name,dependencies,action,optional=optional)


class PartialSemanticRunner(SEMANTIC.SemanticRunner):
    def phase(self,name,dependencies,action,*,optional=False):
        if name=='reconstruct':
            def action(_):raise ReconciliationBlocked('Settled-partial finisher forbids reconstruction action')
        return super().phase(name,dependencies,action,optional=optional)


def finish(row,manifest_ref,*,semantic=False,attempt_id='v1'):
    run_id=SEMANTIC.RUN_ID if semantic else row['run_id'];path=ROOT/'data/flow-runs'/run_id/'run.json'
    if not path.is_file():return {'id':row['id'],'run_id':run_id,'status':'blocked_missing_run'}
    state=P.read(path)
    try:admission=admitted_partial(state,row,run_id)
    except(ValueError,KeyError)as error:return {'id':row['id'],'run_id':run_id,'status':'blocked_partial_admission','reason':str(error),'new_pipeline_execution':False}
    if state['status']in {'complete','complete_with_gaps'}:
        return {'id':row['id'],'run_id':run_id,'status':'already_finished','run':V1.reference(path),'admission':admission,'new_pipeline_execution':False}
    try:
        if semantic:
            revision=state.get('semantic_revision',{})
            if revision.get('full_frame_consistency')is not True or revision.get('prior_discovery')!=V1.reference(SEMANTIC.PRIOR/'discovery.json'):
                raise ValueError('Corrected bottle semantic revision proof differs')
            runner=PartialSemanticRunner(run_id,mode='finish')
        else:
            proof=FINISH.compact_proof(state,row,manifest_ref);runner=PartialNormalRunner(run_id,proof=proof)
    except(ValueError,KeyError,OSError)as error:
        return {'id':row['id'],'run_id':run_id,'status':'blocked_branch_contract','reason':str(error),'new_pipeline_execution':False}
    folder=ROOT/'output/new-captures-20261103';snapshot=folder/f'{row["id"]}-settled-partial-v3-{attempt_id}-predecessor.json'
    with snapshot.open('xb')as stream:stream.write(path.read_bytes())
    started=time.monotonic()
    try:result=runner.execute(V1.PROFILE,input_path=ROOT/row['input_path'],publication_mode='admin_review',overnight_task_budget_plan=V1.BUDGET)
    except ReconciliationBlocked as error:
        result=runner.state;result.update(status='settled_partial_finish_requires_attention',phase=None,error=str(error));runner.save()
    return {'id':row['id'],'run_id':run_id,'status':result['status'],'error':result.get('error'),'admission':admission,
        'run':V1.reference(runner.path),'predecessor':V1.reference(snapshot),'wall_seconds':time.monotonic()-started,
        'new_pipeline_execution':True,'new_reconstruction_attempts':0,'failed_components_retried':False,
        'missing_parts_preserved_as_gaps':True,'automatic_publication':False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',choices=['all','charger','phone','laptop','bottle'],default='all')
    parser.add_argument('--attempt-id',default='v1');parser.add_argument('--allow-settled-partial',action='store_true');a=parser.parse_args(argv)
    import re
    if not a.allow_settled_partial or not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,31}',a.attempt_id):raise ValueError('Explicit bounded --allow-settled-partial required')
    doc,manifest_ref=V1.load_manifest(ROOT/V1.MANIFEST);folder=ROOT/'output/new-captures-20261103';outcomes=[]
    with(ROOT/'.state/capture-gpu-priority-20261103.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for row in doc['captures']:
            if a.capture!='all'and row['id']!=a.capture:continue
            receipt=folder/f'{row["id"]}-settled-partial-v3-{a.attempt_id}-result.json'
            if receipt.exists():
                saved=P.read(receipt)
                if saved.get('manifest')!=manifest_ref:raise ValueError('Saved partial finish belongs to another manifest')
                outcome=saved['outcome']
            else:
                P.write(folder/'settled-partial-v3-progress.json',{'status':'running','capture_id':row['id'],'manifest':manifest_ref,'outcomes':outcomes,'updated_utc':P.now()})
                outcome=finish(row,manifest_ref,semantic=row['id']=='bottle',attempt_id=a.attempt_id)
                V1.immutable(receipt,{'format':'interaction-twin-settled-partial-post-gpu-finish','version':3,
                    'manifest':manifest_ref,'driver':V1.reference(Path(__file__)),'outcome':outcome,
                    'completed_utc':P.now(),'allow_settled_partial_explicit':True,'automatic_publication':False})
            outcomes.append(outcome);print(json.dumps(outcome),flush=True)
        P.write(folder/'settled-partial-v3-progress.json',{'status':'complete'if all(r['status']in {'complete','complete_with_gaps','already_finished'}for r in outcomes)else'needs_attention',
            'manifest':manifest_ref,'outcomes':outcomes,'completed_utc':P.now(),'automatic_publication':False})
    return 0 if all(r['status']in {'complete','complete_with_gaps','already_finished'}for r in outcomes)else 1


if __name__=='__main__':raise SystemExit(main())
