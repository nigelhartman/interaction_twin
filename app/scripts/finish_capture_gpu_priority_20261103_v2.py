#!/usr/bin/env python3
"""Finish missing post-GPU phases with exact normal/compact cache contracts.

This explicitly selected batch never runs perception or reconstruction again.
Compact segment dependencies are recovered from their immutable original intent
and failed predecessor; the original successful compact masks must hash-verify.
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
from scripts import recover_capture_compact_sam_20261103 as COMPACT
from scripts.reconcile_capture_gpu_priority_20261103_v2 import ReconciliationBlocked
P=V1.P
PRESERVE={'prepare','discover','segment','audit','select','reconstruct'}
PAID={'discover','segment','audit','procedure_proposal','mechanism_topology'}


def compact_proof(state,row,manifest_ref):
    saved=state.get('compact_continuation')
    if saved is None:return None
    intent=P.read(V1.checked(saved['intent']));predecessor=P.read(V1.checked(saved['predecessor']))
    if (intent.get('format')!='interaction-twin-known-413-compact-continuation'or intent.get('manifest')!=manifest_ref
        or intent.get('capture_id')!=row['id']or intent.get('run_id')!=row['run_id']
        or intent.get('maximum_new_compact_requests')!=1 or intent.get('automatic_publication')is not False
        or intent.get('adapter')!=V1.reference(COMPACT.ADAPTER)):
        raise ValueError('Saved compact continuation binding changed')
    case=ROOT/'data'/('flow-'+row['run_id'].removeprefix('run-'))[:60]
    proof=COMPACT.known_413(predecessor,case)
    if proof!=intent.get('original_proof'):raise ValueError('Saved original uncharged413 proof changed')
    return proof


class FinishRunner(P.Runner):
    def __init__(self,run_id,*,proof):super().__init__(run_id);self.compact_proof=proof

    def phase(self,name,dependencies,action,*,optional=False):
        if name=='segment'and self.compact_proof:
            dependencies=[*dependencies,COMPACT.ADAPTER,ROOT/self.compact_proof['request']['path'],ROOT/self.compact_proof['encoding']['path']]
        prior=next((r for r in self.state['phases']if r.get('name')==name),None)
        if name in PRESERVE or(name in PAID and prior):
            signature=P.fingerprint({'profile':self.profile,'source':self.state.get('source'),
                'inputs':P.file_receipts(dependencies),'runner_code':P.LOADED_SOURCE_SHA256})
            if (not prior or prior.get('status')not in V1.SUCCESS or prior.get('signature')!=signature
                or not P.verify_receipts(prior.get('files'))
                or(name=='reconstruct'and prior.get('result',{}).get('pending_components'))):
                raise ReconciliationBlocked('Finish requires exact cached '+name+'; no perception/model or uncertain paid request was resent.')
        return super().phase(name,dependencies,action,optional=optional)


def finish(row,manifest_ref,*,runner_class=FinishRunner):
    path=ROOT/'data/flow-runs'/row['run_id']/'run.json'
    if not path.is_file():return {'id':row['id'],'status':'blocked_missing_run'}
    state=P.read(path)
    if state.get('status')in {'complete','complete_with_gaps'}and V1.verified_checkpoint(state):
        return {'id':row['id'],'status':'already_finished','run':V1.reference(path),'new_pipeline_execution':False}
    if (state.get('status')!='paused_after_reconstruct'or not V1.verified_checkpoint(state)
        or not V1.gpu_outcome(state)['all_selected_native_models_ready']):
        return {'id':row['id'],'status':'blocked_gpu_incomplete_or_ambiguous','run':V1.reference(path),'new_pipeline_execution':False}
    try:proof=compact_proof(state,row,manifest_ref)
    except(ValueError,KeyError,OSError)as error:return {'id':row['id'],'status':'blocked_compact_binding','reason':str(error),'new_pipeline_execution':False}
    predecessor=ROOT/'output/new-captures-20261103'/(row['id']+'-finish-v2-predecessor.json')
    with predecessor.open('xb')as stream:stream.write(path.read_bytes())
    started=time.monotonic();runner=runner_class(row['run_id'],proof=proof)
    try:result=runner.execute(V1.PROFILE,input_path=ROOT/row['input_path'],publication_mode='admin_review',overnight_task_budget_plan=V1.BUDGET)
    except ReconciliationBlocked as error:
        result=runner.state;result.update(status='finish_requires_attention',phase=None,error=str(error));runner.save()
    return {'id':row['id'],'run_id':row['run_id'],'status':result['status'],'error':result.get('error'),
        'run':V1.reference(runner.path),'predecessor':V1.reference(predecessor),
        'compact_segment_contract_preserved':proof is not None,'perception_or_model_resends':0,
        'wall_seconds':time.monotonic()-started,'new_pipeline_execution':True,'automatic_publication':False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',type=Path,default=ROOT/V1.MANIFEST)
    parser.add_argument('--capture',choices=['all',*[r[0]for r in V1.SOURCES]],default='all');args=parser.parse_args(argv)
    doc,manifest_ref=V1.load_manifest(args.manifest);folder=ROOT/'output/new-captures-20261103';outcomes=[]
    with(ROOT/'.state/capture-gpu-priority-20261103.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for row in doc['captures']:
            if args.capture!='all'and row['id']!=args.capture:continue
            receipt=folder/(row['id']+'-finish-v2-result.json')
            if receipt.exists():
                saved=P.read(receipt)
                if saved.get('manifest')!=manifest_ref:raise ValueError('Saved finish belongs to another manifest')
                outcome=saved['outcome']
            else:
                P.write(folder/'finish-v2-progress.json',{'status':'running','capture_id':row['id'],'manifest':manifest_ref,'outcomes':outcomes,'updated_utc':P.now()})
                outcome=finish(row,manifest_ref)
                V1.immutable(receipt,{'format':'interaction-twin-post-gpu-finish','version':2,'manifest':manifest_ref,
                    'driver':V1.reference(Path(__file__)),'outcome':outcome,'completed_utc':P.now(),'automatic_publication':False})
            outcomes.append(outcome);print(json.dumps(outcome),flush=True)
        P.write(folder/'finish-v2-progress.json',{'status':'complete'if all(o['status']in {'complete','complete_with_gaps','already_finished'}for o in outcomes)else'needs_attention',
            'manifest':manifest_ref,'outcomes':outcomes,'completed_utc':P.now(),'automatic_publication':False})
    return 0 if all(o['status']in {'complete','complete_with_gaps','already_finished'}for o in outcomes)else 1


if __name__=='__main__':raise SystemExit(main())
