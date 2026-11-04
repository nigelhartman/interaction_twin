#!/usr/bin/env python3
"""One explicitly reviewed compact continuation after a known uncharged HTTP413.

Original requests, models, source frames, discovery boxes, run IDs and budget
baselines are retained. Only the failed segmentation phase gets a new attempt;
the new adapter uses same-size CRF18 input and the unchanged Meta budget client.
"""
from __future__ import annotations
import argparse
import fcntl
import json
from pathlib import Path
import sqlite3
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_capture_gpu_priority_20261103 as V1
from scripts.reconcile_capture_gpu_priority_20261103_v2 import ReconciliationBlocked
P=V1.P
ADAPTER=ROOT/'scripts/stage2_meta_video_compact_20261103.py'


def known_413(state,case):
    phase=next((p for p in state.get('phases',[])if p.get('name')=='segment'),None)
    if state.get('status')!='failed'or not phase or phase.get('status')!='failed':raise ValueError('Only a failed segmentation attempt is eligible')
    directory=case/'segmentation'/('flow-components-'+str(phase['attempt']))
    path=directory/'api/accounting/meta-request.json';receipt=P.read(path)
    if (receipt.get('model')!='sam-3.1'or receipt.get('endpoint')!='/v1/responses'or receipt.get('kind')!='sam'
        or receipt.get('status')!='rejected_before_generation'or receipt.get('http_status')!=413
        or receipt.get('estimated_cost_usd')!='0.000000'or receipt.get('automatic_retries')is not False):
        raise ValueError('Original request is not a known uncharged HTTP413')
    ledger=ROOT/'.secrets/meta-budget.sqlite3'
    db=sqlite3.connect(ledger.resolve().as_uri()+'?mode=ro',uri=True)
    try:row=db.execute('SELECT model,endpoint,kind,state,reason,actual FROM requests WHERE request_id=?',(receipt['request_id'],)).fetchone()
    finally:db.close()
    if row!=('sam-3.1','/v1/responses','sam','released','provider_http_rejected_before_generation:413',None):
        raise ValueError('Original rejected request lacks matching released ledger proof')
    encoding=directory/'video-input/encoding.json';document=P.read(encoding)
    video=directory/'video-input'/document['video']
    if P.sha(video)!=document['video_sha256']or video.stat().st_size!=document['video_bytes']:raise ValueError('Original rejected encoding changed')
    return {'request_id':receipt['request_id'],'request':V1.reference(path),'encoding':V1.reference(encoding),
            'video_sha256':document['video_sha256'],'video_bytes':document['video_bytes'],
            'ledger_state':'released','ledger_reason':'provider_http_rejected_before_generation:413'}


class CompactRecoveryRunner(P.Runner):
    def __init__(self,run_id,*,proof):super().__init__(run_id);self.compact_proof=proof

    def command(self,args,timeout):
        values=[str(v)for v in args]
        exact=[str(ROOT/'.venv-sam/bin/python'),str(ROOT/'scripts/run_overnight_stage.py'),
               '--task-budget-plan',V1.BUDGET,'stage2_meta_video_segment.py']
        if 'stage2_meta_video_segment.py'in values:
            if values[:5]!=exact or len(values)!=11 or values[5::2]!=['--case','--config','--run']:
                raise ValueError('Only the exact reviewed video segmentation command may be replaced')
            values=[exact[0],str(ADAPTER),*values[5:],'--allow-paid-api']
        return super().command(values,timeout)

    def phase(self,name,dependencies,action,*,optional=False):
        if name in {'prepare','discover'}:
            prior=next((r for r in self.state['phases']if r.get('name')==name),None)
            signature=P.fingerprint({'profile':self.profile,'source':self.state.get('source'),
                'inputs':P.file_receipts(dependencies),'runner_code':P.LOADED_SOURCE_SHA256})
            if not prior or prior.get('status')not in V1.SUCCESS or prior.get('signature')!=signature or not P.verify_receipts(prior.get('files')):
                raise ReconciliationBlocked('Saved '+name+' cannot be reused exactly; no prior paid request resent.')
        if name=='segment':dependencies=[*dependencies,ADAPTER,ROOT/self.compact_proof['request']['path'],ROOT/self.compact_proof['encoding']['path']]
        result=super().phase(name,dependencies,action,optional=optional)
        if name=='reconstruct':raise V1.GPUCheckpoint()
        return result


def recover(row,manifest_ref):
    run_path=ROOT/'data/flow-runs'/row['run_id']/'run.json'
    if not run_path.is_file():return {'id':row['id'],'status':'blocked_missing_run'}
    state=P.read(run_path)
    if state.get('status')in {'paused_after_reconstruct','complete','complete_with_gaps'}:
        return {'id':row['id'],'status':'not_a_413_recovery','gpu':V1.gpu_outcome(state),'run':V1.reference(run_path)}
    case=ROOT/'data'/('flow-'+row['run_id'].removeprefix('run-'))[:60]
    try:proof=known_413(state,case)
    except(ValueError,FileNotFoundError,KeyError,sqlite3.Error)as error:
        return {'id':row['id'],'status':'blocked_not_known_413','reason':str(error),'run':V1.reference(run_path)}
    folder=ROOT/'output/new-captures-20261103'
    intent=folder/('compact-continuation-'+proof['request_id']+'.json')
    # One durable changed-payload intent per original rejected request. No resend
    # of a failed/unknown compact request, even if a local process is interrupted.
    V1.immutable(intent,{'format':'interaction-twin-known-413-compact-continuation','version':1,
        'manifest':manifest_ref,'capture_id':row['id'],'run_id':row['run_id'],'original_proof':proof,
        'adapter':V1.reference(ADAPTER),'same_frame_count_and_coordinates':True,'lossy_compression_disclosed':True,
        'maximum_new_compact_requests':1,'created_utc':P.now(),'automatic_publication':False})
    predecessor=folder/(row['id']+'-compact-continuation-predecessor.json')
    with predecessor.open('xb')as f:f.write(run_path.read_bytes())
    started=time.monotonic();runner=CompactRecoveryRunner(row['run_id'],proof=proof);blocked=None
    try:
        result=runner.execute(V1.PROFILE,input_path=ROOT/row['input_path'],publication_mode='admin_review',overnight_task_budget_plan=V1.BUDGET)
    except V1.GPUCheckpoint:
        result=runner.state;result.update(status='paused_after_reconstruct'if V1.verified_checkpoint(result)else'gpu_checkpoint_requires_attention',phase=None,error=None,
            compact_continuation={'intent':V1.reference(intent),'predecessor':V1.reference(predecessor),'old_rejected_attempt_preserved':True,'automatic_publication':False})
        runner.save()
    except ReconciliationBlocked as error:
        blocked=str(error);result=runner.state;result.update(status='failed',phase=None,error=blocked);runner.save()
    return {'id':row['id'],'run_id':row['run_id'],'status':result['status'],'reason':blocked or result.get('error'),
        'gpu':V1.gpu_outcome(result),'run':V1.reference(run_path),'intent':V1.reference(intent),
        'predecessor':V1.reference(predecessor),'wall_seconds':time.monotonic()-started,
        'original_discovery_resent':False,'original_rejected_request_resent':False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',type=Path,default=ROOT/V1.MANIFEST)
    parser.add_argument('--allow-paid-api',action='store_true');args=parser.parse_args(argv)
    if not args.allow_paid_api:raise ValueError('Root-reviewed compact continuation requires explicit --allow-paid-api')
    doc,manifest_ref=V1.load_manifest(args.manifest);folder=ROOT/'output/new-captures-20261103';outcomes=[]
    with (ROOT/'.state/capture-gpu-priority-20261103.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for row in doc['captures']:
            receipt=folder/(row['id']+'-compact-continuation-result.json')
            if receipt.exists():
                saved=P.read(receipt)
                if saved.get('manifest')!=manifest_ref:raise ValueError('Compact continuation belongs to another manifest')
                outcome=saved['outcome']
            else:
                print(json.dumps({'status':'inspecting_known_413','capture_id':row['id']}),flush=True)
                P.write(folder/'compact-continuation-progress.json',{'status':'running','capture_id':row['id'],'manifest':manifest_ref,'outcomes':outcomes,'updated_utc':P.now()})
                outcome=recover(row,manifest_ref)
                V1.immutable(receipt,{'format':'interaction-twin-compact-sam-continuation-result','version':1,
                    'manifest':manifest_ref,'driver':V1.reference(Path(__file__)),'adapter':V1.reference(ADAPTER),
                    'outcome':outcome,'completed_utc':P.now(),'automatic_publication':False})
            outcomes.append(outcome);print(json.dumps(outcome),flush=True)
        summary={'format':'interaction-twin-compact-sam-continuation-batch','version':1,'manifest':manifest_ref,
            'outcomes':outcomes,'completed_utc':P.now(),'automatic_publication':False,'parent_allocation_modified':False}
        final=folder/'batch-compact-continuation-result.json'
        if not final.exists():V1.immutable(final,summary)
        P.write(folder/'compact-continuation-progress.json',summary)
    return 0


if __name__=='__main__':raise SystemExit(main())
