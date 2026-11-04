#!/usr/bin/env python3
"""Explicit four-capture private GPU-first batch; never retries an uncertain run.

This new driver preserves the existing pipeline and budget modules. A boundary
after the real reconstruction phase defers later CPU/API work, with the same
request identity available for an explicit ``--mode finish`` invocation.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pipeline as P

MANIFEST='output/new-captures-20261103/batch-manifest.json'
PROFILE='meta_sam3d_capture_priority_20261103'
BUDGET='output/overnight-pipeline-20261102/task-budget-plan.json'
SOURCES=[('charger','20261103_151242_036b1705.mp4'),('phone','20261103_151424_c0ceb3f7.mp4'),
         ('laptop','20261103_151533_67c51eba.mp4'),('bottle','20261103_151932_1383256a.mp4')]
SUCCESS={'complete','complete_with_gaps','reused_verified','reused_verified_with_gaps'}


class GPUCheckpoint(BaseException):
    """A deliberate, non-error boundary, caught only by this trusted driver."""


class PriorityRunner(P.Runner):
    def phase(self,name,dependencies,action,*,optional=False):
        result=super().phase(name,dependencies,action,optional=optional)
        if name=='reconstruct':raise GPUCheckpoint()
        return result


def reference(path):
    path=Path(path)
    return {'path':path.relative_to(ROOT).as_posix(),'sha256':P.sha(path),'bytes':path.stat().st_size}


def checked(ref):
    if not isinstance(ref,dict)or set(ref)!={'path','sha256','bytes'}:raise ValueError('Exact file reference required')
    path=ROOT/ref['path']
    if path.is_symlink()or path.resolve()!=path or not path.is_file()or reference(path)!=ref:raise ValueError('Pinned batch input changed')
    return path


def load_manifest(path):
    path=Path(path).resolve()
    if path!=ROOT/MANIFEST or path.is_symlink():raise ValueError('Only the fixed approved batch manifest is accepted')
    doc=P.read(path)
    if (doc.get('format')!='interaction-twin-gpu-priority-capture-batch'or doc.get('version')!=1
        or doc.get('profile')!=PROFILE or doc.get('overnight_task_budget_plan')!=BUDGET
        or doc.get('publication_mode')!='admin_review'or doc.get('automatic_publication')is not False
        or doc.get('gpu_parent_allocation')!='16889584'):
        raise ValueError('Approved private batch contract differs')
    if doc.get('profile_ref',{}).get('path')!='profiles/'+PROFILE+'.json':raise ValueError('Fixed new profile required')
    checked(doc['profile_ref']);profile=P.load_profile(PROFILE)
    if any(profile.get(k)!=v for k,v in {'fps':1.5,'max_duration_s':65,'max_frames':120,'reconstruct_limit':4,
        'max_dimension':960,'reconstruction_provider':'sam3d_native','segmentation_provider':'meta_sam31_api',
        'reasoning_provider':'meta_api','mechanism_pipeline_mode':'muse_constrained_v1'}.items()):
        raise ValueError('Approved bounded whole-video profile differs')
    rows=doc.get('captures')
    if not isinstance(rows,list)or len(rows)!=len(SOURCES):raise ValueError('Exactly four approved captures required')
    for row,(identifier,filename)in zip(rows,SOURCES):
        expected='data/new-captures-20261103/sources/'+filename
        if (row.get('id')!=identifier or row.get('run_id')!='run-capture-'+identifier+'-20261103-v1'
            or row.get('input_path')!=expected or row.get('source_ref',{}).get('path')!=expected
            or row.get('owner_context_is_observation')is not False
            or not isinstance(row.get('owner_context'),str)or len(row['owner_context'])>4000):
            raise ValueError('Fixed capture identity or fallible context differs')
        source=checked(row['source_ref']);P.validate_source(source,profile)
    return doc,reference(path)


def verified_checkpoint(state):
    row=next((r for r in state.get('phases',[])if r.get('name')=='reconstruct'),None)
    return bool(row and row.get('status')in SUCCESS and P.verify_receipts(row.get('files')))


def gpu_outcome(state):
    phase=next((r for r in state.get('phases',[])if r.get('name')=='reconstruct'),{})
    result=phase.get('result',{});components=result.get('components',[])
    ready=[r['track_id']for r in components if r.get('output',{}).get('status')=='native_candidate_ready']
    return {'selected_component_count':len(components),'native_ready_component_ids':ready,
            'all_selected_native_models_ready':bool(components)and len(ready)==len(components),
            'pending_components':result.get('pending_components',[]),'failures':result.get('failures',[]),
            'physical_accuracy_verified':False}


def run_capture(row,mode,manifest_ref,*,runner_class=None):
    state_path=ROOT/'data/flow-runs'/row['run_id']/'run.json'
    previous=P.read(state_path)if state_path.is_file()else None
    # Never let ordinary Runner retry a paid phase or an ambiguous interruption.
    if previous:
        if previous.get('status')not in {'paused_after_reconstruct','complete','complete_with_gaps'}or not verified_checkpoint(previous):
            return {'id':row['id'],'run_id':row['run_id'],'status':'blocked_existing_attempt',
                    'reason':'Inspect preserved run; no paid retry or new namespace was created.','run':reference(state_path)}
        if mode=='gpu-first'or previous['status']in {'complete','complete_with_gaps'}:
            return {'id':row['id'],'run_id':row['run_id'],'status':'reused_verified_checkpoint',
                    'run':reference(state_path),'gpu':gpu_outcome(previous),'new_pipeline_execution':False}
    elif mode=='finish':
        return {'id':row['id'],'run_id':row['run_id'],'status':'blocked_no_gpu_checkpoint',
                'reason':'Run GPU-first explicitly before finishing.'}
    started=time.monotonic()
    cls=runner_class or(PriorityRunner if mode=='gpu-first'else P.Runner)
    runner=cls(row['run_id'])
    try:
        state=runner.execute(PROFILE,input_path=ROOT/row['input_path'],publication_mode='admin_review',overnight_task_budget_plan=BUDGET)
    except GPUCheckpoint:
        if not verified_checkpoint(runner.state):
            runner.state.update(status='gpu_checkpoint_requires_attention',phase=None,error='Reconstruction receipt did not verify; no subsequent phase executed.')
        else:
            runner.state.update(status='paused_after_reconstruct',phase=None,error=None,
                gpu_priority_checkpoint={'manifest':manifest_ref,'owner_context_is_observation':False,
                    'owner_context_passed_to_models':False,'future_stages_deferred':True,
                    'automatic_publication':False,'parent_allocation_modified':False})
        runner.save();state=runner.state
    return {'id':row['id'],'run_id':row['run_id'],'status':state['status'],'wall_seconds':time.monotonic()-started,
            'run':reference(runner.path),'gpu':gpu_outcome(state),'new_pipeline_execution':True,
            'error':state.get('error')}


def immutable(path,value):
    with path.open('x')as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=ROOT/MANIFEST)
    parser.add_argument('--mode',choices=['gpu-first','finish'],default='gpu-first')
    args=parser.parse_args(argv)
    doc,manifest_ref=load_manifest(args.manifest)
    folder=ROOT/'output/new-captures-20261103';folder.mkdir(exist_ok=True)
    code_ref=reference(Path(__file__).resolve());started=P.now();outcomes=[]
    lock_path=ROOT/'.state/capture-gpu-priority-20261103.lock';lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for row in doc['captures']:
            receipt=folder/(row['id']+'-'+args.mode+'-result.json')
            if receipt.exists():
                saved=P.read(receipt)
                if saved.get('manifest')!=manifest_ref or saved.get('mode')!=args.mode:raise ValueError('Prior batch receipt belongs to another request')
                outcome=saved['outcome']
            else:
                P.write(folder/'progress.json',{'mode':args.mode,'status':'running','capture_id':row['id'],'run_id':row['run_id'],'updated_utc':P.now(),'manifest':manifest_ref,'outcomes':outcomes})
                print(json.dumps({'status':'starting','capture_id':row['id'],'run_id':row['run_id']}),flush=True)
                outcome=run_capture(row,args.mode,manifest_ref)
                immutable(receipt,{'format':'interaction-twin-capture-gpu-priority-result','version':1,
                    'mode':args.mode,'manifest':manifest_ref,'driver':code_ref,'outcome':outcome,
                    'completed_utc':P.now(),'automatic_publication':False,'parent_allocation_modified':False})
            outcomes.append(outcome);print(json.dumps(outcome),flush=True)
        summary={'format':'interaction-twin-capture-gpu-priority-batch-result','version':1,'mode':args.mode,
            'status':'batch_checkpoint_complete'if all(o['status']in {'paused_after_reconstruct','complete','complete_with_gaps','reused_verified_checkpoint'}for o in outcomes)else'batch_needs_attention',
            'manifest':manifest_ref,'driver':code_ref,'started_utc':started,'completed_utc':P.now(),
            'outcomes':outcomes,'automatic_publication':False,'parent_allocation_modified':False}
        final=folder/('batch-'+args.mode+'-result.json')
        if not final.exists():immutable(final,summary)
        P.write(folder/'progress.json',summary)
    return 0 if summary['status']=='batch_checkpoint_complete'else 1


if __name__=='__main__':raise SystemExit(main())
