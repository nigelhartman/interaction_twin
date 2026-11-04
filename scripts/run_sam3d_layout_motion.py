#!/usr/bin/env python3
"""Prepare or execute one fresh local SAM3D-layout/CoTracker motion smoke test."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import sam3d_layout_motion as motion
import upgrade_bundles as U

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','check','run','worker'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--model-metadata',type=Path);p.add_argument('--projection-directory',type=Path)
    p.add_argument('--expected-plan-sha256',help='Reviewed frozen plan digest, required before learned execution')
    a=p.parse_args()
    if a.command=='prepare':
        if not a.model_metadata or not a.projection_directory:p.error('prepare requires model metadata and raw projection directory')
        plan=motion.prepare(a.model_metadata,a.projection_directory,a.output)
        print(json.dumps({'status':plan['status'],'queries':plan['queries'],'render_samples':plan['render_samples'],'frame_count':len(plan['rows'])},indent=2));return 0
    if a.command=='check':
        plan,out=motion.verify_plan(a.output);print(json.dumps({'status':'frozen_motion_plan_verified','plan_sha256':U.sha(out/'plan.json'),'frame_count':len(plan['rows']),'queries':plan['queries']},indent=2));return 0
    if a.command=='worker':
        _,out=motion.verify_plan(a.output)
        motion.require(a.expected_plan_sha256==U.sha(out/'plan.json'),'Reviewed plan SHA is required before model execution')
        result=motion.run(a.output);print(json.dumps({k:result[k] for k in ['status','supported_nonreference_samples','unsupported_nonreference_samples','wall_seconds']},indent=2));return 0
    plan,out=motion.verify_plan(a.output)
    motion.require(a.expected_plan_sha256==U.sha(out/'plan.json'),'Reviewed plan SHA is required before model execution')
    env=dict(os.environ,PYTORCH_ENABLE_MPS_FALLBACK='0')
    try:r=subprocess.run([sys.executable,'-I',str(Path(__file__).resolve()),'worker','--output',str(out),'--expected-plan-sha256',a.expected_plan_sha256],env=env,capture_output=True,text=True,timeout=720)
    except subprocess.TimeoutExpired as error:
        previous=U.read(out/'experiment.json') if (out/'experiment.json').is_file() else {}
        previous.update(status='720_second_wall_timeout_no_retry',physical_accuracy_verified=False,simulation_ready=False)
        U.write(out/'experiment.json',previous)
        U.write(out/'timeout.json',{'status':'720_second_wall_timeout_no_retry','physical_accuracy_verified':False})
        (out/'run.log').write_text((error.stdout or b'').decode(errors='replace')+(error.stderr or b'').decode(errors='replace')+'\nWorker killed at bounded wall timeout; partial results retained.\n')
        print(json.dumps({'status':'timeout_no_retry','output':str(out)}));return 2
    (out/'run.log').write_text(r.stdout+r.stderr or 'Worker returned without console output.\n');print(r.stdout+r.stderr);return r.returncode

if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,OSError,KeyError,TypeError) as e:
        print(json.dumps({'status':'fresh_motion_unsupported','error':str(e)}));raise SystemExit(2)
