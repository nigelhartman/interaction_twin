#!/usr/bin/env python3
"""Independently check real saved native poses against the viewer matrices.

This checks coordinate/projection implementation and visibility bookkeeping;
it does not validate the recovered physical pose, shape or camera calibration.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from native_appearance import load_native_glb
from motion_lab import verified,sha


def audit(viewer_path,root=ROOT):
    root=Path(root).resolve();viewer_path=Path(viewer_path).resolve();doc=json.loads(viewer_path.read_text())
    if doc['format']!='interval-motion-viewer':raise ValueError('Expected normalized actual viewer data')
    asset=verified(root,doc['asset']['url'],doc['asset']['sha256']);mesh=load_native_glb(asset)
    indices=np.linspace(0,len(mesh.vertices)-1,min(500,len(mesh.vertices)),dtype=int)
    points=mesh.vertices[indices].astype(float);rows=[];withheld=0;references=0
    for episode in doc['episodes']:
        for frame in episode['frames']:
            for branch,row,K in [('candidate',frame,episode['intrinsics']['matrix']),('baseline',frame.get('baseline'),doc['baseline']['intrinsics']['matrix'])]:
                if row is None:continue
                if row['status']=='withheld':
                    if row.get('pose')is not None:raise ValueError('Rejected display pose leaked')
                    withheld+=1;continue
                if row['status']=='reference_unverified':references+=1
                p=row['pose'];q=points@np.asarray(p['R']).T+np.asarray(p['t']);clip=q@np.asarray(K).T
                valid=np.isfinite(q).all(axis=1)&(q[:,2]>0)
                pixels=np.zeros((len(points),2));pixels[valid]=clip[valid,:2]/clip[valid,2,None]
                rows.append({'episode_id':episode['id'],'sample_index':frame['sample_index'],'branch':branch,
                    'K':K,'width':episode['intrinsics']['width'],'height':episode['intrinsics']['height'],
                    'R':p['R'],'t':p['t'],'valid':valid.tolist(),'expected':pixels.tolist()})
    javascript=r'''
import fs from 'node:fs';import assert from 'node:assert/strict';
import * as THREE from './workspace/robot-vendor/three.module.js';
import {sourceCameraMatrices} from './workspace/geometry-camera.mjs';
import {poseMatrix,exactSample,displayPose} from './workspace/motion-math.mjs';
const input=JSON.parse(fs.readFileSync(0,'utf8'));let maximum=0,count=0;
for(const row of input.rows){const transform=new THREE.Matrix4().fromArray(poseMatrix(row.R,row.t)),camera=sourceCameraMatrices([[1,0,0],[0,1,0],[0,0,1]],[0,0,0],row.K,row.width,row.height),view=new THREE.Matrix4().fromArray(camera.world).invert(),projection=new THREE.Matrix4().fromArray(camera.projection);for(let i=0;i<input.points.length;i++){if(!row.valid[i])continue;const p=new THREE.Vector4(...input.points[i],1).applyMatrix4(transform).applyMatrix4(view).applyMatrix4(projection),uv=[(p.x/p.w+1)*row.width/2-.5,(1-p.y/p.w)*row.height/2-.5],error=Math.hypot(uv[0]-row.expected[i][0],uv[1]-row.expected[i][1]);if(!Number.isFinite(error))throw Error('Nonfinite actual projection');maximum=Math.max(maximum,error);count++;}}
assert(maximum<1e-8);assert.equal(exactSample([{timestamp_s:1,status:'supported_nonreference_candidate'}],1.01),null);assert.equal(displayPose({status:'withheld',pose:{R:[[1,0,0],[0,1,0],[0,0,1]],t:[0,0,1]}}),null);console.log(JSON.stringify({actual_saved_pose_rows:input.rows.length,actual_native_vertex_projections:count,maximum_pixel_residual:maximum}));
'''
    process=subprocess.run(['node','--input-type=module','-e',javascript],input=json.dumps({'points':points.tolist(),'rows':rows}),
                           text=True,capture_output=True,cwd=root,timeout=60,check=True)
    result=json.loads(process.stdout)
    result['status']='checked_actual_saved_projections'if result['actual_native_vertex_projections'] else'not_evaluable_no_displayable_pose'
    if not result['actual_native_vertex_projections']:result['maximum_pixel_residual']=None
    result.update(format='real-interval-viewer-math-audit',version=1,viewer_path=str(viewer_path.relative_to(root)),
        viewer_sha256=sha(viewer_path),native_glb_sha256=sha(asset),withheld_rows_hidden=withheld,
        unverified_reference_rows_in_math_check=references,geometry_or_pose_changed=False,
        physical_pose_accuracy_validated=False,GPU_pixels_verified=False,
        modules=[{'path':p,'sha256':sha(root/p)}for p in ['workspace/motion-math.mjs','workspace/geometry-camera.mjs','scripts/audit_motion_viewer.py']])
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--viewer',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=audit(args.viewer);output=Path(args.output)
    if output.exists():raise ValueError('Preserve existing actual audit')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
