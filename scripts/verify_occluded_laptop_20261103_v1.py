#!/usr/bin/env python3
"""Full native triangle and dense FK proof for a private closed-panel candidate."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import recover_occluded_laptop_20261103_v1 as F
import occluded_hinge_recovery_v1 as R
import object_program as O
import sam3d_mechanisms as M
from native_appearance import load_native_glb,render_native_mesh


def build(output=F.OUTPUT):
    start=time.monotonic();folder=ROOT/output;result_file=folder/'result.json'if(folder/'result.json').exists()else folder/'decision.json';result=json.loads(result_file.read_text());scene=json.loads(F.checked(result['scene_ref']).read_text())
    program=json.loads(F.checked(result['object_program_ref']).read_text());O.validate(program,root=ROOT)
    baseline=json.loads((ROOT/F.SCENE).read_text());source=json.loads(F.checked(scene['original_program']).read_text());parts={c['id']:c for c in scene['components']};oldparts={c['id']:c for c in baseline['components']}
    output=folder/'verification';R.require(not output.exists(),'Preserve prior verification');output.mkdir()
    node="""const fs=require('node:fs'),C=require('./showcase/aligned-scene-contract.js');const s=JSON.parse(fs.readFileSync(process.argv[1]));C.validateScene(s,s.source.sha256);const times=[...new Set([...Array.from({length:1001},(_,i)=>s.source.duration_s*i/1000),...s.components[0].poses.map(r=>r.timestamp_s)])].sort((a,b)=>a-b);process.stdout.write(JSON.stringify(times.map(t=>({t,...C.sceneAt(s,t,true)}))));"""
    dense=json.loads(subprocess.run(['node','-e',node,str(ROOT/result['scene_ref']['path'])],cwd=ROOT,capture_output=True,check=True,timeout=30).stdout)
    joint=scene['mechanics']['joints'][0];known=unknown=0;errors=[];gaps=[]
    for frame in dense:
        pp={p['id']:p for p in frame['parts']};pa=pp[joint['parent_id']]['camera_from_native_mesh'];ch=pp[joint['child_id']]['camera_from_native_mesh']
        if pa is None or ch is None:
            R.require(pa is ch is None,'Partial connected pose leaked at unknown clock');unknown+=1;continue
        known+=1
        # The rendered pose derives through the one fixed joint; hinge origins
        # must coincide across every interpolated source-clock observation.
        a=np.asarray(pa);b=np.asarray(ch)
        gaps.append(float(np.linalg.norm(a@np.r_[joint['parent_anchor_native'],1.]-b@np.r_[joint['child_anchor_native'],1.])))
        q=next(j for j in frame['joints']if j['id']==joint['id'])['q_rad']
        expected=M.forward(a,joint['parent_from_joint'],joint['axis'],q,joint['joint_from_child_zero'])
        errors.append(float(np.max(np.abs(b-expected))))
    R.require(known>100 and max(gaps)<1e-8 and max(errors)<1e-8,'Persistent attachment failed dense source clock')
    for identifier,part in parts.items():
        F.checked(part['asset']);R.require(part['asset']==oldparts[identifier]['asset'],'Native source asset changed')
        for old,new in zip(oldparts[identifier]['poses'],part['poses']):
            if old['status']=='estimated':R.require(new['camera_from_native_mesh']==old['camera_from_native_mesh'],'Original supported open matrix changed')
    meshes={c:load_native_glb(F.checked(p['asset']))for c,p in parts.items()};width,height=241,320;k=np.asarray(scene['camera']['K'],float).copy();k[0]*=width/scene['source']['width'];k[1]*=height/scene['source']['height']
    k[0,2]+=(width/scene['source']['width']-1)/2;k[1,2]+=(height/scene['source']['height']-1)/2
    rendered_rows=[];tiles=[]
    for index in [1,16,26,27,30,54,55,66]:
        e=source['evidence'][index-1];source_image=Image.open(ROOT/e['frame_path']).convert('RGB').resize((width,height));rgb=np.full((height,width,3),246,np.uint8);depth=np.full((height,width),np.inf)
        state=parts['laptop_base']['poses'][index-1]['status'];metrics={}
        if state=='estimated':
            for identifier,part in parts.items():
                matrix=part['poses'][index-1]['camera_from_native_mesh'];rotation,translation,scale=M.rotation_pose(matrix);render=render_native_mesh(meshes[identifier],k,rotation,translation,width,height,scale=scale)
                closer=render['mask']&(render['depth']<depth);rgb[closer]=render['rgb'][closer];depth[closer]=render['depth'][closer]
            observed=[]
            replacement=[part['poses'][index-1].get('source_mask')for part in parts.values()if part['poses'][index-1].get('method')=='actual_new_SAM31_mask_and_source_Muse_seam_fixed_joint_fit']
            for item in replacement:
                raw=np.asarray(Image.open(F.checked(item)))>0;observed.append(cv2.resize(raw.astype(np.uint8),(width,height),interpolation=cv2.INTER_NEAREST)>0)
            for observation in ([]if observed else e['observations']):
                if observation['track_id']not in parts or observation['gate_status']!='candidate_geometric_gate_only' or str(observation.get('audit_status','')).startswith('reject_'):continue
                raw=np.asarray(Image.open(ROOT/observation['mask_path']))>0;observed.append(cv2.resize(raw.astype(np.uint8),(width,height),interpolation=cv2.INTER_NEAREST)>0)
            if observed:
                target=np.logical_or.reduce(observed);predicted=np.isfinite(depth)
                metrics={'full_triangle_mask_iou':float(np.count_nonzero(target&predicted)/max(1,np.count_nonzero(target|predicted))),
                         'source_foreground_coverage':float(np.count_nonzero(target&predicted)/max(1,np.count_nonzero(target)))}
        tile=Image.new('RGB',(width*2,height+50),'#f6f7fb');tile.paste(source_image,(0,50));tile.paste(Image.fromarray(rgb),(width,50));draw=ImageDraw.Draw(tile)
        draw.text((8,6),f'Source {index} / {e["timestamp_s"]:.3f}s | {state}',fill='#151923')
        draw.text((8,24),'Source RGB | unchanged native assets + source-gated FK'if state=='estimated'else'Source RGB | withheld: insufficient pose evidence',fill='#3766ed')
        tiles.append(tile);rendered_rows.append({'source_sample_index':index,'timestamp_s':e['timestamp_s'],'status':state,'metrics':metrics,'full_original_triangles_rendered':state=='estimated','retrospective_same_mask_diagnostic_not_heldout':True})
        print(json.dumps(rendered_rows[-1]),flush=True)
    canvas=Image.new('RGB',(width*2*4,(height+50)*2),'white')
    for n,tile in enumerate(tiles):canvas.paste(tile,((n%4)*width*2,(n//4)*(height+50)))
    canvas.save(output/'source-full-triangle-proof.jpg')
    F.write(output/'dense-source-clock.json',dense)
    report={'format':'interaction-twin-closed-panel-recovery-verification','version':1,'scene_ref':result['scene_ref'],'object_program_ref':result['object_program_ref'],
            'object_program_validated':True,'dense_fk':{'clock_count':len(dense),'known_connected_count':known,'unknown_connected_count':unknown,'maximum_native_anchor_gap':max(gaps),'maximum_matrix_error':max(errors)},
            'original_supported_matrices_unchanged':True,'original_native_assets_unchanged':True,'source_frames':rendered_rows,'source_proof':F.ref(output/'source-full-triangle-proof.jpg'),
            'physical_accuracy_verified':False,'contact_and_interpenetration_verified':False,'new_api_calls':0,'new_gpu_calls':0,'wall_seconds':time.monotonic()-start}
    F.write(output/'verification.json',report);return {'verification_ref':F.ref(output/'verification.json'),'source_proof':report['source_proof'],'dense_fk':report['dense_fk']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default=F.OUTPUT);a=p.parse_args();print(json.dumps(build(a.output)),flush=True)
