#!/usr/bin/env python3
"""Private source27 SAM refresh → fixed hinge fit; never approve or publish."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import recover_occluded_laptop_20261103_v1 as F
import source_joint_gap_refinement_v1 as J
import object_program as O
import mechanism_pipeline as P
import sam3d_mechanisms as M
from native_appearance import load_native_glb,render_native_mesh

OUTPUT='data/source-joint-gap-refinements/laptop-opening27-20261103-v1'
BASE='data/occluded-hinge-recoveries/laptop-20261103-v1'
MASK='data/source-mask-refinements/laptop-opening-27-20261103-v1'
CODE=['source_joint_gap_refinement_v1.py','occluded_hinge_recovery_v1.py','scripts/refine_laptop_opening27_20261103_v1.py']


def code_ref(name):
    path=ROOT/name;F.R.require(path.is_file()and not path.is_symlink()and path.resolve().is_relative_to(ROOT),'Ordinary implementation required')
    return {'path':name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}


def plan():
    result=json.loads((ROOT/BASE/'result.json').read_text());scene=json.loads(F.checked(result['scene_ref']).read_text());program=json.loads(F.checked(result['object_program_ref']).read_text());O.validate(program,root=ROOT)
    F.R.require(scene['components'][0]['poses'][26]['status']=='unknown','Source27 must remain a failed original observation before this attempt')
    return {'format':'interaction-twin-fixed-hinge-gap-refinement-plan','version':1,'source_sample_index':27,'timestamp_s':18.070422,
            'baseline_scene':result['scene_ref'],'baseline_object_program':result['object_program_ref'],'new_SAM_packet':F.ref(ROOT/MASK/'plan.json'),
            'new_Muse_source_observation':F.ref(ROOT/F.PACKET/'attempt/model-result.json'),'policy':J.POLICY,
            'implementation':[code_ref(name)for name in CODE],'manual_geometry_or_poses_used':False,'automatic_approval':False,'automatic_publication':False,'new_api_calls':0,'new_gpu_calls':0}


def prepare():
    folder=ROOT/OUTPUT;F.R.require(not folder.exists(),'Fresh gap refinement output required');doc=plan();folder.mkdir(parents=True)
    for row in doc['implementation']:
        dest=folder/'implementation'/(Path(row['path']).name+'.txt');dest.parent.mkdir(exist_ok=True);dest.write_bytes((ROOT/row['path']).read_bytes())
    F.write(folder/'plan.json',doc);return F.ref(folder/'plan.json')


def metrics(target,predicted):
    return {'mask_iou':float(np.count_nonzero(target&predicted)/max(1,np.count_nonzero(target|predicted))),
            'source_foreground_coverage':float(np.count_nonzero(target&predicted)/max(1,np.count_nonzero(target)))}


def build():
    started=time.monotonic();folder=ROOT/OUTPUT;doc=plan();F.R.require(json.loads((folder/'plan.json').read_text())==doc,'Prepared source/fitting plan changed')
    F.R.require(not(folder/'decision.json').exists(),'One source27 fit per immutable plan')
    summary_ref=F.ref(ROOT/MASK/'attempt/summary.json');summary=json.loads(F.checked(summary_ref).read_text());F.R.require(summary['status']=='complete_accounted_candidate_masks','Exact fresh SAM request must complete first')
    F.R.require(summary['plan']==doc['new_SAM_packet'],'SAM source packet differs')
    receipt=json.loads(F.checked(summary['receipt']).read_text());F.R.require(receipt['status']=='complete_accounted'and receipt['model']=='sam-3.1','Accounted actual SAM3.1 masks required')
    source_observations=json.loads(F.checked(doc['new_Muse_source_observation']).read_text());observation=next(r for r in source_observations['frames']if r['sample_index']==27)
    F.R.require(observation['laptop_state']=='partially_open'and observation['visible_laptop_panel']=='both'and observation['hinge_seam_visible'],'Source27 needs two visible inside panels and source seam')
    scene=json.loads(F.checked(doc['baseline_scene']).read_text());parts={c['id']:c for c in scene['components']};joint=scene['mechanics']['joints'][0];masks={};mask_refs={}
    for instance in summary['instances']:
        if instance['track_id'] not in parts:continue
        F.R.require(instance['source']['sample_index']==27 and instance['source']['timestamp_s']==18.070422 and instance['source']['video_sha256']==scene['source']['sha256'],'SAM source identity changed')
        array=np.asarray(Image.open(F.checked(instance['mask'])))>0;F.R.require(array.shape==(960,722)and np.count_nonzero(array)>1000,'Nonempty original-raster fresh mask required')
        masks[instance['track_id']]=array;mask_refs[instance['track_id']]=instance['mask']
    F.R.require(set(masks)==set(parts),'Both new independently prompted panel masks required')
    meshes={cid:load_native_glb(F.checked(part['asset']))for cid,part in parts.items()};normals={}
    for cid,mesh in meshes.items():
        normal=np.linalg.svd(mesh.vertices-mesh.vertices.mean(0),full_matrices=False)[2][-1];anchor=np.asarray(parts[cid]['poses'][43]['camera_from_native_mesh']);normal*=np.sign((anchor[:3,:3]@normal)@(-anchor[:3,3]));normals[cid]=normal
    seam=np.asarray(observation['hinge_seam_uv'])*[721,959];prior=parts['laptop_base']['poses'][27]['camera_from_native_mesh']
    fitted=J.fit(meshes['laptop_base'].vertices,meshes['laptop_lid'].vertices,masks['laptop_base'],masks['laptop_lid'],scene['camera'],joint,prior,normals['laptop_base'],normals['laptop_lid'],seam)
    w,h=241,320;k=np.asarray(scene['camera']['K'],float).copy();k[0]*=w/722;k[1]*=h/960;k[0,2]+=(w/722-1)/2;k[1,2]+=(h/960-1)/2
    depth=np.full((h,w),np.inf);owner=np.full((h,w),-1,int);rgb=np.full((h,w,3),246,np.uint8)
    for n,(cid,key)in enumerate([('laptop_base','parent_matrix'),('laptop_lid','child_matrix')]):
        rotation,translation,scale=M.rotation_pose(fitted[key]);render=render_native_mesh(meshes[cid],k,rotation,translation,w,h,scale=scale);closer=render['mask']&(render['depth']<depth);owner[closer]=n;rgb[closer]=render['rgb'][closer];depth[closer]=render['depth'][closer]
    targets={cid:cv2.resize(mask.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)>0 for cid,mask in masks.items()}
    scores={'parent':metrics(targets['laptop_base'],owner==0),'child':metrics(targets['laptop_lid'],owner==1),'union':metrics(targets['laptop_base']|targets['laptop_lid'],np.isfinite(depth))}
    accepted=J.gate(scores,fitted)
    evidence=json.loads(F.checked(scene['original_program']).read_text());source=evidence['evidence'][26];proof=Image.new('RGB',(w*3,h+58),'#f6f7fb');proof.paste(Image.open(ROOT/source['frame_path']).resize((w,h)),(0,58));proof.paste(Image.fromarray(rgb),(w*2,58));draw=ImageDraw.Draw(proof)
    draw.text((8,6),f'Source27 / actual SAM3.1 + fixed native hinge | {"candidate retained"if accepted else"withheld"}',fill='#151923');draw.text((8,25),f'Source RGB | prior unknown | automatic q={fitted["q_rad"]:.3f}rad / physical accuracy unverified',fill='#3766ed');proof.save(folder/'source27-full-triangle-proof.jpg')
    decision={'format':'interaction-twin-fixed-hinge-gap-refinement','version':1,'accepted':accepted,'plan':F.ref(folder/'plan.json'),'baseline_scene':doc['baseline_scene'],'SAM_summary':summary_ref,'fitting':fitted,'full_original_triangle_metrics':scores,
              'source_proof':F.ref(folder/'source27-full-triangle-proof.jpg'),'same_mask_fit_diagnostic_not_heldout_accuracy':True,'original_native_assets_unchanged':True,'manual_geometry_or_poses_used':False,'new_api_calls':0,'new_gpu_calls':0,'physical_accuracy_verified':False,'wall_seconds':time.monotonic()-started}
    if accepted:
        research=copy.deepcopy(scene);research.pop('mechanics');research['fitter_version']='new_SAM_source27_fixed_hinge_kinematic_gap_refinement_v1'
        extra=[summary_ref,summary['receipt'],summary['native_result'],doc['new_Muse_source_observation'],doc['baseline_scene'],doc['baseline_object_program'],F.ref(folder/'plan.json'),*mask_refs.values()]
        extra += [F.ref(folder/'implementation'/(Path(name).name+'.txt'))for name in CODE]
        by_path={r['path']:r for r in [*research['inputs'],*extra]};research['inputs']=[by_path[k]for k in sorted(by_path)]
        research['research_derivation']={'kind':'fresh_Muse_box_SAM_source27_fixed_native_hinge_refinement','prior':scene['research_derivation'],'prior_research_input_signature':scene['input_signature'],'source_fit_plan':F.ref(folder/'plan.json'),'mask_summary':summary_ref,'new_source_mask_fit':fitted,'full_triangle_gate':scores,'original_assets_changed':False}
        research['input_signature']=hashlib.sha256(json.dumps({'inputs':research['inputs'],'research_derivation':research['research_derivation']},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        for cid,key in [('laptop_base','parent_matrix'),('laptop_lid','child_matrix')]:
            row=next(r for r in research['components']if r['id']==cid)['poses'][26];row.update(status='estimated',camera_from_native_mesh=fitted[key],method='actual_new_SAM31_mask_and_source_Muse_seam_fixed_joint_fit',reason='source27_new_visible_panel_masks_fit_through_retained_native_joint',source_mask=mask_refs[cid],accepted_motion_evidence=False)
        F.write(folder/'source-scene.json',research)
        mechanics=copy.deepcopy(scene['mechanics']);mechanics['source_scene']=F.ref(folder/'source-scene.json');mechanics['wall_seconds']=time.monotonic()-started;j=mechanics['joints'][0];j['samples'][26].update(status='estimated',q_rad=fitted['q_rad'],reason='fresh_SAM31_visible_base_lid_and_Muse_seam_fixed_joint_fit',accepted_motion_evidence=False)
        j['limits_rad']=[min(s['q_rad']for s in j['samples']if s['status']=='estimated'),max(s['q_rad']for s in j['samples']if s['status']=='estimated')]
        j['source_joint_gap_refinement']={'plan':F.ref(folder/'plan.json'),'SAM_summary':summary_ref,'sample_index':27,'timestamp_s':18.070422,'source_triangle_gate':scores,'physical_accuracy_verified':False}
        updated=copy.deepcopy(research);updated['mechanics']=mechanics;F.write(folder/'mechanics.json',mechanics);F.write(folder/'scene.json',updated)
        export=P.export_scene(F.ref(folder/'scene.json'),root=ROOT,publish=False)
        decision.update(scene_ref=F.ref(folder/'scene.json'),object_program_ref=export['object_program'],retained_supported_samples=sum(r['status']=='estimated'for r in updated['components'][0]['poses']),unknown_samples=sum(r['status']=='unknown'for r in updated['components'][0]['poses']))
    F.write(folder/'decision.json',decision);return decision


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','run']);parser.add_argument('--output',default=OUTPUT);args=parser.parse_args();OUTPUT=args.output;print(json.dumps(prepare()if args.command=='prepare'else build()),flush=True)
