#!/usr/bin/env python3
"""Unpublished camera hypotheses: immutable geometry, shared focal multiplier.

These conditional monocular fits do not identify a physical camera. No candidate
scene or existing camera is changed. Holdout losses cannot change thresholds.
"""
from pathlib import Path
import copy,io,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import native_assembly_refinement_v1 as N
import planar_assembly_calibration_v1 as C
import sam3d_mechanisms as M
from native_appearance import load_native_glb,render_native_mask

ROOT=Path(__file__).resolve().parents[1]
def run():
    out=ROOT/'data/laptop-camera-hypotheses-20261104-v1';out.mkdir(exist_ok=False)
    doc_ref=N.ref(ROOT/'data/laptop-mating-20261104-v1/final-candidate-v2/declared-geometry.json',ROOT);doc=json.loads(N.checked(doc_ref,ROOT));scene=json.loads(N.checked(doc['scene_ref'],ROOT));audit=json.loads(N.checked(doc['source_alignment_audit_ref'],ROOT));report=json.loads(N.checked(doc['refinement_report_ref'],ROOT));joint=scene['mechanics']['joints'][0];components={c['id']:c for c in scene['components']};meshes={i:load_native_glb(N.checked(c['asset'],ROOT))for i,c in components.items()if i in {'laptop_base','laptop_lid'}}
    rows={r['source_sample_index']:r for r in audit['rows']};policy={'focal_multipliers':[.75,1.,1.25,1.5],'training_samples':[1,27,44],'heldout_samples':[16,26,30,54,66],'geometry_assets_joint_frames_and_relative_scale':'unchanged_exact_approved','camera_is_physical_calibration':False,'selection_rule':'training_mean_full_triangle_union_IoU_only; any heldout regression greater than .05 rejects; report_only_no_adoption','same_focal_multiplier_all_sources':True}
    N.write(out/'plan.json',{'format':'interaction-twin-camera-hypothesis-plan','version':1,'baseline_derived_assembly_ref':doc_ref,'policy':policy});results=[]
    for factor in policy['focal_multipliers']:
        camera=copy.deepcopy(scene['camera']);camera['K'][0][0]*=factor;camera['K'][1][1]*=factor;items=[]
        for n in policy['training_samples']+policy['heldout_samples']:
            row=rows[n];pm=np.zeros((960,722),bool);cm=pm.copy()
            for b in row['source_mask_refs']:
                v=np.array(__import__('PIL').Image.open(io.BytesIO(N.checked(b['artifact'],ROOT))))>0
                if b['component_id']=='laptop_base':pm=v
                else:cm=v
            prior=components['laptop_base']['poses'][n-1]['camera_from_native_mesh'];f=C.fit_sample(meshes['laptop_base'].vertices,meshes['laptop_lid'].vertices,pm,cm,camera,joint['parent_from_joint'],joint['axis'],joint['joint_from_child_zero'],prior,np.array(report['calibration']['parent_interior_normal']),np.array(report['calibration']['child_interior_normal']),closed=row['closed'],visible_panel=row['visible_panel'],quad=np.array(row['source_quad'])if row.get('source_quad')else None)
            width,height=241,320;k=np.array(camera['K']);k[0]*=width/722;k[1]*=height/960;k[0,2]+=(width/722-1)/2;k[1,2]+=(height/960-1)/2
            rendered={}
            for i,key in [('laptop_base','parent'),('laptop_lid','child')]:
                r,t,sc=M.rotation_pose(f[key]);rendered[i]=render_native_mask(meshes[i],k,r,t,width,height,scale=sc)['mask']
            masks={i:cv2.resize(v.astype(np.uint8),(width,height),interpolation=cv2.INTER_NEAREST)>0 for i,v in [('laptop_base',pm),('laptop_lid',cm)]};union=masks['laptop_base']|masks['laptop_lid'];pred=rendered['laptop_base']|rendered['laptop_lid'];iou=lambda a,b:float((a&b).sum()/max(1,(a|b).sum()))
            item={'source_sample_index':n,'role':'training'if n in policy['training_samples']else'heldout','focal_multiplier':factor,'conditional_union_iou':iou(pred,union),'visible_part_iou':{i:iou(rendered[i],v)for i,v in masks.items()if v.any()},'parent_matrix':f['parent'].tolist(),'child_matrix':f['child'].tolist(),'q_rad':f['q'],'source_mask_refs':row['source_mask_refs']};items.append(item);print(factor,n,item['conditional_union_iou'],flush=True)
        results.append({'focal_multiplier':factor,'training_mean_union_iou':float(np.mean([r['conditional_union_iou']for r in items if r['role']=='training'])),'samples':items});N.write(out/f'hypothesis-{factor:.2f}.json',results[-1])
    selected=max(results,key=lambda r:r['training_mean_union_iou']);baseline=next(r for r in results if r['focal_multiplier']==1);base={r['source_sample_index']:r for r in baseline['samples']};held=[{'source_sample_index':r['source_sample_index'],'delta_from_same_fitter_focal1':r['conditional_union_iou']-base[r['source_sample_index']]['conditional_union_iou']}for r in selected['samples']if r['role']=='heldout'];passed=all(r['delta_from_same_fitter_focal1']>=-.05 for r in held)
    result={'format':'interaction-twin-camera-hypothesis-report','version':1,'plan_ref':N.ref(out/'plan.json',ROOT),'policy':policy,'results':results,'training_selected_multiplier':selected['focal_multiplier'],'heldout_changes':held,'heldout_gate_passed':passed,'adopted':False,'physical_camera_identified':False,'physical_accuracy_verified':False,'new_api_calls':0,'new_gpu_calls':0,'limitation':'Conditional same-mask fits compare camera/pose ambiguity. They are not independent reconstruction accuracy. Current approved camera/geometry/poses remain unchanged.'};N.write(out/'report.json',result)
if __name__=='__main__':run()
