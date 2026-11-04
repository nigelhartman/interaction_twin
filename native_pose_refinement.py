"""Guarded development-only constant parent-frame pose correction.

Full original triangles rank a bounded declarative pose search on explicitly
selected training samples. A fixed, separate DEVELOPMENT set decides whether to
retain the candidate. Final-test/heldout frames are never loaded as target data.
Unknown poses, native meshes/scales and joint frames/angles remain unchanged.
This is estimated alignment, not geometry, camera or physics verification.
"""
from __future__ import annotations
import copy,json,math,time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import sam3d_common_scene as S
import sam3d_mechanisms as M
import upgrade_bundles as U
from native_appearance import load_native_glb,render_native_mask

ROOT=Path(__file__).resolve().parent
CODE=['native_pose_refinement.py','native_scene_audit.py','native_appearance.py','alignment_render.py','sam3d_common_scene.py','sam3d_mechanisms.py']


def require(value,message):
    if not value:raise ValueError(message)


def checked(reference,root):
    path=U._local(reference['path'],root);require(S.ref(path,root)==reference,'Refinement evidence changed');return path


def occlusion_metrics(source,rendered,other):
    source=np.asarray(source,bool);rendered=np.asarray(rendered,bool);other=np.asarray(other,bool)
    require(source.shape==rendered.shape==other.shape,'Refinement masks use different raster frames')
    # An OTHER SAM component is only a possible occluder; retain the full raw
    # overlap separately. Do not discount disagreement inside the target mask.
    discounted=rendered&~(other&~source)
    def iou(a,b):
        union=np.count_nonzero(a|b);return float(np.count_nonzero(a&b)/union)if union else None
    return {'raw_mask_iou':iou(source,rendered),'occluder_accounted_mask_iou':iou(source,discounted),
        'known_other_mask_discounted_pixels':int(np.count_nonzero(rendered&~discounted)),
        'unmodeled_hand_occlusion_discounted':False}


def correction_gate(baseline,candidate,*,minimum_gain=.02,maximum_mean_raw_drop=.01,maximum_individual_raw_drop=.08):
    require(len(baseline)==len(candidate) and len(baseline)>=3,'At least3 fixed development frames required')
    require([r['source_index']for r in baseline]==[r['source_index']for r in candidate],'Development frame selection changed')
    raw_before=np.asarray([r['raw_mean_iou']for r in baseline]);raw_after=np.asarray([r['raw_mean_iou']for r in candidate])
    discounted_before=np.asarray([r['accounted_mean_iou']for r in baseline]);discounted_after=np.asarray([r['accounted_mean_iou']for r in candidate])
    require(np.isfinite(np.r_[raw_before,raw_after,discounted_before,discounted_after]).all(),'Finite development comparisons required')
    component_drops=[]
    for before,after in zip(baseline,candidate):
        require(set(before['components'])==set(after['components']),'Development components changed')
        component_drops.extend(float(before['components'][c]['raw_mask_iou'])-float(after['components'][c]['raw_mask_iou'])for c in before['components'])
    require(component_drops and np.isfinite(component_drops).all(),'Finite component comparisons required')
    gain=float(np.mean(discounted_after-discounted_before));raw_change=float(np.mean(raw_after-raw_before));largest_drop=float(max(component_drops))
    accepted=gain>=minimum_gain and raw_change>=-maximum_mean_raw_drop and largest_drop<=maximum_individual_raw_drop
    return {'accepted':bool(accepted),'accounted_mean_gain':gain,'raw_mean_change':raw_change,'largest_individual_raw_drop':largest_drop,
        'minimum_accounted_gain':minimum_gain,'maximum_mean_raw_drop':maximum_mean_raw_drop,'maximum_individual_raw_drop':maximum_individual_raw_drop,
        'selection_evidence':'fixed development samples only; final-test evidence excluded'}


def apply_correction(scene,parent_id,delta):
    """New scene in memory; every moving attached descendant uses original FK."""
    result=copy.deepcopy(scene);parts={c['id']:c for c in result['components']};old={c['id']:c for c in scene['components']}
    joints=result['mechanics']['joints'];children={j['child_id']for j in joints}
    require(parent_id in parts and parent_id not in children,'Only a known root assembly may receive this correction')
    M.rotation_pose(delta);require(np.isclose(np.linalg.det(np.asarray(delta)[:3,:3]),1.),'Correction cannot change native scale')
    affected={parent_id};pending=list(joints)
    while pending:
        next_level=[j for j in pending if j['parent_id']in affected]
        if not next_level:break
        for joint in next_level:affected.add(joint['child_id'])
        pending=[j for j in pending if j not in next_level]
    for row in parts[parent_id]['poses']:
        if row['status']=='estimated' and row['camera_from_native_mesh']is not None:
            row['camera_from_native_mesh']=(np.asarray(row['camera_from_native_mesh'])@delta).tolist()
            row.update(method='guarded_constant_parent_native_frame_correction',accepted_motion_evidence=False)
    ordered=[];resolved={parent_id};pending=[j for j in joints if j['child_id']in affected]
    while pending:
        next_level=[j for j in pending if j['parent_id']in resolved]
        require(next_level,'Cyclic or ambiguous correction assembly')
        for joint in next_level:ordered.append(joint);resolved.add(joint['child_id'])
        pending=[j for j in pending if j not in next_level]
    for joint in ordered:
        if joint['child_id']not in affected:continue
        parents={r['source_sample_index']:r for r in parts[joint['parent_id']]['poses']};angles={r['source_sample_index']:r for r in joint['samples']}
        for row in parts[joint['child_id']]['poses']:
            original=next(r for r in old[joint['child_id']]['poses']if r['source_sample_index']==row['source_sample_index']);p=parents[row['source_sample_index']];q=angles[row['source_sample_index']]
            if original['status']=='estimated' and p['status']==q['status']=='estimated':
                row['camera_from_native_mesh']=M.forward(p['camera_from_native_mesh'],joint['parent_from_joint'],joint['axis'],q['q_rad'],joint['joint_from_child_zero']).tolist()
                row.update(method='guarded_parent_correction_original_joint_FK',accepted_motion_evidence=False)
            else:row.update(status='unknown',camera_from_native_mesh=None)
        # Preserve historical fitter details, without presenting its previous
        # parent matrices as the final corrected scene transforms.
        if joint.get('shared_parent_refinement'):
            joint['shared_parent_refinement']['scope']='historical pre-correction fitting diagnostics; final poses are in scene components'
    return result


def propose(scene_ref,audit_ref,output,*,parent_id,training_samples,development_samples,root=ROOT,max_dimension=192):
    import native_scene_audit
    root=Path(root).resolve();start=time.monotonic();scene_path=checked(scene_ref,root);scene=U.read(scene_path);audit=native_scene_audit.validate_manifest(checked(audit_ref,root),root=root);program=U.read(checked(scene['original_program'],root))
    require(audit['scene_ref']==scene_ref and audit['interaction_ref']==scene['original_program'],'Audit refers to another candidate')
    allowed={r['source_index']for r in audit['samples']if r['role']=='development'};train=list(training_samples);dev=list(development_samples)
    require(all(type(i)is int for i in train+dev) and 1<=len(train)<=4 and 3<=len(dev)<=12 and len(set(train))==len(train) and len(set(dev))==len(dev) and not set(train)&set(dev),'Bounded disjoint training/development samples required')
    require(set(train+dev)<=allowed and not set(train+dev)&set(scene.get('evaluation_protocol',{}).get('held_out_sample_indices',[])),'Final-test or heldout targets cannot enter correction selection')
    require(type(max_dimension)is int and 128<=max_dimension<=384,'Bounded refinement raster required')
    out=Path(output);out=out if out.is_absolute()else root/out;require(out.is_relative_to(root/'data') and out.resolve()==out and not out.exists() and not any(p.is_symlink()for p in [out,*out.parents]),'Fresh ordinary-data correction output required')
    parts={c['id']:c for c in scene['components']};joints=scene['mechanics']['joints'];require(parent_id in parts and parent_id not in {j['child_id']for j in joints},'Known root parent required')
    affected={parent_id}
    while True:
        expanded=affected|{j['child_id']for j in joints if j['parent_id']in affected}
        if expanded==affected:break
        affected=expanded
    require(len(affected)>=2,'A connected assembly is required')
    meshes={i:load_native_glb(checked(parts[i]['asset'],root))for i in affected};evidence={r['source_index']:r for r in program['evidence']}
    for i in train+dev:
        require(all(next(r for r in parts[c]['poses']if r['source_sample_index']==i)['status']=='estimated'for c in affected),'Training/development assembly estimate unavailable')
    h0,w0=scene['source']['height'],scene['source']['width'];factor=min(1.,max_dimension/max(h0,w0));h,w=round(h0*factor),round(w0*factor);k=np.asarray(scene['camera']['K'],float).copy();k[0]*=w/w0;k[1]*=h/h0;k[0,2]+=(w/w0-1)/2;k[1,2]+=(h/h0-1)/2
    targets={};inputs={r['path']:r for r in [scene_ref,audit_ref,scene['original_program'],*[parts[c]['asset']for c in affected]]}
    for index in train+dev:
        frame=evidence[index];masks={}
        for observation in frame['observations']:
            if observation['gate_status']!='candidate_geometric_gate_only' or str(observation.get('audit_status','')).startswith('reject_'):continue
            reference={'path':observation['mask_path'],'sha256':observation['mask_sha256'],'bytes':U._local(observation['mask_path'],root).stat().st_size};path=checked(reference,root);inputs[reference['path']]=reference
            with Image.open(path)as image:
                pixels=np.asarray(image);require(pixels.ndim==2 and set(np.unique(pixels))<={0,1,255},'Binary SAM mask required');raw=pixels>0
            require(raw.shape==(h0,w0),'Refinement mask differs from original source frame')
            masks[observation['track_id']]=cv2.resize(raw.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)>0
        require(all(c in masks and np.count_nonzero(masks[c])>=4 for c in affected),'Visible source masks required for every corrected assembly component')
        targets[index]=masks
    def evaluate(delta,indices):
        proposed=apply_correction(scene,parent_id,delta);by_id={c['id']:c for c in proposed['components']};rows=[]
        for index in indices:
            renders={};depth=np.full((h,w),np.inf);owner=np.full((h,w),-1,int)
            for n,c in enumerate(sorted(affected)):
                pose=next(r for r in by_id[c]['poses']if r['source_sample_index']==index);rotation,translation,scale=M.rotation_pose(pose['camera_from_native_mesh']);render=render_native_mask(meshes[c],k,rotation,translation,w,h,scale=scale);renders[c]=render
                closer=render['mask']&(render['depth']<depth);owner[closer]=n;depth[closer]=render['depth'][closer]
            metrics={}
            for n,c in enumerate(sorted(affected)):
                other=np.zeros((h,w),bool)
                for identifier,mask in targets[index].items():
                    if identifier!=c:other|=mask
                metrics[c]=occlusion_metrics(targets[index][c],owner==n,other)
            rows.append({'source_index':index,'timestamp_s':evidence[index]['timestamp_s'],'raw_mean_iou':float(np.mean([m['raw_mask_iou']or 0 for m in metrics.values()])),
                'accounted_mean_iou':float(np.mean([m['occluder_accounted_mask_iou']or 0 for m in metrics.values()])),'components':metrics})
        return rows
    config={'parent_id':parent_id,'training_samples':train,'development_samples':dev,'max_dimension':max_dimension,'rotation_degrees':[-12.,0.,12.],
        'translation':'training-only union-mask projected pose; constant native-frame correction applied without development refit','unknown_policy':'preserve_unknown_no_fill','joint_policy':'unchanged_frames_axis_q_and_native_scale',
        'occluders':'only exact other source component SAM masks; raw IoU and full rendered geometry preserved; no guessed hands'}
    out.mkdir(parents=True);snapshots=[]
    for name in CODE:
        target=out/'implementation'/name;target.parent.mkdir(exist_ok=True);target.write_bytes((root/name).read_bytes());snapshots.append(S.ref(target,root))
    S.write(out/'selection-plan.json',{'format':'native-parent-correction-selection-plan','version':1,'scene':scene_ref,'audit':audit_ref,'config':config,'inputs':[inputs[p]for p in sorted(inputs)],'implementation_snapshots':snapshots,'final_test_target_pixels_used':False})
    anchor=train[0];parent=next(r for r in parts[parent_id]['poses']if r['source_sample_index']==anchor);baseline=np.asarray(parent['camera_from_native_mesh']);rotation,translation,scale=M.rotation_pose(baseline)
    assembly=[]
    for c in sorted(affected):
        pose=np.asarray(next(r for r in parts[c]['poses']if r['source_sample_index']==anchor)['camera_from_native_mesh']);vertices=meshes[c].vertices;vertices=vertices[np.linspace(0,len(vertices)-1,min(500,len(vertices)),dtype=int)]
        world=vertices@pose[:3,:3].T+pose[:3,3];assembly.append((world-translation)@rotation)
    assembly=np.concatenate(assembly);union=np.logical_or.reduce([targets[anchor][c]for c in affected]);target=S.mask_shape(union)
    candidates=[{'name':'baseline','delta':np.eye(4),'training':evaluate(np.eye(4),train)}]
    for axis_name,axis in zip('xyz',np.eye(3)):
        for angle in [-12.,12.]:
            proposed_rotation=Rotation.from_rotvec(axis*math.radians(angle)).as_matrix()@rotation;fit=S.projected_pose(assembly,proposed_rotation,1.,k,target,translation[2])
            if fit is None or not .8<=fit[0][2]/translation[2]<=1.25:continue
            matrix=np.eye(4);matrix[:3,:3]=proposed_rotation*scale;matrix[:3,3]=fit[0];delta=np.linalg.inv(baseline)@matrix
            if np.linalg.norm(delta[:3,3])>.35*np.linalg.norm(np.ptp(meshes[parent_id].vertices,axis=0)):continue
            candidates.append({'name':axis_name+('minus'if angle<0 else'plus')+'12','delta':delta,'training':evaluate(delta,train)})
    chosen=max(candidates,key=lambda c:(np.mean([r['accounted_mean_iou']for r in c['training']]),np.mean([r['raw_mean_iou']for r in c['training']]),c['name']=='baseline'))
    baseline_dev=evaluate(np.eye(4),dev);candidate_dev=evaluate(chosen['delta'],dev);gate=correction_gate(baseline_dev,candidate_dev)
    receipt={'format':'interaction-twin-native-parent-correction','version':1,'source_scene':scene_ref,'audit':audit_ref,'part_ledger':scene.get('part_ledger'),'config':config,
        'candidates':[{'name':c['name'],'delta_parent_native':c['delta'].tolist(),'training':c['training']}for c in candidates],
        'selected_training_candidate':chosen['name'],'development_baseline':baseline_dev,'development_candidate':candidate_dev,'retention_gate':gate,
        'decision':'private_candidate_accepted'if gate['accepted']else'baseline_retained','implementation_snapshots':snapshots,
        'final_test_target_pixels_used':False,'new_api_calls':0,'new_gpu_calls':0,'published':False,'physical_accuracy_verified':False,'wall_seconds':time.monotonic()-start}
    if gate['accepted']:
        corrected=apply_correction(scene,parent_id,chosen['delta']);mechanics=corrected['mechanics'];mechanics['source_scene']=scene['mechanics']['source_scene'];mechanics['implementation']=S.source_ref(root/'native_pose_refinement.py',root)
        mechanics['additional_implementations']=[S.source_ref(root/name,root)for name in ['alignment_render.py','native_appearance.py','native_scene_audit.py','sam3d_mechanisms.py']]
        mechanics['parent_pose_correction']={'parent_id':parent_id,'delta_parent_native':chosen['delta'].tolist(),'source_scene':scene_ref,'training_samples':train,'development_samples':dev,'final_test_used':False,'retention_gate':gate}
        corrected['inputs']=[*scene.get('inputs',[]),scene_ref,audit_ref];S.write(out/'mechanics.json',mechanics);S.write(out/'scene.json',corrected)
        receipt['candidate_scene']=S.ref(out/'scene.json',root);receipt['candidate_mechanics']=S.ref(out/'mechanics.json',root)
    S.write(out/'decision.json',receipt)
    return {'decision':receipt['decision'],'receipt':S.ref(out/'decision.json',root),'retention_gate':gate,'candidate_scene':receipt.get('candidate_scene'),'wall_seconds':receipt['wall_seconds']}
