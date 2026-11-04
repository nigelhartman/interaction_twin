"""New CPU SAM2 temporal observations and conditional native-pose hypotheses.

No original masks/poses, assets, owner state or inference guards are modified.
Forward and backward propagation begin from distinct saved SAM source masks.
These are fallible model observations, explicitly used as fitting input rather
than mislabeled independent feature measurements or physical validation.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

import source_temporal_recovery_v1 as T
import sam3d_common_scene as S
import sam3d_mechanisms as M
from native_appearance import load_native_glb,render_native_mask

ROOT=T.ROOT
CHECKPOINT=ROOT/'models/sam2.1_hiera_tiny.pt'
CONFIG='configs/sam2.1/sam2.1_hiera_t.yaml'


def prepare_mask_episode(hypothesis_ref,episode_id,output,*,root=ROOT):
    """Stage a small texture-poor rigid part; never infer an unknown joint."""
    root=Path(root).resolve();out=root/output;T.require(not out.exists(),'New SAM-only episode preparation required');out.mkdir(parents=True)
    d=T.read_checked(hypothesis_ref,root);scene=T.read_checked(d['baseline_scene_ref'],root);program=T.read_checked(d['evidence_program_ref'],root)
    group,episode=next((g,e)for g in d['groups']for e in g['episodes']if e['id']==episode_id)
    T.require(len(group['component_ids'])==1,'SAM-only preparation cannot guess an attached child angle')
    a,b=episode['anchor_before'],episode['anchor_after'];ra,ta,sa=T.matrix_parts(a['camera_from_native_mesh']);rb,tb,sb=T.matrix_parts(b['camera_from_native_mesh'])
    angular_gap=float(Rotation.from_matrix(rb@ra.T).magnitude())
    T.require(b['timestamp_s']-a['timestamp_s']<=1.5 and angular_gap<=math.radians(15)and max(ta[2]/tb[2],tb[2]/ta[2])<=1.35,'Known-part anchors do not justify a same-view bounded SAM hypothesis')
    frames,_,_,_=T.video_frames(program,program['evidence'],scene['camera'],root=root,policy=d['policy'])
    proofs=T.read_checked(d['flow_proofs_ref'],root);episode['driver_before']=copy.deepcopy(a);episode['driver_after']=copy.deepcopy(b)
    episode.update(view_change_break=False,constant_joint_states=[],reason='awaiting_independent_source_SAM_masks_for_texture_poor_part',status='withheld',predictions=[])
    for frame in frames:
        if not a['timestamp_s']<frame.timestamp_s<b['timestamp_s']:continue
        pid=f'{episode_id}@{frame.timestamp_s:.6f}'
        episode['predictions'].append({'timestamp_s':frame.timestamp_s,'source_sample_index':frame.original_index,'source_RGB':frame.evidence,
            'status':'withheld','reason':'awaiting_independent_source_mask_hypothesis','component_poses':{},'joint_states':[],
            'original_unknown_row_preserved':True,'flow_proof_id':pid,'method':'texture_poor_single_rigid_part_pending_source_mask_conditioning'})
        proofs['rows'].append({'id':pid,'timestamp_s':frame.timestamp_s,'target_RGB':frame.evidence,
            'past':{'status':'withheld','reason':'no_reliable_native_surface_corner_tracks'},'future':{'status':'withheld','reason':'no_reliable_native_surface_corner_tracks'},
            'driver_before':group['root_component_id'],'driver_after':group['root_component_id'],'agreement':None})
    T.write_new(out/'flow-proofs.json',proofs);d['flow_proofs_ref']=T.ref(out/'flow-proofs.json',root)
    d['parent_motion_hypothesis_ref']=hypothesis_ref;d['additional_implementation_fingerprints']=[T.ref(Path(__file__),root)]
    T.write_new(out/'hypotheses.json',d)
    return T.ref(out/'hypotheses.json',root)


def build_masks(hypothesis_ref,episode_id,output,*,root=ROOT):
    root=Path(root).resolve();out=root/output;T.require(not out.exists(),'New SAM temporal output required');out.mkdir(parents=True)
    implementation=T.ref(Path(__file__),root);hyp=T.read_checked(hypothesis_ref,root);program=T.read_checked(hyp['evidence_program_ref'],root);scene=T.read_checked(hyp['baseline_scene_ref'],root)
    group,episode=next((g,e)for g in hyp['groups']for e in g['episodes']if e['id']==episode_id)
    T.require(episode['driver_before']['component_id']==episode['driver_after']['component_id']and not episode['view_change_break'],'SAM temporal propagation may not bridge a known surface-face change')
    a,b=episode['driver_before'],episode['driver_after'];frames,k,size,video=T.video_frames(program,program['evidence'],scene['camera'],root=root,policy=hyp['policy'])
    selected=[f for f in frames if a['timestamp_s']<=f.timestamp_s<=b['timestamp_s']]
    T.require(3<=len(selected)<=96 and b['timestamp_s']-a['timestamp_s']<=4.,'Bounded SAM source-consistent clip required')
    inputs=out/'frames';inputs.mkdir();capture=cv2.VideoCapture(str(root/video['path']));records=[]
    try:
        for i,frame in enumerate(selected):
            if frame.evidence.get('image_ref'):rgb=np.asarray(Image.open(root/frame.evidence['image_ref']['path']).convert('RGB'))
            else:
                capture.set(cv2.CAP_PROP_POS_FRAMES,frame.evidence['decoded_frame_index']);ok,bgr=capture.read();T.require(ok,'Exact SAM source frame cannot be decoded');rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            rgb=cv2.resize(rgb,size,interpolation=cv2.INTER_AREA);path=inputs/f'{i:06d}.jpg';Image.fromarray(rgb).save(path,quality=95)
            records.append({'frame_index':i,'timestamp_s':frame.timestamp_s,'source_sample_index':frame.original_index,'source_RGB':frame.evidence,'rgb_ref':T.ref(path,root)})
    finally:capture.release()
    checkpoint=T.ref(CHECKPOINT,root)
    plan={'format':'interaction-twin-bidirectional-local-SAM-temporal-plan','version':1,'hypothesis_ref':hypothesis_ref,
        'episode_id':episode_id,'source_ref':video,'evidence_program_ref':hyp['evidence_program_ref'],'baseline_scene_ref':hyp['baseline_scene_ref'],
        'anchor_before':a,'anchor_after':b,'frames':records,'checkpoint_fingerprint':checkpoint,
        'implementation_fingerprint':implementation,'model':'Meta SAM2.1 Tiny','config':CONFIG,'device':'cpu',
        'manual_points_boxes_masks_or_poses':False,'new_hosted_API_or_remote_GPU_calls':0,
        'future_and_past_masks_are_independent_propagations':True,'minimum_forward_backward_mask_IoU':.72,
        'physical_accuracy_verified':False,'mask_observations_are_ground_truth':False}
    T.write_new(out/'plan.json',plan)
    import torch
    from sam2.build_sam import build_sam2_video_predictor
    torch.set_num_threads(4);started=time.monotonic()
    predictor=build_sam2_video_predictor(CONFIG,str(CHECKPOINT),device='cpu',apply_postprocessing=False)
    masks={}
    for label,anchor,index,reverse in [('past',a,0,False),('future',b,len(records)-1,True)]:
        seed=np.asarray(Image.open(root/anchor['source_mask_ref']['path']))>0
        seed=cv2.resize(seed.astype(np.uint8),size,interpolation=cv2.INTER_NEAREST)>0
        folder=out/label;folder.mkdir();direction={}
        with torch.inference_mode():
            state=predictor.init_state(str(inputs),offload_video_to_cpu=True,offload_state_to_cpu=True)
            predictor.add_new_mask(state,frame_idx=index,obj_id=1,mask=seed)
            for number,ids,logits in predictor.propagate_in_video(state,start_frame_idx=index,reverse=reverse):
                binary=(logits[list(ids).index(1),0]>0).cpu().numpy()
                T.require(binary.shape==seed.shape,'SAM source raster changed')
                path=folder/f'{number:06d}.png';Image.fromarray(binary.astype(np.uint8)*255).save(path);direction[int(number)]={'mask':binary,'ref':T.ref(path,root)}
        masks[label]=direction;del state
    del predictor
    consensus=out/'consensus';consensus.mkdir();results=[]
    for row in records:
        i=row['frame_index'];first=masks['past'][i]['mask'];second=masks['future'][i]['mask']
        overlap=int((first&second).sum());union=int((first|second).sum());iou=overlap/max(1,union)
        accepted=iou>=.72 and overlap>=64;consensus_ref=None
        if accepted:
            path=consensus/f'{i:06d}.png';Image.fromarray((first&second).astype(np.uint8)*255).save(path);consensus_ref=T.ref(path,root)
        results.append({**row,'past_mask_ref':masks['past'][i]['ref'],'future_mask_ref':masks['future'][i]['ref'],
            'forward_backward_mask_IoU':float(iou),'status':'estimated'if accepted else'withheld','consensus_mask_ref':consensus_ref,
            'consensus':'intersection_of_independent_source_mask_propagations','model_observation_not_ground_truth':True})
    report={'format':'interaction-twin-bidirectional-local-SAM-temporal-observations','version':1,'plan_ref':T.ref(out/'plan.json',root),
        'source_ref':video,'evidence_program_ref':hyp['evidence_program_ref'],'baseline_scene_ref':hyp['baseline_scene_ref'],
        'episode_id':episode_id,'rows':results,'actual_local_SAM2_CPU_states':2,'new_hosted_API_or_remote_GPU_calls':0,
        'physical_accuracy_verified':False,'target_masks_will_be_fitting_inputs_not_independent_tests':True,'wall_seconds':time.monotonic()-started}
    T.require(T.ref(Path(__file__),root)==implementation,'Executing SAM temporal helper changed')
    T.write_new(out/'observations.json',report)
    return T.ref(out/'observations.json',root)


def fit_consensus_mask(vertices,mask,k,prior,*,maximum_rotation_degrees=3.):
    """Bounded source-mask fitting, with unchanged native scale and joint."""
    rotation,translation,scale=T.matrix_parts(prior);target=S.mask_shape(mask)
    sparse=vertices[np.linspace(0,len(vertices)-1,min(1800,len(vertices)),dtype=int)]
    candidates=[rotation]
    for axis in np.eye(3):
        for degrees in [-maximum_rotation_degrees,maximum_rotation_degrees]:candidates.append(Rotation.from_rotvec(axis*math.radians(degrees)).as_matrix()@rotation)
    scored=[]
    for r in candidates:
        result=S.projected_pose(sparse,r,scale,k,target,translation[2])
        if result is None:continue
        t,uv,_=result;delta=float(Rotation.from_matrix(r@rotation.T).magnitude())
        # Monocular silhouette depth is ambiguous. Keep the source-supported
        # neighbouring pose gauge within eight percent during this correction.
        t[2]=float(np.clip(t[2],translation[2]/1.08,translation[2]*1.08))
        center_delta=t[:2]-translation[:2];maximum=.08*translation[2]
        if np.linalg.norm(center_delta)>maximum:t[:2]=translation[:2]+center_delta*maximum/np.linalg.norm(center_delta)
        camera=sparse@r.T*scale+t;homogeneous=camera@k.T;uv=homogeneous[:,:2]/homogeneous[:,2,None]
        loss,iou=S.silhouette_score(uv,target);scored.append((loss+.015*delta,r,t,float(iou),delta))
    T.require(scored,'No positive-depth conditional source-mask pose')
    _,r,t,iou,delta=min(scored,key=lambda x:x[0]);matrix=np.eye(4);matrix[:3,:3]=r*scale;matrix[:3,3]=t
    return matrix,{'candidate_count':len(scored),'convex_fit_iou_diagnostic':iou,'rotation_delta_from_neighbor_prior_rad':delta,
        'maximum_rotation_search_rad':math.radians(maximum_rotation_degrees),'native_uniform_scale_unchanged':scale,'target_mask_used_for_fitting':True}


def augment(hypothesis_ref,observations_ref,output,*,root=ROOT):
    root=Path(root).resolve();out=root/output;T.require(not out.exists(),'New conditional SAM motion derivative required');out.mkdir(parents=True)
    implementation=T.ref(Path(__file__),root);hyp=T.read_checked(hypothesis_ref,root);scene=T.read_checked(hyp['baseline_scene_ref'],root)
    observations=T.read_checked(observations_ref,root)
    T.require(all(observations[k]==hyp[k]for k in ['source_ref','evidence_program_ref','baseline_scene_ref']),'SAM motion observations belong to another exact source/scene')
    report_rows={round(r['timestamp_s'],6):r for r in observations['rows']};group,episode=next((g,e)for g in hyp['groups']for e in g['episodes']if e['id']==observations['episode_id'])
    T.require(not episode['view_change_break'],'Conditional mask poses may not hide a view-change break')
    parts={c['id']:c for c in scene['components']};meshes={cid:load_native_glb(root/parts[cid]['asset']['path'])for cid in group['component_ids']}
    joints=[j for j in scene.get('mechanics',{}).get('joints',[])if j['parent_id']==group['root_component_id']]
    q={s['id']:s['q_rad']for s in episode['constant_joint_states']};assembly=[meshes[group['root_component_id']].vertices]
    for joint in joints:
        relative=M.forward(np.eye(4),joint['parent_from_joint'],joint['axis'],q[joint['id']],joint['joint_from_child_zero'])
        assembly.append(meshes[joint['child_id']].vertices@relative[:3,:3].T+relative[:3,3])
    vertices=np.concatenate(assembly);k=np.asarray(hyp['camera']['K']);size=(hyp['camera']['width'],hyp['camera']['height'])
    proofs=T.read_checked(hyp['flow_proofs_ref'],root);by_proof={p['id']:p for p in proofs['rows']};accepted=0
    confirmed=[p for p in episode['predictions']if p['status']=='estimated'and p.get('support')in {'past_and_future_RGB_agree','one_sided_near_anchor_RGB'}]
    def render_poses(poses):
        full=np.zeros((size[1],size[0]),bool)
        for cid,pose in poses.items():
            r,t,s=T.matrix_parts(pose);full|=render_native_mask(meshes[cid],k,r,t,*size,scale=s)['mask']
        return full
    def coupled(matrix):
        poses={group['root_component_id']:matrix.tolist()}
        for joint in joints:poses[joint['child_id']]=M.forward(matrix,joint['parent_from_joint'],joint['axis'],q[joint['id']],joint['joint_from_child_zero']).tolist()
        return poses
    for prediction in episode['predictions']:
        if prediction['status']=='estimated':continue
        row=report_rows.get(round(prediction['timestamp_s'],6))
        if row is None or row['status']!='estimated':continue
        mask=np.asarray(Image.open(root/row['consensus_mask_ref']['path']))>0
        if mask.shape!=(size[1],size[0]):mask=cv2.resize(mask.astype(np.uint8),size,interpolation=cv2.INTER_NEAREST)>0
        before=[p for p in confirmed if p['timestamp_s']<prediction['timestamp_s']];after=[p for p in confirmed if p['timestamp_s']>prediction['timestamp_s']]
        original_anchors=not before or not after
        if original_anchors:
            first={'timestamp_s':episode['anchor_before']['timestamp_s'],'component_poses':{group['root_component_id']:episode['anchor_before']['camera_from_native_mesh']},'flow_proof_id':'original-scene-before'}
            last={'timestamp_s':episode['anchor_after']['timestamp_s'],'component_poses':{group['root_component_id']:episode['anchor_after']['camera_from_native_mesh']},'flow_proof_id':'original-scene-after'}
        else:first=max(before,key=lambda p:p['timestamp_s']);last=min(after,key=lambda p:p['timestamp_s'])
        if last['timestamp_s']-first['timestamp_s']>1.5:continue
        alpha=(prediction['timestamp_s']-first['timestamp_s'])/(last['timestamp_s']-first['timestamp_s'])
        prior=T.blend(first['component_poses'][group['root_component_id']],last['component_poses'][group['root_component_id']],alpha)
        poses=coupled(prior);full=render_poses(poses);overlap=int((full&mask).sum());prior_iou=overlap/max(1,int((full|mask).sum()));prior_coverage=overlap/max(1,int(mask.sum()))
        if prior_iou>=.72 and prior_coverage>=.80:
            matrix=prior;details={'candidate_count':1,'convex_fit_iou_diagnostic':None,'rotation_delta_from_neighbor_prior_rad':0.,
                'maximum_rotation_search_rad':0.,'native_uniform_scale_unchanged':T.matrix_parts(prior)[2],'target_mask_used_for_fitting':True,
                'pose_selection':'nearest_RGB_confirmed_SE3_prediction_verified_by_new_bidirectional_SAM_masks'}
        else:
            try:matrix,details=fit_consensus_mask(vertices,mask,k,prior)
            except ValueError:continue
            poses=coupled(matrix);full=render_poses(poses)
        overlap=int((full&mask).sum());iou=overlap/max(1,int((full|mask).sum()));coverage=overlap/max(1,int(mask.sum()))
        if iou<.60 or coverage<.75:continue
        details.update({'observations_ref':observations_ref,'consensus_mask_ref':row['consensus_mask_ref'],'source_RGB_ref':row['rgb_ref'],
            'forward_backward_mask_IoU':row['forward_backward_mask_IoU'],'full_triangle_conditional_mask_iou':float(iou),
            'conditional_foreground_coverage':float(coverage),'independent_validation':False,'prior_root_matrix':prior.tolist(),
            'fitted_root_matrix':matrix.tolist(),'same_source_view_segment':True,
            'prior_anchor_timestamps_s':[first['timestamp_s'],last['timestamp_s']],
            'maximum_source_confirmed_interpolation_gap_s':1.5,'prior_full_triangle_mask_iou':float(prior_iou)})
        if original_anchors:details['original_prior_anchor_rows']=[episode['anchor_before'],episode['anchor_after']]
        else:details.update(prior_source_motion_ref=hypothesis_ref,prior_anchor_proof_ids=[first['flow_proof_id'],last['flow_proof_id']])
        proof=by_proof[prediction['flow_proof_id']];proof['conditional_SAM_mask_fit']=details
        prediction.update(status='estimated',reason=None,component_poses=poses,joint_states=copy.deepcopy(episode['constant_joint_states']),
            support='source_SAM_mask_conditioned_pose',method='bidirectional_local_SAM2_CPU_fixed_native_scale_joint_pose',
            view_segment_id=episode['id']+':same_source_view',evidence_kind='conditional_model_mask_fit')
        accepted+=1
    T.write_new(out/'flow-proofs.json',proofs)
    hyp['flow_proofs_ref']=T.ref(out/'flow-proofs.json',root);hyp['parent_motion_hypothesis_ref']=hypothesis_ref
    hyp['additional_implementation_fingerprints']=[implementation];hyp['source_SAM_temporal_observations_ref']=observations_ref
    T.require(T.ref(Path(__file__),root)==implementation,'Executing conditional SAM temporal helper changed')
    T.write_new(out/'hypotheses.json',hyp)
    return {'hypotheses':T.ref(out/'hypotheses.json',root),'new_conditional_mask_estimates':accepted}
