"""Bounded independently initialized, mask-constrained motion episodes.

Native generated geometry stays byte-identical. Each short episode is a separate
reference-conditioned hypothesis, never a recovered assembly or physical pose.
No API, network, downloads, manual coordinates, or old-pose initialization.
"""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from alignment_fit import solve_pose
from native_appearance import load_native_glb, ray_native_mesh_points, render_native_mesh
from scripts import stage4_native_pose as P
from scripts import stage4_native_track as T
from observed_cotracker import cycle_endpoints, no_reacquisition
from stage5 import validate_interaction
import interval_motion_search as SEARCH


POLICY={
    'version':'interval-motion-v1','maximum_episodes':2,'episode_radius_s':1.,'maximum_frames_per_episode':9,
    'minimum_reference_spacing_s':2.25,'minimum_episode_frames':3,
    'temporal_evaluation_rule':'abs(source_timestamp_s-round(source_timestamp_s))<=0.02',
    'reference_rule':'image-only maximum mask area*log1p masked Laplacian variance; non-evaluation, original plausible audited mask',
    'minimum_reference_mask_pixels':128,'minimum_mask_area_ratio_adjacent':.35,'maximum_mask_area_ratio_adjacent':2.85,
    'maximum_centroid_motion_in_mask_extent':1.,'maximum_source_gap_factor':1.6,
    'crop_margin_fraction':.2,'crop_minimum_margin_px':16,'crop_uses_reference_selection_frames_only':True,
    'maximum_query_points':96,'model_maximum_side_px':384,'maximum_cycle_error_source_px':2.5,
    'pose_search_width':192,'random_orientation_seeds':96,'diverse_pose_refinements':4,'pose_refine_evaluations':160,
    'appearance_rank_weight':.25,'reference_ambiguity_rank_margin':.03,
    'minimum_training_points':12,'minimum_training_inliers':8,'minimum_training_inlier_fraction':.65,
    'minimum_disjoint_heldout_points':6,'maximum_heldout_median_px':3.,'maximum_heldout_p90_px':8.,
    'minimum_full_projection_iou':.45,'minimum_source_mask_coverage':.6,
    'point_split':'automatic query ID modulo3; remainder0 held out before any PnP',
    'maximum_runtime_s':900,'device':'mps','fallback':False,'new_api_calls':0,
    'manual_coordinates_or_geometry_or_poses':False,'physical_pose_accuracy_measured':False,
    'loss_policy':'first failed visibility/mask/cycle gate stops support in each direction; no reacquisition',
    'episode_merge_policy':'no merging, interpolation, identity recovery, or cross-episode pose continuity',
}
SUPPORTED='supported_nonreference_candidate'
REFERENCE='reference_unverified'
WITHHELD='withheld'


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def evaluation_time(time_s):return abs(float(time_s)-round(float(time_s)))<=.02


def file_path(root,value):
    path=Path(root)/str(value).removeprefix('/files/') if str(value).startswith('/files/') else Path(value)
    if not path.is_absolute():path=Path(root)/path
    path=path.resolve()
    if not path.is_relative_to(Path(root).resolve()/'data') or not path.is_file():raise ValueError('Expected existing saved project data artifact')
    return path


def public_url(root,path):return '/files/'+str(Path(path).resolve().relative_to(Path(root).resolve()))


def verify_worker_lock(root):
    """The orchestrator, not this engine, owns the fixed process-wide flock."""
    path=Path(root)/'data/flow-runs/.model-worker.lock'
    fd=os.open(path,os.O_RDWR|os.O_NOFOLLOW)
    try:
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        fcntl.flock(fd,fcntl.LOCK_UN)
        raise RuntimeError('The fixed local model-worker lock must already be held by the orchestrator')
    finally:os.close(fd)


def stable_pair(first,second):
    if not first['available'] or not second['available']:return False
    ratio=second['mask_area_pixels']/max(1,first['mask_area_pixels'])
    if not POLICY['minimum_mask_area_ratio_adjacent']<=ratio<=POLICY['maximum_mask_area_ratio_adjacent']:return False
    a,b=np.asarray(first['mask_centroid_xy']),np.asarray(second['mask_centroid_xy'])
    extent=max(first['mask_extent_px'],second['mask_extent_px'],1.)
    return bool(np.linalg.norm(a-b)/extent<=POLICY['maximum_centroid_motion_in_mask_extent'])


def choose_episodes(rows,maximum=2):
    if not isinstance(maximum,int) or isinstance(maximum,bool) or not 1<=maximum<=POLICY['maximum_episodes']:raise ValueError('At most two bounded episodes')
    times=[r['timestamp_s'] for r in rows]
    cadence=float(np.median(np.diff(times))) if len(times)>1 else 0.
    candidates=[i for i,r in enumerate(rows) if r['available'] and r['audit_verdict']=='plausible_visible_mask' and not evaluation_time(r['timestamp_s']) and r['mask_area_pixels']>=POLICY['minimum_reference_mask_pixels']]
    candidates.sort(key=lambda i:(-rows[i]['quality_score'],rows[i]['timestamp_s']))
    chosen=[]
    for anchor in candidates:
        ref=rows[anchor]
        if any(abs(ref['timestamp_s']-e['reference_timestamp_s'])<POLICY['minimum_reference_spacing_s'] for e in chosen):continue
        start=end=anchor
        for direction in [-1,1]:
            current=anchor
            while 0<=current+direction<len(rows):
                nxt=current+direction
                if abs(rows[nxt]['timestamp_s']-ref['timestamp_s'])>POLICY['episode_radius_s']+.001:break
                if abs(rows[nxt]['timestamp_s']-rows[current]['timestamp_s'])>max(.001,cadence*POLICY['maximum_source_gap_factor']) or not stable_pair(rows[current],rows[nxt]):break
                current=nxt
                if direction<0:start=current
                else:end=current
        episode_rows=rows[start:end+1]
        if len(episode_rows)<POLICY['minimum_episode_frames'] or len(episode_rows)>POLICY['maximum_frames_per_episode']:continue
        chosen.append({'id':f'episode-{len(chosen)+1:02d}','reference_sample_index':ref['sample_index'],
            'reference_timestamp_s':ref['timestamp_s'],'start_s':episode_rows[0]['timestamp_s'],'end_s':episode_rows[-1]['timestamp_s'],
            'reference_quality_score':ref['quality_score'],'frames':episode_rows})
        if len(chosen)==maximum:break
    return sorted(chosen,key=lambda e:e['start_s'])


def material(interaction_path,track_id,root):
    root=Path(root).resolve();interaction_path=file_path(root,interaction_path);program=read(interaction_path)
    validate_interaction(program,root=root,verify_files=True)
    track=next(t for t in program['tracklets'] if t['id']==track_id)
    assets=[a for a in program['assets'] if a['track_id']==track_id]
    if len(assets)!=1:raise ValueError('A single immutable native asset is required for this component')
    asset=assets[0];mesh_path=file_path(root,asset['path'])
    if sha(mesh_path)!=asset['sha256']:raise ValueError('Native mesh hash mismatch')
    candidates=[]
    for item in program['inputs']:
        if item['kind']=='model_conditional_motion_experiment':
            path=file_path(root,item['path']);pack=read(path)
            if pack['object_id']==track_id:candidates.append((path,pack))
    if len(candidates)!=1:raise ValueError('An exact preserved baseline motion experiment is required')
    baseline_path,baseline=candidates[0]
    if baseline['provenance']['source_video_sha256']!=program['source']['sha256'] or baseline['provenance']['mesh_sha256']!=asset['sha256']:
        raise ValueError('Baseline source video/native mesh differs from the proposed interval input')
    rows=[]
    for evidence in program['evidence']:
        observations=[o for o in evidence['observations'] if o['track_id']==track_id]
        if not observations:continue
        if len(observations)!=1:raise ValueError('Duplicate component observation')
        observation=observations[0]
        image_path=file_path(root,evidence['frame_path']);mask_path=file_path(root,observation['mask_path'])
        image=np.asarray(Image.open(image_path).convert('RGB'));mask=np.asarray(Image.open(mask_path).convert('L'))>0
        if image.shape[:2]!=mask.shape or sha(image_path)!=evidence['frame_sha256'] or sha(mask_path)!=observation['mask_sha256']:raise ValueError('Source pixels/mask identity mismatch')
        ys,xs=np.where(mask);area=int(mask.sum());verdict=observation['audit_status']
        available=bool(area and observation['gate_status']=='candidate_geometric_gate_only' and verdict not in ['reject_wrong_or_lost_mask','uncertain'])
        gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        lap_var=float(cv2.Laplacian(gray,cv2.CV_64F)[mask].var()) if area else 0.
        rows.append({'sample_index':evidence['source_index'],'timestamp_s':evidence['timestamp_s'],
            'source_image_url':public_url(root,image_path),'source_image_sha256':evidence['frame_sha256'],
            'mask_url':public_url(root,mask_path),'mask_sha256':observation['mask_sha256'],
            'width':image.shape[1],'height':image.shape[0],'gate_status':observation['gate_status'],
            'audit_verdict':verdict,'available':available,'mask_area_pixels':area,
            'mask_centroid_xy':[float(xs.mean()),float(ys.mean())] if area else None,
            'mask_extent_px':float(max(xs.max()-xs.min()+1,ys.max()-ys.min()+1)) if area else 0.,
            'masked_laplacian_variance':lap_var,'quality_score':float(area*math.log1p(lap_var)),
            'temporal_split':'evaluation' if evaluation_time(evidence['timestamp_s']) else 'reference_selection'})
    return program,track,asset,mesh_path,baseline_path,baseline,rows


def plan(interaction_path,track_id,out,*,root):
    root=Path(root).resolve();out=Path(out).resolve()
    if out.exists() or not out.is_relative_to(root/'data'):raise ValueError('A fresh data interval output is required')
    program,track,asset,mesh_path,baseline_path,baseline,rows=material(interaction_path,track_id,root)
    episodes=choose_episodes(rows)
    out.mkdir(parents=True)
    value={'format':'short-interval-motion-plan','version':1,'case_id':program['case_id'],'component_id':track_id,'track_id':track_id,'label':track['label'],
        'interaction_path':str(file_path(root,interaction_path).relative_to(root)),'interaction_sha256':sha(file_path(root,interaction_path)),
        'source':program['source'],'source_video_sha256':program['source']['sha256'],
        'asset':asset,'mesh_sha256':asset['sha256'],'baseline':{'path':str(baseline_path.relative_to(root)),'sha256':sha(baseline_path)},
        'intrinsics':baseline['intrinsics'],'policy':POLICY,'episodes':episodes,
        'reference_selection_used_pose_or_heldout_errors':False,'old_pose_initialization_used':False,
        'host_timestamp_utc':datetime.now(timezone.utc).isoformat(),'client_task_date':'2026-10-31',
        'implementation_sha256':sha(__file__),'new_api_calls':0,'new_model_inference_calls':0,
        'scope_note':'Raw SAM masks and model audit verdicts remain fallible. Reference choices and intervals are frozen before tracking outcomes.'}
    write(out/'plan.json',value)
    targets={r['sample_index']:r for e in episodes for r in e['frames']}
    audit={'format':'short-interval-mask-scope-audit-request','version':1,
        'plan_path':str((out/'plan.json').relative_to(root)),'plan_sha256':sha(out/'plan.json'),
        'case_id':program['case_id'],'track_id':track_id,'interaction_sha256':value['interaction_sha256'],
        'source_video_sha256':value['source_video_sha256'],'targets':[targets[k] for k in sorted(targets)],
        'required_observation_fields':['track_id','sample_index','verdict','single_component_visible','component_boundary_plausible','other_components_inside_mask','notes'],
        'required_identity_fields':['source_image_sha256','mask_sha256'],
        'allowed_verdicts':['plausible_visible_mask','reject_wrong_or_lost_mask','uncertain'],
        'rules':'Assess exact untouched source RGB and proposed target mask only. Do not use motion-fit results. Unassessed/rejected/uncertain masks remain withheld.'}
    write(out/'audit-request.json',audit)
    (out/'plan-implementation.py').write_bytes(Path(__file__).read_bytes())
    return value


def load_audit(plan_value,plan_path,audit_path,root):
    targets={r['sample_index']:r for e in plan_value['episodes'] for r in e['frames']}
    if audit_path is None:
        return {i:{'verdict':r['audit_verdict'],'scope_reviewed':False,'scope_safe':True,
                   'source':'original_program_mask_audit'} for i,r in targets.items()},None
    path=file_path(root,audit_path);value=read(path)
    for key,expected in [('plan_sha256',sha(plan_path)),('interaction_sha256',plan_value['interaction_sha256']),('source_video_sha256',plan_value['source_video_sha256'])]:
        if value.get(key)!=expected:raise ValueError('Supplemental audit identity mismatch: '+key)
    found={}
    for row in value['result']['observations']:
        index=row['sample_index'];target=targets.get(index)
        if not target or index in found or row['track_id']!=plan_value['track_id']:raise ValueError('Supplemental audit duplicate/unknown target')
        if any(row.get(key)!=target[key] for key in ['source_image_sha256','mask_sha256']):raise ValueError('Supplemental source RGB/mask hash mismatch')
        if row.get('verdict') not in ['plausible_visible_mask','reject_wrong_or_lost_mask','uncertain']:raise ValueError('Unknown supplemental audit verdict')
        scope=[row.get(k) for k in ['single_component_visible','component_boundary_plausible','other_components_inside_mask']]
        if any(not isinstance(item,bool) for item in scope):raise ValueError('Explicit model scope booleans are required')
        found[index]={'verdict':row['verdict'],'scope_reviewed':True,'scope_safe':scope[0] and scope[1] and not scope[2],
            'source':'supplemental_exact_plan_scope_audit','single_component_visible':scope[0],
            'component_boundary_plausible':scope[1],'other_components_inside_mask':scope[2],'notes':row.get('notes','')}
    if set(found)!=set(targets):raise ValueError('Supplemental audit must account for every exact planned target')
    return found,{'path':str(path.relative_to(Path(root).resolve())),'sha256':sha(path),'plan_sha256':value['plan_sha256']}


def audit_reasons(row,assessment):
    reasons=[]
    if row['gate_status']!='candidate_geometric_gate_only':reasons.append('source_mask_failed_geometric_gate')
    if assessment['verdict']!='plausible_visible_mask':reasons.append('source_mask_not_plausible_in_model_audit')
    if not assessment['scope_safe']:reasons.append('source_mask_component_scope_not_plausible')
    return reasons


def pose_reasons(train_count,inlier_count,errors,iou,coverage,source_reasons):
    errors=np.asarray(errors,float);reasons=list(source_reasons)
    if train_count<POLICY['minimum_training_points']:reasons.append('fewer_than_12_training_points')
    if inlier_count<POLICY['minimum_training_inliers'] or inlier_count/max(1,train_count)<POLICY['minimum_training_inlier_fraction']:reasons.append('insufficient_training_consensus')
    if len(errors)<POLICY['minimum_disjoint_heldout_points']:reasons.append('fewer_than_6_disjoint_heldout_points')
    if not len(errors) or not np.isfinite(errors).all() or np.median(errors)>POLICY['maximum_heldout_median_px'] or np.percentile(errors,90)>POLICY['maximum_heldout_p90_px']:reasons.append('heldout_error_above_3px_median_or_8px_p90')
    if not np.isfinite(iou) or not np.isfinite(coverage) or iou<POLICY['minimum_full_projection_iou'] or coverage<POLICY['minimum_source_mask_coverage']:reasons.append('raw_full_mesh_silhouette_gate_failed')
    return sorted(set(reasons))


def reassess_baseline(record,row,assessment):
    original=record.get('status','missing_baseline_observation') if record else 'missing_baseline_observation'
    result={'baseline_status':original,'baseline_original_supported':original=='supported_conditional_motion_candidate',
        'baseline_reference':original.startswith('reference_'),'baseline_reassessed_status':WITHHELD,'baseline_reassessed_reasons':[]}
    if record is None:
        result['baseline_reassessed_reasons']=['no_exact_baseline_source_timestamp'];return result
    if result['baseline_reference']:
        result['baseline_reassessed_status']=REFERENCE;return result
    if not record.get('candidate_rotation') or not record.get('candidate_translation'):
        result['baseline_reassessed_reasons']=['no_baseline_numeric_pose_candidate'];return result
    reasons=pose_reasons(len(record.get('training_visible_point_ids',[])),len(record.get('training_inlier_point_ids',[])),
        record.get('heldout_residual_px',[]),record.get('raw_full_projection_iou_with_sam',0),record.get('source_mask_coverage',0),audit_reasons(row,assessment))
    result.update(baseline_reassessed_status=WITHHELD if reasons else SUPPORTED,baseline_reassessed_reasons=reasons)
    return result


def episode_crop(rows,masks):
    bounds=[]
    for row,mask in zip(rows,masks):
        if row['temporal_split']!='reference_selection':continue
        y,x=np.where(mask)
        if len(x):bounds.append([x.min(),y.min(),x.max()+1,y.max()+1])
    if not bounds:raise ValueError('No TRAIN/reference-selection source masks for crop')
    b=np.asarray(bounds);lo=b[:,:2].min(0);hi=b[:,2:].max(0)
    margin=np.maximum(POLICY['crop_minimum_margin_px'],np.ceil((hi-lo)*POLICY['crop_margin_fraction'])).astype(int)
    lo=np.maximum(lo-margin,0);hi=np.minimum(hi+margin,[masks[0].shape[1],masks[0].shape[0]])
    return [int(v) for v in [*lo,*hi]]


def serial(value):
    if isinstance(value,np.ndarray):return serial(value.tolist())
    if isinstance(value,np.generic):return serial(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [serial(v) for v in value]
    return value


def fit_reference(mesh,image,mask,k,deadline):
    height,width=mask.shape;work=(POLICY['pose_search_width'],round(height*POLICY['pose_search_width']/width))
    small=cv2.resize(mask.astype(np.uint8),work,interpolation=cv2.INTER_NEAREST).astype(bool)
    small_k=P.resized_intrinsics(k,(width,height),work)
    centre=(mesh.vertices.min(0)+mesh.vertices.max(0))/2;centred=mesh.vertices-centre
    coarse=[]
    for index,rotation in enumerate(P.generic_orientations(POLICY['random_orientation_seeds'])):
        if time.monotonic()>deadline:raise TimeoutError('Fixed episode runtime budget exhausted')
        translation=P.initial_translation(centred,rotation,small_k,small)
        overlap=P.iou(SEARCH.silhouette(mesh,rotation,translation,small_k,work,centre),small)
        coarse.append((overlap,index,rotation,translation))
    coarse.sort(key=lambda r:(-r[0],r[1]));seeds=[]
    for item in coarse:
        if all(P.rotation_distance_deg(item[2],old[2])>=35 for old in seeds):seeds.append(item)
        if len(seeds)==POLICY['diverse_pose_refinements']:break
    candidates=[]
    for overlap,index,rotation,translation in seeds:
        if time.monotonic()>deadline:raise TimeoutError('Fixed episode runtime budget exhausted')
        fit=SEARCH.refine(mesh,centre,small_k,small,rotation,translation,POLICY['pose_refine_evaluations'])
        fit['translation']=fit['translation']-fit['rotation']@centre
        rendered=render_native_mesh(mesh,k,fit['rotation'],fit['translation'],width,height)
        intersection=int((rendered['mask']&mask).sum())
        fit.update(seed_index=index,search_proxy_seed_iou=overlap,iou_full_resolution=P.iou(rendered['mask'],mask),
            source_mask_coverage=intersection/max(1,int(mask.sum())),
            appearance=P.appearance_similarity(image,rendered['rgb'],rendered['mask']&mask))
        fit['rank_score']=P.reference_rank_score(fit['iou_full_resolution'],fit['appearance'],POLICY['appearance_rank_weight'])
        candidates.append(fit)
    candidates.sort(key=lambda f:-f['rank_score']);best=candidates[0]
    ambiguous=any(P.rotation_distance_deg(best['rotation'],c['rotation'])>45 and best['rank_score']-c['rank_score']<POLICY['reference_ambiguity_rank_margin'] for c in candidates[1:])
    return best,{'method':'generic SO(3) original-triangle search proxy, exact native finalist reranking and bounded same-reference appearance ranking',
        'search_proxy_policy':SEARCH.POLICY,'finalist_ranking_uses_exact_native_mask_and_rgb':True,
        'heldout_motion_gates_use_exact_native_rendering':True,
        'old_pose_initialization_used':False,'reference_fit_is_physical_truth':False,
        'reference_orientation_ambiguous':ambiguous,'candidates':serial(candidates),'selected_candidate_index':0,
        'reference_image_used_to_fit_pose':True,'evaluation_images_used_to_fit_reference':False}


class OfflineTracker:
    def __init__(self,root,out,deadline):
        self.root=Path(root).resolve();self.out=Path(out);self.deadline=deadline;self.calls=[];self.model=None

    def infer(self,images,queries,rows,crop,scope):
        if time.monotonic()>self.deadline:raise TimeoutError('Fixed model runtime budget exhausted')
        import torch
        from cotracker.predictor import CoTrackerPredictor
        from importlib.metadata import distribution
        import inspect
        T.validate_tracking_device('mps',torch)
        checkpoint=self.root/'models/cotracker3/scaled_offline.pth';provenance=read(checkpoint.parent/'provenance.json')
        if sha(checkpoint)!=provenance['checkpoint_sha256']:raise ValueError('Prepared CoTracker checkpoint changed')
        package=json.loads(distribution('cotracker').read_text('direct_url.json'))
        if package['vcs_info']['commit_id']!=provenance['code_commit']:raise ValueError('Prepared CoTracker installed commit changed')
        predictor_path=Path(inspect.getsourcefile(CoTrackerPredictor))
        if self.model is None:
            self.model=CoTrackerPredictor(checkpoint=str(checkpoint)).to('mps').eval()
            (self.out/'cotracker-predictor.py').write_bytes(predictor_path.read_bytes())
        self.model.interp_shape=T.tracking_input_shape(images[0].shape,POLICY['model_maximum_side_px'])[0]
        query=np.asarray(queries,np.float32)
        if query.ndim!=2 or query.shape[1]!=3 or not np.isfinite(query).all():raise ValueError('Automatic query array malformed')
        h,w=images[0].shape[:2]
        if (query[:,0]<0).any() or (query[:,0]>=len(images)).any() or (query[:,1:]<0).any() or (query[:,1]>w-1).any() or (query[:,2]>h-1).any():raise ValueError('Automatic queries leave actual source crop')
        call={'scope':scope,'source_sample_indices':[r['sample_index'] for r in rows],
            'source_image_sha256':[r['source_image_sha256'] for r in rows],'queries':len(query),
            'device':'mps','fallback':False,'crop_xyxy':crop,'native_input_hw':[h,w],
            'model_interp_hw':list(self.model.interp_shape),'checkpoint_sha256':provenance['checkpoint_sha256'],
            'code_commit':provenance['code_commit'],'predictor_sha256':sha(predictor_path),
            'host_started_at_utc':datetime.now(timezone.utc).isoformat(),'status':'started'}
        self.calls.append(call);write(self.out/'inference-calls.json',{'actual_model_calls':len(self.calls),'calls':self.calls,'new_api_calls':0})
        tensor=torch.from_numpy(np.stack(images)).permute(0,3,1,2)[None].float().to('mps')
        started=time.monotonic()
        try:
            with torch.inference_mode():
                uv,visible=self.model(tensor,queries=torch.from_numpy(query)[None].to('mps'),backward_tracking=True)
                torch.mps.synchronize();uv=uv[0].cpu().numpy().astype(float);visible=visible[0].cpu().numpy().astype(bool)
            call.update(status='complete',inference_seconds=time.monotonic()-started)
        except Exception as error:
            call.update(status='failed',error_type=type(error).__name__,message=str(error));raise
        finally:write(self.out/'inference-calls.json',{'actual_model_calls':len(self.calls),'calls':self.calls,'new_api_calls':0})
        return uv,visible


def frame_record(row,assessment,reference_sample):
    return {**row,'status':WITHHELD,'reasons':[],'reference':row['sample_index']==reference_sample,
        'evaluation_timestamp_reserved_before_reference_selection':row['temporal_split']=='evaluation',
        'candidate_rotation':None,'candidate_translation':None,'physical_pose_accuracy_measured':False,
        'mask_model_audit_verdict':assessment['verdict'],'scope_assessment':assessment,
        'training_visible_point_ids':[],'heldout_visible_point_ids':[],'training_inlier_point_ids':[],
        'train_count':0,'inlier_count':0,'heldout_errors_px':[],'heldout_median_px':None,'heldout_p90_px':None,
        'silhouette_iou':None,'mask_coverage':None}


def contiguous_scope_segment(rows,assessment,anchor):
    safe=[not audit_reasons(r,assessment[r['sample_index']]) for r in rows]
    if not safe[anchor]:return []
    first=last=anchor
    while first>0 and safe[first-1]:first-=1
    while last+1<len(rows) and safe[last+1]:last+=1
    return list(range(first,last+1))


def baseline_barrier_reasons(reference_sample,target_sample,rows,assessment):
    low,high=sorted([reference_sample,target_sample])
    barriers=[r['sample_index'] for r in rows if low<=r['sample_index']<=high and audit_reasons(r,assessment[r['sample_index']])]
    return ['known_audited_mask_loss_between_baseline_reference_and_target'] if barriers else []


def track_episode(episode,assessment,mesh,k,tracker,root,out,deadline,*,reference_provider=None,fixed_crop=None,expected_query_uv=None):
    rows=episode['frames'];masks=[np.asarray(Image.open(file_path(root,r['mask_url'])).convert('L'))>0 for r in rows]
    images=[np.asarray(Image.open(file_path(root,r['source_image_url'])).convert('RGB')) for r in rows]
    anchor=next(i for i,r in enumerate(rows) if r['sample_index']==episode['reference_sample_index'])
    result={k:v for k,v in episode.items() if k!='frames'}
    result.update(intrinsics={'matrix':k.tolist(),'width':rows[0]['width'],'height':rows[0]['height'],'calibrated':False},
        gauge='independent_reference_conditioned_candidate',physical_scale_known=False,
        frames=[frame_record(r,assessment[r['sample_index']],episode['reference_sample_index']) for r in rows],
        status='prepared',queries={'training_point_ids':[],'heldout_point_ids':[]})
    ref_reasons=audit_reasons(rows[anchor],assessment[rows[anchor]['sample_index']])
    if ref_reasons:
        for rec in result['frames']:rec['reasons']=['reference_mask_vetoed']+ref_reasons
        result['status']='unsupported_reference_mask';return result
    active=contiguous_scope_segment(rows,assessment,anchor)
    result['inferred_source_sample_indices']=[rows[i]['sample_index'] for i in active]
    result['scope_loss_policy']='Only the contiguous audited/geometrically eligible source segment containing the frozen reference enters learned inference; omitted planned rows remain withheld.'
    if len(active)<POLICY['minimum_episode_frames']:
        for rec in result['frames']:rec['reasons']=['insufficient_contiguous_audited_source_frames']
        result['status']='unsupported_audited_segment_too_short';return result
    best,reference=(fit_reference(mesh,images[anchor],masks[anchor],k,deadline) if reference_provider is None
                    else reference_provider(mesh,images[anchor],masks[anchor],k,deadline))
    write(out/'reference-fit.json',reference);result['reference_fit']={**reference,'report_url':public_url(root,out/'reference-fit.json')}
    seeds=T.query_points(images[anchor],masks[anchor],POLICY['maximum_query_points'])
    if expected_query_uv is not None and not np.array_equal(seeds,np.asarray(expected_query_uv,dtype=seeds.dtype)):
        raise ValueError('Controlled density reference query pixels differ from the frozen automatic queries')
    canonical,hits=ray_native_mesh_points(seeds,mesh,k,best['rotation'],best['translation'])
    train,held=T.split_ids(hits)
    result['queries']={'count':len(seeds),'reference_mesh_ray_hits':int(hits.sum()),
        'training_point_ids':train.tolist(),'heldout_point_ids':held.tolist(),'split':POLICY['point_split'],
        'automatic_query_uv_source':seeds.tolist(),'manual_queries':False}
    if len(train)<POLICY['minimum_training_points'] or len(held)<POLICY['minimum_disjoint_heldout_points'] or best['iou_full_resolution']<POLICY['minimum_full_projection_iou'] or best['source_mask_coverage']<POLICY['minimum_source_mask_coverage']:
        for rec in result['frames']:rec['reasons']=['reference_fit_or_ray_hit_support_insufficient']
        result['status']='unsupported_reference_geometry';return result
    model_rows=[rows[i] for i in active];model_masks=[masks[i] for i in active];model_anchor=active.index(anchor)
    crop=episode_crop(model_rows,model_masks) if fixed_crop is None else list(fixed_crop)
    if len(crop)!=4 or any(type(v) is not int for v in crop) or not (0<=crop[0]<crop[2]<=rows[0]['width'] and 0<=crop[1]<crop[3]<=rows[0]['height']):
        raise ValueError('Expected a bounded fixed source-pixel crop')
    x0,y0,x1,y1=crop
    cropped=[images[i][y0:y1,x0:x1].copy() for i in active]
    queries=np.c_[np.full(len(seeds),model_anchor),seeds-np.array([x0,y0])]
    local_raw,local_visible=tracker.infer(cropped,queries,model_rows,crop,episode['id']+'_forward')
    local_raw+=np.array([x0,y0]);support=T.support_masks(local_raw,local_visible,model_masks,model_anchor)
    instantaneous=support['instantaneous_support']&(local_raw[:,:,0]>=x0)&(local_raw[:,:,0]<x1)&(local_raw[:,:,1]>=y0)&(local_raw[:,:,1]<y1)
    endpoints=cycle_endpoints(instantaneous,model_anchor)
    reverse_queries=np.array([[len(active)-1-end,*local_raw[end,j]-np.array([x0,y0])] for j,end in enumerate(endpoints)])
    local_backward,local_back_visible=tracker.infer(cropped[::-1],reverse_queries,model_rows[::-1],crop,episode['id']+'_cycle_reverse')
    local_backward=local_backward[::-1]+np.array([x0,y0]);local_back_visible=local_back_visible[::-1]
    back_support=T.support_masks(local_backward,local_back_visible,model_masks,model_anchor)['instantaneous_support']
    local_cycle=np.linalg.norm(local_raw-local_backward,axis=2)
    local_valid=no_reacquisition(instantaneous&back_support&np.isfinite(local_cycle)&(local_cycle<=POLICY['maximum_cycle_error_source_px']),model_anchor)&hits[None]
    raw=np.full((len(rows),len(seeds),2),np.nan);backward=np.full_like(raw,np.nan)
    visible=np.zeros(raw.shape[:2],bool);back_visible=np.zeros_like(visible);valid=np.zeros_like(visible);cycle=np.full(visible.shape,np.inf)
    raw[active]=local_raw;backward[active]=local_backward;visible[active]=local_visible;back_visible[active]=local_back_visible;valid[active]=local_valid;cycle[active]=local_cycle
    np.savez_compressed(out/'tracks.npz',raw_model_uv=raw,model_visibility=visible,reverse_model_uv=backward,
        reverse_visibility=back_visible,cycle_error_px=cycle,valid_without_reacquisition=valid,
        reference_ray_hits=hits,canonical_xyz=canonical,query_uv=seeds,training_point_ids=train,heldout_point_ids=held,
        source_sample_indices=np.array([r['sample_index'] for r in rows]),source_pts_s=np.array([r['timestamp_s'] for r in rows]))
    result.update(crop_xyxy=crop,tracks_url=public_url(root,out/'tracks.npz'),tracks_sha256=sha(out/'tracks.npz'))
    for j,rec in enumerate(result['frames']):
        if time.monotonic()>deadline:raise TimeoutError('Fixed episode runtime budget exhausted')
        if j not in active:
            rec['reasons']=['outside_contiguous_audited_visible_segment']+audit_reasons(rec,assessment[rec['sample_index']]);continue
        present=train[valid[j,train]];checks=held[valid[j,held]]
        rec.update(training_visible_point_ids=present.tolist(),heldout_visible_point_ids=checks.tolist(),train_count=len(present))
        if j==anchor:
            rotation,translation=best['rotation'],best['translation'];inliers=np.arange(len(present));residual=[]
        else:
            fit=solve_pose(canonical[present],raw[j,present],k,best['rotation'],best['translation']) if len(present)>=6 else {'success':False,'reason':'fewer_than_6_valid_training_correspondences'}
            if not fit['success']:
                rec['reasons']=[fit['reason']]+audit_reasons(rec,assessment[rec['sample_index']]);continue
            rotation,translation=fit['rotation'],fit['translation'];inliers=fit['inlier_indices'];residual=fit['residual_px']
        projected=T.project(canonical[checks],k,rotation,translation)
        depths=(canonical[checks]@rotation.T+translation)[:,2]
        errors=np.linalg.norm(projected-raw[j,checks],axis=1);errors[~np.isfinite(depths)|(depths<=0)]=np.inf
        rendered=render_native_mesh(mesh,k,rotation,translation,rec['width'],rec['height'])
        intersection=int((rendered['mask']&masks[j]).sum());overlap=P.iou(rendered['mask'],masks[j]);coverage=intersection/max(1,int(masks[j].sum()))
        rec.update(raw_candidate_rotation=rotation.tolist(),raw_candidate_translation=translation.tolist(),
            training_inlier_point_ids=present[inliers].tolist(),inlier_count=len(inliers),training_residual_px=serial(residual),
            heldout_errors_px=serial(errors),heldout_median_px=float(np.median(errors)) if len(errors) and np.isfinite(errors).all() else None,
            heldout_p90_px=float(np.percentile(errors,90)) if len(errors) and np.isfinite(errors).all() else None,
            silhouette_iou=overlap,mask_coverage=coverage,scored_behind_camera_points=int((depths<=0).sum()))
        reasons=pose_reasons(len(present),len(inliers),errors,overlap,coverage,audit_reasons(rec,assessment[rec['sample_index']]))
        rec['reasons']=reasons
        if j==anchor:rec['status']=REFERENCE
        elif not reasons:rec['status']=SUPPORTED
        if rec['status'] in [REFERENCE,SUPPORTED]:rec.update(candidate_rotation=rotation.tolist(),candidate_translation=translation.tolist())
    result['status']='complete_with_gaps' if any(r['status']==WITHHELD for r in result['frames']) else 'complete_conditional_episode'
    return result


def snapshot(root,out):
    files=['interval_motion.py','interval_motion_search.py','alignment_fit.py','alignment_render.py','native_appearance.py','native_alignment.py','observed_cotracker.py',
           'scripts/stage4_native_pose.py','scripts/stage4_native_track.py','scripts/stage4_interval_motion.py']
    directory=out/'implementation';directory.mkdir()
    records={}
    for name in files:
        source=Path(root)/name;target=directory/name.replace('/','__');target.write_bytes(source.read_bytes());target.chmod(0o444)
        records[name]={'sha256':sha(source),'snapshot_url':public_url(root,target)}
    return records


def run(interaction_path,track_id,out,*,root,audit_path=None,plan_path=None):
    root=Path(root).resolve();out=Path(out).resolve();started=time.monotonic();deadline=started+POLICY['maximum_runtime_s']
    verify_worker_lock(root)
    if plan_path is None:
        value=plan(interaction_path,track_id,out,root=root);plan_path=out/'plan.json'
    else:
        plan_path=file_path(root,plan_path);value=read(plan_path)
        if out.exists() or not out.is_relative_to(root/'data'):raise ValueError('A fresh interval result directory is required')
        out.mkdir(parents=True);(out/'plan.json').write_bytes(plan_path.read_bytes())
    if value['interaction_sha256']!=sha(file_path(root,interaction_path)) or value['track_id']!=track_id or value['policy']!=POLICY:
        raise ValueError('Frozen interval plan identity or fixed policy changed')
    assessment,audit=load_audit(value,plan_path,audit_path,root)
    program,track,asset,mesh_path,baseline_path,baseline,_=material(interaction_path,track_id,root)
    base={'format':'short-interval-motion','version':1,'status':'running','case_id':value['case_id'],
        'algorithm_version':'short_interval_motion_v2_search_proxy','candidate_search_policy':SEARCH.POLICY,
        'track_id':track_id,'component_id':track_id,'label':track['label'],'interaction_sha256':value['interaction_sha256'],
        'interaction_path':value['interaction_path'],
        'source':value['source'],'source_video_sha256':value['source_video_sha256'],'asset':asset,'mesh_sha256':asset['sha256'],
        'baseline':value['baseline'],'plan':{'path':str(plan_path.relative_to(root)),'sha256':sha(plan_path)},
        'supplemental_audit':audit,'audit_binding':audit,'policy':POLICY,'implementation':snapshot(root,out),'episodes':[],
        'baseline_reassessment_policy':{'version':1,'gates':{key:POLICY[key] for key in [
            'minimum_training_points','minimum_training_inliers','minimum_training_inlier_fraction',
            'minimum_disjoint_heldout_points','maximum_heldout_median_px','maximum_heldout_p90_px',
            'minimum_full_projection_iou','minimum_source_mask_coverage']},
            'uses_saved_numeric_candidates_only':True,'uses_same_current_audit_and_original_geometric_gates':True,
            'new_inference_or_pose_refit':False,'original_outputs_overwritten':False},
        'frame_count':0,'supported_nonreference_frames':0,'reference_frames':0,'withheld_nonreference_frames':0,
        'coordinate_convention':'camera_from_native_mesh; OpenCV x-right,y-down,z-forward; native mesh @ R.T+t',
        'physical_pose_accuracy_measured':False,'usable_physical_pose_recovered':False,'common_assembly_established':False,
        'physical_scale_known':False,'new_api_calls':0,'old_pose_initialization_used':False,'manual_geometry_or_poses':False,
        'host_started_at_utc':datetime.now(timezone.utc).isoformat(),'client_task_date':'2026-10-31'}
    write(out/'interval-motion.json',base)
    try:
        mesh=load_native_glb(mesh_path);k=np.asarray(value['intrinsics']['matrix'],float)
        tracker=OfflineTracker(root,out,deadline)
        for episode in value['episodes']:
            print(json.dumps({'phase':'interval_reference','track_id':track_id,'episode_id':episode['id'],'reference_sample_index':episode['reference_sample_index']}),flush=True)
            directory=out/episode['id'];directory.mkdir()
            result=track_episode(episode,assessment,mesh,k,tracker,root,directory,deadline)
            write(directory/'episode.json',serial(result));base['episodes'].append(result)
            write(out/'interval-motion.json',serial(base))
        rows=[];old={r['sample_index']:r for r in baseline['frames']}
        planned={r['sample_index']:r for e in value['episodes'] for r in e['frames']}
        for episode in base['episodes']:
            for rec in episode['frames']:
                prior=old.get(rec['sample_index'])
                if prior and (prior['timestamp_s']!=rec['timestamp_s'] or prior['source_frame_sha256']!=rec['source_image_sha256'] or prior['sam_mask_sha256']!=rec['mask_sha256']):raise ValueError('Exact baseline frame/mask/PTS join failed')
                shared=reassess_baseline(prior,rec,assessment[rec['sample_index']])
                barriers=baseline_barrier_reasons(baseline['reference_sample_index'],rec['sample_index'],list(planned.values()),assessment)
                if barriers and not shared['baseline_reference']:
                    shared['baseline_reassessed_status']=WITHHELD
                    shared['baseline_reassessed_reasons']=sorted(set(shared['baseline_reassessed_reasons']+barriers))
                shared.update(sample_index=rec['sample_index'],timestamp_s=rec['timestamp_s'],episode_id=episode['id'],
                    baseline_supported=shared['baseline_reassessed_status']==SUPPORTED,candidate_status=rec['status'],
                    candidate_supported=rec['status']==SUPPORTED,candidate_reference=rec['reference'],
                    source_image_sha256=rec['source_image_sha256'],mask_sha256=rec['mask_sha256'],
                    evaluation_timestamp_reserved_before_reference_selection=rec['temporal_split']=='evaluation',
                    shared_scope_assessment=assessment[rec['sample_index']])
                shared['eligible_matched_nonreference']=not shared['baseline_reference'] and not shared['candidate_reference']
                rows.append(shared)
        matched=[r for r in rows if r['eligible_matched_nonreference']]
        evaluated=[r for r in matched if r['evaluation_timestamp_reserved_before_reference_selection']]
        base['comparison']={'exact_same_mesh':True,'exact_same_normalized_source_video':True,'exact_pts_rows':rows,
            'baseline_reassessment_policy':'Original numeric candidates, train/inlier/heldout/silhouette/coverage values; identical current audit/scope and original geometric-mask gates. Original outputs untouched.',
            'baseline_reassessment_uses_new_model_inference':False,
            'matched_nonreference_count':len(matched),'baseline_shared_audit_supported':sum(r['baseline_supported'] for r in matched),
            'candidate_supported':sum(r['candidate_supported'] for r in matched),
            'reserved_evaluation_count':len(evaluated),'baseline_reserved_evaluation_supported':sum(r['baseline_supported'] for r in evaluated),
            'candidate_reserved_evaluation_supported':sum(r['candidate_supported'] for r in evaluated),
            'different_point_sets_and_reference_hypotheses':True,'physical_accuracy_comparison_performed':False}
        records=[r for e in base['episodes'] for r in e['frames']]
        base.update(frame_count=len(records),supported_nonreference_frames=sum(r['status']==SUPPORTED for r in records),
            reference_frames=sum(r['reference'] for r in records),withheld_nonreference_frames=sum(not r['reference'] and r['status']!=SUPPORTED for r in records),
            status='complete_with_gaps' if records and any(r['status']==SUPPORTED for r in records) else 'unsupported',
            actual_model_calls=len(tracker.calls),wall_seconds=time.monotonic()-started,
            limitations=['Native geometry, silhouette-fit reference orientation, intrinsics and physical scale are unverified. A low residual is not measured physical pose accuracy.',
                'Each episode is an independent reference-conditioned hypothesis. No interpolation, occlusion bridge, identity recovery, cross-episode pose merge or common assembly is produced.',
                'Scored query IDs are disjoint from PnP training IDs, but CoTracker observations share a learned sequence and generated reference hypothesis; these checks are correlated image evidence.',
                'Reserved temporal evaluation images do not select or fit a reference; all episode images jointly provide learned 2D tracking context. No reconstruction training/test isolation is claimed for the tracker.',
                'The exact source masks and model scope audits remain fallible. Unassessed/rejected/uncertain masks are withheld.',
                'Newly visible surfaces, hidden contact, seating, joint axes, physical robot paths and simulator readiness are not established.'])
        if sha(mesh_path)!=base['mesh_sha256'] or sha(baseline_path)!=base['baseline']['sha256'] or sha(file_path(root,interaction_path))!=base['interaction_sha256']:raise ValueError('Immutable source/native/baseline input changed')
        base['host_completed_at_utc']=datetime.now(timezone.utc).isoformat();write(out/'interval-motion.json',serial(base))
        return base
    except Exception as error:
        base.update(status='failed',error={'type':type(error).__name__,'message':str(error)},wall_seconds=time.monotonic()-started)
        write(out/'interval-motion.json',serial(base));raise
