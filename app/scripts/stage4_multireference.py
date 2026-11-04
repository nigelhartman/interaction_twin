#!/usr/bin/env python3
"""Dense crop tracking of one fixed generated mesh, with bounded auto references.

Every reference is independently initialized; branches may disagree and are not
claimed as identity recovery, calibrated pose or a physical assembly.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import sys
import time
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageOps
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))
from scripts import stage4_dense_jabra as dense
from scripts import stage4_track_generated as motion
from scripts import stage4_generated_pose as init
from alignment_fit import solve_pose
from alignment_render import ray_mesh_points,render_mesh
from multireference_tracking import choose_references,merge_branches,scope_veto,evaluation_frame,contiguous_nonempty_segment
read,write,sha=motion.read,motion.write,motion.sha
CASE=ROOT/'data/recording-jabra-5fps-v1'
BASE=ROOT/'data/recording-144929-1fps-v3'
ASSEMBLY=ROOT/'data/jabra-assembly-v1'
REFS={'container_base':ASSEMBLY/'pose-container_base-alternate/experiment.json','container_lid':ASSEMBLY/'pose-container_lid/experiment.json'}


def material(tid):
    reference=read(REFS[tid]);meta_path=Path(reference['inputs']['manifest']);meta=read(meta_path)
    source=read(CASE/'manifest.json')['source']
    if source['sha256']!=reference['source_video_sha256'] or sha(CASE/source['file'])!=source['sha256']:
        raise ValueError('Dense frames and fixed mesh reference must share the exact source video')
    mesh_path=meta_path.parent/meta['outputs']['glb_path']
    if sha(mesh_path)!=reference['inputs']['mesh_sha256'] or sha(meta_path)!=reference['inputs']['manifest_sha256']:
        raise ValueError('Frozen model mesh or metadata changed')
    seg=CASE/'segmentation/automatic-components-v1';tracks=read(seg/'tracks.json')
    if read(seg/'annotations.json')['manual_seed_boxes_used'] is not False:
        raise ValueError('Model proposals required')
    tr=next(t for t in tracks['tracks'] if t['track_id']==tid)
    all_rows=read(CASE/'frames/index.json')['frames'];rows=[all_rows[r['sample_index']-1] for r in tr['frames']]
    images=[np.array(Image.open(CASE/r['file']).convert('RGB')) for r in rows]
    mask_paths=[seg/r['raw_mask_file'] for r in tr['frames']]
    masks=[np.array(Image.open(p).convert('L'))>127 for p in mask_paths]
    if any(not m.any() for m in masks):
        anchor=min(range(len(rows)),key=lambda i:abs(rows[i]['timestamp_s']-reference['source_timestamp_s']))
        selected=contiguous_nonempty_segment([m.any() for m in masks],anchor)
        rows=[rows[i] for i in selected];images=[images[i] for i in selected]
        masks=[masks[i] for i in selected];mask_paths=[mask_paths[i] for i in selected]
    import trimesh
    mesh=trimesh.load(mesh_path,force='mesh',process=False)
    return reference,meta_path,mesh_path,rows,images,masks,mask_paths,np.asarray(mesh.vertices),np.asarray(mesh.faces),np.asarray(mesh.visual.vertex_colors)[:,:3]


def infer_branch(tid,out,reference=None,branch_id='existing-reference',clip_radius_s=None):
    out=Path(out).resolve()
    if out.exists():raise ValueError('Preserve earlier inference output')
    old,meta_path,mesh_path,rows,images,masks,mask_paths,vertices,faces,colors=material(tid)
    reference=old if reference is None else reference
    if clip_radius_s is not None:
        selected=[i for i,row in enumerate(rows) if abs(row['timestamp_s']-reference['source_timestamp_s'])<=clip_radius_s+.001]
        rows=[rows[i] for i in selected];images=[images[i] for i in selected];masks=[masks[i] for i in selected];mask_paths=[mask_paths[i] for i in selected]
    anchor_row=dense.nearest_sample(rows,reference['source_timestamp_s'],.0001);anchor=rows.index(anchor_row)
    query_mask=masks[anchor]
    if reference.get('initialization_mask_file'):
        source_mask=Path(reference['initialization_mask_file'])
        if sha(source_mask)!=reference['initialization_mask_sha256']:raise ValueError('Initialization source mask changed')
        if sha(CASE/anchor_row['file'])!=reference['source_frame_sha256']:raise ValueError('Reinitialization source RGB differs')
        query_mask=query_mask&(np.array(Image.open(source_mask).convert('L'))>127)
    seeds=motion.query_points(images[anchor],query_mask)
    k=np.array(reference['intrinsics']['matrix']);rotation=np.array(reference['candidate']['rotation']);translation=np.array(reference['candidate']['translation'])
    canonical,hits=ray_mesh_points(seeds,vertices,faces,k,rotation,translation);train,held=motion.split_ids(hits)
    if len(train)<12 or len(held)<6:raise ValueError('Insufficient raycast training or held-out support')
    out.mkdir(parents=True);start=time.monotonic()
    record={'schema_version':'0.2','status':'running','object_id':tid,'case_path':str(CASE),'branch_id':branch_id,
      'provenance':{'reference_file':str(REFS[tid]) if reference is old else reference.get('reference_manifest_file'),
        'reference_sha256':sha(REFS[tid]) if reference is old else reference.get('reference_manifest_sha256'),
        'model_metadata_file':str(meta_path),'model_metadata_sha256':sha(meta_path),'mesh_file':str(mesh_path),'mesh_sha256':sha(mesh_path),
        'source_video_sha256':read(CASE/'manifest.json')['source']['sha256'],
        'original_source_sha256':read(CASE/'manifest.json')['source']['original_sha256'],
        'annotations_sha256':sha(CASE/'segmentation/automatic-components-v1/annotations.json'),
        'track_index_sha256':sha(CASE/'segmentation/automatic-components-v1/tracks.json'),
        'manual_geometry_used':False,'manual_pose_used':False,'manual_seed_boxes_used':False,'manual_corners_used':False},
      'intrinsics':reference['intrinsics'],'fixed_native_mesh_scale':1.,'physical_scale_known':False,
      'reference_sample_index':anchor_row['index'],'reference_timestamp_s':anchor_row['timestamp_s'],
      'reference':reference,'queries':{'training_point_ids':train.tolist(),'heldout_point_ids':held.tolist(),'query_count':len(seeds),'mesh_ray_hits':int(hits.sum()),'split':'IDs modulo3 remainder0 held out before pose fit'},
      'reference_initialization_independently_verified':False,'usable_physical_pose_recovered':False,'physical_pose_accuracy_measured':False,
      'inference_segment_source_interval_s':[rows[0]['timestamp_s'],rows[-1]['timestamp_s']],
      'empty_observation_policy':'Empty SAM masks split learned sequences. Only the contiguous nonempty segment containing the automatic reference is inferred; omitted source times remain unsupported. No fallback crop or temporal bridge.',
      'policy':'Reference-conditioned independent branch; absorbing loss in both directions, no point reacquisition or interpolation. No geometry modification.'}
    record['queries']['reference_mask_policy']='exact old initialization SAM intersected with dense SAM' if reference.get('initialization_mask_file') else 'dense raw SAM interior only'
    write(out/'experiment.json',record)
    uv,visible,crop_uv,transforms,runtime=dense.crop_cotracker(images,masks,seeds,anchor,out)
    bounds=np.isfinite(crop_uv).all(2)&(crop_uv>=0).all(2)&(crop_uv<=383).all(2)
    support=motion.support_masks(uv,visible&bounds,masks,anchor);valid=support['valid_no_reacquisition']&hits[None]
    np.savez_compressed(out/'tracks.npz',raw_model_uv=uv,raw_crop_model_uv=crop_uv,model_visibility=visible,
      crop_bounds=bounds,query_uv=seeds,canonical_xyz=canonical,reference_ray_hits=hits,valid_without_reacquisition=valid,
      sam_membership=support['sam_membership'],image_bounds=support['image_bounds'],training_point_ids=train,heldout_point_ids=held,
      timestamps_s=np.array([r['timestamp_s'] for r in rows]),sample_indices=np.array([r['index'] for r in rows]))
    record.update(status='learned_inference_complete_evaluation_pending',runtime=runtime,wall_seconds=time.monotonic()-start,
      raw_tracks_sha256=sha(out/'tracks.npz'),source_frame_sha256=[sha(CASE/r['file']) for r in rows],sam_mask_sha256=[sha(p) for p in mask_paths])
    write(out/'experiment.json',record)
    print(json.dumps({'object_id':tid,'branch':branch_id,'frames':len(rows),'inference_seconds':runtime['inference_seconds'],'out':str(out)}),flush=True)


def load_audits(paths,expected_mask_hashes=None,expected_image_hashes=None):
    verdicts={};scope={};provenance=[]
    for path in paths:
        path=Path(path).resolve();d=read(path);provenance.append({'file':str(path),'sha256':sha(path)})
        if expected_mask_hashes is not None:
            evidence_path=path.parent/'evidence.json'
            if not evidence_path.exists():raise ValueError('Audit must retain exact image/mask input evidence')
            evidence=read(evidence_path)
            if isinstance(evidence,dict):evidence=evidence.get('observations',evidence.get('evidence',evidence.get('frames',[])))
            ev={(r['track_id'],r['sample_index']):r for r in evidence}
            for item in d['result']['observations']:
                key=(item['track_id'],item['sample_index'])
                if key not in expected_mask_hashes:continue
                e=ev.get(key,{})
                actual=e.get('raw_mask_sha256',e.get('mask_sha256',e.get('source_mask_sha256')))
                if actual!=expected_mask_hashes[key]:raise ValueError('Audit used a different mask; no index-only verdict transfer')
                if expected_image_hashes is not None:
                    actual_rgb=e.get('source_frame_sha256',e.get('image_sha256',e.get('source_image_sha256')))
                    if actual_rgb!=expected_image_hashes[key]:raise ValueError('Audit used a different source image')
        for item in d['result']['observations']:
            key=(item['track_id'],item['sample_index'])
            if 'verdict' in item:verdicts[key]=item['verdict']
            if all(k in item for k in ('single_component_visible','component_boundary_plausible','other_components_inside_mask')):
                scope[key]=item
    return verdicts,scope,provenance


def inherited_scope(tid,rows):
    oldrows=read(BASE/'frames/index.json')['frames'];scope=read(ASSEMBLY/'scope/audit.json')
    result={}
    for item in scope['result']['observations']:
        if item['track_id']!=tid:continue
        oldtime=oldrows[item['sample_index']-1]['timestamp_s']
        if oldtime<rows[0]['timestamp_s']-.0001 or oldtime>rows[-1]['timestamp_s']+.0001:continue
        new=dense.nearest_sample(rows,oldtime,.0001)
        result[new['index']]=item
    return result


def evaluate(out,audit_paths):
    out=Path(out).resolve();record=read(out/'experiment.json');tid=record['object_id']
    _,_,_,rows,images,masks,mask_paths,vertices,faces,colors=material(tid)
    arrays=np.load(out/'tracks.npz');uv=arrays['raw_model_uv'];valid=arrays['valid_without_reacquisition'];canonical=arrays['canonical_xyz'];train=arrays['training_point_ids'];held=arrays['heldout_point_ids']
    selected=set(arrays['sample_indices'].tolist());keep=[i for i,row in enumerate(rows) if row['index'] in selected]
    rows=[rows[i] for i in keep];images=[images[i] for i in keep];masks=[masks[i] for i in keep];mask_paths=[mask_paths[i] for i in keep]
    if record['raw_tracks_sha256']!=sha(out/'tracks.npz'):raise ValueError('Learned track archive changed')
    if record['source_frame_sha256']!=[sha(CASE/r['file']) for r in rows] or record['sam_mask_sha256']!=[sha(p) for p in mask_paths]:raise ValueError('Inference source changed')
    initial=CASE/'segmentation/automatic-components-v1/model-audit-v1/audit.json'
    expected={(tid,row['index']):sha(maskpath) for row,maskpath in zip(rows,mask_paths)}
    expected_rgb={(tid,row['index']):sha(CASE/row['file']) for row in rows}
    verdicts,scopes,provenance=load_audits([initial]+[Path(p) for p in audit_paths],expected,expected_rgb);prior=inherited_scope(tid,rows)
    withheld=read(CASE/'segmentation/automatic-components-v1/quality.json')['tracks'][tid]['withheld_samples']
    k=np.array(record['intrinsics']['matrix']);rotation=np.array(record['reference']['candidate']['rotation']);translation=np.array(record['reference']['candidate']['translation'])
    audit_key=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()[:16]
    review_root=out/'evaluations'/audit_key
    if review_root.exists():raise ValueError('Preserve existing evaluation; identical audit inputs already evaluated')
    records=[];(review_root/'details').mkdir(parents=True);(review_root/'full_projection_masks').mkdir()
    for j,row in enumerate(rows):
        sample=row['index'];present=train[valid[j,train]];checks=held[valid[j,held]];verdict=verdicts.get((tid,sample),'unassessed')
        rec={'sample_index':sample,'timestamp_s':row['timestamp_s'],'status':'unsupported','reasons':[],
          'source_frame_sha256':record['source_frame_sha256'][j],'sam_mask_sha256':record['sam_mask_sha256'][j],
          'training_visible_point_ids':present.tolist(),'heldout_visible_point_ids':checks.tolist(),'mask_model_audit_verdict':verdict,
          'usable_pose_exported':False,'reference_initialization_independently_verified':False,'unverified_orientation_branch':True}
        fit=solve_pose(canonical[present],uv[j,present],k,rotation,translation) if len(present)>=6 else {'success':False,'reason':'fewer_than_6_valid_training_correspondences'}
        overlay=images[j].copy()
        if fit['success']:
            r,t=fit['rotation'],fit['translation'];errors=np.linalg.norm(motion.project(canonical[checks],k,r,t)-uv[j,checks],axis=1)
            rendered=render_mesh(vertices,faces,k,r,t,row['width'],row['height'],vertex_colors=colors)
            intersection=int((rendered['mask']&masks[j]).sum());iou=intersection/max(1,int((rendered['mask']|masks[j]).sum()));coverage=intersection/max(1,int(masks[j].sum()))
            rec.update(candidate_rotation=r.tolist(),candidate_translation=t.tolist(),training_inliers=len(fit['inlier_indices']),
              heldout_residual_px=errors.tolist(),heldout_median_px=float(np.median(errors)) if len(errors) else None,
              heldout_p90_px=float(np.percentile(errors,90)) if len(errors) else None,raw_full_projection_iou_with_sam=iou,source_mask_coverage=coverage)
            rec['reasons']=motion.pose_gate(len(present),len(fit['inlier_indices']),errors,iou,coverage,verdict)
            # Both prior and new scope vetoes persist. They are never erased by a
            # looser visual plausibility judgment or a favorable numeric fit.
            if sample in prior:
                rec['inherited_model_scope_audit']=prior[sample];rec['reasons']+=scope_veto(prior[sample])
            if (tid,sample) in scopes:
                rec['fresh_model_scope_audit']=scopes[(tid,sample)];rec['reasons']+=scope_veto(scopes[(tid,sample)])
            if sample in withheld:rec['reasons'].append('source_mask_failed_geometric_gate')
            rec['reasons']=list(dict.fromkeys(rec['reasons']))
            if sample==record['reference_sample_index']:rec['status']='reference_candidate_unverified_geometry_and_pose'
            elif not rec['reasons']:rec.update(status='supported_conditional_motion_candidate',conditional_pose_exported=True)
            else:rec['status']='unsupported_candidate_projection_retained'
            raw=rendered['mask'];overlay[raw]=(.45*overlay[raw]+.55*rendered['rgb'][raw]).astype(np.uint8)
            edges=cv2.morphologyEx(masks[j].astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0;overlay[edges]=[0,255,180]
            Image.fromarray(rendered['mask'].astype(np.uint8)*255).save(review_root/'full_projection_masks'/f'{sample:06d}.png')
        else:rec['reasons']=[fit['reason']]
        Image.fromarray(overlay).save(review_root/'details'/f'{sample:06d}.jpg',quality=92);rec['overlay_file']=str((review_root/'details'/f'{sample:06d}.jpg').relative_to(out));records.append(rec)
    record.update(status='complete_conditional_tracking_experiment',frames=records,frame_count=len(records),sampling_fps=5.,
      supported_nonreference_frames=sum(r['status']=='supported_conditional_motion_candidate' for r in records),reference_frames=1,
      audit_inputs=provenance,prior_scope_veto_retained=True,scope_policy='Raw SAM full-mesh silhouette comparison; no occlusion-aware clipping or substitution in this run.',
      raw_projections_clipped_to_sam=False,evaluation='Disjoint point IDs are internal checks conditional on unverified mesh/ref/K; reserved temporal samples cannot select additional refs.')
    previous=read(out/'experiment.json')
    if 'frames' in previous:
        archive=out/'evaluation-history';archive.mkdir(exist_ok=True)
        dest=archive/('experiment-'+sha(out/'experiment.json')[:16]+'.json')
        if not dest.exists():shutil.copy2(out/'experiment.json',dest)
    write(review_root/'experiment.json',record)
    write(out/'experiment.json',record);print(json.dumps({'object_id':tid,'branch':record['branch_id'],'supported':record['supported_nonreference_frames'],'out':str(out)}),flush=True)


def fit_reference(tid,row,out,initialization_mask_file=None):
    """Fit same immutable mesh at a separately selected/audited source view."""
    out=Path(out);out.mkdir(parents=True)
    old,meta_path,mesh_path,rows,images,masks,mask_paths,vertices,faces,colors=material(tid)
    j=next(i for i,r in enumerate(rows) if r['index']==row['sample_index']);rgb,target=images[j],masks[j]
    if initialization_mask_file is not None:target=np.array(Image.open(initialization_mask_file).convert('L'))>127
    h,w=target.shape;work=(192,round(h*192/w));smalltarget=cv2.resize(target.astype(np.uint8),work,interpolation=cv2.INTER_NEAREST).astype(bool)
    k=np.array(old['intrinsics']['matrix']);smallk=init.resized_intrinsics(k,(w,h),work);center=(vertices.min(0)+vertices.max(0))/2;centered=vertices-center
    coarse=[]
    for i,r in enumerate(init.generic_orientations(96)):
        t=init.initial_translation(centered,r,smallk,smalltarget);s=init.iou(init.silhouette(centered,faces,r,t,smallk,work),smalltarget);coarse.append((s,i,r,t))
    coarse.sort(key=lambda x:-x[0]);seeds=[]
    for item in coarse:
        if all(init.rotation_distance_deg(item[2],a[2])>=35 for a in seeds):seeds.append(item)
        if len(seeds)==4:break
    fits=[]
    for score,index,r,t in seeds:
        fit=init.refine(centered,faces,smallk,smalltarget,r,t,160);fit['translation']=fit['translation']-fit['rotation']@center
        rendered=render_mesh(vertices,faces,k,fit['rotation'],fit['translation'],w,h,vertex_colors=colors)
        overlap=init.iou(rendered['mask'],target);appearance=init.appearance_similarity(rgb,rendered['rgb'],rendered['mask']&target)
        fit.update(iou_full_resolution=overlap,appearance=appearance,combined_rank=init.reference_rank_score(overlap,appearance,.25),seed_index=index)
        fits.append(fit)
    fits.sort(key=lambda f:-f['combined_rank']);best=fits[0]
    def serial(f):return {key:value.tolist() if isinstance(value,np.ndarray) else value for key,value in f.items()}
    result={'schema_version':'0.2','status':'unverified_independent_reference_candidate','object_id':tid,'source_timestamp_s':rows[j]['timestamp_s'],
      'source_sample_index':rows[j]['index'],'source_video_sha256':read(CASE/'manifest.json')['source']['sha256'],
      'mesh_file':str(mesh_path),'mesh_sha256':sha(mesh_path),'intrinsics':old['intrinsics'],'candidate':serial(best),
      'all_candidates':[serial(f) for f in fits],'selection_evidence':row,
      'source_frame_sha256':sha(CASE/rows[j]['file']),'sam_mask_sha256':sha(mask_paths[j]),'physical_pose_accuracy_measured':False,
      'reference_rotation_ambiguous':any(init.rotation_distance_deg(best['rotation'],f['rotation'])>45 and best['combined_rank']-f['combined_rank']<.03 for f in fits[1:]),
      'fixed_native_mesh_scale':1.,'manual_geometry_used':False,'manual_pose_used':False,'manual_view_choice':False,
      'policy':'Same immutable mesh fit by generic SO(3) search and full projected silhouette/appearance. New reference never selected using evaluation pose errors.'}
    if initialization_mask_file is not None:
        result.update(initialization_mask_file=str(Path(initialization_mask_file).resolve()),initialization_mask_sha256=sha(initialization_mask_file))
    write(out/'reference.json',result);result.update(reference_manifest_file=str(out/'reference.json'),reference_manifest_sha256=sha(out/'reference.json'))
    return result


def late_lid_reference(run):
    """One automatic anchor within the already-supported other-component overlap."""
    run=Path(run).resolve();out=run/'reference-late-01'
    if out.exists():raise ValueError('Preserve previous late initialization')
    base=read(ROOT/'data/jabra-local-v2/motion-container_base-v1/experiment.json')
    ear=read(CASE/'alignment/generated-earbud-crop-motion-v1/experiment.json')
    times=[]
    for a in base['frames']:
        if not a['status'].startswith(('supported_','reference_')):continue
        if any(abs(a['timestamp_s']-b['timestamp_s'])<.001 and b['status'].startswith(('supported_','reference_')) for b in ear['frames']):times.append(a['timestamp_s'])
    if not times:raise ValueError('No observed base/earbud overlap: no desired time invented')
    selection=read(BASE/'automatic-reconstruction-inputs-v1/container_lid/selection.json')
    scope=read(ASSEMBLY/'scope/audit.json');checks={(r['track_id'],r['sample_index']):r for r in scope['result']['observations']}
    eligible=[]
    for candidate in selection['all_ranked_candidates']:
        audit=checks.get(('container_lid',candidate['sample_index']))
        if audit and not scope_veto(audit) and audit['reconstruction_view_usable'] and audit['interior_state']=='not_applicable' and candidate['timestamp_s']>=16 and min(times)-.001<=candidate['timestamp_s']<=max(times)+.001:
            eligible.append(candidate)
    if not eligible:raise ValueError('No existing scope-safe late reference within actual other-component overlap')
    chosen=sorted(eligible,key=lambda r:(-r['score'],r['timestamp_s']))[0]
    rows=read(CASE/'frames/index.json')['frames'];dense_row=dense.nearest_sample(rows,chosen['timestamp_s'],.0001)
    evidence=dict(chosen,source_old_sample_index=chosen['sample_index'],sample_index=dense_row['index'],
      selection_rule='Intersect existing conditional base+earbud supported source-time interval with old scope-safe lid candidates>=16s, then highest original image quality score. Exactly one bounded independent reference; no pose residual/evaluation outcome ranking.',
      mask_verdict='plausible_visible_mask',scope_safe=True,quality_score=chosen['score'])
    write(run/'late-reference-selection.json',{'chosen':evidence,'eligible':eligible,'overlap_time_bounds_s':[min(times),max(times)],'source_scope_audit_sha256':sha(ASSEMBLY/'scope/audit.json'),
       'temporal_evaluation_policy':'Source18.001578s reference excluded; all other original1fps source timestamps in bounded clip remain held-out for temporal reporting. No pose-outcome ranking in selection.', 'manual_view_choice':False})
    reference=fit_reference('container_lid',evidence,out,BASE/chosen['mask_file'])
    return reference


def additional(tid,run,scope_paths,maximum):
    """Only fresh scope-safe images and non-evaluation PTS may be selected."""
    run=Path(run);old,_,_,rows,images,masks,maskpaths,*_=material(tid)
    expected={(tid,row['index']):sha(maskpath) for row,maskpath in zip(rows,maskpaths)}
    expected_rgb={(tid,row['index']):sha(CASE/row['file']) for row in rows}
    verdicts,scopes,provenance=load_audits(scope_paths,expected,expected_rgb);candidates=[]
    for row,image,mask in zip(rows,images,masks):
        scope=scopes.get((tid,row['index']));safe=scope is not None and not scope_veto(scope) and scope.get('reconstruction_view_usable',True)
        if scope and scope.get('interior_state') not in {'empty_visible','not_applicable'}:safe=False
        yy,xx=np.where(mask);sharp=float(cv2.Laplacian(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY),cv2.CV_64F)[mask].var())
        border=bool(xx.min()<=1 or yy.min()<=1 or xx.max()>=mask.shape[1]-2 or yy.max()>=mask.shape[0]-2)
        score=np.log1p(mask.sum())+min(6.,np.log1p(sharp))-(5 if border else 0)
        candidates.append({'sample_index':row['index'],'timestamp_s':row['timestamp_s'],'scope_safe':safe,
          'mask_verdict':verdicts.get((tid,row['index']),'unassessed'),'quality_score':float(score),'source_mask_area':int(mask.sum()),
          'source_sharpness':sharp,'border_touching':border,'model_scope':scope})
    chosen=choose_references(candidates,old['source_timestamp_s'],maximum,2.)
    write(run/'automatic-reference-selection.json',{'rule':'fresh scope-safe complete component; plausible model audit; source image quality descending; >=2s temporal separation; exclude all source PTS within.02s of an integer(second) reserved before selection; never use pose/evaluation scores','audit_inputs':provenance,'candidates':candidates,'chosen':chosen,'manual_view_choice':False})
    for n,row in enumerate(chosen,1):
        path=run/f'reference-extra-{n:02d}';reference=fit_reference(tid,row,path)
        if reference['candidate']['iou_full_resolution']<.75:
            write(path/'withheld.json',{'reason':'same-reference silhouette IoU below predeclared.75 initializationgate'});continue
        infer_branch(tid,run/f'branch-extra-{n:02d}',reference,f'additional-{n:02d}')


def merge(run):
    run=Path(run).resolve();paths=sorted(run.glob('branch*/experiment.json'));branches=[read(p) for p in paths]
    if not branches or any('frames' not in b for b in branches):raise ValueError('Evaluate every branch before merging')
    records=merge_branches(branches);record=dict(branches[0]);record.pop('reference',None)
    folders={b['branch_id']:p.parent.name for b,p in zip(branches,paths)}
    for rec in records:
        if rec.get('overlay_file') and rec.get('selected_branch_id'):
            rec['overlay_file']=str(Path(folders[rec['selected_branch_id']])/rec['overlay_file'])
        if rec.get('diagnostic_overlay_file'):
            rec['diagnostic_overlay_file']=str(Path(folders[rec['diagnostic_branch_id']])/rec['diagnostic_overlay_file'])
        if rec.get('selected_branch_id')=='late-independent-01' or rec.get('diagnostic_branch_id')=='late-independent-01':
            rec['acceptance_gate_reasons']=rec['reasons'].copy()
            rec['informational_reasons']=['independent_reference_orientation_unverified']
            rec['reasons']=rec['reasons']+rec['informational_reasons']
    record.update(branch_id='merged-independent-references',status='complete_conditional_multireference_experiment',frames=records,
      branch_inputs=[{'file':str(p),'sha256':sha(p)} for p in paths],
      reference_sample_indices=[b['reference_sample_index'] for b in branches],
      supported_nonreference_frames=sum(r['status']=='supported_conditional_motion_candidate' for r in records),reference_frames=sum(r['status'].startswith('reference_') for r in records),
      merge_policy='nearest reference in source time among passing candidates, never residual-ranked. Supported branches disagreeing by>35deg or>25%relative translation are withheld. No averaging/interpolation/point reacquisition.',
      physical_assembly_verified=False,hinge_or_contact_recovered=False)
    if (run/'experiment.json').exists():
        archive=run/'merge-history';archive.mkdir(exist_ok=True)
        dest=archive/('experiment-'+sha(run/'experiment.json')[:16]+'.json')
        if not dest.exists():shutil.copy2(run/'experiment.json',dest)
    write(run/'experiment.json',record);print(json.dumps({'merged':str(run/'experiment.json'),'supported':record['supported_nonreference_frames'],'references':record['reference_frames']}))


def qa(run):
    run=Path(run).resolve();record=read(run/'experiment.json');rows=record['frames']
    source_rows={r['index']:r for r in read(CASE/'frames/index.json')['frames']}
    if sha(record['provenance']['mesh_file'])!=record['provenance']['mesh_sha256']:raise ValueError('Mesh lineage failed')
    if record['fixed_native_mesh_scale']!=1.:raise ValueError('Native scale drifted')
    for rec in rows:
        if rec['timestamp_s']!=source_rows[rec['sample_index']]['timestamp_s']:raise ValueError('Source PTS mismatch')
        if rec['status']=='supported_conditional_motion_candidate':
            blocking=[r for r in rec['reasons'] if r!='independent_reference_orientation_unverified']
            if blocking or rec.get('mask_model_audit_verdict')!='plausible_visible_mask':raise ValueError('Invalid accepted evidence')
        if 'candidate_rotation' in rec:
            r=np.array(rec['candidate_rotation']);t=np.array(rec['candidate_translation'])
            if not np.isfinite(t).all() or not np.allclose(r.T@r,np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(r),1.,atol=1e-5):raise ValueError('Nonrigid or nonfinite pose')
        if rec.get('overlay_file') and not (run/rec['overlay_file']).is_file():raise ValueError('Missing raw projection diagnostic')
    branch_checks=[]
    for item in record['branch_inputs']:
        p=Path(item['file']);b=read(p);a=np.load(p.parent/'tracks.npz')
        if sha(p)!=item['sha256'] or sha(p.parent/'tracks.npz')!=b['raw_tracks_sha256']:raise ValueError('Branch evidence changed')
        train=set(a['training_point_ids'].tolist());held=set(a['heldout_point_ids'].tolist())
        if train&held:raise ValueError('Training/heldout point IDs overlap')
        anchor=next(i for i,r in enumerate(b['frames']) if r['sample_index']==b['reference_sample_index'])
        valid=a['valid_without_reacquisition']
        for direction in [1,-1]:
            previous=anchor
            for j in range(anchor+direction,len(valid) if direction==1 else -1,direction):
                if np.any(valid[j]&~valid[previous]):raise ValueError('Forbidden point reacquisition')
                previous=j
        branch_checks.append({'branch_id':b['branch_id'],'training_heldout_ids_disjoint':True,'absorbing_loss_verified':True,'raw_track_hash_verified':True})
    # Deterministic inspection choices; no pose/geometry is edited from this QA.
    refs=[r for r in rows if r['status'].startswith('reference_')]
    supported=[r for r in rows if r['status']=='supported_conditional_motion_candidate']
    anchor_record=refs[-1] if refs else None
    anchor_t=anchor_record['timestamp_s'] if anchor_record else rows[0]['timestamp_s']
    chosen_branch=anchor_record.get('selected_branch_id') if anchor_record else None
    branch_supported=[r for r in supported if r.get('selected_branch_id')==chosen_branch] or supported
    rejected=[r for r in rows if r['status'].startswith('unsupported') and (r.get('overlay_file') or r.get('diagnostic_overlay_file'))]
    choices=[]
    for rec in [anchor_record,branch_supported[0] if branch_supported else None,
                next((r for r in rejected if r['timestamp_s']>anchor_t),None),rejected[-1] if rejected else None]:
        if rec is not None and rec['sample_index'] not in {r['sample_index'] for r in choices}:choices.append(rec)
    canvas=Image.new('RGB',(1200,680),'#111b27');draw=ImageDraw.Draw(canvas)
    draw.text((16,10),record['object_id']+' | full source LEFT + automatic display crop RIGHT; raw files retained',font=motion.font(18),fill='white')
    seg=CASE/'segmentation/automatic-components-v1'
    track=next(t for t in read(seg/'tracks.json')['tracks'] if t['track_id']==record['object_id'])
    maskmap={r['sample_index']:seg/r['raw_mask_file'] for r in track['frames']}
    for j,rec in enumerate(choices):
        x=(j%2)*600;y=45+(j//2)*310;image=Image.open(run/rec.get('overlay_file',rec.get('diagnostic_overlay_file')))
        thumb=ImageOps.contain(image,(135,235));canvas.paste(thumb,(x+10+(135-thumb.width)//2,y+55))
        mask=np.array(Image.open(maskmap[rec['sample_index']]).convert('L'))>127;yy,xx=np.where(mask)
        if len(xx):
            pad=max(25,round(max(np.ptp(xx),np.ptp(yy))*.35));box=(max(0,int(xx.min())-pad),max(0,int(yy.min())-pad),min(mask.shape[1],int(xx.max())+pad+1),min(mask.shape[0],int(yy.max())+pad+1))
            crop=ImageOps.contain(image.crop(box),(430,235));canvas.paste(crop,(x+150+(430-crop.width)//2,y+55))
        draw.text((x+10,y),f"#{rec['sample_index']} | {rec['timestamp_s']:.6f}s | {rec['status'][:38]}",font=motion.font(16),fill='#85efd4' if rec['status'].startswith('supported') else '#ffc98c')
        draw.text((x+10,y+23),', '.join(rec['reasons'])[:72] or 'conditional/unverified reference and geometry',font=motion.font(13),fill='#d6e0e8')
    canvas.save(run/'qa-preview.jpg',quality=94)
    result={'source_timestamps_verified':True,'mesh_sha256_verified':True,'constant_native_scale_verified':True,'rigid_candidate_matrices_verified':True,
       'branch_checks':branch_checks,'raw_projection_paths_verified':True,'supported':record['supported_nonreference_frames'],'preview_samples':[r['sample_index'] for r in choices],
       'visual_qa_file':'qa-preview.jpg','unsupported_samples_never_interpolated':True,'physical_pose_accuracy_verified':False,'full_browser_rendering_tested':False}
    write(run/'qa.json',result);print(json.dumps(result))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['infer','evaluate','additional','merge','qa','late-fit','late-infer']);p.add_argument('--object-id',choices=list(REFS));p.add_argument('--out',type=Path,required=True);p.add_argument('--audit',type=Path,action='append',default=[]);p.add_argument('--max-extra-refs',type=int,default=2);a=p.parse_args()
    if a.action=='infer':infer_branch(a.object_id,a.out)
    elif a.action=='evaluate':evaluate(a.out,a.audit)
    elif a.action=='additional':additional(a.object_id,a.out,a.audit,a.max_extra_refs)
    elif a.action=='merge':merge(a.out)
    elif a.action=='qa':qa(a.out)
    elif a.action=='late-fit':late_lid_reference(a.out)
    else:
        p=a.out/'reference-late-01/reference.json';reference=read(p);reference.update(reference_manifest_file=str(p.resolve()),reference_manifest_sha256=sha(p))
        infer_branch('container_lid',a.out/'branch-late-01',reference,'late-independent-01',4.)
if __name__=='__main__':main()
