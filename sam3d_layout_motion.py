"""Fresh conditional motion from one genuine SAM3D raw-layout reference.

Standalone experiment only. Never reads another asset's canonical points or
poses, changes geometry, interpolates, reanchors, selects models or runs physics.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import time

import numpy as np
from PIL import Image
import sam3d_milestone as M
import upgrade_bundles as U
from native_appearance import load_native_glb, ray_native_mesh_points, render_native_mesh
from native_alignment import native_representation_receipt
from sam3d_source_projection import recover_estimated_camera, raw_layout_projection

ROOT=Path(__file__).resolve().parent
DEPENDENCY_NAMES=['sam3d_layout_motion.py','scripts/run_sam3d_layout_motion.py','scripts/stage4_native_track.py',
                  'native_appearance.py','native_alignment.py','portable_twin.py','stage5.py','alignment_render.py','alignment_fit.py',
                  'sam3d_source_projection.py','sam3d_coordinates.py']
TRACKER_FILES=['models/cotracker3/scaled_offline.pth','models/cotracker3/provenance.json']
spec=importlib.util.spec_from_file_location('fresh_native_tracking_helpers',ROOT/'scripts/stage4_native_track.py')
T=importlib.util.module_from_spec(spec);spec.loader.exec_module(T)


def require(value,message):
    if not value:raise ValueError(message)


def ref(path,root):
    root=Path(root).resolve();path=Path(path)
    if not path.is_absolute():path=root/path
    try:relative=path.relative_to(root).as_posix()
    except ValueError:raise ValueError('Diagnostic input is outside the workspace') from None
    if relative in DEPENDENCY_NAMES+TRACKER_FILES:
        current=root
        for part in Path(relative).parts:
            current=current/part;require(not current.is_symlink(),'Diagnostic source symlinks are not permitted')
        require(path.is_file(),'Frozen diagnostic source must be a regular file')
    else:path=U._local(path,root)
    return {'path':path.relative_to(root).as_posix(),'sha256':U.sha(path),'bytes':path.stat().st_size}


def verify_identity(model,glb_sha,projection):
    require(model.get('backend')=='Meta SAM 3D Objects' and model.get('fixture') in (None,False)
            and model.get('sam3d_output') is True and model.get('learned_inference_executed') is True
            and model.get('geometry_reconstructed_by_model') is True
            and model.get('manual_geometry_fallback') is False and model.get('manual_pose_used') is False,
            'A genuine-result SAM3D native candidate is required; no baseline geometry')
    require(projection.get('native_glb',{}).get('sha256')==glb_sha
            and projection.get('source_binding',{}).get('track_id')==model.get('object_id')
            and projection['source_binding']['source']['video_sha256']==model['source']['video_sha256'],
            'Projection seed and new native geometry/source/component differ')
    require(projection.get('optimized') is False and projection.get('old_pose_reused') is False,
            'Only the automatic raw-layout seed is permitted')


def fresh_ray_queries(seeds,mesh,k,r,t,*,ray_function=ray_native_mesh_points):
    """Exact new-mesh ray hits with an outer eight-ray memory bound."""
    seeds=np.asarray(seeds,float);require(seeds.ndim==2 and seeds.shape[1]==2 and len(seeds)<=96,'Bounded Nx2 fresh queries required')
    xyz=np.full((len(seeds),3),np.nan);hits=np.zeros(len(seeds),bool)
    for start in range(0,len(seeds),8):
        p,valid=ray_function(seeds[start:start+8],mesh,k,r,t)
        xyz[start:start+len(p)]=p;hits[start:start+len(valid)]=valid
    train,held=T.split_ids(hits)
    return xyz,hits,train,held


def fit_frame(canonical,uv,valid,train,held,k,r,t,*,solver=T.solve_pose):
    train=np.asarray(train,int);held=np.asarray(held,int)
    require(not np.intersect1d(train,held).size and np.all(train%3!=0) and np.all(held%3==0),
            'Training/heldout query identities must be fixed and disjoint before fitting')
    present=train[np.asarray(valid,bool)[train]];checks=held[np.asarray(valid,bool)[held]]
    fit=solver(canonical[present],uv[present],k,r,t) if len(present)>=6 else {'success':False,'reason':'fewer_than_6_valid_training_correspondences'}
    errors=np.empty(0);positive=True
    if fit['success'] and len(checks):
        camera=canonical[checks]@fit['rotation'].T+fit['translation']
        positive=bool(np.isfinite(camera).all() and np.all(camera[:,2]>1e-8))
        errors=np.linalg.norm(T.project(canonical[checks],k,fit['rotation'],fit['translation'])-uv[checks],axis=1) if positive else np.full(len(checks),np.inf)
    return fit,present,checks,errors,positive


def conditional_gate(train_count,inlier_count,errors,iou,coverage,verdict,*,rendered,positive,withheld):
    reasons=T.pose_gate(train_count,inlier_count,errors,iou,coverage,verdict)
    if not rendered:
        reasons=[('raw_full_mesh_silhouette_not_evaluated' if x=='raw_full_mesh_silhouette_gate_failed' else x) for x in reasons]
        if 'raw_full_mesh_silhouette_not_evaluated' not in reasons:reasons.append('raw_full_mesh_silhouette_not_evaluated')
    if not positive:reasons.append('heldout_nonpositive_or_nonfinite_depth')
    if withheld:reasons.append('source_mask_failed_geometric_gate')
    return reasons


def source_rows_by_id(frames,track_rows):
    by_id={r['index']:r for r in frames}
    require(len(by_id)==len(frames),'Unique original source sample IDs required')
    rows=[]
    for row in track_rows:
        require(row['sample_index'] in by_id,'Track source sample is missing')
        source=by_id[row['sample_index']]
        require(source['timestamp_s']==row['timestamp_s'],'Track/source PTS differ')
        rows.append(source)
    return rows


def validate_queries(query,plan):
    require(set(query)=={'query_uv','canonical_xyz','reference_ray_hits','training_point_ids','heldout_point_ids'},'Exact fresh query arrays required')
    uv=query['query_uv'];xyz=query['canonical_xyz'];hits=query['reference_ray_hits']
    require(uv.shape==(plan['queries']['count'],2) and 18<=len(uv)<=96 and np.isfinite(uv).all()
            and xyz.shape==(len(uv),3) and hits.shape==(len(uv),) and hits.dtype==np.bool_
            and np.isfinite(xyz[hits]).all(),'Fresh query shapes or finite native hits differ')
    train,held=T.split_ids(hits)
    require(np.array_equal(query['training_point_ids'],train) and np.array_equal(query['heldout_point_ids'],held)
            and train.tolist()==plan['queries']['training_ids'] and held.tolist()==plan['queries']['heldout_ids']
            and len(train)>=12 and len(held)>=6 and int(hits.sum())==plan['queries']['fresh_new_mesh_ray_hits'],
            'Fresh stable training/heldout point identities differ')


def validate_scope(plan):
    rows=plan['rows']
    require([r['sample_index'] for r in rows]==list(range(38,48)) and plan['anchor_row']==2 and plan['anchor_sample_index']==40,
            'This frozen smoke is samples38..47 with anchor40 only')
    require(all(a['timestamp_s']<b['timestamp_s'] for a,b in zip(rows,rows[1:])) and 0<rows[-1]['timestamp_s']-rows[0]['timestamp_s']<=5,
            'Ordered native source PTS within five seconds required')
    require(plan['render_samples']==[40,45] and [r['sample_index'] for r in rows if r['audit_verdict']=='plausible_visible_mask']==[40,45],
            'Only cached-audited samples40 and45 may receive full mesh renders')
    require(all(plan[k] is False for k in ['old_geometry_or_poses_read','reanchor','reacquisition','interpolation','operator_fallback',
                                         'automatic_retry','program_or_catalog_promotion','physical_accuracy_verified','simulation_ready'])
            and plan['maximum_tracker_runs']==1 and plan['maximum_full_mesh_renders']==2
            and plan['device']=='mps' and plan['wall_timeout_seconds']==720 and plan['model_max_dimension']==384
            and plan['queries']['ray_batch_limit']==8 and plan['queries']['old_canonical_points_read'] is False,
            'Frozen smoke safety scope differs')


def dependencies(root):
    return [ref(root/name,root) for name in DEPENDENCY_NAMES]


def prepare(model_metadata,projection_directory,output,*,first_sample=38,last_sample=47,anchor_sample=40,root=ROOT):
    root=Path(root).resolve();out=M._directory(output,root,exists=False)
    require(not out.exists(),'Fresh motion experiment directory required')
    metadata_path=U._local(model_metadata,root);model=U.read(metadata_path)
    require(model.get('outputs')=={'glb_path':'asset.glb'},'An unchanged imported native asset is required')
    glb=U._local(metadata_path.parent/'asset.glb',root);glb_sha=U.sha(glb)
    projection_directory=M._directory(projection_directory,root);projection=U.read(projection_directory/'plan.json')
    diagnostic=U.read(projection_directory/'diagnostic.json')
    verify_identity(model,glb_sha,projection)
    require(diagnostic.get('plan_sha256')==U.sha(projection_directory/'plan.json')
            and diagnostic.get('status')=='one_raw_projection_rendered' and diagnostic.get('source_video_pose_accepted') is False,
            'Frozen raw projection diagnostic differs')
    source=model['source'];selection=U._local(source['selection_manifest_path'],root)
    require(U.sha(selection)==source['selection_manifest_sha256'],'Automatic source selection changed')
    case=next((p for p in selection.parents if p.is_relative_to(root/'data') and (p/'manifest.json').is_file()),None)
    require(case is not None,'Source case unavailable')
    manifest=U.read(case/'manifest.json');require(manifest['source']['sha256']==source['video_sha256'],'Source video identity differs')
    video=U._local(case/manifest['source']['file'],root);require(U.sha(video)==source['video_sha256'],'Source video bytes differ')
    segmentation=case/'segmentation/flow-components-1';config=U.read(segmentation/'annotations.json')
    require(config.get('manual_seed_boxes_used') is False,'Explicit automatic SAM prompts required')
    tracks=U.read(segmentation/'tracks.json');track=next(t for t in tracks['tracks'] if t['track_id']==model['object_id'])
    index=U.read(case/'frames/index.json')
    require(index['time_domain']==U.TIME_DOMAIN,'Original source PTS required')
    selected=[r for r in track['frames'] if first_sample<=r['sample_index']<=last_sample]
    require([r['sample_index'] for r in selected]==list(range(first_sample,last_sample+1)) and 2<=len(selected)<=12,'Fixed contiguous source window required')
    require(source['sample_index']==anchor_sample and first_sample<=anchor_sample<=last_sample,'Raw seed anchor differs')
    audit_path=segmentation/'model-audit-v1/audit.json';audit=U.read(audit_path)
    require(audit['source_sha256']==source['video_sha256'] and audit['source_track_index_sha256']==U.sha(segmentation/'tracks.json')
            and audit.get('manual_corrections_used') is False,'Cached audit does not bind exact source track/masks')
    verdicts={(r['track_id'],r['sample_index']):r['verdict'] for r in audit['result']['observations']}
    quality=U.read(segmentation/'quality.json');withheld=quality['tracks'][model['object_id']]['withheld_samples'];rows=[]
    for row,frame in zip(selected,source_rows_by_id(index['frames'],selected)):
        image=U._local(case/frame['file'],root);mask=U._local(segmentation/row['raw_mask_file'],root)
        require(U.sha(image)==row['source_frame_sha256'] and U.sha(mask)==row['raw_mask_sha256'],'Source frame/SAM bytes differ')
        rows.append({'sample_index':row['sample_index'],'timestamp_s':row['timestamp_s'],'frame':ref(image,root),
                     'mask':ref(mask,root),'width':frame['width'],'height':frame['height'],
                     'audit_verdict':verdicts.get((model['object_id'],row['sample_index']),'unassessed'),
                     'withheld_by_source_quality':row['sample_index'] in withheld})
    require(0<rows[-1]['timestamp_s']-rows[0]['timestamp_s']<=5 and all(a['timestamp_s']<b['timestamp_s'] for a,b in zip(rows,rows[1:])),
            'At most five seconds of strictly ordered source samples')
    anchor=next(i for i,r in enumerate(rows) if r['sample_index']==anchor_sample)
    image=np.asarray(Image.open(root/rows[anchor]['frame']['path']).convert('RGB'));mask=np.asarray(Image.open(root/rows[anchor]['mask']['path']))>0
    require(np.array_equal(image,np.asarray(Image.open(metadata_path.parent/source['image_path']).convert('RGB')))
            and np.array_equal(mask,np.asarray(Image.open(metadata_path.parent/source['mask_path']))>0),'Reference pixels must be the exact native model source')
    raw_dir=metadata_path.parent/'raw';raw_meta=U.read(raw_dir/'metadata.json');inventory={r['file']:r['sha256'] for r in raw_meta['outputs']['files']}
    require(inventory['object.glb']==glb_sha==U.sha(raw_dir/'object.glb')
            and inventory['layout-and-pointmap.npz']==U.sha(raw_dir/'layout-and-pointmap.npz'),'Fresh native output hashes differ')
    with U._safe_layout((raw_dir/'layout-and-pointmap.npz').read_bytes()) as layout:
        require(layout['pointmap'].shape==(*mask.shape,3),'Exact native pointmap mapping required')
        camera=recover_estimated_camera(layout['pointmap']);pose=raw_layout_projection(layout['rotation'],layout['translation'],layout['scale'])
    require(camera==projection['camera'] and pose==projection['raw_projection'],'Automatic raw seed/K changed')
    mesh=load_native_glb(glb);k=np.asarray(camera['K_estimated']);r=np.asarray(pose['R']);t=np.asarray(pose['t'])
    seeds=T.query_points(image,mask);canonical,hits,train,held=fresh_ray_queries(seeds,mesh,k,r,t)
    require(len(train)>=12 and len(held)>=6,'Insufficient fresh new-mesh reference ray support; no fallback')
    arrays={'query_uv':seeds,'canonical_xyz':canonical,'reference_ray_hits':hits,'training_point_ids':train,'heldout_point_ids':held}
    buffer=__import__('io').BytesIO();np.savez_compressed(buffer,**arrays);query_bytes=buffer.getvalue()
    sources=[ref(metadata_path,root),ref(glb,root),ref(video,root),ref(selection,root),
             *[ref(p,root) for p in [case/'manifest.json',case/'frames/index.json',segmentation/'annotations.json',segmentation/'tracks.json',segmentation/'quality.json',audit_path,
                 projection_directory/'plan.json',projection_directory/'diagnostic.json',raw_dir/'metadata.json',raw_dir/'layout-and-pointmap.npz',root/'models/cotracker3/scaled_offline.pth',root/'models/cotracker3/provenance.json']]]
    source_files=sources+[r[k] for r in rows for k in ['frame','mask']]
    plan={'format':'fresh-sam3d-layout-motion-plan','version':1,'status':'prepared_not_run','object_id':model['object_id'],
          'source_video_sha256':source['video_sha256'],'native_glb':ref(glb,root),'model_metadata':ref(metadata_path,root),
          'rows':rows,'anchor_row':anchor,'anchor_sample_index':anchor_sample,'reference_pose':pose,'camera':camera,
          'native_appearance':native_representation_receipt(mesh),'dependencies':dependencies(root),'source_files':source_files,
          'query_file':{'path':'reference-queries.npz','sha256':U.digest(query_bytes),'bytes':len(query_bytes)},
          'queries':{'count':len(seeds),'fresh_new_mesh_ray_hits':int(hits.sum()),'ray_batch_limit':8,'training_ids':train.tolist(),'heldout_ids':held.tolist(),
                     'split':'stable y,x query IDs; modulo3 remainder0 heldout, fixed before any fit','old_canonical_points_read':False},
          'render_samples':[r['sample_index'] for r in rows if r['audit_verdict']=='plausible_visible_mask'],
          'maximum_full_mesh_renders':2,'maximum_tracker_runs':1,'device':'mps','model_max_dimension':384,
          'operator_fallback':False,'wall_timeout_seconds':720,'automatic_retry':False,
          'old_geometry_or_poses_read':False,'reanchor':False,'reacquisition':False,'interpolation':False,
          'program_or_catalog_promotion':False,'physical_accuracy_verified':False,'simulation_ready':False}
    validate_scope(plan);validate_queries(arrays,plan)
    out.mkdir(parents=True,exist_ok=False);(out/'reference-queries.npz').write_bytes(query_bytes);U.write(out/'plan.json',plan)
    return plan


def verify_plan(directory,*,root=ROOT):
    root=Path(root).resolve();out=M._directory(directory,root);p=U.read(out/'plan.json')
    require(p.get('format')=='fresh-sam3d-layout-motion-plan' and p.get('status')=='prepared_not_run','Frozen unexecuted motion plan required')
    require(not (out/'experiment.json').exists(),'One attempt only; preserve previous results')
    require(p['dependencies']==dependencies(root),'Project dependency bytes changed')
    for f in p['source_files']+[p['native_glb'],p['model_metadata']]:
        actual=ref(f['path'],root);require(actual==f,'Frozen source or new mesh bytes changed')
    require(p['query_file']['path']=='reference-queries.npz' and U.sha(out/'reference-queries.npz')==p['query_file']['sha256'],'Fresh canonical query bytes changed')
    validate_scope(p)
    with np.load(out/'reference-queries.npz',allow_pickle=False) as arrays:validate_queries({k:arrays[k] for k in arrays.files},p)
    return p,out


def run(directory,*,root=ROOT):
    root=Path(root).resolve();plan,out=verify_plan(directory,root=root);start=time.monotonic()
    base={'format':'fresh-sam3d-layout-motion-experiment','version':1,'status':'running','plan_sha256':U.sha(out/'plan.json'),
          'object_id':plan['object_id'],'native_glb_sha256':plan['native_glb']['sha256'],'old_geometry_or_poses_read':False,
          'physical_accuracy_verified':False,'simulation_ready':False,'program_or_catalog_changed':False,
          'model_runs_attempted':0,'model_runs':0,'learned_tracker_executed':False}
    U.write(out/'experiment.json',base)
    try:
        mesh=load_native_glb(root/plan['native_glb']['path']);require(native_representation_receipt(mesh)==plan['native_appearance'],'Native ray/render interpretation changed')
        with np.load(out/'reference-queries.npz',allow_pickle=False) as a:query={k:a[k] for k in a.files}
        validate_queries(query,plan)
        images=[np.asarray(Image.open(root/r['frame']['path']).convert('RGB')) for r in plan['rows']]
        masks=[np.asarray(Image.open(root/r['mask']['path']))>0 for r in plan['rows']]
        base.update(model_runs_attempted=1,learned_tracker_executed=None);U.write(out/'experiment.json',base)
        raw_uv,visible,runtime=T.cotracker(images,query['query_uv'],plan['anchor_row'],'mps',384)
        require(runtime.get('learned_tracker_executed') is True,'Tracker did not confirm actual learned execution')
        base.update(model_runs=1,learned_tracker_executed=True,runtime=runtime);U.write(out/'experiment.json',base)
        support=T.support_masks(raw_uv,visible,masks,plan['anchor_row'])
        valid=support['valid_no_reacquisition']&query['reference_ray_hits'][None]
        k=np.asarray(plan['camera']['K_estimated']);r=np.asarray(plan['reference_pose']['R']);t=np.asarray(plan['reference_pose']['t'])
        train=query['training_point_ids'];held=query['heldout_point_ids'];canonical=query['canonical_xyz'];records=[];renders=[]
        np.savez_compressed(out/'fresh-tracks.npz',raw_model_uv=raw_uv,model_visibility=visible,valid_no_reacquisition=valid,
                            canonical_xyz=canonical,query_uv=query['query_uv'],training_ids=train,heldout_ids=held)
        for j,row in enumerate(plan['rows']):
            require(time.monotonic()-start<700,'Bounded motion budget exhausted; preserve partial results')
            fit,present,checks,errors,positive=fit_frame(canonical,raw_uv[j],valid[j],train,held,k,r,t)
            rec={'sample_index':row['sample_index'],'timestamp_s':row['timestamp_s'],'status':'unsupported','reasons':[],
                 'training_visible_point_ids':present.tolist(),'heldout_visible_point_ids':checks.tolist(),
                 'source_frame_sha256':row['frame']['sha256'],'sam_mask_sha256':row['mask']['sha256'],
                 'mask_model_audit_verdict':row['audit_verdict'],'usable_pose_exported':False,'conditional_pose_exported':False}
            rendered=None
            if fit['success']:
                rec.update(candidate_rotation=fit['rotation'].tolist(),candidate_translation=fit['translation'].tolist(),
                           training_inlier_point_ids=present[fit['inlier_indices']].tolist(),
                           heldout_residual_px=[float(v) if np.isfinite(v) else None for v in errors],
                           heldout_median_px=float(np.median(errors)) if len(errors) and np.isfinite(errors).all() else None,
                           heldout_p90_px=float(np.percentile(errors,90)) if len(errors) and np.isfinite(errors).all() else None)
                iou=coverage=0.
                if row['sample_index'] in plan['render_samples']:
                    rendered=render_native_mesh(mesh,k,fit['rotation'],fit['translation'],row['width'],row['height'])
                    intersection=int((rendered['mask']&masks[j]).sum());iou=intersection/max(1,int((rendered['mask']|masks[j]).sum()));coverage=intersection/max(1,int(masks[j].sum()))
                    rec.update(raw_full_projection_iou_with_sam=iou,source_mask_coverage=coverage)
                rec['reasons']=conditional_gate(len(present),len(fit['inlier_indices']),errors,iou,coverage,row['audit_verdict'],
                                               rendered=rendered is not None,positive=positive,withheld=row['withheld_by_source_quality'])
                if j==plan['anchor_row']:rec['status']='reference_candidate_unverified_geometry_and_pose'
                elif not rec['reasons'] and rendered is not None:rec['status']='supported_conditional_motion_candidate';rec['conditional_pose_exported']=True
                else:rec['status']='unsupported_candidate_projection_retained'
            else:rec['reasons']=[fit['reason']]
            records.append(rec);renders.append(rendered)
            U.write(out/'partial-frames.json',{'frames':records,'physical_accuracy_verified':False})
        U.write(out/'point-support.json',{'source_video_sha256':plan['source_video_sha256'],'native_glb_sha256':plan['native_glb']['sha256'],
                'training_ids':train.tolist(),'heldout_ids':held.tolist(),'raw_tracks_file':'fresh-tracks.npz',
                'no_reacquisition':True,'old_canonical_points_read':False,'physical_accuracy_verified':False})
        T.make_review(out,images,records,masks,renders,raw_uv,valid,set(held.tolist()))
        base.update(status='complete_conditional_native_motion_diagnostic',runtime=runtime,frames=records,
                    supported_nonreference_samples=[r['sample_index'] for r in records if r['status']=='supported_conditional_motion_candidate'],
                    unsupported_nonreference_samples=[r['sample_index'] for r in records if r['status'].startswith('unsupported')],
                    full_mesh_rendered_samples=[r['sample_index'] for r in records if 'raw_full_projection_iou_with_sam' in r],
                    wall_seconds=time.monotonic()-start,
                    evaluation='Training and heldout IDs fixed before fit; CoTracker residuals are correlated learned checks conditional on inferred mesh/reference/camera, not physical truth.',
                    limitations=['Estimated camera from the same inferred pointmap; no independent calibration.',
                                 'Only cached-audited samples40/45 may be rendered or conditionally accepted; all unassessed samples remain unsupported.',
                                 'No new surfaces, reanchoring, interpolation, assembly, contact, seating or robot simulation.'])
        U.write(out/'experiment.json',base)
        return base
    except BaseException as error:
        base.update(status='failed_or_interrupted',error=str(error),wall_seconds=time.monotonic()-start)
        U.write(out/'experiment.json',base);raise
