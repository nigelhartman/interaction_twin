"""Controlled 3fps-subset/12fps-context comparison in a separate source domain.

Both methods share byte-identical common JPEGs/masks, native asset, original
automatic reference pose, native ray query IDs, crop, camera and score gates.
"""
from __future__ import annotations
import copy
from datetime import datetime,timezone
from pathlib import Path
import time
import numpy as np
from PIL import Image
import interval_motion as M
from native_appearance import load_native_glb

POLICY={'version':'controlled-density-v1','maximum_frames':32,'maximum_runtime_s':900,
    'comparison':'3fps common native-frame subset versus12fps context using SAME propagated masks on common frames',
    'reference':'same hash-bound automatic reference fitted before density sampling; no new pose search or manual input',
    'same_native_mesh':True,'same_reference_pose_intrinsics_queries_crop_and_score_gates':True,
    'canonical_point_equivalence_absolute_tolerance':1e-10,'new_api_calls':0,
    'native_sampling_domain_separate_from_original_program':True,'physical_pose_accuracy_measured':False}


def make_variants(plan):
    episode=plan['episodes'][0]
    dense=copy.deepcopy(episode);dense['id']='density-12fps'
    sparse=copy.deepcopy(episode);sparse['id']='control-3fps'
    sparse['frames']=[r for r in sparse['frames'] if r.get('original_program_sample_index') is not None]
    if not 3<=len(sparse['frames'])<len(dense['frames'])<=POLICY['maximum_frames']:raise ValueError('A bounded paired sparse/dense source domain is required')
    if not any(r['sample_index']==sparse['reference_sample_index'] for r in sparse['frames']):raise ValueError('Frozen reference must be in exact common source subset')
    return sparse,dense


def validate_plan_manifest(plan,manifest):
    """Refuse new-domain retargeting behind an attested input manifest."""
    if len(plan.get('episodes',[]))!=1 or plan['episodes'][0]['frames']!=manifest['frames']:
        raise ValueError('Density plan source rows differ from frozen input manifest')
    for key in ['track_id','source_video_sha256','mesh_sha256','interaction_sha256']:
        if plan[key]!=manifest[key]:raise ValueError('Density plan identity differs from frozen input manifest')
    for key in ['fixed_reference_fit','fixed_intrinsics','fixed_original_query_uv_source','fixed_original_crop_xyxy']:
        if plan[key]!=manifest[key]:raise ValueError('Density plan fixed reference settings differ from frozen input manifest')


def same_point_arrays(first,second):
    a=np.load(first);b=np.load(second)
    exact_keys=['query_uv','reference_ray_hits','training_point_ids','heldout_point_ids']
    checks={key:np.array_equal(a[key],b[key],equal_nan=True) for key in exact_keys}
    checks['canonical_xyz']=bool(np.allclose(a['canonical_xyz'],b['canonical_xyz'],atol=POLICY['canonical_point_equivalence_absolute_tolerance'],rtol=0,equal_nan=True))
    finite=np.isfinite(a['canonical_xyz'])&np.isfinite(b['canonical_xyz'])
    return checks,float(np.max(np.abs(a['canonical_xyz'][finite]-b['canonical_xyz'][finite]))) if finite.any() else 0.


def compare(control,dense):
    control_rows={r['sample_index']:r for r in control['frames']};dense_rows={r['sample_index']:r for r in dense['frames']};rows=[]
    for sample,a in sorted(control_rows.items()):
        b=dense_rows.get(sample)
        if b is None or any(a[k]!=b[k] for k in ['timestamp_s','source_image_sha256','mask_sha256','reference']):raise ValueError('Controlled shared source frame/mask/PTS/reference mismatch')
        rows.append({'sample_index':sample,'native_decoded_frame_index_zero_based':a['native_decoded_frame_index_zero_based'],
            'timestamp_s':a['timestamp_s'],'native_pts_exact_s':a['native_pts_exact_s'],'source_image_sha256':a['source_image_sha256'],'mask_sha256':a['mask_sha256'],
            'original_program_sample_index':a['original_program_sample_index'],'reference':a['reference'],
            'evaluation_timestamp_reserved_before_reference_selection':a['evaluation_timestamp_reserved_before_reference_selection'],
            'control_status':a['status'],'dense_status':b['status'],'control_supported':a['status']==M.SUPPORTED,'dense_supported':b['status']==M.SUPPORTED,
            'control_reasons':a['reasons'],'dense_reasons':b['reasons']})
    matched=[r for r in rows if not r['reference']];evaluated=[r for r in matched if r['evaluation_timestamp_reserved_before_reference_selection']]
    return {'exact_common_pts_rows':rows,'matched_nonreference_count':len(matched),
        'control_supported':sum(r['control_supported'] for r in matched),'dense_supported':sum(r['dense_supported'] for r in matched),
        'reserved_evaluation_count':len(evaluated),'control_reserved_evaluation_supported':sum(r['control_supported'] for r in evaluated),
        'dense_reserved_evaluation_supported':sum(r['dense_supported'] for r in evaluated),
        'only_context_sampling_changes_between_control_and_dense':True,
        'comparison_to_original_sam31_3fps_is_not_a_density_isolation':True,'physical_accuracy_measured':False}


def run(input_plan,audit_path,out,*,root):
    root=Path(root).resolve();out=Path(out).resolve();input_plan=M.file_path(root,input_plan);audit_path=M.file_path(root,audit_path)
    if out.exists() or not out.is_relative_to(root/'data'):raise ValueError('A fresh separate density experiment output is required')
    M.verify_worker_lock(root);started=time.monotonic();deadline=started+POLICY['maximum_runtime_s']
    plan=M.read(input_plan);manifest_path=M.file_path(root,plan['input_manifest']['path'])
    if M.sha(manifest_path)!=plan['input_manifest']['sha256']:raise ValueError('Native density input manifest changed')
    manifest=M.read(manifest_path)
    validate_plan_manifest(plan,manifest)
    if manifest['status']!='complete_masks_unaudited' or not manifest['reference_analysis_mask_byte_identity_verified']:raise ValueError('Expected complete genuine local propagation with exact original seed')
    parent_path=M.file_path(root,manifest['parent_interval_result']['path'])
    if M.sha(parent_path)!=manifest['parent_interval_result']['sha256']:raise ValueError('Frozen parent automatic motion result changed')
    parent=M.read(parent_path);previous=next(e for e in parent['episodes'] if e['id']==manifest['selected_episode_id'])
    if parent['policy']!=M.POLICY:raise ValueError('Controlled motion scoring policy differs from frozen parent')
    if previous['reference_fit']!=manifest['fixed_reference_fit'] or previous['intrinsics']!=manifest['fixed_intrinsics'] or previous['crop_xyxy']!=manifest['fixed_original_crop_xyxy'] or previous['queries']['automatic_query_uv_source']!=manifest['fixed_original_query_uv_source']:raise ValueError('Controlled original reference/camera/crop/queries mismatch')
    mesh_path=M.file_path(root,manifest['asset']['path']);source=M.file_path(root,manifest['source']['path'])
    interaction=M.file_path(root,manifest['interaction_path'])
    if M.sha(interaction)!=manifest['interaction_sha256']:raise ValueError('Original semantic interaction changed')
    if M.sha(mesh_path)!=manifest['mesh_sha256'] or M.sha(source)!=manifest['source_video_sha256']:raise ValueError('Native geometry or source video changed')
    for row in plan['episodes'][0]['frames']:
        for url_key,hash_key in [('source_image_url','source_image_sha256'),('mask_url','mask_sha256')]:
            if M.sha(M.file_path(root,row[url_key]))!=row[hash_key]:raise ValueError('Prepared density source pixels or mask changed')
    assessment,audit=M.load_audit(plan,input_plan,audit_path,root)
    sparse,dense=make_variants(plan);out.mkdir(parents=True)
    (out/'density-implementation.py').write_bytes(Path(__file__).read_bytes());implementation=M.snapshot(root,out)
    protocol={'format':'short-interval-density-experiment','version':1,'status':'running','policy':POLICY,'motion_policy':M.POLICY,
        'case_id':manifest['case_id'],'component_id':manifest['track_id'],'track_id':manifest['track_id'],'label':manifest['label'],
        'source':manifest['source'],'source_video_sha256':manifest['source_video_sha256'],'asset':manifest['asset'],'mesh_sha256':manifest['mesh_sha256'],
        'interaction_path':manifest['interaction_path'],'interaction_sha256':manifest['interaction_sha256'],
        'source_extension':{'path':str(manifest_path.relative_to(root)),'sha256':M.sha(manifest_path)},
        'plan':{'path':str(input_plan.relative_to(root)),'sha256':M.sha(input_plan)},'audit_binding':audit,
        'parent_interval_result':manifest['parent_interval_result'],'inherited_automatic_reference_pose_used':True,
        'manual_geometry_or_poses':False,'new_api_calls':0,'physical_pose_accuracy_measured':False,'usable_physical_pose_recovered':False,
        'common_assembly_established':False,'original_semantic_program_extended':False,'implementation':implementation,
        'density_implementation_sha256':M.sha(__file__),'variants':[],
        'host_started_at_utc':datetime.now(timezone.utc).isoformat(),'client_task_date':'2026-10-31'}
    M.write(out/'density-experiment.json',protocol)
    mesh=load_native_glb(mesh_path);k=np.asarray(manifest['fixed_intrinsics']['matrix'],float);tracker=M.OfflineTracker(root,out,deadline)
    frozen=manifest['fixed_reference_fit'];seed_image=np.asarray(Image.open(M.file_path(root,manifest['seed']['source_image_url'])).convert('RGB'))
    seed_mask=np.asarray(Image.open(M.file_path(root,manifest['seed']['mask_url'])).convert('L'))>0
    def fixed_reference(mesh_arg,image_arg,mask_arg,k_arg,deadline_arg):
        if not np.array_equal(image_arg,seed_image) or not np.array_equal(mask_arg,seed_mask) or not np.array_equal(k_arg,k):raise ValueError('Frozen automatic reference pixels/mask/intrinsics changed')
        best=copy.deepcopy(frozen['candidates'][0]);best['rotation']=np.asarray(best['rotation'],float);best['translation']=np.asarray(best['translation'],float)
        receipt=copy.deepcopy(frozen);receipt.update(method='Frozen earlier automatic reference; same exact native mesh/source image/mask/camera',
            inherited_automatic_reference_pose_used=True,inherited_parent_result=manifest['parent_interval_result'],
            reference_fit_is_physical_truth=False,manual_pose_used=False,reference_search_repeated=False,
            old_pose_initialization_used=True,old_pose_source='earlier automatic interval reference; not original baseline pose')
        return best,receipt
    try:
        results=[]
        for variant in [sparse,dense]:
            name=variant['id'];directory=out/name;directory.mkdir()
            print(__import__('json').dumps({'phase':'density_variant','variant':name,'frames':len(variant['frames'])}),flush=True)
            episode=M.track_episode(variant,assessment,mesh,k,tracker,root,directory,deadline,
                reference_provider=fixed_reference,fixed_crop=manifest['fixed_original_crop_xyxy'],expected_query_uv=manifest['fixed_original_query_uv_source'])
            record={key:protocol[key] for key in ['case_id','component_id','track_id','label','source','source_video_sha256','asset','mesh_sha256','interaction_path','interaction_sha256','source_extension','plan','audit_binding','parent_interval_result','physical_pose_accuracy_measured','usable_physical_pose_recovered','common_assembly_established','original_semantic_program_extended','implementation','density_implementation_sha256','motion_policy']}
            record.update(format='short-interval-density-run',version=1,status='complete_with_gaps' if any(r['status']==M.SUPPORTED for r in episode['frames']) else 'unsupported',
                context_requested_fps=3 if name=='control-3fps' else 12,episodes=[episode],frame_count=len(episode['frames']),
                supported_nonreference_frames=sum(r['status']==M.SUPPORTED for r in episode['frames']),
                reference_frames=sum(r['reference'] for r in episode['frames']),
                withheld_nonreference_frames=sum(not r['reference'] and r['status']!=M.SUPPORTED for r in episode['frames']),
                coordinate_convention=parent['coordinate_convention'],inherited_automatic_reference_pose_used=True,
                manual_geometry_or_poses=False,new_api_calls=0)
            M.write(directory/'interval-motion.json',M.serial(record));results.append(record)
            protocol['variants'].append({'id':name,'context_requested_fps':record['context_requested_fps'],
                'path':str((directory/'interval-motion.json').relative_to(root)),'sha256':M.sha(directory/'interval-motion.json'),
                'status':record['status'],'frame_count':record['frame_count'],'supported_nonreference_frames':record['supported_nonreference_frames']})
            M.write(out/'density-experiment.json',M.serial(protocol))
        if all('tracks_url' in r['episodes'][0] for r in results):
            a=M.file_path(root,results[0]['episodes'][0]['tracks_url']);b=M.file_path(root,results[1]['episodes'][0]['tracks_url'])
            checks,delta=same_point_arrays(a,b)
            prior_tracks=M.file_path(root,previous['tracks_url']);prior_checks,prior_delta=same_point_arrays(a,prior_tracks)
            if not all(checks.values()) or not all(prior_checks.values()):raise ValueError('Controlled native query/ray point correspondence identities changed')
            protocol['point_identity_proof']={'control_vs_dense_checks':checks,'control_vs_original_checks':prior_checks,
                'maximum_control_dense_canonical_difference':delta,'maximum_control_original_canonical_difference':prior_delta,
                'original_tracks_sha256':M.sha(prior_tracks)}
        else:protocol['point_identity_proof']={'not_evaluated':'One variant was source/reference-vetoed before learned tracking'}
        protocol.update(status='complete_with_gaps',comparison=compare(results[0]['episodes'][0],results[1]['episodes'][0]),
            actual_cotracker_model_calls=len(tracker.calls),wall_seconds=time.monotonic()-started,
            limitations=['This is a separate native-PTS input domain. Original semantic program evidence identities and files are unchanged.',
                'The paired control changes the mask generator from originalSAM3.1 video tracking to SAM2Tiny propagation. Only the paired3fps/12fps comparison isolates context sampling with matching common-frame masks.',
                'Higher cadence can expose an occlusion absent at sparse samples. Missing/uncertain audited samples split context and remain withheld; no interpolation is performed.',
                'The automatic generated-mesh reference, camera intrinsics, scale and geometry remain unverified; image agreement is not physical6D accuracy.'])
        if M.sha(mesh_path)!=protocol['mesh_sha256'] or M.sha(manifest_path)!=protocol['source_extension']['sha256'] or M.sha(parent_path)!=protocol['parent_interval_result']['sha256']:raise ValueError('Frozen density inputs changed')
        protocol['host_completed_at_utc']=datetime.now(timezone.utc).isoformat();M.write(out/'density-experiment.json',M.serial(protocol));return protocol
    except Exception as error:
        protocol.update(status='failed',error={'type':type(error).__name__,'message':str(error)},wall_seconds=time.monotonic()-started)
        M.write(out/'density-experiment.json',M.serial(protocol));raise
