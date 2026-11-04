"""Read-only adapter for a separately sampled, controlled motion comparison.

The original program is never extended. Native-frame IDs belong only to the
hash-bound source extension; the new 3 fps control is not the old SAM3.1 run.
"""
from pathlib import Path
import hashlib
import json

from motion_lab import (CONVENTION, STATUSES, finite, intrinsics, path, pose,
                        read, sha, url, verified)
from motion_lab import normalize as normalize_original

LOADED_SOURCE_SHA256=sha(__file__)


def check(condition,message):
    if not condition:raise ValueError(message)


def completed(doc,expected_format):
    check(doc.get('format')==expected_format and doc.get('version')==1 and
          doc.get('status')in {'complete_with_gaps','unsupported'},'Require completed controlled-density output')
    for field in ['physical_pose_accuracy_measured','usable_physical_pose_recovered',
                  'common_assembly_established','original_semantic_program_extended','manual_geometry_or_poses']:
        check(doc.get(field)is False,'Density result must retain physical uncertainty and unchanged original program')
    check(type(doc.get('new_api_calls'))is int and doc['new_api_calls']==0 and
          doc.get('inherited_automatic_reference_pose_used')is True,'Require automatic frozen-reference density provenance')


def normalize(root,input_path):
    root=Path(root).resolve();input_path=path(root,str(input_path));doc=read(input_path)
    completed(doc,'short-interval-density-experiment')
    receipts={input_path}
    def artifact(binding):
        target=verified(root,binding['path'],binding['sha256']);receipts.add(target);return target
    program_path=artifact({'path':doc['interaction_path'],'sha256':doc['interaction_sha256']})
    program=read(program_path);source=artifact(doc['source']);asset=artifact(doc['asset'])
    check(doc['track_id']==doc['component_id'] and doc['case_id']==program['case_id'] and
          doc['component_id']in {t['id']for t in program['tracklets']} and
          doc['source_video_sha256']==doc['source']['sha256']==program['source']['sha256'] and
          doc['mesh_sha256']==doc['asset']['sha256'] and doc['asset']['physical_scale_known']is False,
          'Density source, program, component or native asset identity mismatch')
    extension_path=artifact(doc['source_extension']);extension=read(extension_path)
    check(extension.get('format')=='short-interval-density-input' and extension.get('version')==1 and
          extension.get('separate_sampling_domain')is True and extension.get('published_interaction_program')is False and
          extension.get('original_program_evidence_was_extended_or_overwritten')is False,
          'Native-frame extension must remain separate from original program evidence')
    for key in ['interaction_sha256','source_video_sha256','mesh_sha256','case_id','track_id']:
        check(extension.get(key)==doc.get(key),'Density source extension identity mismatch')
    parent_path=artifact(doc['parent_interval_result']);parent=read(parent_path)
    parent_validation=normalize_original(root,parent_path)
    check(extension['parent_interval_result']==doc['parent_interval_result'] and
          parent.get('format')=='short-interval-motion' and parent.get('case_id')==doc['case_id'] and
          parent.get('interaction_sha256')==doc['interaction_sha256'] and parent.get('mesh_sha256')==doc['mesh_sha256'],
          'Density parent interval identity mismatch')
    parent_episode=next((e for e in parent['episodes']if e['id']==extension['selected_episode_id']),None)
    check(parent_episode is not None,'Missing frozen parent interval')
    parent_reference=next((f for f in parent_episode['frames']if f['sample_index']==extension['original_reference_sample_index']),None)
    check(parent_reference is not None and parent_reference['status']=='reference_unverified','Require actual unverified automatic parent reference')
    frozen_pose=pose(parent_reference);K=intrinsics(extension['fixed_intrinsics'])
    check(K==intrinsics(parent_episode['intrinsics']) and extension['fixed_original_query_uv_source']==parent_episode['queries']['automatic_query_uv_source'] and
          extension['fixed_original_crop_xyxy']==parent_episode['crop_xyxy'],'Frozen reference camera, query or crop changed')
    artifact(doc['plan']);artifact(doc['audit_binding'])
    evidence={r['source_index']:r for r in program['evidence']}
    original_reference_evidence=evidence[extension['original_reference_sample_index']]
    original_reference_mask=next(o for o in original_reference_evidence['observations']if o['track_id']==doc['component_id'])
    native={};previous=-1
    from PIL import Image
    for row in extension['frames']:
        i=row['sample_index'];time=row['timestamp_s']
        check(type(i)is int and i>0 and i==row['native_decoded_frame_index_zero_based']+1 and
              i not in native and finite(time) and time>previous and abs(float(row['native_pts_exact_s'])-time)<1e-7,
              'Invalid native decoded-frame identity or PTS')
        previous=time;image=artifact({'path':row['source_image_url'],'sha256':row['source_image_sha256']})
        mask=artifact({'path':row['mask_url'],'sha256':row['mask_sha256']})
        with Image.open(image)as im,Image.open(mask)as ma:
            check(im.size==ma.size==(K['width'],K['height']),'Density source/mask camera dimensions differ')
        old_index=row.get('original_program_sample_index')
        if old_index is not None:
            original=evidence.get(old_index)
            check(original is not None and original['frame_sha256']==row['source_image_sha256'] and
                  abs(original['timestamp_s']-time)<=2e-6 and row.get('original_3fps_jpeg_copied_byte_identical')is True,
                  'Common RGB must be byte-identical original program evidence at the exact native PTS')
            artifact({'path':original['frame_path'],'sha256':original['frame_sha256']})
        if i==extension['reference_sample_index']:
            check(old_index==extension['original_reference_sample_index'] and row['mask_sha256']==original_reference_mask['mask_sha256'],
                  'Density analysis reference mask must equal the original SAM3.1 seed bytes')
            artifact({'path':original_reference_mask['mask_path'],'sha256':original_reference_mask['mask_sha256']})
        else:
            native_mask=artifact({'path':row['native_sam2_mask_url'],'sha256':row['native_sam2_mask_sha256']})
            check(sha(native_mask)==row['mask_sha256'],'Analysis mask differs from unedited native SAM2 mask')
        native[i]=row
    check(len(native)==extension['frame_count'] and len(native)<=32,'Unexpected native extension frame count')
    variants={};variant_paths={}
    for binding in doc['variants']:
        name=binding['id'];check(name in {'control-3fps','density-12fps'}and name not in variants,'Unknown or duplicate density branch')
        target=artifact(binding);variant=read(target);completed(variant,'short-interval-density-run')
        check(variant.get('context_requested_fps')==(3 if name=='control-3fps'else 12),'Density branch context label changed')
        for key in ['case_id','component_id','track_id','interaction_sha256','source_video_sha256','mesh_sha256','source_extension','plan','audit_binding','parent_interval_result','motion_policy']:
            check(variant.get(key)==doc.get(key),'Control and density must share identities, audit and all scoring gates')
        check(variant['coordinate_convention']==CONVENTION and len(variant['episodes'])==1,'Unknown density pose convention')
        episode=variant['episodes'][0]
        check(intrinsics(episode['intrinsics'])==K and episode['reference_sample_index']==extension['reference_sample_index'] and
              episode['crop_xyxy']==extension['fixed_original_crop_xyxy'],'Density camera, frozen reference or crop changed')
        for key in ['training_point_ids','heldout_point_ids','automatic_query_uv_source','reference_mesh_ray_hits']:
            check(episode['queries'][key]==parent_episode['queries'][key],'Frozen reference queries or point partitions changed')
        training=set(episode['queries']['training_point_ids']);held=set(episode['queries']['heldout_point_ids'])
        check(not training&held and all(type(i)is int and i%3 for i in training)and all(type(i)is int and i%3==0 for i in held),'Density point split changed')
        frames={};last=-1
        for frame in episode['frames']:
            i=frame['sample_index'];saved=native.get(i)
            check(saved is not None and i not in frames and frame['timestamp_s']>last,'Unknown or repeated native sample in density branch')
            last=frame['timestamp_s']
            for key in ['timestamp_s','native_pts_exact_s','source_image_sha256','mask_sha256','source_image_url','mask_url','original_program_sample_index','temporal_split']:
                check(frame.get(key)==saved.get(key),'Density frame identity differs from explicit native extension')
            status=frame['status'];reference=frame.get('reference')is True
            check(status in STATUSES and reference==(i==extension['reference_sample_index']),'Density reference/status identity mismatch')
            check(not(reference and status=='supported_nonreference_candidate'),'Reference cannot count as supported motion')
            if status=='withheld':check(frame.get('candidate_rotation')is None and frame.get('candidate_translation')is None,'Withheld density candidate leaked a display pose')
            else:pose(frame)
            if reference:check(status=='reference_unverified' and pose(frame)==frozen_pose,'Density reference pose differs from frozen parent')
            reserved=frame.get('evaluation_timestamp_reserved_before_reference_selection')
            check(type(reserved)is bool and reserved==(frame['temporal_split']=='evaluation') and not(reference and reserved),
                  'Density reserved temporal identity changed')
            train=set(frame.get('training_visible_point_ids',[]));test=set(frame.get('heldout_visible_point_ids',[]))
            check(not train&test and train<=training and test<=held,'Density visible query partition changed')
            frames[i]=frame
        check(extension['reference_sample_index']in frames,'Density branch omitted its reference')
        if name=='density-12fps':check(set(frames)==set(native),'Dense branch must retain every native source sample, including gaps')
        else:check(set(frames)=={i for i,r in native.items()if r.get('original_program_sample_index')is not None},'3fps control must use exactly the common original samples')
        variants[name]=(variant,episode,frames);variant_paths[name]=target
    check(set(variants)=={'control-3fps','density-12fps'},'Missing paired density branch')
    # Check actual canonical points, rather than accepting the group proof booleans.
    import numpy as np
    arrays=[]
    for episode in [parent_episode,variants['control-3fps'][1],variants['density-12fps'][1]]:
        target=artifact({'path':episode['tracks_url'],'sha256':episode['tracks_sha256']})
        with np.load(target,allow_pickle=False)as archive:
            arrays.append({k:archive[k].copy()for k in ['query_uv','reference_ray_hits','training_point_ids','heldout_point_ids','canonical_xyz']})
    for reference in arrays[1:]:
        for key in arrays[0]:
            check(np.array_equal(arrays[0][key],reference[key],equal_nan=True),'Canonical points or fixed query geometry differ across context conditions')
    control=variants['control-3fps'][2];dense=variants['density-12fps'][2]
    expected_common=set(control)&set(dense);common={};counts={'fair_same_pts_samples':0,'candidate_supported':0,'baseline_supported':0,'both_supported':0,'references_excluded':0}
    reserved_counts={'samples':0,'candidate_supported':0,'control_supported':0}
    for row in doc['comparison']['exact_common_pts_rows']:
        i=row['sample_index'];check(i in expected_common and i not in common,'Unknown or repeated common density comparison row')
        a,b=control[i],dense[i]
        for key in ['timestamp_s','native_pts_exact_s','source_image_sha256','mask_sha256','original_program_sample_index']:
            check(row[key]==a[key]==b[key],'Matched density PTS, image or mask identity changed')
        for key,expected in [('control_status',a['status']),('dense_status',b['status']),('control_supported',a['status']=='supported_nonreference_candidate'),('dense_supported',b['status']=='supported_nonreference_candidate'),('reference',a['reference'])]:
            check(row[key]==expected,'Density comparison status/count flags differ from actual saved rows')
        check(a['reference']==b['reference'],'Paired reference flags differ')
        check(row['evaluation_timestamp_reserved_before_reference_selection']==a['evaluation_timestamp_reserved_before_reference_selection']==b['evaluation_timestamp_reserved_before_reference_selection'],
              'Common density temporal evaluation identity changed')
        if a['reference']:counts['references_excluded']+=1
        else:
            counts['fair_same_pts_samples']+=1;counts['candidate_supported']+=row['dense_supported'];counts['baseline_supported']+=row['control_supported'];counts['both_supported']+=row['dense_supported']and row['control_supported']
            if row['evaluation_timestamp_reserved_before_reference_selection']:
                reserved_counts['samples']+=1;reserved_counts['candidate_supported']+=row['dense_supported'];reserved_counts['control_supported']+=row['control_supported']
        common[i]=row
    check(set(common)==expected_common,'Density comparison must include every exact common source sample')
    summary=doc['comparison']
    for source_key,key in [('matched_nonreference_count','fair_same_pts_samples'),('dense_supported','candidate_supported'),('control_supported','baseline_supported')]:
        check(summary[source_key]==counts[key],'Density headline differs from recomputed exact common counts')
    for source_key,key in [('reserved_evaluation_count','samples'),('dense_reserved_evaluation_supported','candidate_supported'),('control_reserved_evaluation_supported','control_supported')]:
        check(summary[source_key]==reserved_counts[key],'Density reserved timestamp counts differ from actual saved rows')
    output_frames=[]
    def display(frame):
        result={k:frame[k]for k in ['sample_index','timestamp_s','status','reference','reasons','evaluation_timestamp_reserved_before_reference_selection','temporal_split','original_program_sample_index']}
        result.update(image_url=frame['source_image_url'],image_sha256=frame['source_image_sha256'],mask_url=frame['mask_url'],mask_sha256=frame['mask_sha256'],pose=pose(frame)if frame['status']!='withheld'else None)
        for k in ['train_count','inlier_count','heldout_median_px','heldout_p90_px','silhouette_iou','mask_coverage']:
            if finite(frame.get(k)):result[k]=frame[k]
        result['audit_note']=frame.get('scope_assessment',{}).get('notes')
        return result
    for i,frame in dense.items():
        row=display(frame);row['baseline']=display(control[i])if i in control else None
        row['fair_comparison']=i in common and not frame['reference'];output_frames.append(row)
    eid='density-context-comparison'
    episode={'id':eid,'start_s':extension['selected_interval_s'][0],'end_s':extension['selected_interval_s'][1],
             'reference_sample_index':extension['reference_sample_index'],'reference_timestamp_s':parent_reference['timestamp_s'],
             'intrinsics':K,'gauge':'shared_frozen_reference_conditioned_comparison','status':doc['status'],'frames':output_frames}
    identity={'input_sha256':sha(input_path),'adapter_sha256':LOADED_SOURCE_SHA256,'parent_adapter_sha256':parent_validation['adapter_sha256']}
    return {'format':'interval-motion-viewer','version':1,'id':'motion-density-'+hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20],
            'case_id':doc['case_id'],'component_id':doc['component_id'],'label':doc['label']+' · 12 fps '+extension['selected_episode_id'],'status':doc['status'],
            'raw_url':url(root,input_path),'input_sha256':sha(input_path),'adapter_sha256':LOADED_SOURCE_SHA256,
            'source':{'url':url(root,source),'sha256':doc['source']['sha256'],'original_sha256':doc['source'].get('original_sha256')},
            'asset':{'url':url(root,asset),'sha256':doc['asset']['sha256'],'backend':doc['asset'].get('backend'),'physical_scale_known':False},
            'coordinate_convention':CONVENTION,'episodes':[episode],'baseline':{'url':url(root,variant_paths['control-3fps']),'sha256':sha(variant_paths['control-3fps']),'intrinsics':K,'label':'New 3 fps control'},
            'interaction_path':doc['interaction_path'],'interaction_sha256':doc['interaction_sha256'],'comparison_counts':counts,
            'parent_validation':{'input_sha256':parent_validation['input_sha256'],'adapter_sha256':parent_validation['adapter_sha256']},
            'comparison_kind':'density_control','display_labels':{'candidate':'12 fps context','baseline':'3 fps control','comparison':'Matched common frames','comparison_note':'Identical source pixels and masks · reference excluded'},
            'reserved_comparison_counts':reserved_counts,
            'description':'Controlled context experiment. Both branches use the same generated model, frozen automatic reference, queries, source masks and scoring gates. Only the video context density changes.',
            'episode_note':'The two branches share one unverified starting pose. Additional native video frames are a separate sampling domain; the original semantic program is unchanged.',
            'extra_links':[{'label':'Native-frame source manifest ↗','url':url(root,extension_path)},{'label':'Original SAM3.1 experiment ↗','url':url(root,parent_path)}],
            'physical_pose_accuracy_measured':False,'common_assembly_established':False,
            'limitations':doc['limitations']+['Headline counts compare only identical common source timestamps, images and masks, excluding the shared reference. Additional 12 fps poses do not by themselves prove better tracking accuracy.',
                'The new 3 fps control uses propagated SAM2 masks. It is a new controlled condition, not a replacement of the earlier SAM3.1 experiment.',
                'Native decoded-frame IDs belong to a separate source extension; no original program evidence was extended or overwritten.',
                'Only exact saved samples are displayed. There is no interpolation, hidden contact, joint, common assembly or physically verified robot motion.'],
            'artifact_receipts':[{'path':str(p.relative_to(root)),'sha256':sha(p)}for p in sorted(receipts)]}
