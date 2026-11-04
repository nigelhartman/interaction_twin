"""Separate native-PTS density extension, seeded only by saved SAM3.1 mask.

This is not a replacement semantic interaction program. Common 3fps source
JPEGs are copied byte-for-byte; additional samples retain exact decoded PTS.
"""
from __future__ import annotations
import copy
from datetime import datetime,timezone
from decimal import Decimal
import json
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import numpy as np
from PIL import Image
import interval_motion as M

POLICY={'format_version':1,'requested_fps':12,'maximum_frames':32,
    'selection':'nearest native PTS to12fps targets within frozen first-chronological episode; union exact original3fps sample indices',
    'maximum_program_pts_rounding_difference_s':.000002,
    'reference_selection':'chronologically first episode of existing frozen source-only plan; no motion scores select the interval',
    'common_rgb_policy':'copy all original3fps JPEGs byte-for-byte after nativePTS and dimensions verify',
    'mask_prompt':'one byte-identical saved hosted SAM3.1 binary mask at the frozen reference; no other prompts',
    'model':'local SAM2.1 Tiny','device':'mps','operator_fallback':False,'maximum_runtime_s':900,
    'published_interaction_program':False,'manual_boxes_points_geometry_poses':False,'new_api_calls':0}


def command(arguments):
    return subprocess.run([str(x) for x in arguments],check=True,text=True,capture_output=True)


def select_native_frames(native_pts,common_rows,start,end):
    start,end=Decimal(str(start)),Decimal(str(end));pts=[Decimal(str(t)) for t in native_pts]
    tolerance=Decimal(str(POLICY['maximum_program_pts_rounding_difference_s']))
    common={}
    for row in common_rows:
        target=Decimal(str(row['timestamp_s']));index=min(range(len(pts)),key=lambda i:abs(pts[i]-target))
        if abs(pts[index]-target)>tolerance:raise ValueError('Original source timestamp does not match an exact decoded native frame')
        if index in common:raise ValueError('Original3fps source indices collapse to one decoded frame')
        common[index]=row
    available=[i for i,t in enumerate(pts) if start-tolerance<=t<=end+tolerance]
    if not available:raise ValueError('No native frames in frozen interval')
    selected=set(common);target_index=0
    while start+Decimal(target_index)/POLICY['requested_fps']<=end+tolerance:
        target=start+Decimal(target_index)/POLICY['requested_fps'];selected.add(min(available,key=lambda i:abs(pts[i]-target)));target_index+=1
    selected=sorted(selected)
    if len(selected)>POLICY['maximum_frames']:raise ValueError('Native density sample bound exceeded')
    if any(i not in available for i in selected):raise ValueError('A selected native frame leaves the frozen interval')
    return selected,common


def select_episode(plan,episode_id=None):
    ordered=sorted(plan['episodes'],key=lambda e:(e['start_s'],e['id']))
    if episode_id is None:return ordered[0]
    matches=[e for e in ordered if e['id']==episode_id]
    if len(matches)!=1:raise ValueError('Expected one exact already-frozen episode ID')
    return matches[0]


def extract(plan_path,prior_result_path,out,*,root,episode_id=None):
    root=Path(root).resolve();out=Path(out).resolve();plan_path=M.file_path(root,plan_path);prior_result_path=M.file_path(root,prior_result_path)
    if out.exists() or not out.is_relative_to(root/'data'):raise ValueError('A fresh separate data density domain is required')
    plan=M.read(plan_path);prior=M.read(prior_result_path)
    if prior['plan']['sha256']!=M.sha(plan_path) or prior['interaction_sha256']!=plan['interaction_sha256'] or prior['mesh_sha256']!=plan['mesh_sha256'] or prior['source_video_sha256']!=plan['source_video_sha256']:raise ValueError('Parent interval result identity mismatch')
    episode=select_episode(plan,episode_id)
    policy=copy.deepcopy(POLICY)
    if episode_id is not None:
        policy.update(format_version=2,
            selection='nearest native PTS to12fps targets within every frozen episode, separately; union exact original3fps sample indices',
            reference_selection='explicit already-frozen episode from all-episode chronological family; original automatic reference unchanged; no motion scores choose or refit it')
    previous=next(e for e in prior['episodes'] if e['id']==episode['id'])
    if previous.get('status') not in ['complete_conditional_episode','complete_with_gaps'] or not previous.get('reference_fit'):raise ValueError('Frozen selected reference has no automatic fit')
    reference=next(r for r in episode['frames'] if r['sample_index']==episode['reference_sample_index'])
    source=M.file_path(root,plan['source']['path'])
    if M.sha(source)!=plan['source_video_sha256']:raise ValueError('Normalized source video changed')
    out.mkdir(parents=True);(out/'preparation-implementation.py').write_bytes(Path(__file__).read_bytes())
    metadata={'format':'short-interval-density-input','version':1,'status':'preparing_frames','policy':policy,
        'case_id':plan['case_id'],'track_id':plan['track_id'],'label':plan['label'],
        'parent_plan':{'path':str(plan_path.relative_to(root)),'sha256':M.sha(plan_path)},
        'parent_interval_result':{'path':str(prior_result_path.relative_to(root)),'sha256':M.sha(prior_result_path)},
        'interaction_path':plan['interaction_path'],'interaction_sha256':plan['interaction_sha256'],
        'source':plan['source'],'source_video_sha256':plan['source_video_sha256'],'asset':plan['asset'],'mesh_sha256':plan['mesh_sha256'],
        'published_interaction_program':False,'separate_sampling_domain':True,
        'original_program_evidence_was_extended_or_overwritten':False,'selected_episode_id':episode['id'],
        'selected_interval_s':[episode['start_s'],episode['end_s']],'original_reference_sample_index':episode['reference_sample_index'],
        'fixed_reference_fit':previous['reference_fit'],'fixed_intrinsics':previous['intrinsics'],
        'fixed_original_query_uv_source':previous['queries']['automatic_query_uv_source'],
        'fixed_original_crop_xyxy':previous['crop_xyxy'],
        'seed':{'source_image_url':reference['source_image_url'],'source_image_sha256':reference['source_image_sha256'],
            'mask_url':reference['mask_url'],'mask_sha256':reference['mask_sha256'],'origin':'exact_saved_hosted_Meta_SAM3.1_mask',
            'manual_points_or_boxes':False},
        'host_started_at_utc':datetime.now(timezone.utc).isoformat(),'client_task_date':'2026-10-31',
        'new_api_calls':0,'new_model_inference_calls':0,'physical_pose_accuracy_measured':False,'implementation_sha256':M.sha(__file__)}
    M.write(out/'dense-input.json',metadata)
    probe=['ffprobe','-v','error','-select_streams','v:0','-show_frames','-show_entries','frame=best_effort_timestamp_time,width,height','-of','json',source]
    decoded=json.loads(command(probe).stdout)['frames'];pts=[r['best_effort_timestamp_time'] for r in decoded]
    selected,common=select_native_frames(pts,episode['frames'],episode['start_s'],episode['end_s'])
    frames=out/'frames';frames.mkdir();expression='+'.join(f'eq(n\\,{i})' for i in selected)
    extraction=['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-n','-i',source,'-map','0:v:0','-vf','select='+expression,'-fps_mode','passthrough','-q:v','3',frames/'%06d.jpg']
    command(extraction);extracted=sorted(frames.glob('*.jpg'))
    if len(extracted)!=len(selected):raise ValueError('Exact decoded extraction frame count mismatch')
    records=[]
    for local,(index,path) in enumerate(zip(selected,extracted)):
        native=decoded[index];original=common.get(index);before_hash=M.sha(path)
        with Image.open(path) as image:size=image.size
        if size!=(reference['width'],reference['height']) or size!=(native['width'],native['height']):raise ValueError('Full-frame source dimensions changed')
        equal_pixels=None
        if original:
            original_path=M.file_path(root,original['source_image_url'])
            equal_pixels=np.array_equal(np.asarray(Image.open(path).convert('RGB')),np.asarray(Image.open(original_path).convert('RGB')))
            # The original JPEG is the established observation. PTS/native-index
            # association is attested separately; never recompress common input.
            shutil.copyfile(original_path,path)
        records.append({'sample_index':index+1,'local_index_zero_based':local,'timestamp_s':float(pts[index]),
            'native_pts_exact_s':pts[index],'native_decoded_frame_index_zero_based':index,
            'source_image_url':M.public_url(root,path),'source_image_sha256':M.sha(path),'width':size[0],'height':size[1],
            'original_program_sample_index':original['sample_index'] if original else None,
            'original_program_timestamp_s':original['timestamp_s'] if original else None,
            'program_pts_rounding_difference_s':float(Decimal(pts[index])-Decimal(str(original['timestamp_s']))) if original else None,
            'original_3fps_jpeg_copied_byte_identical':bool(original),'independent_extracted_jpeg_sha256_before_common_copy':before_hash,
            'independent_extracted_common_rgb_equal':equal_pixels,
            'temporal_split':'evaluation' if M.evaluation_time(float(pts[index])) else 'reference_selection'})
    anchor=next(r for r in records if r['original_program_sample_index']==episode['reference_sample_index'])
    metadata.update(status='frames_prepared',frames=records,frame_count=len(records),
        reference_sample_index=anchor['sample_index'],reference_local_index_zero_based=anchor['local_index_zero_based'],
        source_frame_domain='new native decoded frame IDs/PTS, not original semantic evidence IDs',
        extraction={'ffprobe_command':[str(x) for x in probe],'ffmpeg_command':[str(x) for x in extraction],
            'ffmpeg_version':command(['ffmpeg','-version']).stdout.splitlines()[0],
            'reference_RGB_byte_identity_verified':anchor['source_image_sha256']==reference['source_image_sha256'],
            'common_3fps_frame_count':len(common),'common_sampling_domain_join_uses_exact_native_index_and_recorded_PTS_rounding_only':True})
    M.write(out/'dense-input.json',metadata)
    return metadata


def extract_family(plan_path,prior_result_path,out,*,root):
    """Prepare every existing frozen episode; never select a winning subset."""
    root=Path(root).resolve();out=Path(out).resolve();plan_path=M.file_path(root,plan_path);prior_result_path=M.file_path(root,prior_result_path)
    if out.exists() or not out.is_relative_to(root/'data'):raise ValueError('A fresh separate episode family directory is required')
    plan=M.read(plan_path);episodes=sorted(plan['episodes'],key=lambda e:(e['start_s'],e['id']))
    if not 1<=len(episodes)<=M.POLICY['maximum_episodes']:raise ValueError('Frozen episode family exceeds declared component bound')
    out.mkdir(parents=True)
    receipt={'format':'short-interval-density-family','version':1,'status':'preparing_inputs',
        'plan':{'path':str(plan_path.relative_to(root)),'sha256':M.sha(plan_path)},
        'prior':{'path':str(prior_result_path.relative_to(root)),'sha256':M.sha(prior_result_path)},
        'frozen_episode_ids':[e['id'] for e in episodes],
        'selection':'ALL existing source-only planned episodes in chronological order; original references/geometry/gates/crops/queries remain unchanged',
        'implementation_sha256':M.sha(__file__),'new_api_calls':0,'physical_accuracy_measured':False,
        'host_started_at_utc':datetime.now(timezone.utc).isoformat(),'client_task_date':'2026-10-31','inputs':[]}
    M.write(out/'family.json',receipt)
    for episode in episodes:
        directory=out/episode['id'];value=extract(plan_path,prior_result_path,directory,root=root,episode_id=episode['id'])
        receipt['inputs'].append({'episode_id':episode['id'],'directory':str(directory.relative_to(root)),
            'reference_original_program_sample_index':value['original_reference_sample_index'],
            'selected_interval_s':value['selected_interval_s'],'frame_count':value['frame_count']})
        M.write(out/'family.json',receipt)
    receipt['status']='frames_prepared';M.write(out/'family.json',receipt);return receipt


def propagate(directory,*,root):
    root=Path(root).resolve();directory=Path(directory).resolve();path=M.file_path(root,directory/'dense-input.json');metadata=M.read(path)
    if metadata['status']!='frames_prepared':raise ValueError('Fresh prepared frames are required; preserve previous model attempts')
    M.verify_worker_lock(root)
    import torch,sam2
    from sam2.build_sam import build_sam2_video_predictor
    if not torch.backends.mps.is_available() or __import__('os').environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':raise RuntimeError('MPS without operator fallback is required; no installation/download attempted')
    checkpoint=root/'models/sam2.1_hiera_tiny.pt';config='configs/sam2.1/sam2.1_hiera_t.yaml'
    if not checkpoint.is_file():raise ValueError('Prepared local SAM2Tiny checkpoint unavailable')
    source=M.file_path(root,metadata['source']['path'])
    if M.sha(source)!=metadata['source_video_sha256']:raise ValueError('Normalized source video changed')
    for row in metadata['frames']:
        if M.sha(M.file_path(root,row['source_image_url']))!=row['source_image_sha256']:raise ValueError('Prepared source image changed')
    seed_path=M.file_path(root,metadata['seed']['mask_url'])
    if M.sha(seed_path)!=metadata['seed']['mask_sha256']:raise ValueError('Exact SAM3.1 seed changed')
    seed=np.asarray(Image.open(seed_path).convert('L'))>0
    native=directory/'native_masks';masks=directory/'masks';native.mkdir();masks.mkdir();started=time.monotonic()
    source_files=[Path(sam2.__file__).parent/'sam2_video_predictor.py',Path(sam2.__file__).parent/'build_sam.py',Path(sam2.__file__).parent/config]
    snapshots=directory/'sam2-implementation';snapshots.mkdir()
    runtime_code=[]
    for code in source_files:
        target=snapshots/code.name;target.write_bytes(code.read_bytes());runtime_code.append({'source':str(code),'sha256':M.sha(code),'snapshot_url':M.public_url(root,target)})
    metadata.update(status='running_sam2',model={'name':'Meta SAM2.1 Tiny','checkpoint_path':str(checkpoint.relative_to(root)),
        'checkpoint_sha256':M.sha(checkpoint),'runtime_code':runtime_code,'config':config,'device':'mps','operator_fallback':False,
        'prompt_calls':1,'prompt_type':'predictor.add_new_mask','forward_reverse_propagation_passes':2,
        'storage_offload_to_cpu':True,'torch_version':torch.__version__,'model_download_performed':False},
        new_model_inference_calls=1)
    M.write(path,metadata)
    try:
        predictor=build_sam2_video_predictor(config,str(checkpoint),device='mps',apply_postprocessing=False)
        outputs={};anchor=metadata['reference_local_index_zero_based'];calls=[]
        with torch.inference_mode():
            state=predictor.init_state(str(directory/'frames'),offload_video_to_cpu=True,offload_state_to_cpu=True)
            predictor.add_new_mask(state,frame_idx=anchor,obj_id=1,mask=seed)
            for reverse in [False,True]:
                call={'reverse':reverse,'start_frame_idx':anchor,'returned_local_indices':[]};calls.append(call)
                for local,ids,logits in predictor.propagate_in_video(state,start_frame_idx=anchor,reverse=reverse):
                    if time.monotonic()-started>POLICY['maximum_runtime_s']:raise TimeoutError('Bounded local SAM2 runtime exhausted')
                    mask=(logits[list(ids).index(1),0]>0).cpu().numpy()
                    if mask.shape!=seed.shape:raise ValueError('Native propagated mask changed full-frame dimensions')
                    raw=native/(f'{local+1:06d}-'+('reverse.png' if reverse else 'forward.png'))
                    Image.fromarray(mask.astype(np.uint8)*255).save(raw);call['returned_local_indices'].append(int(local))
                    if (not reverse and local>=anchor) or (reverse and local<anchor):outputs[int(local)]=raw
                torch.mps.synchronize()
        if set(outputs)!=set(range(len(metadata['frames']))):raise ValueError('SAM2 did not return every bounded source frame')
        for row in metadata['frames']:
            local=row['local_index_zero_based'];target=masks/f'{local+1:06d}.png';raw=outputs[local]
            if local==anchor:shutil.copyfile(seed_path,target);origin='byte-identical original hostedSAM3.1 seed; local SAM2seed output preserved separately'
            else:shutil.copyfile(raw,target);origin='native local SAM2.1 Tiny propagated binary output, unedited'
            binary=np.asarray(Image.open(target).convert('L'))>0;area=int(binary.sum());ys,xs=np.where(binary)
            fraction=area/binary.size;eligible=bool(area>=16 and fraction<=.8)
            row.update(mask_url=M.public_url(root,target),mask_sha256=M.sha(target),mask_origin=origin,
                native_sam2_mask_url=M.public_url(root,raw),native_sam2_mask_sha256=M.sha(raw),
                mask_area_pixels=area,mask_fraction=fraction,gate_status='candidate_geometric_gate_only' if eligible else 'withheld_geometric_gate',
                audit_verdict='not_model_audited',available=eligible,
                mask_centroid_xy=[float(xs.mean()),float(ys.mean())] if area else None,
                mask_extent_px=float(max(xs.max()-xs.min()+1,ys.max()-ys.min()+1)) if area else 0.)
        metadata.update(status='complete_masks_unaudited',propagation_calls=calls,wall_seconds=time.monotonic()-started,
            reference_analysis_mask_byte_identity_verified=metadata['frames'][anchor]['mask_sha256']==metadata['seed']['mask_sha256'])
        M.write(path,metadata)
        plan={'format':'short-interval-density-motion-plan','version':1,'track_id':metadata['track_id'],'component_id':metadata['track_id'],
            'case_id':metadata['case_id'],'interaction_sha256':metadata['interaction_sha256'],'source_video_sha256':metadata['source_video_sha256'],
            'mesh_sha256':metadata['mesh_sha256'],'input_manifest':{'path':str(path.relative_to(root)),'sha256':M.sha(path)},
            'policy':metadata['policy'],'fixed_reference_fit':metadata['fixed_reference_fit'],'fixed_intrinsics':metadata['fixed_intrinsics'],
            'fixed_original_query_uv_source':metadata['fixed_original_query_uv_source'],'fixed_original_crop_xyxy':metadata['fixed_original_crop_xyxy'],
            'episodes':[{'id':'density-episode-01','reference_sample_index':metadata['reference_sample_index'],
                'reference_timestamp_s':metadata['frames'][anchor]['timestamp_s'],'start_s':metadata['frames'][0]['timestamp_s'],
                'end_s':metadata['frames'][-1]['timestamp_s'],'frames':metadata['frames']}]}
        M.write(directory/'motion-plan.json',plan)
        M.write(directory/'audit-request.json',{'format':'short-interval-mask-scope-audit-request','version':1,
            'plan_path':str((directory/'motion-plan.json').relative_to(root)),'plan_sha256':M.sha(directory/'motion-plan.json'),
            'track_id':metadata['track_id'],'interaction_sha256':metadata['interaction_sha256'],'source_video_sha256':metadata['source_video_sha256'],
            'sampling_domain':'new native-PTS extension, originalprogram evidence unchanged','targets':metadata['frames'],
            'required_observation_fields':['track_id','sample_index','source_image_sha256','mask_sha256','verdict','single_component_visible','component_boundary_plausible','other_components_inside_mask','notes'],
            'scope_definition':'single_component_visible means one coherent intended component INSIDE raw proposed mask; other objects outside mask alone do not veto'})
        return metadata
    except Exception as error:
        metadata.update(status='failed',error={'type':type(error).__name__,'message':str(error)},wall_seconds=time.monotonic()-started)
        M.write(path,metadata);raise
