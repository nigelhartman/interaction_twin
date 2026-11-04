"""New disjoint phone-image diagnostics with existing local SAM2.1 Tiny/MPS.

The exact test PTS, original native/derived assets and predicted poses are frozen
before target RGB is decoded. Saved SAM3.1 masks alone seed local SAM2; no target
pixels fit geometry or poses. Outputs are fallible model observations, never
human ground truth, physical validation or publication approval.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation,Slerp

import charger_tracking_recovery_20261103_v1 as C
import sam3d_mechanisms as M
from native_appearance import load_native_glb,render_native_mesh

ROOT=Path(__file__).resolve().parent
OUTPUT='output/assembly-quality-20261104/phone-midpoint-evaluation-v2'
PROGRAM='data/flow-capture-phone-20261103-v1/flow-procedure-1/interaction.json'
BASE='data/mechanism-scenes/648d3924ab106192ea1cdad94c6cdf0b2d44d5c61bba3090cd5dbb1fcb00f298/constrained/scene.json'
PRIOR='data/geometry-quality-20261104/phone-source-refit-v1'
PAIRS=[(4,5),(8,9),(26,27),(37,38)]
CHECKPOINT='models/sam2.1_hiera_tiny.pt'


def _command(args):
    return subprocess.run(args,check=True,capture_output=True,text=True).stdout


def _interpolate(a,b,fraction):
    ra,ta,sa=M.rotation_pose(a);rb,tb,sb=M.rotation_pose(b)
    C.require(np.isclose(sa,sb),'Common original native scale required')
    matrix=np.eye(4);matrix[:3,:3]=sa*Slerp([0.,1.],Rotation.from_matrix([ra,rb]))(fraction).as_matrix()
    matrix[:3,3]=ta*(1-fraction)+tb*fraction;return matrix.tolist()


def prepare():
    folder=ROOT/OUTPUT;folder.mkdir(exist_ok=False)
    program=C.read(ROOT/PROGRAM);baseline=C.read(ROOT/BASE);prior=C.read(ROOT/PRIOR/'scene.json')
    source=C.ref(program['source']['path']);C.require(source['sha256']==program['source']['sha256'],'Exact native source required')
    evidence={e['source_index']:e for e in program['evidence']}
    component=next(c for c in baseline['components']if c['id']=='phone_blue')
    before={r['source_sample_index']:r for r in component['poses']}
    previous=next(c for c in prior['components']if c['id']=='phone_blue')
    after={r['source_sample_index']:copy.deepcopy(r)for r in previous['poses']}
    old_predictions=C.read(ROOT/PRIOR/'heldout-predictions-before-target-read.json')
    # Preserve the FAILED old heldout prediction as a diagnostic; it is NOT
    # silently marked known in any scene or reclassified as a passing holdout.
    for p in old_predictions:
        if after[p['sample_index']]['camera_from_native_mesh']is None:
            after[p['sample_index']]['camera_from_native_mesh']=p['matrix']
            after[p['sample_index']]['diagnostic_only_prior_failed_prediction']=True
    probe_args=['ffprobe','-v','error','-select_streams','v:0','-show_frames','-show_entries',
                'frame=best_effort_timestamp_time,width,height','-of','json',str(ROOT/source['path'])]
    decoded=json.loads(_command(probe_args))['frames']
    original_times=[e['timestamp_s']for e in program['evidence']];rows=[]
    for index,(left,right)in enumerate(PAIRS,1):
        a,b=evidence[left],evidence[right];requested=(a['timestamp_s']+b['timestamp_s'])/2
        native,frame=min(enumerate(decoded),key=lambda pair:abs(float(pair[1]['best_effort_timestamp_time'])-requested))
        pts=float(frame['best_effort_timestamp_time']);fraction=(pts-a['timestamp_s'])/(b['timestamp_s']-a['timestamp_s'])
        C.require(all(abs(pts-t)>.05 for t in original_times),'New test PTS must be disjoint from ALL prior evidence')
        C.require(0<fraction<1 and frame['width']==baseline['source']['width']and frame['height']==baseline['source']['height'],'Exact in-between source frame coordinates required')
        obs=next(o for o in a['observations']if o['track_id']=='phone_blue')
        for table in [before,after]:
            C.require(table[left]['camera_from_native_mesh']is not None and table[right]['camera_from_native_mesh']is not None,'Saved neighboring diagnostic prediction required')
        rows.append({'test_id':index,'left_source_index':left,'right_source_index':right,
            'requested_timestamp_s':requested,'timestamp_s':pts,'decoded_frame_index_zero_based':native,
            'fraction':fraction,'seed_rgb':C.ref(a['frame_path']),'seed_mask':C.ref(obs['mask_path']),
            'seed_mask_origin':'saved_hosted_Meta_SAM3.1_mask_not_new_SAM3_inference',
            'baseline_matrix':_interpolate(before[left]['camera_from_native_mesh'],before[right]['camera_from_native_mesh'],fraction),
            'candidate_matrix':_interpolate(after[left]['camera_from_native_mesh'],after[right]['camera_from_native_mesh'],fraction),
            'candidate_pose_origin':'frozen_prior_automatic_subset_fits_including_explicit_failed_source5_prediction_no_target_fitting'})
    assets=[{'id':'native_original','asset':component['asset'],'pose':'baseline_matrix'},
        {'id':'prior_subset','asset':previous['asset'],'pose':'candidate_matrix'},
        {'id':'new_competing_subset','asset':C.ref('data/geometry-quality-20261104/phone-competing-slab-candidates-v1/competing_slab_2/asset.glb'),'pose':'candidate_matrix'},
        {'id':'source_affine_2','asset':C.ref('data/geometry-quality-20261104/phone-affine-candidates-v1/source_affine_2/asset.glb'),'pose':'candidate_matrix'}]
    plan={'format':'interaction-twin-phone-new-midpoint-local-evaluation','version':1,
        'implementation':C.ref(__file__),'source':source,'source_program':C.ref(PROGRAM),'baseline_scene':C.ref(BASE),
        'prior_rejected_report':C.ref(ROOT/PRIOR/'report.json'),'prior_candidate_scene':C.ref(ROOT/PRIOR/'scene.json'),
        'prior_source5_failure':{'before_iou':.8873720136518771,'after_iou':.7474869320466425,'still_failed':True},
        'assets':assets,'test_frames':rows,'camera':baseline['camera'],'source_size_wh':[baseline['source']['width'],baseline['source']['height']],
        'segmentation':{'model':'Meta SAM2.1 Tiny','device':'mps','checkpoint':C.ref(CHECKPOINT),
            'config':'configs/sam2.1/sam2.1_hiera_t.yaml','init_state_count':4,'frames_per_state':2,
            'prompt':'only exact earlier saved SAM3.1 mask','manual_points_boxes_or_masks':False,
            'new_hosted_inference':False,'labels_are_fallible_model_observations_not_ground_truth':True},
        'test_target_RGB_decoded_or_viewed_before_plan':False,'target_pixels_used_for_pose_or_geometry_fitting':False,
        'minimum_test_iou':.65,'maximum_test_regression':.05,'render_height':288,
        'automatic_approval':False,'automatic_publication':False,'physical_accuracy_verified':False,
        'new_api_calls':0,'new_remote_gpu_calls':0,'network_access_required':False,'new_downloads':False,
        'limitations':['Frozen previous-fit poses are diagnostics, not a new accepted reconstruction.',
            'Masks are locally propagated model hypotheses seeded by prior fallible SAM3.1 masks.',
            'Passing these four samples cannot waive the original source5 failure or missing native surfaces.']}
    C.write_new(folder/'selection-plan.json',plan);return C.ref(folder/'selection-plan.json')


def run():
    folder=ROOT/OUTPUT;plan=C.read(folder/'selection-plan.json')
    C.require(C.ref(__file__)==plan['implementation'],'Frozen local evaluation helper changed')
    for ref in [plan['source'],plan['source_program'],plan['baseline_scene'],plan['prior_rejected_report'],plan['prior_candidate_scene'],plan['segmentation']['checkpoint']]+[a['asset']for a in plan['assets']]:
        C.require(C.ref(ref['path'])==ref,'Frozen local input changed')
    attempt=folder/'attempt';attempt.mkdir(exist_ok=False);C.write_new(attempt/'intent.json',{'plan':C.ref(folder/'selection-plan.json'),'network_required':False})
    import torch
    from sam2.build_sam import build_sam2_video_predictor
    C.require(torch.backends.mps.is_available(),'Local MPS unavailable; no CPU or download fallback')
    started=time.monotonic();frames=attempt/'decoded';frames.mkdir()
    selected='+'.join(f'eq(n\\,{r["decoded_frame_index_zero_based"]})'for r in plan['test_frames'])
    args=['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-n','-i',str(ROOT/plan['source']['path']),'-map','0:v:0','-vf',f'select={selected}','-fps_mode','passthrough','-q:v','2',str(frames/'%06d.jpg')]
    _command(args);C.require(len(list(frames.glob('*.jpg')))==4,'Exact four decoded target RGB frames required')
    predictor=build_sam2_video_predictor(plan['segmentation']['config'],str(ROOT/CHECKPOINT),device='mps',apply_postprocessing=False)
    observations=[]
    for row in plan['test_frames']:
        target=attempt/f'test-{row["test_id"]}';target.mkdir();inputs=target/'inputs';inputs.mkdir()
        shutil.copyfile(ROOT/row['seed_rgb']['path'],inputs/'000000.jpg')
        shutil.copyfile(frames/f'{row["test_id"]:06d}.jpg',inputs/'000001.jpg')
        seed=np.asarray(Image.open(ROOT/row['seed_mask']['path']).convert('L'))>0
        outputs={}
        with torch.inference_mode():
            state=predictor.init_state(str(inputs),offload_video_to_cpu=True,offload_state_to_cpu=True)
            predictor.add_new_mask(state,frame_idx=0,obj_id=1,mask=seed)
            for index,ids,logits in predictor.propagate_in_video(state,start_frame_idx=0):
                binary=(logits[list(ids).index(1),0]>0).cpu().numpy()
                C.require(binary.shape==seed.shape and binary.any(),'Nonempty original-coordinate local SAM2 output required')
                path=target/f'mask-{index}.png';Image.fromarray(binary.astype(np.uint8)*255).save(path);outputs[index]=C.ref(path)
            torch.mps.synchronize()
        C.require(set(outputs)=={0,1},'Both local SAM2 observations required')
        observations.append({**row,'rgb':C.ref(inputs/'000001.jpg'),'mask':outputs[1],
            'seed_reprojection_mask':outputs[0],'mask_origin':'actual_local_Meta_SAM2.1_Tiny_MPS_forward_propagation',
            'not_human_ground_truth':True,'identity_unverified':True})
        C.write_new(target/'observation.json',observations[-1]);del state
        print('SAM2 local test complete',row['test_id'],flush=True)
    del predictor;torch.mps.empty_cache()
    meshes={a['id']:load_native_glb(ROOT/a['asset']['path'])for a in plan['assets']}
    w0,h0=plan['source_size_wh'];h=plan['render_height'];w=round(w0*h/h0);k=np.asarray(plan['camera']['K']).copy();k[0]*=w/w0;k[1]*=h/h0;k[0,2]+=(w/w0-1)/2;k[1,2]+=(h/h0-1)/2
    sheet=Image.new('RGB',(1500,1400),'#f5f6fa');draw=ImageDraw.Draw(sheet);scores=[]
    for number,row in enumerate(observations):
        mask=cv2.resize(np.asarray(Image.open(ROOT/row['mask']['path'])).astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)>0
        rgb=np.asarray(Image.open(ROOT/row['rgb']['path']).convert('RGB').resize((w,h)))
        items=[('Fresh source + local SAM2',rgb,mask)];values={}
        for a in plan['assets']:
            rotation,translation,scale=M.rotation_pose(row[a['pose']]);rendered=render_native_mesh(meshes[a['id']],k,rotation,translation,w,h,scale=scale)
            predicted=rendered['mask'];iou=float(np.count_nonzero(predicted&mask)/max(1,np.count_nonzero(predicted|mask)))
            pixels=np.full((h,w,3),[237,240,247],np.uint8);pixels[predicted]=rendered['rgb'][predicted]
            items.append((f'{a["id"]} | IoU {iou:.3f}',pixels,predicted));values[a['id']]=iou
        union=np.logical_or.reduce([r[2]for r in items]);ys,xs=np.nonzero(union);padding=12
        crop=[max(0,int(xs.min())-padding),max(0,int(ys.min())-padding),min(w,int(xs.max())+padding+1),min(h,int(ys.max())+padding+1)]
        for column,(label,pixels,binary)in enumerate(items):
            if column==0:
                pixels=pixels.copy();pixels[binary]=(pixels[binary]*.76+np.array([38,98,255])*.24).astype(np.uint8)
            image=Image.fromarray(pixels).crop(crop);image.thumbnail((288,300));x=column*300+(300-image.width)//2;y=number*350+40+(300-image.height)//2;sheet.paste(image,(x,y));draw.text((column*300+8,number*350+7),label,fill='#151923')
        baseline=values['native_original'];scores.append({'test_id':row['test_id'],'timestamp_s':row['timestamp_s'],'source_ref':row['rgb'],'mask_ref':row['mask'],'full_triangle_mask_iou':values,'candidate_test_gates':{key:value>=.65 and value>=baseline-.05 for key,value in values.items()if key!='native_original'}})
    sheet.save(attempt/'comparison-sheet.png')
    report={'format':'interaction-twin-phone-disjoint-midpoint-diagnostics','version':1,'plan':C.ref(folder/'selection-plan.json'),
        'observations':observations,'test_results':scores,'comparison_sheet':C.ref(attempt/'comparison-sheet.png'),
        'prior_source5_failure':plan['prior_source5_failure'],'old_heldout_rows_relabelled_as_passing':False,
        'target_pixels_used_for_fitting':False,'default_eligible':False,'automatic_publication':False,
        'new_api_calls':0,'new_remote_gpu_calls':0,'local_inference':{'model':'SAM2.1 Tiny','device':'mps','video_state_initializations':4,'processed_rgb_frames':8},
        'wall_seconds':time.monotonic()-started,'physical_accuracy_verified':False,'limitations':plan['limitations']}
    C.write_new(attempt/'report.json',report);return C.ref(attempt/'report.json')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','run']);a=p.parse_args()
    print(json.dumps(prepare()if a.action=='prepare'else run()),flush=True)
