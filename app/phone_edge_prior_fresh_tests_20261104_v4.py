"""Two newly sampled mask-withheld RGB tracking tests for the edge prior.

New target PTS are frozen before RGB extraction. Predictions are frozen before
actual local SAM2/CPU target observations. The observed masks remain fallible,
not ground truth; source RGB remains a permitted input to the tracker.
"""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation,Slerp

import phone_source_regeneration_20261104_v1 as P
import phone_source_prism_20261104_v1 as G
import phone_source_prism_evaluation_20261104_v1 as E
import phone_source_prism_ray_tracking_20261104_v2 as R
from native_appearance import load_native_glb,render_native_mesh

ROOT=P.ROOT
OUTPUT=ROOT/'output/submission-polish-20261104-v1/phone-edge-prior-v4/fresh-tests-v1'
ASSET=ROOT/'data/geometry-quality-20261104/phone-multiview-edge-prior-v4/asset.glb'
PAIRS=[(15,16),(35,36)]


def command(argv):return subprocess.run(argv,check=True,capture_output=True,timeout=120).stdout


def build():
    OUTPUT.mkdir(exist_ok=False)
    base=json.loads(E.BASE.read_text());k=np.asarray(base['camera']['K'])
    component=next(c for c in base['components']if c['id']=='phone_blue');old={r['source_sample_index']:r for r in component['poses']}
    unit=json.loads((G.OUTPUT/'full-scene-v1/unit-gauge-poses.json').read_text())
    correction=ROOT/'output/submission-polish-20261104-v1/phone-edge-prior-v4/source-training-unit-poses.json'
    unit.update(json.loads(correction.read_text()))
    program=json.loads((P.CASE/'flow-procedure-1/interaction.json').read_text());evidence={r['source_index']:r for r in program['evidence']}
    source=P.CASE/'source/video.mp4'
    frames=json.loads(command(['ffprobe','-v','error','-select_streams','v:0','-show_frames',
        '-show_entries','frame=best_effort_timestamp_time','-of','json',str(source)]))['frames']
    prior_times=[e['timestamp_s']for e in program['evidence']]
    for path in [ROOT/'output/assembly-quality-20261104/phone-midpoint-evaluation-v2/selection-plan.json',
                 G.OUTPUT/'new-disjoint-tests-v1/selection-plan.json']:
        prior_times.extend(r['timestamp_s']for r in json.loads(path.read_text())['test_frames'])
    tests=[]
    for number,(left,right)in enumerate(PAIRS,1):
        requested=(evidence[left]['timestamp_s']+evidence[right]['timestamp_s'])/2
        index,frame=min(enumerate(frames),key=lambda p:abs(float(p[1]['best_effort_timestamp_time'])-requested))
        pts=float(frame['best_effort_timestamp_time'])
        assert all(abs(pts-t)>.05 for t in prior_times)
        a=np.asarray(old[left]['camera_from_native_mesh']);b=np.asarray(old[right]['camera_from_native_mesh'])
        scale=float(np.cbrt(np.linalg.det(a[:3,:3])));f=(pts-evidence[left]['timestamp_s'])/(evidence[right]['timestamp_s']-evidence[left]['timestamp_s'])
        matrix=np.eye(4);matrix[:3,:3]=scale*Slerp([0,1],Rotation.from_matrix([a[:3,:3]/scale,b[:3,:3]/scale]))(f).as_matrix();matrix[:3,3]=a[:3,3]*(1-f)+b[:3,3]*f
        tests.append({'test_id':number,'reference_indices':[left,right],'native_frame_index_zero_based':index,
            'timestamp_s':pts,'baseline_interpolated_matrix':matrix.tolist()})
    checkpoint=ROOT/'models/sam2.1_hiera_tiny.pt'
    plan={'format':'interaction-twin-phone-edge-prior-new-source-frame-test-plan','version':1,
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'implementation':P.ref(Path(__file__)),
        'asset':P.ref(ASSET),'edge_prior':P.ref(ASSET.parent/'derivation.json'),
        'source_program':P.ref(P.CASE/'flow-procedure-1/interaction.json'),'source':P.ref(source),
        'source_pose_corrections':P.ref(correction),'tests':tests,'target_RGB_decoded_before_plan':False,
        'target_RGB_used_for_tracking':True,'target_masks_used_for_geometry_pose_or_reference_selection':False,
        'prediction_algorithm':P.ref(ROOT/'phone_source_prism_ray_tracking_20261104_v2.py'),
        'minimum_IoU':.65,'maximum_baseline_regression':.05,
        'segmentation':{'model':'Meta SAM2.1 Tiny','device':'cpu','checkpoint':P.ref(checkpoint),
            'config':'configs/sam2.1/sam2.1_hiera_t.yaml','seed':'exact earlier saved SAM3.1 mask',
            'manual_points_or_boxes':False,'not_ground_truth':True},
        'new_API_or_GPU_calls':0,'automatic_publication':False,'physical_accuracy_verified':False}
    P.save(OUTPUT/'selection-plan.json',plan)
    decoded=OUTPUT/'decoded';decoded.mkdir()
    selection='+'.join(f'eq(n\\,{r["native_frame_index_zero_based"]})'for r in tests)
    command(['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-n','-i',str(source),'-map','0:v:0',
        '-vf',f'select={selection}','-fps_mode','passthrough','-q:v','2',str(decoded/'%06d.jpg')])
    mesh=load_native_glb(ASSET);planes=np.unique(np.round(ConvexHull(mesh.vertices).equations,10),axis=0)
    predictions=[]
    for test in tests:
        tref=P.ref(decoded/f'{test["test_id"]:06d}.jpg');target=np.asarray(Image.open(ROOT/tref['path']).convert('RGB'));options=[]
        for index in test['reference_indices']:
            rgb,mask,sref=G.frame(index)
            result=R.track(rgb,target,mask,np.asarray(unit[str(index)]),k,planes)
            options.append({'reference_index':index,'reference':sref,**result})
        known=[o for o in options if o['status']=='estimated'];selected=max(known,key=lambda o:o['confidence_for_reference_selection'])if known else None
        predictions.append({**test,'target_RGB':tref,'reference_options':options,'selected':selected})
    P.save(OUTPUT/'predictions-before-new-target-mask-inference.json',predictions)
    import torch
    from sam2.build_sam import build_sam2_video_predictor
    torch.set_num_threads(4)
    predictor=build_sam2_video_predictor(plan['segmentation']['config'],str(checkpoint),device='cpu',apply_postprocessing=False)
    observations=[]
    for pred in predictions:
        folder=OUTPUT/f'test-{pred["test_id"]}';folder.mkdir();inputs=folder/'inputs';inputs.mkdir()
        left=pred['reference_indices'][0];rgb,seed,sref=G.frame(left)
        shutil.copyfile(ROOT/sref['rgb']['path'],inputs/'000000.jpg');shutil.copyfile(ROOT/pred['target_RGB']['path'],inputs/'000001.jpg')
        masks={}
        with torch.inference_mode():
            state=predictor.init_state(str(inputs),offload_video_to_cpu=True,offload_state_to_cpu=True)
            predictor.add_new_mask(state,frame_idx=0,obj_id=1,mask=seed)
            for index,ids,logits in predictor.propagate_in_video(state,start_frame_idx=0):
                binary=(logits[list(ids).index(1),0]>0).cpu().numpy();assert binary.shape==seed.shape and binary.any()
                file=folder/f'mask-{index}.png';Image.fromarray(binary.astype(np.uint8)*255).save(file);masks[index]=P.ref(file)
        observations.append({**pred,'target_mask':masks[1],'seed_mask':sref['mask'],
            'mask_origin':'actual_local_SAM2.1_Tiny_CPU_forward_propagation_not_ground_truth'})
        del state
    del predictor
    P.save(OUTPUT/'fresh-observations.json',observations)
    original=load_native_glb(ROOT/component['asset']['path']);previous=load_native_glb(G.OUTPUT/'asset.glb')
    sk=k.copy();sk[0]*=.3;sk[1]*=.3;sk[0,2]+=(.3-1)/2;sk[1,2]+=(.3-1)/2
    results=[];sheet=Image.new('RGB',(1200,800),'#f6f8fc')
    from phone_multiview_edge_prior_20261104_v4 import image_panel
    for row,item in enumerate(observations):
        target=cv2.resize((np.asarray(Image.open(ROOT/item['target_mask']['path']))>0).astype(np.uint8),(216,288),interpolation=cv2.INTER_NEAREST)>0
        m=np.asarray(item['baseline_interpolated_matrix']);scale=float(np.cbrt(np.linalg.det(m[:3,:3])))
        before=render_native_mesh(original,sk,m[:3,:3]/scale,m[:3,3],216,288,scale=scale)
        def iou(mask):return float((mask&target).sum()/max(1,(mask|target).sum()))
        baseline=iou(before['mask']);value=None;priorvalue=None;after=None
        if item['selected']is not None:
            m=np.asarray(item['selected']['matrix']);after=render_native_mesh(mesh,sk,m[:3,:3],m[:3,3],216,288)
            value=iou(after['mask']);priorvalue=iou(render_native_mesh(previous,sk,m[:3,:3],m[:3,3],216,288)['mask'])
        passed=value is not None and value>=.65 and value>=baseline-.05
        results.append({'test_id':item['test_id'],'timestamp_s':item['timestamp_s'],'target_RGB':item['target_RGB'],
            'target_mask':item['target_mask'],'baseline_original_SAM_IoU':baseline,
            'previous_prism_same_new_pose_IoU':priorvalue,'new_edge_prior_IoU':value,
            'passes_unrelaxed_gate':passed,'target_mask_used_for_fit':False,
            'RGB_tracking_status':'estimated'if item['selected']is not None else'unknown'})
        rgb=cv2.resize(np.asarray(Image.open(ROOT/item['target_RGB']['path']).convert('RGB')),(216,288))
        image_panel(sheet,rgb,target,0,row*400,400,400,f'NEW SOURCE {item["timestamp_s"]:.6f}s')
        bp=np.full((288,216,3),[238,241,247],np.uint8);bp[before['mask']]=before['rgb'][before['mask']]
        image_panel(sheet,bp,before['mask'],400,row*400,400,400,f'Original SAM3D | IoU {baseline:.3f}')
        if after is not None:
            ap=np.full((288,216,3),[238,241,247],np.uint8);ap[after['mask']]=after['rgb'][after['mask']]
            image_panel(sheet,ap,after['mask'],800,row*400,400,400,f'Edge prior | IoU {value:.3f}')
        else:ImageDraw.Draw(sheet).text((820,row*400+100),'Unknown: RGB confidence insufficient',fill='#536079')
    sheet.save(OUTPUT/'comparison.png')
    report={'format':'interaction-twin-phone-edge-prior-new-source-frame-tests','version':1,
        'plan':P.ref(OUTPUT/'selection-plan.json'),'predictions_before_SAM2':P.ref(OUTPUT/'predictions-before-new-target-mask-inference.json'),
        'actual_observations':P.ref(OUTPUT/'fresh-observations.json'),'comparison':P.ref(OUTPUT/'comparison.png'),
        'results':results,'passing_tests':sum(r['passes_unrelaxed_gate']for r in results),
        'new_target_frames':2,'target_masks_used_for_fit':False,'actual_local_SAM2_CPU_states':2,
        'new_API_or_GPU_calls':0,'automatic_publication':False,'physical_accuracy_verified':False,
        'limitations':['New distinct source PTS are withheld from geometry and source training; target RGB is allowed for tracking.',
            'SAM2 target masks are fallible model observations propagated from saved SAM3.1 seeds, not independent human ground truth.',
            'The previous prism is rendered at the same new pose for a conditional shape-only diagnostic, not a separately retuned tracker.',
            'No2D mask result establishes true3D depth, physical contact, grasping or insertion.']}
    P.save(OUTPUT/'report.json',report)
    return {'report':P.ref(OUTPUT/'report.json'),'passing_tests':report['passing_tests'],'results':results}


if __name__=='__main__':
    print(json.dumps(build(),indent=2))
