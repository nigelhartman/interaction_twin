"""Four new disjoint source-frame tests using actual local SAM2.1 Tiny/CPU.

The target RGB is a permitted tracking input. All candidate predictions are
frozen before SAM2 runs or target masks are read. Seeds are exact prior SAM3.1
masks; no manual points, new network inference, checkpoint download or promotion.
"""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation,Slerp

import phone_source_regeneration_20261104_v1 as P
import phone_source_prism_20261104_v1 as G
import phone_source_prism_evaluation_20261104_v1 as E
import phone_source_prism_rgb_tracking_20261104_v1 as T
from native_appearance import load_native_glb,render_native_mesh

ROOT=P.ROOT
OUTPUT=G.OUTPUT/'new-disjoint-tests-v1'
PAIRS=[(9,10),(12,13),(21,22),(45,46)]
CHECKPOINT=ROOT/'models/sam2.1_hiera_tiny.pt'


def command(argv):
    return subprocess.run(argv,check=True,capture_output=True,timeout=120).stdout


def build():
    OUTPUT.mkdir(exist_ok=False);started=time.monotonic()
    source=P.CASE/'source/video.mp4'
    program=json.loads((P.CASE/'flow-procedure-1/interaction.json').read_text());ev={e['source_index']:e for e in program['evidence']}
    scene=json.loads((G.OUTPUT/'full-scene-v1/scene.json').read_text());k=np.asarray(scene['camera']['K'])
    unit=json.loads((G.OUTPUT/'full-scene-v1/unit-gauge-poses.json').read_text())
    baseline=json.loads(E.BASE.read_text());oldpart=next(c for c in baseline['components']if c['id']=='phone_blue')
    oldposes={p['source_sample_index']:p for p in oldpart['poses']}
    native=json.loads(command(['ffprobe','-v','error','-select_streams','v:0','-show_frames',
        '-show_entries','frame=best_effort_timestamp_time,width,height','-of','json',str(source)]))['frames']
    previous_tests=json.loads((ROOT/'output/assembly-quality-20261104/phone-midpoint-evaluation-v2/selection-plan.json').read_text())
    oldtimes=[r['timestamp_s']for r in previous_tests['test_frames']]
    tests=[]
    for number,(left,right)in enumerate(PAIRS,1):
        target=(ev[left]['timestamp_s']+ev[right]['timestamp_s'])/2
        frameid,frame=min(enumerate(native),key=lambda p:abs(float(p[1]['best_effort_timestamp_time'])-target))
        timestamp=float(frame['best_effort_timestamp_time'])
        assert all(abs(timestamp-e['timestamp_s'])>.05 for e in program['evidence'])
        assert all(abs(timestamp-t)>.05 for t in oldtimes)
        olda=np.asarray(oldposes[left]['camera_from_native_mesh']);oldb=np.asarray(oldposes[right]['camera_from_native_mesh'])
        scale=float(np.cbrt(np.linalg.det(olda[:3,:3])));fraction=(timestamp-ev[left]['timestamp_s'])/(ev[right]['timestamp_s']-ev[left]['timestamp_s'])
        oldmatrix=np.eye(4);oldmatrix[:3,:3]=scale*Slerp([0,1],Rotation.from_matrix([olda[:3,:3]/scale,oldb[:3,:3]/scale]))(fraction).as_matrix()
        oldmatrix[:3,3]=olda[:3,3]*(1-fraction)+oldb[:3,3]*fraction
        tests.append({'test_id':number,'reference_source_index':left,'right_source_index':right,
            'native_frame_index_zero_based':frameid,'timestamp_s':timestamp,'baseline_matrix':oldmatrix.tolist(),
            'reference_pose_available':str(left)in unit,'reference_unit_pose':unit.get(str(left))})
    plan={'format':'interaction-twin-phone-source-prism-new-disjoint-mask-tests','version':1,
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'implementation':P.ref(Path(__file__)),
        'source':P.ref(source),'program':P.ref(P.CASE/'flow-procedure-1/interaction.json'),
        'candidate':P.ref(G.OUTPUT/'asset.glb'),'full_scene':P.ref(G.OUTPUT/'full-scene-v1/scene.json'),
        'test_frames':tests,'target_RGB_decoded_before_plan':False,'target_RGB_allowed_for_tracking':True,
        'target_masks_used_for_pose_or_geometry_fit':False,'minimum_test_IoU':.65,'maximum_baseline_regression':.05,
        'segmentation':{'model':'Meta SAM2.1 Tiny','device':'cpu','checkpoint':P.ref(CHECKPOINT),
            'config':'configs/sam2.1/sam2.1_hiera_t.yaml','seed':'exact_earlier_saved_SAM3.1_mask',
            'manual_points_boxes_or_masks':False,'fallible_model_observations_not_ground_truth':True},
        'automatic_publication':False,'physical_accuracy_verified':False,'network_access_required':False,
        'new_API_or_remote_GPU_calls':0}
    P.save(OUTPUT/'selection-plan.json',plan)
    attempt=OUTPUT/'attempt';attempt.mkdir();decoded=attempt/'decoded';decoded.mkdir()
    selected='+'.join(f'eq(n\\,{r["native_frame_index_zero_based"]})'for r in tests)
    command(['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-n','-i',str(source),'-map','0:v:0',
        '-vf',f'select={selected}','-fps_mode','passthrough','-q:v','2',str(decoded/'%06d.jpg')])
    predictions=[]
    thickness=json.loads((G.OUTPUT/'report.json').read_text())['estimated_edge_thickness_over_length']
    for test in tests:
        rgb,mask,ref=G.frame(test['reference_source_index'])
        targetref=P.ref(decoded/f'{test["test_id"]:06d}.jpg')
        target=np.asarray(Image.open(ROOT/targetref['path']).convert('RGB'))
        if test['reference_pose_available']:
            predicted=T.track(rgb,target,mask,np.asarray(test['reference_unit_pose']),k,thickness)
        else:predicted={'status':'unknown','reason':'reference_pose_unknown_no_extrapolation'}
        predictions.append({**test,'reference':ref,'target_RGB':targetref,'prediction':predicted})
    P.save(attempt/'predictions-before-new-SAM2-inference.json',predictions)
    import torch
    from sam2.build_sam import build_sam2_video_predictor
    torch.set_num_threads(4)
    predictor=build_sam2_video_predictor(plan['segmentation']['config'],str(CHECKPOINT),device='cpu',apply_postprocessing=False)
    observations=[]
    for test in predictions:
        folder=attempt/f'test-{test["test_id"]}';folder.mkdir();inputs=folder/'inputs';inputs.mkdir()
        shutil.copyfile(ROOT/test['reference']['rgb']['path'],inputs/'000000.jpg')
        shutil.copyfile(ROOT/test['target_RGB']['path'],inputs/'000001.jpg')
        seed=np.asarray(Image.open(ROOT/test['reference']['mask']['path']))>0
        outputs={}
        with torch.inference_mode():
            state=predictor.init_state(str(inputs),offload_video_to_cpu=True,offload_state_to_cpu=True)
            predictor.add_new_mask(state,frame_idx=0,obj_id=1,mask=seed)
            for index,ids,logits in predictor.propagate_in_video(state,start_frame_idx=0):
                binary=(logits[list(ids).index(1),0]>0).cpu().numpy()
                assert binary.shape==seed.shape and binary.any()
                path=folder/f'mask-{index}.png';Image.fromarray(binary.astype(np.uint8)*255).save(path)
                outputs[index]=P.ref(path)
        observations.append({**test,'target_mask':outputs[1],'seed_reprojection_mask':outputs[0],
            'origin':'actual_local_Meta_SAM2.1_Tiny_CPU_forward_propagation','not_ground_truth':True})
        del state
        print('Actual new local SAM2 test complete',test['test_id'],flush=True)
    del predictor
    P.save(attempt/'new-observations.json',observations)
    mesh=load_native_glb(G.OUTPUT/'asset.glb');original=load_native_glb(ROOT/oldpart['asset']['path'])
    sk=k.copy();sk[0]*=.3;sk[1]*=.3;sk[0,2]+=(.3-1)/2;sk[1,2]+=(.3-1)/2
    scores=[];sheet=Image.new('RGB',(1080,1440),'#f7f8fb');draw=ImageDraw.Draw(sheet)
    for number,row in enumerate(observations):
        target=cv2.resize((np.asarray(Image.open(ROOT/row['target_mask']['path']))>0).astype(np.uint8),(216,288),interpolation=cv2.INTER_NEAREST)>0
        oldmatrix=np.asarray(row['baseline_matrix']);scale=float(np.cbrt(np.linalg.det(oldmatrix[:3,:3])))
        before=render_native_mesh(original,sk,oldmatrix[:3,:3]/scale,oldmatrix[:3,3],216,288,scale=scale)
        def iou(binary):return float((binary&target).sum()/max(1,(binary|target).sum()))
        baselineiou=iou(before['mask']);value=None;after=None
        pred=row['prediction']
        if pred['status']=='estimated':
            m=np.asarray(pred['matrix']);after=render_native_mesh(mesh,sk,m[:3,:3],m[:3,3],216,288);value=iou(after['mask'])
        passed=value is not None and value>=.65 and value>=baselineiou-.05
        scores.append({'test_id':row['test_id'],'timestamp_s':row['timestamp_s'],'RGB':row['target_RGB'],
            'mask':row['target_mask'],'candidate_pose_status':pred['status'],'baseline_full_triangle_IoU':baselineiou,
            'candidate_full_triangle_IoU':value,'passes_unrelaxed_gate':passed})
        rgb=cv2.resize(np.asarray(Image.open(ROOT/row['target_RGB']['path']).convert('RGB')),(216,288))
        items=[('Actual new RGB + local SAM2',rgb,target),
            (f'Original native | IoU {baselineiou:.3f}',before['rgb'],before['mask'])]
        if after is not None:items.append((f'Source prism | IoU {value:.3f}',after['rgb'],after['mask']))
        union=np.logical_or.reduce([r[2]for r in items]);ys,xs=np.nonzero(union)
        crop=[max(0,int(xs.min())-12),max(0,int(ys.min())-12),min(216,int(xs.max())+13),min(288,int(ys.max())+13)]
        for col,(label,pixels,binary)in enumerate(items):
            if col:pixels=pixels.copy();pixels[~binary]=[238,241,247]
            part=Image.fromarray(pixels).crop(crop);part.thumbnail((340,300))
            sheet.paste(part,(col*360+(360-part.width)//2,number*360+48))
            draw.text((col*360+8,number*360+8),f'New test{row["test_id"]} | {row["timestamp_s"]:.6f}s',fill='#121824')
            draw.text((col*360+8,number*360+25),label,fill='#121824')
    sheet.save(attempt/'comparison.png')
    report={'format':'interaction-twin-phone-source-prism-new-disjoint-test-report','version':1,
        'selection_plan':P.ref(OUTPUT/'selection-plan.json'),'predictions_before_SAM2':P.ref(attempt/'predictions-before-new-SAM2-inference.json'),
        'observations':P.ref(attempt/'new-observations.json'),'comparison':P.ref(attempt/'comparison.png'),
        'results':scores,'all_unrelaxed_gates_pass':all(r['passes_unrelaxed_gate']for r in scores),
        'actual_local_SAM2_CPU_inference_states':4,'new_API_or_GPU_calls':0,
        'target_mask_used_for_fitting':False,'target_RGB_used_for_tracking':True,
        'automatic_publication':False,'physical_accuracy_verified':False,
        'elapsed_seconds':time.monotonic()-started,
        'limitations':['Fresh disjoint targets are tested against fallible locally propagated masks, not human ground truth.',
            'New RGB drives automatic tracking; this does not establish blind temporal pose prediction.',
            'Planar feature assumptions can fail at thin edges and changing reflections; failed/unknown predictions remain explicit.',
            'Passing projection gates cannot establish hidden geometry, metric thickness, port contact, grasping or insertion.']}
    P.save(OUTPUT/'report.json',report)
    return {'report':P.ref(OUTPUT/'report.json'),'all_unrelaxed_gates_pass':report['all_unrelaxed_gates_pass'],'results':scores}


if __name__=='__main__':
    print(json.dumps(build(),indent=2))
