#!/usr/bin/env python3
"""Measure contemporaneous component depth, then fit a fixed pairwise scale gauge."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from relative_gauge import log_depth_measurement,pair_edges,solve_gauge
from alignment_render import render_mesh
from stage4_generated_observer import pose_matrix
from stage4_generated_pose import sha256

ALLOWED={'supported_conditional_motion_candidate','reference_candidate_unverified_geometry_and_pose'}
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def load_sources(cases,source_hash):
    result={}
    for case in cases:
        case=Path(case).resolve();manifest=read(case/'manifest.json')
        if manifest['source']['sha256']!=source_hash:raise ValueError('Mask/source case video mismatch')
        rows={r['index']:r for r in read(case/'frames/index.json')['frames']}
        seg=case/'segmentation/automatic-components-v1'
        for track in read(seg/'tracks.json')['tracks']:
            for f in track['frames']:
                row=rows[f['sample_index']];image=case/row['file'];mask=seg/f['raw_mask_file']
                key=(track['track_id'],round(row['timestamp_s'],6),sha256(image),sha256(mask))
                result[key]=(image,mask)
    return result


def cached_depth(caches,timestamp,image,source_hash):
    pixels=None
    for metadata_path in caches:
        metadata_path=Path(metadata_path).resolve();meta=read(metadata_path)
        if meta.get('fov_x_deg')!=60.:continue
        for row in meta.get('frames',[]):
            if abs(row['timestamp_s']-timestamp)>1e-6 or row.get('source_video_sha256',source_hash)!=source_hash:continue
            path=metadata_path.parent/row['file']
            if sha256(path)!=row['sha256']:raise ValueError('Cached learned depth changed')
            if row.get('source_rgb_sha256')!=sha256(image):
                # Exported PNG bytes can differ; prove equality of decoded pixels.
                model_meta_path=row.get('model_metadata_file')
                if not model_meta_path:continue
                model_meta=read(model_meta_path)
                original=Path(model_meta_path).parent/model_meta['source']['image_path']
                if sha256(original)!=row['source_rgb_sha256']:raise ValueError('Cache reference image changed')
                pixels=np.array(Image.open(image).convert('RGB')) if pixels is None else pixels
                if not np.array_equal(pixels,np.array(Image.open(original).convert('RGB'))):continue
            raw=np.load(path)
            return raw['depth'].copy(),raw['valid'].copy(),{'mode':'reused_actual_MoGe_prediction_after_exact_pixel_check','file':str(path),'sha256':sha256(path),'metadata_sha256':sha256(metadata_path)}
    return None


def run(args):
    out=args.out.resolve()
    if out.exists():raise ValueError('Fresh output required')
    packs=[read(p) for p in args.experiment];ids=[p['object_id'] for p in packs]
    if len(ids)!=len(set(ids)):raise ValueError('One experiment per component required')
    source_hash=packs[0]['provenance']['source_video_sha256'];K=np.array(packs[0]['intrinsics']['matrix'])
    for p in packs:
        if p['provenance']['source_video_sha256']!=source_hash or not np.allclose(p['intrinsics']['matrix'],K):raise ValueError('One video and common K required')
        if p.get('fixed_native_mesh_scale')!=1.:raise ValueError('Native model scale must remain fixed')
        if any(p['provenance'].get(k) for k in ['manual_geometry_used','manual_pose_used','manual_corners_used']):raise ValueError('Manual geometry/poses prohibited')
        if sha256(p['provenance']['mesh_file'])!=p['provenance']['mesh_sha256']:raise ValueError('Learned mesh changed')
    sources=load_sources(args.mask_case,source_hash)
    groups={}
    for p in packs:
        for f in p['frames']:
            if f['status'] not in ALLOWED:continue
            pose_matrix(f)
            key=(p['object_id'],round(f['timestamp_s'],6),f['source_frame_sha256'],f['sam_mask_sha256'])
            if key not in sources:raise ValueError('Pose does not match exact source image/mask hashes')
            groups.setdefault(round(f['timestamp_s'],6),[]).append((p,f,*sources[key]))
    selected=[(t,rows) for t,rows in sorted(groups.items()) if len(rows)>=2]
    # Bounded deterministic temporal coverage; never choose times by depth fit.
    if len(selected)>args.max_frames:
        selected=[selected[i] for i in sorted(set(np.linspace(0,len(selected)-1,args.max_frames).round().astype(int))) ]
    out.mkdir(parents=True);model=None;modelprov=None;frames=[];failures=[]
    import trimesh
    meshes={p['object_id']:trimesh.load(p['provenance']['mesh_file'],force='mesh',process=False) for p in packs}
    for t,rows in selected:
        image=rows[0][2];rgb=np.array(Image.open(image).convert('RGB'));h,w=rgb.shape[:2]
        for _,_,other,_ in rows[1:]:
            if not np.array_equal(rgb,np.array(Image.open(other).convert('RGB'))):raise ValueError('Contemporaneous poses do not share actual source pixels')
        found=cached_depth(args.depth_cache,t,image,source_hash)
        if found is None:
            if not args.infer_missing:
                failures.append({'timestamp_s':t,'reason':'No exact cached depth; local inference not requested'});continue
            if model is None:
                os.environ['HF_HOME']=str(ROOT/'models/hf-cache');os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
                import torch
                from moge.model.v2 import MoGeModel
                if not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':raise RuntimeError('Actual MPS required, no fallback')
                modeldir=ROOT/'models/moge-2-vits-normal';modelprov=read(modeldir/'provenance.json')
                if sha256(modeldir/'model.pt')!=modelprov['checkpoint_sha256']:raise ValueError('Depth weights changed')
                model=MoGeModel.from_pretrained(str(modeldir/'model.pt')).to('mps').eval()
            ratio=640/max(w,h);size=(round(w*ratio),round(h*ratio))
            tensor=torch.from_numpy(np.array(Image.fromarray(rgb).resize(size,Image.Resampling.LANCZOS)).copy()).permute(2,0,1).float().to('mps')/255
            start=time.monotonic()
            with torch.inference_mode():pred=model.infer(tensor,num_tokens=1200,use_fp16=False,apply_mask=True,fov_x=60.);torch.mps.synchronize()
            depth=pred['depth'].cpu().numpy();valid=pred['mask'].cpu().numpy().astype(bool)
            depth_path=out/f'depth-{t:.6f}.npz';np.savez_compressed(depth_path,depth=depth,valid=valid)
            prov={'mode':'actual_local_MoGe2Small_MPS','file':str(depth_path),'sha256':sha256(depth_path),'model_provenance':modelprov,'inference_seconds':time.monotonic()-start}
        else:depth,valid,prov=found
        depth=cv2.resize(depth,(w,h),interpolation=cv2.INTER_LINEAR)
        valid=cv2.resize(valid.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)>0
        record={'timestamp_s':t,'source_image_sha256':sha256(image),'depth_provenance':prov,'objects':[]}
        for pack,f,_,maskpath in rows:
            tid=pack['object_id'];mesh=meshes[tid];mask=np.array(Image.open(maskpath).convert('L'))>127
            support=cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)&valid
            rendered=render_mesh(np.asarray(mesh.vertices),np.asarray(mesh.faces),K,np.array(f['candidate_rotation']),np.array(f['candidate_translation']),w,h)
            support &= rendered['mask']
            try:
                measurement=log_depth_measurement(depth,rendered['depth'],support)
                item={'object_id':tid,'status':'measured',**measurement}
            except ValueError as exc:item={'object_id':tid,'status':'withheld_insufficient_common_pixels','reason':str(exc)}
            item.update(pose_timestamp_s=f['timestamp_s'],pose_status=f['status'],mask_sha256=sha256(maskpath),mesh_sha256=pack['provenance']['mesh_sha256'])
            record['objects'].append(item)
        frames.append(record);write(out/'measurements.json',{'source_video_sha256':source_hash,'frames':frames,'failed_frames':failures})
    edges=pair_edges(frames);prior=read(args.prior_gauge);anchor_scale=prior['selected_scales'][args.anchor]
    result=solve_gauge(ids,edges,args.anchor,anchor_scale)
    result.update(edges=edges,frames=frames,failed_frames=failures,camera_K=K.tolist(),fov_x_deg=60.,same_K_as_pose_fitting=True,
                  source_video_sha256=source_hash,physical_assembly_verified=False,hinge_or_contact_recovered=False,fallback_equal_depth_used=False,
                  inputs=[{'file':str(p.resolve()),'sha256':sha256(p)} for p in args.experiment],prior_gauge_file=str(args.prior_gauge.resolve()),prior_gauge_sha256=sha256(args.prior_gauge))
    for edge in edges:
        prior_log=np.log(prior['selected_scales'][edge['a']]/prior['selected_scales'][edge['b']])
        edge['prior_reference_gauge_log_error']=float(abs(prior_log-edge['log_ratio']))
    result['limitations']=['Same-frame pair ratios cancel multiplicative depth gauge only. Depth shape bias and assumed camera remain.',
                          'Single-timestamp edges are explicitly not temporally validated. Depth and generated geometry are not measurements.',
                          'No metric dimensions, mechanical joint, hidden insertion, contact or successful seating established.']
    write(out/'metadata.json',result)
    print(json.dumps({k:result[k] for k in ['status','selected_scales','unplaced_components','temporal_stability_verified_for_all_edges']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,nargs='+',required=True);p.add_argument('--mask-case',type=Path,nargs='+',required=True)
    p.add_argument('--depth-cache',type=Path,nargs='*',default=[]);p.add_argument('--prior-gauge',type=Path,required=True)
    p.add_argument('--anchor',default='small_object_a');p.add_argument('--max-frames',type=int,default=9)
    p.add_argument('--infer-missing',action='store_true');p.add_argument('--out',type=Path,required=True)
    run(p.parse_args())
