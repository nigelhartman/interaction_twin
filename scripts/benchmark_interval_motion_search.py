#!/usr/bin/env python3
"""Prospective fixed-pose TRAIN-reference proxy/native raster comparison."""
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import interval_motion as M
import interval_motion_search as S
from native_appearance import load_native_glb
from scripts import stage4_native_pose as P

def main(out):
    out=Path(out)
    if out.exists():raise ValueError('A fresh benchmark output is required')
    out.mkdir(parents=True)
    names=['jabra-case-base','jabra-case-lid','jabra-earbud-a']
    protocol={'method':S.POLICY,'random_orientation_indices':[0,8,24],
        'reference_rule':'chronologically first frozen episode reference, no fitted pose or held-out result',
        'plans':[f'data/interval-motion-20261031/{name}-3fps-plan-v1/plan.json' for name in names],
        'search_width':192,'evaluates_motion_or_heldout_errors':False,'new_model_or_api_calls':0,
        'benchmark_is_not_pixel_equivalence_or_physical_accuracy_validation':True}
    M.write(out/'protocol.json',protocol);(out/'implementation.py').write_bytes(Path(__file__).read_bytes());(out/'proxy-implementation.py').write_bytes(Path(S.__file__).read_bytes())
    rows=[]
    for plan_path in protocol['plans']:
        plan=M.read(ROOT/plan_path);episode=plan['episodes'][0]
        row=next(r for r in episode['frames'] if r['sample_index']==episode['reference_sample_index'])
        mask=np.asarray(Image.open(M.file_path(ROOT,row['mask_url'])).convert('L'))>0
        height,width=mask.shape;size=(192,round(height*192/width));small=cv2.resize(mask.astype(np.uint8),size,interpolation=cv2.INTER_NEAREST).astype(bool)
        k=P.resized_intrinsics(np.asarray(plan['intrinsics']['matrix']),(width,height),size)
        mesh_path=M.file_path(ROOT,plan['asset']['path']);before=M.sha(mesh_path);mesh=load_native_glb(mesh_path)
        centre=(mesh.vertices.min(0)+mesh.vertices.max(0))/2;centered=mesh.vertices-centre
        rotations=P.generic_orientations(M.POLICY['random_orientation_seeds'])
        for index in protocol['random_orientation_indices']:
            rotation=rotations[index];translation=P.initial_translation(centered,rotation,k,small)
            start=time.monotonic();proxy=S.silhouette(mesh,rotation,translation,k,size,centre);proxy_time=time.monotonic()-start
            start=time.monotonic();native=P.silhouette(mesh,rotation,translation,k,size,centre);native_time=time.monotonic()-start
            distance_native=cv2.distanceTransform((~native).astype(np.uint8),cv2.DIST_L2,5)
            distance_proxy=cv2.distanceTransform((~proxy).astype(np.uint8),cv2.DIST_L2,5)
            rows.append({'component_id':plan['component_id'],'reference_sample_index':row['sample_index'],
                'reference_source_image_sha256':row['source_image_sha256'],'reference_mask_sha256':row['mask_sha256'],
                'mesh_sha256':before,'generic_orientation_index':index,'rotation':rotation.tolist(),
                'translation_centered_mesh':translation.tolist(),'origin_offset':centre.tolist(),'intrinsics':k.tolist(),'size_wh':list(size),
                'proxy_native_iou':S.iou(proxy,native),'proxy_pixels':int(proxy.sum()),'native_pixels':int(native.sum()),
                'proxy_seconds':proxy_time,'native_seconds':native_time,
                'maximum_proxy_pixel_to_nearest_native_sample_px':float(distance_native[proxy].max()) if proxy.any() and native.any() else None,
                'maximum_native_pixel_to_nearest_proxy_sample_px':float(distance_proxy[native].max()) if proxy.any() and native.any() else None})
        if M.sha(mesh_path)!=before:raise ValueError('Native asset changed')
    report={'format':'short-interval-reference-search-benchmark','version':1,'protocol_sha256':M.sha(out/'protocol.json'),
        'proxy_implementation_sha256':M.sha(S.__file__),'results':rows,
        'median_search_speedup':float(np.median([r['native_seconds']/max(1e-12,r['proxy_seconds']) for r in rows])),
        'minimum_proxy_native_iou':min(r['proxy_native_iou'] for r in rows),'new_model_or_api_calls':0,
        'conclusion':'Search proxy boundaries differ. Only candidate generation is accelerated; exact native rendering still reranks final reference candidates and scores every motion pose.'}
    M.write(out/'report.json',report)
    print(json.dumps({k:report[k] for k in ['median_search_speedup','minimum_proxy_native_iou','new_model_or_api_calls']}))

if __name__=='__main__':main(ROOT/sys.argv[1])
