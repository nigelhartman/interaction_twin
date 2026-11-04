#!/usr/bin/env python3
"""Source/mask/full native triangle proof for the private charger revision."""
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import charger_tracking_recovery_20261103_v1 as C
import charger_mask_refit_20261103_v1 as F
import sam3d_mechanisms as M
from native_appearance import load_native_glb, render_native_mesh

SAMPLES = [36, 42, 48, 54]
OUTPUT = 'output/new-capture-alignment-20261103/phone-charger-audit-v1'


def build():
    folder = ROOT / OUTPUT; report = C.read(ROOT/F.OUTPUT/'report.json')
    baseline = C.read(C.local(report['baseline']['path'])); candidate = C.read(C.local(report['candidate_scene']['path']))
    old = next(c for c in baseline['components'] if c['id'] == 'adapter_late'); new = next(c for c in candidate['components'] if c['id'] == 'adapter_late')
    mesh = load_native_glb(C.local(new['asset']['path'])); program = C.read(ROOT/C.PROGRAM); width, height = candidate['source']['width'], candidate['source']['height']
    factor = 720 / height; w, h = round(width*factor), 720
    k = np.asarray(candidate['camera']['K']); k[0] *= w/width; k[1] *= h/height; k[0,2] += (w/width-1)/2; k[1,2] += (h/height-1)/2
    canvas = Image.new('RGB', (1600, 1610), '#f7f8fb'); draw = ImageDraw.Draw(canvas)
    draw.text((22, 15), 'Charger body: source-bound SAM recovery / unchanged full native triangles', fill='#151923')
    draw.text((22, 39), 'Automatic mask refit. Same-mask diagnostics; shape, depth, contact and physical scale remain estimates.', fill='#637085')
    for col, title in enumerate(['Source crop', 'Previous 3D estimate', 'Fresh SAM3.1 body mask', 'Revised native 3D', 'Revised source overlay']): draw.text((col*320+16, 80), title, fill='#151923')
    proof = []
    for n, index in enumerate(SAMPLES):
        evidence = next(e for e in program['evidence'] if e['source_index'] == index); before = next(r for r in old['poses'] if r['source_sample_index'] == index); after = next(r for r in new['poses'] if r['source_sample_index'] == index)
        rgb = cv2.resize(np.asarray(Image.open(C.local(evidence['frame_path'])).convert('RGB')), (w,h), interpolation=cv2.INTER_AREA)
        mask = cv2.resize((np.asarray(Image.open(C.local(after['source_mask']['path'])))>0).astype(np.uint8), (w,h), interpolation=cv2.INTER_NEAREST)>0
        r,t,s = M.rotation_pose(after['camera_from_native_mesh']); rendered = render_native_mesh(mesh,k,r,t,w,h,scale=s)
        union = mask | rendered['mask']; y,x = np.where(union); span=max(x.max()-x.min()+1,y.max()-y.min()+1); padding=max(20,int(span*.25))
        box = [max(0,int(x.min()-padding)),max(0,int(y.min()-padding)),min(w,int(x.max()+padding+1)),min(h,int(y.max()+padding+1))]
        native=np.full_like(rgb,[231,235,243]);native[rendered['mask']]=rendered['rgb'][rendered['mask']]
        overlay=rgb.copy();overlay[rendered['mask']]=(overlay[rendered['mask']]*.45+rendered['rgb'][rendered['mask']]*.55).astype(np.uint8)
        sam=rgb.copy();sam[mask]=(sam[mask]*.45+np.array([59,102,245])*.55).astype(np.uint8)
        panels=[rgb,None,sam,native,overlay]
        if before['status']=='estimated':
            br,bt,bs=M.rotation_pose(before['camera_from_native_mesh']);b=render_native_mesh(mesh,k,br,bt,w,h,scale=bs);oldrgb=np.full_like(rgb,[231,235,243]);oldrgb[b['mask']]=b['rgb'][b['mask']];panels[1]=oldrgb
        for col,pixels in enumerate(panels):
            x0,y0=col*320+10,110+n*370
            if pixels is None:
                draw.rounded_rectangle((x0,y0,x0+299,y0+299),radius=10,fill='#e7ebf3');draw.text((x0+22,y0+135),'Unknown / no supported pose',fill='#637085')
            else:
                crop=Image.fromarray(pixels).crop(box);crop=ImageOps.contain(crop,(300,300));canvas.paste(crop,(x0+(300-crop.width)//2,y0+(300-crop.height)//2))
            draw.text((x0,y0+308),f'#{index} / {evidence["timestamp_s"]:.3f}s',fill='#151923')
        diagnostic=next(r for r in report['comparison']if r['source_sample_index']==index)
        draw.text((20,110+n*370+337),f'Conditional full-triangle mask IoU {diagnostic["candidate_full_native"]["full_triangle_raw_mask_iou"]:.3f}; all {len(mesh.faces):,} faces unchanged',fill='#637085')
        proof.append({'sample_index':index,'timestamp_s':evidence['timestamp_s'],'source_rgb':C.ref(evidence['frame_path']),
            'fresh_mask':after['source_mask'],'before_status':before['status'],'after_status':after['status'],
            'camera_raster':[w,h],'editorial_crop_xyxy':box,'crop_selection':'union of fresh observed mask and full revised native render plus25percent padding',
            'face_count':len(mesh.faces),'native_geometry_decimated_or_modified':False})
    path=folder/'charger-full-native-before-after.jpg';canvas.save(path,quality=95)
    retained=[r for r in report['comparison']if r['candidate_status']=='estimated'];common=[r for r in retained if r.get('baseline_full_native')]
    summary={'format':'interaction-twin-charger-tracking-proof','version':1,'proof_image':C.ref(path),
        'report':C.ref(ROOT/F.OUTPUT/'report.json'),'scene':report['candidate_scene'],'native_asset':new['asset'],'samples':proof,
        'adapter_late_estimated_before':report['baseline_estimated_count'],'adapter_late_estimated_after':report['candidate_estimated_count'],
        'newly_estimated_samples':report['newly_supported_source_samples'],'fresh_refits_retained':len(retained),
        'median_retained_full_triangle_same_mask_iou':float(np.median([r['candidate_full_native']['full_triangle_raw_mask_iou']for r in retained])),
        'matched_existing_count':len(common),'matched_existing_baseline_median':float(np.median([r['baseline_full_native']['full_triangle_raw_mask_iou']for r in common])),
        'matched_existing_candidate_median':float(np.median([r['candidate_full_native']['full_triangle_raw_mask_iou']for r in common])),
        'largest_matched_existing_raw_iou_drop':float(max(r['baseline_full_native']['full_triangle_raw_mask_iou']-r['candidate_full_native']['full_triangle_raw_mask_iou']for r in common)),
        'independent_accuracy_or_physical_validation':False,'automatic_publication':False}
    C.write_new(folder/'charger-tracking-proof.json',summary);return summary


if __name__=='__main__':print(json.dumps(build()),flush=True)
