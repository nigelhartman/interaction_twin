#!/usr/bin/env python3
"""One-frame local segmentation check; deliberately not a tracking pipeline."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--case',required=True)
p.add_argument('--frame-index',type=int,default=1)
p.add_argument('--box',type=float,nargs=4,required=True,metavar=('X0','Y0','X1','Y1'))
p.add_argument('--device',choices=['mps','cpu'],default='mps')
a=p.parse_args()
if '/' in a.case or a.case.startswith('.'):
    p.error('Use a case slug.')
if a.device=='mps' and not torch.backends.mps.is_available():
    p.error('MPS is unavailable in this execution context. No silent CPU fallback was used.')
case=ROOT/'data'/a.case
index=json.loads((case/'frames/index.json').read_text())['frames']
row=next((r for r in index if r['index']==a.frame_index),None)
if row is None:p.error('Frame index not found.')
image=Image.open(case/row['file']).convert('RGB'); w,h=image.size
x0,y0,x1,y1=a.box
if not 0<=x0<x1<=w or not 0<=y0<y1<=h:p.error('Box is outside the image.')
checkpoint=ROOT/'models/sam2.1_hiera_tiny.pt'
if not checkpoint.is_file():p.error('Missing SAM 2.1 Tiny checkpoint.')
out=case/'segmentation/smoke';out.mkdir(parents=True,exist_ok=True)
started=time.monotonic()
model=build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',str(checkpoint),device=a.device,apply_postprocessing=False)
predictor=SAM2ImagePredictor(model)
with torch.inference_mode():
    predictor.set_image(np.array(image,copy=True))
    masks,scores,logits=predictor.predict(box=np.array(a.box,dtype=np.float32),multimask_output=True)
if a.device=='mps':torch.mps.synchronize()
best=int(np.argmax(scores));mask=masks[best].astype(bool)
if not 0<int(mask.sum())<w*h:raise RuntimeError('Degenerate segmentation mask.')
Image.fromarray(mask.astype(np.uint8)*255).save(out/'mask.png')
overlay=np.asarray(image).astype(np.float32).copy()
overlay[mask]=overlay[mask]*.52+np.array([35,190,164])*.48
marked=Image.fromarray(overlay.astype(np.uint8));dr=ImageDraw.Draw(marked);dr.rectangle(a.box,outline='#ffad36',width=3)
marked.save(out/'overlay.jpg',quality=94)
preview=Image.new('RGB',(w*2,h+56),'#edf2f5');preview.paste(image,(0,56));preview.paste(marked,(w,56))
d=ImageDraw.Draw(preview);f=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',24)
d.text((16,14),'Original frame',fill='#183244',font=f)
d.text((w+16,14),'SAM 2.1 Tiny / '+a.device.upper()+' / manual box prompt',fill='#183244',font=f)
preview.save(out/'preview.jpg',quality=94)
meta={'provider':'meta_sam2_local','model':'sam2.1_hiera_tiny','device':a.device,
      'torch_version':torch.__version__,'inference_dtype':'float32','cpu_fallback_enabled':False,
      'scope':'single_frame_access_smoke_test_not_video_tracking',
      'input_frame':row['file'],'timestamp_s':row['timestamp_s'],
      'prompt':{'type':'box','coordinates_xyxy':a.box,'origin':'assistant_visual_annotation','human_confirmed':False},
      'selected_mask_index':best,'model_predicted_mask_quality':float(scores[best]),'measured_iou':None,
      'mask_area_pixels':int(mask.sum()),'wall_time_s':round(time.monotonic()-started,3),
      'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
(out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(meta,indent=2))
