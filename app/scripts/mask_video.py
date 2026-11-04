#!/usr/bin/env python3
"""Camera-only sampled-video mask pass, authorized as an additional Stage 1 check."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
import torch
from sam2.build_sam import build_sam2_video_predictor

ROOT=Path(__file__).resolve().parents[1]
def font(size):
    try:return ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',size)
    except OSError:return ImageFont.load_default(size=size)
def save_json(path,data):path.write_text(json.dumps(data,indent=2)+'\n')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',required=True)
    parser.add_argument('--box',nargs=4,type=float,default=[350,85,695,320])
    parser.add_argument('--device',choices=['mps','cpu'],default='mps')
    parser.add_argument('--run-name',default='camera-full-1fps')
    parser.add_argument('--prompts',type=Path,help='Optional JSON seed/correction boxes, using 1-based source sample indices.')
    args=parser.parse_args()
    for slug in [args.case,args.run_name]:
        if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,63}',slug):parser.error('Use lowercase slug names.')
    if args.device=='mps' and not torch.backends.mps.is_available():parser.error('MPS unavailable; no silent CPU fallback.')
    case=ROOT/'data'/args.case;manifest=json.loads((case/'manifest.json').read_text());rows=json.loads((case/'frames/index.json').read_text())['frames']
    expected=sorted((case/'frames').glob('*.jpg'),key=lambda p:int(p.stem))
    if expected!=[case/r['file'] for r in rows]:parser.error('JPEG directory does not match timestamp index.')
    w,h=rows[0]['width'],rows[0]['height']
    if any((r['width'],r['height'])!=(w,h) for r in rows):parser.error('Frame sizes must be constant.')
    prompts=json.loads(args.prompts.read_text()) if args.prompts else [{'sample_index':1,'box_xyxy':args.box,'origin':'assistant_visual_annotation','human_confirmed':False}]
    if not prompts or not any(p['sample_index']==1 for p in prompts):parser.error('A seed on sample 1 is required.')
    for p in prompts:
        x0,y0,x1,y1=p['box_xyxy']
        if not 1<=p['sample_index']<=len(rows) or not 0<=x0<x1<=w or not 0<=y0<y1<=h:parser.error('Invalid prompt index or box.')
    out=case/'segmentation'/args.run_name
    if out.exists():parser.error('Run directory exists. Use another run-name to preserve evidence.')
    for name in ['masks','overlays','comparison']:(out/name).mkdir(parents=True,exist_ok=True)
    checkpoint=ROOT/'models/sam2.1_hiera_tiny.pt'
    started=time.monotonic()
    meta={'status':'running','created_at_utc':datetime.now(timezone.utc).isoformat(),'model':'sam2.1_hiera_tiny','provider':'meta_sam2_local','device':args.device,
          'torch_version':torch.__version__,'cpu_fallback_enabled':False,'cpu_storage_offload':True,'dtype':'float32',
          'scope':'camera_assembly_only_sampled_full_video; not Stage 2 component tracking',
          'source_sha256':manifest['source']['sha256'],'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
          'sampling':manifest['sampling'],'object_id':1,'object_name':'camera_assembly','prompts':prompts,
          'ground_truth_available':False,'accuracy_measured':False,'frame_count_expected':len(rows)}
    save_json(out/'metadata.json',meta)
    try:
        print('Loading SAM 2.1 Tiny video predictor on '+args.device,flush=True)
        predictor=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_t.yaml',str(checkpoint),device=args.device,apply_postprocessing=False)
        with torch.inference_mode():
            state=predictor.init_state(str(case/'frames'),offload_video_to_cpu=True,offload_state_to_cpu=True)
            for p in prompts:
                predictor.add_new_points_or_box(state,frame_idx=p['sample_index']-1,obj_id=1,box=np.array(p['box_xyxy'],dtype=np.float32))
            stats=[];previous=None;inference_started=time.monotonic()
            for frame_idx,obj_ids,logits in predictor.propagate_in_video(state):
                row=rows[frame_idx];j=list(obj_ids).index(1);mask=(logits[j,0]>0).cpu().numpy()
                area=int(mask.sum());fraction=area/(w*h)
                flags=[]
                if area==0:flags.append('empty_mask')
                if fraction>.8:flags.append('mask_covers_over_80_percent')
                if previous and area and max(area/previous,previous/area)>3:flags.append('area_change_over_3x')
                ys,xs=np.where(mask)
                bbox=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)] if area else None
                name=f"{row['index']:06d}"
                Image.fromarray(mask.astype(np.uint8)*255).save(out/'masks'/(name+'.png'))
                image=Image.open(case/row['file']).convert('RGB');pixels=np.array(image,dtype=np.float32)
                pixels[mask]=pixels[mask]*.52+np.array([35,190,164])*.48
                overlay=Image.fromarray(pixels.astype(np.uint8));overlay.save(out/'overlays'/(name+'.jpg'),quality=92)
                preview=Image.new('RGB',(w*2,h+64),'#edf2f5');preview.paste(image,(0,64));preview.paste(overlay,(w,64));d=ImageDraw.Draw(preview)
                label=f"sample {row['index']:02d}/{len(rows)} | source {row['timestamp_s']:.3f}s"
                d.text((16,12),'Original | '+label,font=font(23),fill='#183244')
                d.text((w+16,12),'SAM 2.1 Tiny / '+args.device.upper()+' | camera mask',font=font(23),fill='#183244')
                preview.save(out/'comparison'/(name+'.jpg'),quality=92)
                stats.append({'sample_index':row['index'],'timestamp_s':row['timestamp_s'],'input_file':row['file'],
                              'mask_file':'masks/'+name+'.png','mask_area_pixels':area,'area_fraction':round(fraction,6),'bbox_xyxy':bbox,'heuristic_flags':flags})
                previous=area
                if len(stats)%10==0 or len(stats)==len(rows):
                    save_json(out/'progress.json',{'completed':len(stats),'total':len(rows),'elapsed_s':round(time.monotonic()-started,2)})
                    print(f"Processed {len(stats)}/{len(rows)} samples; {time.monotonic()-started:.1f}s elapsed",flush=True)
            if args.device=='mps':torch.mps.synchronize()
            inference_s=time.monotonic()-inference_started
        if len(stats)!=len(rows):raise RuntimeError('Not every sampled frame was processed.')
        save_json(out/'frames.json',{'time_domain':'original_source_presentation_seconds','frames':stats})
        # 1 sampled frame per second, held for display; 30fps encoding is duplication, not mask inference.
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-framerate','1','-i',str(out/'comparison/%06d.jpg'),'-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-r','30','-movflags','+faststart',str(out/'camera-mask-preview.mp4')],check=True)
        sheet_files=[]
        for batch in range(0,len(rows),24):
            group=rows[batch:batch+24];sheet=Image.new('RGB',(1304,58+((len(group)+3)//4)*207),'#edf2f5');d=ImageDraw.Draw(sheet)
            d.text((12,14),'Camera mask / source timestamps / SAM 2.1 Tiny '+args.device.upper(),font=font(22),fill='#183244')
            for i,row in enumerate(group):
                x=12+(i%4)*320;y=58+(i//4)*207
                thumb=ImageOps.contain(Image.open(out/'overlays'/f"{row['index']:06d}.jpg"),(308,174))
                sheet.paste(thumb,(x+(308-thumb.width)//2,y+(174-thumb.height)//2))
                d.text((x,y+179),f"#{row['index']:03d} / {row['timestamp_s']:.3f}s",font=font(19),fill='#183244')
            name=f'contact-sheet-{batch//24+1:02d}.jpg';sheet.save(out/name,quality=93);sheet_files.append(name)
        selected=[rows[round(i*(len(rows)-1)/11)] for i in range(min(12,len(rows)))]
        overview=Image.new('RGB',(1284,58+math_ceil(len(selected),3)*255),'#edf2f5');d=ImageDraw.Draw(overview)
        d.text((12,14),'Full-video camera masks | 1 fps sampled preview',font=font(23),fill='#183244')
        for i,row in enumerate(selected):
            x=12+(i%3)*420;y=58+(i//3)*255
            thumb=ImageOps.contain(Image.open(out/'overlays'/f"{row['index']:06d}.jpg"),(408,226));overview.paste(thumb,(x,y))
            d.text((x,y+230),f"#{row['index']:03d} / {row['timestamp_s']:.3f}s",font=font(19),fill='#183244')
        overview.save(out/'overview.jpg',quality=94)
        meta.update(status='complete',processed_frames=len(stats),inference_and_frame_output_s=round(inference_s,3),wall_time_s=round(time.monotonic()-started,3),
                    empty_masks=sum(s['mask_area_pixels']==0 for s in stats),flagged_samples=[s['sample_index'] for s in stats if s['heuristic_flags']],
                    preview_playback='75 sampled frames held at 1 sample/second; encoded at 30fps by repetition; not dense masks',
                    contact_sheets=sheet_files,review_status='not_yet_visually_reviewed')
        save_json(out/'metadata.json',meta)
        print(json.dumps(meta,indent=2),flush=True)
    except Exception as exc:
        meta.update(status='failed',error=str(exc),wall_time_s=round(time.monotonic()-started,3));save_json(out/'metadata.json',meta);raise

def math_ceil(a,b):return (a+b-1)//b
if __name__=='__main__':main()
