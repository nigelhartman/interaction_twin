#!/usr/bin/env python3
"""Render raw, visible-part-resolved candidate masks with explicit unknown regions."""
import argparse,json,math,subprocess,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from perception import resolve_masks

def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def font(n):
    try:return ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',n)
    except OSError:return ImageFont.load_default(size=n)
def rgb(s):return np.array([int(s[k:k+2],16) for k in [1,3,5]])
def main():
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--run',default='components-v1');a=p.parse_args()
    case=ROOT/'data'/a.case;out=case/'segmentation'/a.run;rows=read(case/'frames/index.json')['frames'];cfg=read(out/'annotations.json');meta=read(out/'metadata.json')
    if meta['status']!='complete':raise ValueError('Tracking is not complete')
    specs={x['id']:x for x in cfg['tracks']};tracks=read(out/'tracks.json')['tracks'];lookup={t['track_id']:{r['sample_index']:r for r in t['frames']} for t in tracks}
    for d in ['overlays','comparison','resolved_masks']:(out/d).mkdir(exist_ok=True)
    stats=[];w,h=rows[0]['width'],rows[0]['height']
    for row in rows:
        i=row['index'];raw={};withheld=[];reasons={};raw_stats={}
        for tid,spec in specs.items():
            tr=lookup[tid].get(i)
            if not tr:continue
            mask=np.asarray(Image.open(out/tr['raw_mask_file']))>0;raw[tid]=mask
            raw_stats[tid]={'area':int(mask.sum()),'bbox':tr['raw_bbox_xyxy']}
            reason=[]
            if tr['status'].startswith('withheld'):reason.append('annotated_occlusion_or_ambiguity')
            max_area=50000 if tid=='cover' else (65000 if tid!='camera' else w*h)
            if mask.sum()>max_area:reason.append('oversized_component_mask')
            if not mask.any():reason.append('empty_prediction')
            if reason:withheld.append(tid);reasons[tid]=reason
        resolved,unknown=resolve_masks(raw,specs,withheld)
        base=np.array(Image.open(case/row['file']).convert('RGB'));pixels=base.astype(float)
        records=[]
        for tid,mask in resolved.items():
            dest=out/'resolved_masks'/tid;dest.mkdir(exist_ok=True)
            Image.fromarray(mask.astype(np.uint8)*255).save(dest/f'{i:06d}.png')
            pixels[mask]=pixels[mask]*.55+rgb(specs[tid]['color'])*.45
            ys,xs=np.where(mask)
            records.append({'track_id':tid,'visible_mask_file':f'resolved_masks/{tid}/{i:06d}.png','area_pixels':int(mask.sum()),'bbox_xyxy':[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)] if len(xs) else None,'identity_status':specs[tid]['identity_status']})
        if unknown is not None:pixels[unknown]=pixels[unknown]*.6+np.array([150,150,150])*.4
        overlay=Image.fromarray(pixels.astype('uint8'));d=ImageDraw.Draw(overlay)
        for r in records:
            if r['bbox_xyxy'] and r['area_pixels']>80:
                x0,y0,x1,y1=r['bbox_xyxy'];label=specs[r['track_id']]['label'];y=max(0,y0-24)
                box=d.textbbox((x0,y),label,font=font(18));d.rectangle(box,fill='#173244');d.text((x0,y),label,font=font(18),fill='white')
        overlay.save(out/'overlays'/f'{i:06d}.jpg',quality=93)
        canvas=Image.new('RGB',(w*2,h+124),'#edf2f5');canvas.paste(Image.fromarray(base),(0,124));canvas.paste(overlay,(w,124));d=ImageDraw.Draw(canvas)
        d.text((16,10),f"Source {row['timestamp_s']:.3f}s | sample {i:02d}/{len(rows)}",font=font(25),fill='#173244')
        d.text((w+16,10),'SAM 2.1 Tiny / MPS | candidate part tracks',font=font(25),fill='#173244')
        x=16
        for tid in ['camera','cover','out_left','out_right']:
            s=specs[tid];d.rectangle((x,52,x+17,69),fill=s['color']);d.text((x+24,48),s['label'],font=font(20),fill='#173244');x+=220
        d.text((w+16,49),'? = identity unresolved after occlusion; grey = conflicting masks',font=font(20),fill='#173244')
        d.text((16,84),'Masks cover visible pixels only. Small parts take priority over body.',font=font(19),fill='#526574')
        d.text((w+16,84),'Withheld: '+(', '.join(withheld) if withheld else 'none'),font=font(18),fill='#526574')
        canvas.save(out/'comparison'/f'{i:06d}.jpg',quality=92)
        stats.append({'sample_index':i,'timestamp_s':row['timestamp_s'],'components':records,'withheld':reasons,'raw_predictions':raw_stats,'ambiguous_pixels':int(unknown.sum()) if unknown is not None else 0})
    fps=read(case/'manifest.json')['sampling']['fps']
    video=out/'components-preview.mp4'
    if video.exists():raise ValueError('Preview already exists; preserve previous render')
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-framerate',str(fps),'-i',str(out/'comparison/%06d.jpg'),'-c:v','libx264','-crf','19','-pix_fmt','yuv420p','-r','30','-movflags','+faststart',str(video)],check=True)
    sheet_files=[]
    for begin in range(0,len(rows),24):
        group=rows[begin:begin+24];sheet=Image.new('RGB',(1304,58+math.ceil(len(group)/4)*212),'#edf2f5');d=ImageDraw.Draw(sheet);d.text((12,15),'Stage 2 / part mask candidates / original source timestamps',font=font(22),fill='#173244')
        for j,row in enumerate(group):
            x=12+j%4*320;y=58+j//4*212;thumb=ImageOps.contain(Image.open(out/'overlays'/f"{row['index']:06d}.jpg"),(308,174));sheet.paste(thumb,(x,y));d.text((x,y+180),f"#{row['index']:03d} / {row['timestamp_s']:.3f}s",font=font(19),fill='#173244')
        name=f'contact-sheet-{begin//24+1:02d}.jpg';sheet.save(out/name,quality=94);sheet_files.append(name)
    # Raw track inspections expose failures hidden by compositing/withholding.
    for tid,spec in specs.items():
        tr=lookup[tid];indexes=sorted(tr);selected=sorted(set(indexes[round(j*(len(indexes)-1)/max(1,min(11,len(indexes)-1)))] for j in range(min(12,len(indexes)))))
        sheet=Image.new('RGB',(984,58+math.ceil(len(selected)/3)*212),'#edf2f5');d=ImageDraw.Draw(sheet);d.text((12,15),'RAW '+tid+' / seeds & uncertainty recorded separately',font=font(21),fill='#173244')
        for j,i in enumerate(selected):
            row=rows[i-1];mask=np.asarray(Image.open(out/tr[i]['raw_mask_file']))>0;base=np.array(Image.open(case/row['file']).convert('RGB'));pixels=base.astype(float);pixels[mask]=pixels[mask]*.5+rgb(spec['color'])*.5
            thumb=ImageOps.contain(Image.fromarray(pixels.astype('uint8')),(308,174));x=12+j%3*324;y=58+j//3*212;sheet.paste(thumb,(x,y));d.text((x,y+180),f"#{i:03d} / {row['timestamp_s']:.3f}s",font=font(18),fill='#173244')
        sheet.save(out/f'raw-review-{tid}.jpg',quality=94)
    write(out/'frames.json',{'composition':'visible part candidates override camera; peer overlaps become unknown; raw masks preserved','outside_track_ranges':'not_assessed_not_absent','frames':stats})
    summary={'frames':len(rows),'tracks':{},'composition':'part priority, peer overlap withheld','accuracy_measured':False,'review_status':'pending','contact_sheets':sheet_files}
    for tid,spec in specs.items():
        summary['tracks'][tid]={'sample_range':spec['sample_range'],'raw_predictions':len(lookup[tid]),'candidate_visible_samples':sum(any(c['track_id']==tid and c['area_pixels']>0 for c in f['components']) for f in stats),'withheld_samples':[f['sample_index'] for f in stats if tid in f['withheld']],'identity_status':spec['identity_status']}
    write(out/'quality.json',summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
