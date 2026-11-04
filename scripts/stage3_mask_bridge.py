#!/usr/bin/env python3
"""Expose single-part SAM candidates through the shared mask contract and review sheets."""
import argparse,json,math,re
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def font(n):
    try:return ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',n)
    except OSError:return ImageFont.load_default(size=n)
def main():
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--run',required=True);a=p.parse_args()
    if any(not re.fullmatch(r'[a-z0-9][a-z0-9-]*',s) for s in [a.case,a.run]):p.error('Invalid case/run')
    case=ROOT/'data'/a.case;out=case/'segmentation'/a.run;meta=read(out/'metadata.json');tracks=read(out/'tracks.json')['tracks'];rows=read(case/'frames/index.json')['frames']
    if meta['status']!='complete' or len(tracks)!=1:raise ValueError('Requires one complete component track')
    if (out/'frames.json').exists():raise ValueError('Mask bridge already exists')
    track=tracks[0];lookup={r['sample_index']:r for r in track['frames']};frames=[]
    for row in rows:
        rec=lookup[row['index']];withheld=rec['status']!='candidate_visible'
        frames.append({'sample_index':row['index'],'timestamp_s':row['timestamp_s'],'components':[] if withheld else [{'track_id':track['track_id'],'visible_mask_file':rec['raw_mask_file'],'area_pixels':rec['raw_area_pixels'],'bbox_xyxy':rec['raw_bbox_xyxy']}], 'withheld':{track['track_id']:[rec['status']]} if withheld else {},'ambiguous_pixels':0})
    (out/'frames.json').write_text(json.dumps({'composition':'single raw SAM candidate; no peer/parent composition needed','frames':frames},indent=2)+'\n')
    for begin in range(0,len(rows),24):
        group=rows[begin:begin+24];sheet=Image.new('RGB',(1304,58+math.ceil(len(group)/4)*212),'#edf2f5');d=ImageDraw.Draw(sheet);d.text((12,15),'Stage 3 / denser SAM battery-mask candidates',font=font(22),fill='#173244')
        for j,row in enumerate(group):
            rec=lookup[row['index']];mask=np.asarray(Image.open(out/rec['raw_mask_file']))>0;base=np.array(Image.open(case/row['file']).convert('RGB'));base[mask]=(base[mask]*.6+np.array([255,174,67])*.4).astype(np.uint8)
            thumb=ImageOps.contain(Image.fromarray(base),(308,174));x=12+j%4*320;y=58+j//4*212;sheet.paste(thumb,(x,y));d.text((x,y+180),f"#{row['index']:03d} / {row['timestamp_s']:.3f}s",font=font(19),fill='#173244')
        sheet.save(out/f'mask-review-{begin//24+1:02d}.jpg',quality=95)
    print(out/'frames.json')
if __name__=='__main__':main()
