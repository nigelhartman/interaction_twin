#!/usr/bin/env python3
"""Track reviewed component candidates; preserve raw masks and identity ambiguity."""
import argparse, hashlib, json, re, time, tempfile, os, sys, shutil
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from sam2.build_sam import build_sam2_video_predictor
ROOT=Path(__file__).resolve().parents[1]
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def load(p):return json.loads(p.read_text())
def validate_config(cfg,rows):
    ids=set();n=len(rows)
    for track in cfg['tracks']:
        tid=track['id']
        if not re.fullmatch(r'[a-z][a-z0-9_]*',tid) or tid in ids:raise ValueError('Invalid or duplicate track ID')
        ids.add(tid);a,b=track['sample_range']
        if not 1<=a<=b<=n:raise ValueError('Invalid track range')
        if not track['prompts']:raise ValueError('Track needs a prompt')
        for p in track['prompts']:
            k=p['sample_index']
            if not a<=k<=b:raise ValueError('Prompt outside track range')
            w,h=rows[k-1]['width'],rows[k-1]['height']
            if 'box_xyxy' in p:
                x0,y0,x1,y1=p['box_xyxy']
                if not 0<=x0<x1<=w or not 0<=y0<y1<=h:raise ValueError('Invalid prompt box')
            if len(p.get('points',[]))!=len(p.get('labels',[])):raise ValueError('Point/label mismatch')
            for x,y in p.get('points',[]):
                if not 0<=x<w or not 0<=y<h:raise ValueError('Point outside image')
    return True

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--run',default='components-v1');p.add_argument('--device',choices=['mps','cpu'],default='mps');p.add_argument('--reuse-from');a=p.parse_args()
    if any(not re.fullmatch(r'[a-z0-9][a-z0-9-]*',s) for s in [a.case,a.run]):p.error('Invalid case/run slug')
    if a.device=='mps' and not torch.backends.mps.is_available():p.error('MPS unavailable; no silent CPU fallback')
    case=ROOT/'data'/a.case;rows=load(case/'frames/index.json')['frames'];cfg=load(a.config);validate_config(cfg,rows)
    out=case/'segmentation'/a.run
    if out.exists():p.error('Run exists; use a fresh --run')
    out.mkdir(parents=True);write(out/'annotations.json',cfg)
    ckpt=ROOT/'models/sam2.1_hiera_tiny.pt';start=time.monotonic()
    meta={'status':'running','model':'sam2.1_hiera_tiny','device':a.device,'torch':torch.__version__,'source_sha256':load(case/'manifest.json')['source']['sha256'],'checkpoint_sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest(),'human_confirmed':False,'seed_origin':cfg.get('seed_origin','assistant_visual_annotation'),'cpu_storage_offload':True,'cpu_operator_fallback_enabled':False,'scope':'stage_2_component_candidate_tracking','accuracy_measured':False}
    write(out/'metadata.json',meta)
    predictor=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_t.yaml',str(ckpt),device=a.device,apply_postprocessing=False)
    all_records=[]
    previous=case/'segmentation'/a.reuse_from if a.reuse_from else None
    old_specs={t['id']:t for t in load(previous/'annotations.json')['tracks']} if previous else {}
    if previous:
        old_meta=load(previous/'metadata.json')
        if old_meta['status']!='complete' or old_meta['source_sha256']!=meta['source_sha256'] or old_meta['checkpoint_sha256']!=meta['checkpoint_sha256']:
            raise ValueError('Cannot reuse masks from an incomplete run or a different source/checkpoint')
    try:
        for track in cfg['tracks']:
            tid=track['id'];lo,hi=track['sample_range'];dest=out/'raw_masks'/tid
            if previous and old_specs.get(tid)==track:
                dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copytree(previous/'raw_masks'/tid,dest,copy_function=os.link)
                rec=load(previous/'tracks'/f'{tid}.json');rec['reused_from']=a.reuse_from
                write(out/'tracks'/f'{tid}.json',rec);all_records.append(rec);print('Reused unchanged track '+tid,flush=True);continue
            dest.mkdir(parents=True)
            print(f'Tracking {tid}: samples {lo}-{hi}',flush=True);t0=time.monotonic()
            with tempfile.TemporaryDirectory(prefix='.sam-frames-',dir=out) as temp:
                frame_dir=Path(temp)
                for j,row in enumerate(rows[lo-1:hi]):os.symlink((case/row['file']).resolve(),frame_dir/f'{j:06d}.jpg')
                with torch.inference_mode():
                    state=predictor.init_state(str(frame_dir),offload_video_to_cpu=True,offload_state_to_cpu=True)
                    for prompt in track['prompts']:
                        kw={'frame_idx':prompt['sample_index']-lo,'obj_id':1}
                        if 'box_xyxy' in prompt:kw['box']=np.array(prompt['box_xyxy'],dtype=np.float32)
                        if prompt.get('points'):kw.update(points=np.array(prompt['points'],dtype=np.float32),labels=np.array(prompt['labels'],dtype=np.int32))
                        predictor.add_new_points_or_box(state,**kw)
                    first=min(q['sample_index'] for q in track['prompts'])-lo
                    done={}
                    def save(local,obj_ids,logits):
                        row=rows[lo-1+local];mask=(logits[0,0]>0).cpu().numpy();name=f"{row['index']:06d}.png";Image.fromarray(mask.astype(np.uint8)*255).save(dest/name)
                        ys,xs=np.where(mask);area=int(mask.sum());bbox=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)] if area else None
                        withheld=any(x<=row['index']<=y for x,y in track.get('withhold_ranges',[]))
                        done[row['index']]={'sample_index':row['index'],'timestamp_s':row['timestamp_s'],'raw_mask_file':str((dest/name).relative_to(out)),'raw_area_pixels':area,'raw_bbox_xyxy':bbox,'status':'withheld_due_to_occlusion_or_ambiguity' if withheld else ('candidate_visible' if area else 'no_mask'),'identity_status':track['identity_status']}
                    for f,ids,logits in predictor.propagate_in_video(state,start_frame_idx=first):save(f,ids,logits)
                    if first:
                        for f,ids,logits in predictor.propagate_in_video(state,start_frame_idx=first,reverse=True):save(f,ids,logits)
                    if a.device=='mps':torch.mps.synchronize()
                    del state
                if len(done)!=hi-lo+1:raise RuntimeError('Incomplete track result '+tid)
                rec={'track_id':tid,'candidate_entity_ids':track['candidate_entity_ids'],'sample_range':[lo,hi],'wall_time_s':round(time.monotonic()-t0,3),'frames':sorted(done.values(),key=lambda d:d['sample_index'])}
                write(out/'tracks'/f'{tid}.json',rec);all_records.append(rec)
            print(f'Completed {tid}, {time.monotonic()-t0:.1f}s; total {time.monotonic()-start:.1f}s',flush=True)
        write(out/'tracks.json',{'tracks':all_records,'outside_track_ranges':'not_assessed_not_evidence_of_absence'})
        meta.update(status='complete',wall_time_s=round(time.monotonic()-start,3),track_count=len(all_records),mask_count=sum(len(r['frames']) for r in all_records),review_status='pending')
        write(out/'metadata.json',meta);print(json.dumps(meta,indent=2))
    except Exception as exc:
        meta.update(status='failed',error=str(exc),wall_time_s=round(time.monotonic()-start,3));write(out/'metadata.json',meta);raise
if __name__=='__main__':main()
