#!/usr/bin/env python3
"""Explicit local checkerboard calibration; no model inference or API calls."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import calibration_core as C
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input','--source',dest='source',required=True);p.add_argument('--out','--output',dest='output',required=True)
    p.add_argument('--inner-rows',type=int,required=True);p.add_argument('--inner-cols',type=int,required=True)
    p.add_argument('--square-size-mm',type=float);p.add_argument('--max-frames',type=int,default=50);p.add_argument('--max-seconds',type=float,default=120)
    a=p.parse_args();r=C.run(ROOT/a.source,ROOT/a.output,root=ROOT,
        board={'inner_rows':a.inner_rows,'inner_cols':a.inner_cols,'square_size_mm':a.square_size_mm},max_frames=a.max_frames,max_seconds=a.max_seconds)
    print(json.dumps({'status':r['status'],'camera_record':r['camera_record'],'elapsed_s':r['elapsed_s']}))
