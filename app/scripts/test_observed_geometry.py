#!/usr/bin/env python3
"""Run one isolated, local-only source-pixel SfM feasibility experiment."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import observed_geometry

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--program',required=True)
    p.add_argument('--track',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--max-views',type=int,default=45)
    p.add_argument('--max-ba-evaluations',type=int,default=70)
    p.add_argument('--no-sensitivity',action='store_true')
    p.add_argument('--camera-mode',choices=['assumed','moge_prior'],default='assumed')
    args=p.parse_args()
    camera_kwargs={}
    if args.camera_mode=='moge_prior':
        if not args.no_sensitivity:
            p.error('Learned camera mode requires --no-sensitivity; compare camera hypotheses separately')
        from automatic_camera_prior import CameraPrior
        camera_kwargs['camera_provider']=CameraPrior(ROOT)
    result=observed_geometry.run(ROOT/args.program,args.track,ROOT/args.output,root=ROOT,
                                 max_views=args.max_views,sensitivity=not args.no_sensitivity,
                                 max_nfev=args.max_ba_evaluations,**camera_kwargs)
    print(json.dumps({'output':args.output,'status':result['status']}))

if __name__=='__main__': main()
