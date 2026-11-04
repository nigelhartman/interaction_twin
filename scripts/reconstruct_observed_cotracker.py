#!/usr/bin/env python3
"""Offline CoTracker TRAIN-only source tracks followed by mathematical SfM."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import observed_cotracker

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--program',required=True)
    parser.add_argument('--track',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--max-views',type=int,default=49)
    parser.add_argument('--max-ba-evaluations',type=int,default=70)
    parser.add_argument('--device',choices=['mps'],default='mps')
    parser.add_argument('--no-sensitivity',action='store_true')
    args=parser.parse_args()
    try:
        observed_cotracker.run(ROOT/args.program,args.track,ROOT/args.output,root=ROOT,max_views=args.max_views,
                              max_nfev=args.max_ba_evaluations,sensitivity=not args.no_sensitivity,device=args.device)
    except Exception as error:
        output=(ROOT/args.output).resolve()
        if output.is_relative_to(ROOT/'data') and output.is_dir() and (output/'protocol.json').is_file():
            (output/'failure.json').write_text(json.dumps({'status':'failed','error_type':type(error).__name__,
                                                          'message':str(error),'new_api_calls':0})+'\n')
        raise
