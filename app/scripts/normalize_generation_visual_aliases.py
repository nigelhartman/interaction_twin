#!/usr/bin/env python3
"""Normalize only supplied response aliases offline, preserving raw evidence."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import generation_visual_alias_v1 as A

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet');parser.add_argument('--output');parser.add_argument('--verify')
    args=parser.parse_args()
    if args.verify:result=A.verify(args.verify)
    else:
        if not args.packet or not args.output:parser.error('--packet and --output required')
        result=A.build(args.packet,args.output)
    print(json.dumps(result,indent=2,allow_nan=False))
