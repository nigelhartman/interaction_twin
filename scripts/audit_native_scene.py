#!/usr/bin/env python3
"""Run explicit local CPU diagnostics; never inference, repair or publication."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from native_scene_audit import build_audit
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--scene',required=True);parser.add_argument('--output',required=True)
parser.add_argument('--max-frames',type=int,default=12);parser.add_argument('--max-dimension',type=int,default=384)
args=parser.parse_args();print(json.dumps(build_audit(args.scene,args.output,max_frames=args.max_frames,max_dimension=args.max_dimension),indent=2))
