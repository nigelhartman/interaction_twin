#!/usr/bin/env python3
"""Private bounded CPU pose experiment. Never runs inference or publishes assets."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import sam3d_common_scene as S
from native_pose_refinement import propose
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--scene',required=True);parser.add_argument('--audit',required=True);parser.add_argument('--output',required=True)
parser.add_argument('--parent',required=True);parser.add_argument('--training-sample',type=int,action='append',required=True);parser.add_argument('--development-sample',type=int,action='append',required=True)
parser.add_argument('--max-dimension',type=int,default=192)
args=parser.parse_args();root=Path(__file__).resolve().parents[1]
print(json.dumps(propose(S.ref(root/args.scene,root),S.ref(root/args.audit,root),args.output,parent_id=args.parent,
    training_samples=args.training_sample,development_samples=args.development_sample,max_dimension=args.max_dimension),indent=2))
