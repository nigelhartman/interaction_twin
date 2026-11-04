#!/usr/bin/env python3
"""Private source-declaration check; no flexible fitting, inference or publication."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import rigidity_feasibility as R
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--scene',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
print(json.dumps(R.build(args.scene,args.output),indent=2))
