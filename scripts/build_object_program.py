#!/usr/bin/env python3
"""Export a separate descriptive object program from saved constrained mechanics."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import object_program
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    print(json.dumps(object_program.build(args.scene,args.output),indent=2))
