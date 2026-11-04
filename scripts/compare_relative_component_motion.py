#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from relative_component_motion import run
if __name__=='__main__':
    p=argparse.ArgumentParser(description='Conditional simultaneous relative rotation; no invented scale or translation')
    p.add_argument('--first',required=True);p.add_argument('--second',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();report=run(ROOT/a.first,ROOT/a.second,ROOT/a.output,root=ROOT)
    print(json.dumps({'status':report['status'],'simultaneous_pose_pairs':report['simultaneous_pose_pairs']}))
