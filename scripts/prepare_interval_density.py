#!/usr/bin/env python3
"""Separate12fps native-PTS interval preparation; orchestration owns model flock."""
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from interval_motion_dense_input import extract,extract_family,propagate
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['extract','extract-family','propagate']);p.add_argument('--plan');p.add_argument('--prior');p.add_argument('--out',required=True)
    a=p.parse_args()
    if a.action in ['extract','extract-family']:
        if not a.plan or not a.prior:p.error('extract requires exact frozen plan and priorresult')
        v=(extract if a.action=='extract' else extract_family)(ROOT/a.plan,ROOT/a.prior,ROOT/a.out,root=ROOT)
    else:v=propagate(ROOT/a.out,root=ROOT)
    print(json.dumps({'status':v['status'],'frame_count':v.get('frame_count'),'track_id':v.get('track_id'),'input_count':len(v.get('inputs',[]))}))
