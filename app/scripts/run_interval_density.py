#!/usr/bin/env python3
"""Paired bounded native-PTS density experiment; orchestration owns worker flock."""
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from interval_motion_density import run
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);p.add_argument('--audit',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();value=run(ROOT/a.plan,ROOT/a.audit,ROOT/a.out,root=ROOT)
    print(json.dumps({'status':value['status'],'actual_cotracker_model_calls':value['actual_cotracker_model_calls'],'wall_seconds':value['wall_seconds'],'comparison':value['comparison']}))
