#!/usr/bin/env python3
"""Bounded short-interval motion; model-worker flock is owned by orchestration."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import interval_motion
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--interaction',required=True);p.add_argument('--track',required=True);p.add_argument('--out',required=True)
    p.add_argument('--audit');p.add_argument('--audit-plan');p.add_argument('--plan-only',action='store_true')
    a=p.parse_args()
    if a.plan_only:
        value=interval_motion.plan(ROOT/a.interaction,a.track,ROOT/a.out,root=ROOT)
        print(json.dumps({'phase':'plan_complete','track_id':a.track,'episodes':len(value['episodes']),'target_masks':len({r['sample_index'] for e in value['episodes'] for r in e['frames']})}))
    else:
        value=interval_motion.run(ROOT/a.interaction,a.track,ROOT/a.out,root=ROOT,audit_path=ROOT/a.audit if a.audit else None,plan_path=ROOT/a.audit_plan if a.audit_plan else None)
        print(json.dumps({k:value[k] for k in ['status','track_id','frame_count','supported_nonreference_frames','withheld_nonreference_frames']}))
