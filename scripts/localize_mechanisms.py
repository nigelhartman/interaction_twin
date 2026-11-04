#!/usr/bin/env python3
"""Offline prepare or explicit guarded one-shot Muse attachment-seam refinement."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mechanism_localization as M
p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
a=sub.add_parser('prepare');a.add_argument('--proposal',required=True);a.add_argument('--interaction',required=True);a.add_argument('--output',required=True)
a=sub.add_parser('run');a.add_argument('--prepared',required=True);a.add_argument('--task-budget-plan',required=True);a.add_argument('--allow-paid-api',action='store_true')
a=p.parse_args()
if a.command=='prepare':
 r=M.prepare(a.proposal,a.interaction,a.output)
 print(json.dumps({'request_key':r['request_key'],'observations':len(r['context']['observations']),'images':len(r['context']['images']),
                   'maximum_estimated_reservation_usd':r['maximum_estimated_reservation_usd'],'automatic_retries':False},indent=2))
else:
 r=M.run_prepared(a.prepared,a.task_budget_plan,allow_paid_api=a.allow_paid_api)
 print(json.dumps({'joint_hints':len(r['joint_hints']),'visible_hints':sum(len(j['hinge_hints']) for j in r['joint_hints']),
                   'estimated_cost_usd':r['reasoning']['estimated_cost_usd'],'physical_accuracy_verified':False},indent=2))
