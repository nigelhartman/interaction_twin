#!/usr/bin/env python3
"""Prepare offline or explicitly run one exact approved source/render visual review."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import native_visual_review as V


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','run'])
    parser.add_argument('--audit');parser.add_argument('--output');parser.add_argument('--prepared')
    parser.add_argument('--role',choices=V.ROLES,default='critic');parser.add_argument('--round',type=int,default=1)
    parser.add_argument('--previous-report');parser.add_argument('--allow-paid-api',action='store_true')
    parser.add_argument('--authorized-plan-sha256')
    args=parser.parse_args()
    if args.mode=='prepare':
        if not args.audit or not args.output:parser.error('prepare requires --audit and --output')
        result=V.prepare(args.audit,args.output,role=args.role,round_number=args.round,previous_report=args.previous_report)
    else:
        if not args.prepared:parser.error('run requires --prepared')
        result=V.run_prepared(args.prepared,allow_paid_api=args.allow_paid_api,authorized_plan_sha256=args.authorized_plan_sha256)
    print(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__':main()
