#!/usr/bin/env python3
"""Prepare offline, or run an exactly authorized private five-frame hand packet."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import hand_occlusion_evidence as H
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['prepare','run'])
parser.add_argument('--audit');parser.add_argument('--output');parser.add_argument('--prepared');parser.add_argument('--training-sample',type=int,action='append');parser.add_argument('--development-sample',type=int,action='append')
parser.add_argument('--allow-paid-api',action='store_true');parser.add_argument('--authorized-plan-sha256');args=parser.parse_args()
if args.mode=='prepare':
    if not args.audit or not args.output:parser.error('prepare requires --audit and --output')
    result=H.prepare(args.audit,args.output,training_samples=args.training_sample or[],development_samples=args.development_sample or[])
else:
    if not args.prepared:parser.error('run requires --prepared')
    result=H.run_prepared(args.prepared,authorized_plan_sha256=args.authorized_plan_sha256,allow_paid_api=args.allow_paid_api)
print(json.dumps(result,indent=2))
