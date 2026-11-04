#!/usr/bin/env python3
"""Prepare or explicitly register a finished private run; no model work."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    prepare=commands.add_parser('prepare');prepare.add_argument('--state',required=True);prepare.add_argument('--plan',required=True)
    apply=commands.add_parser('register');apply.add_argument('--plan',required=True);apply.add_argument('--receipt',required=True)
    apply.add_argument('--local-structural-review-only',action='store_true',required=True)
    args=parser.parse_args()
    import admin_completed_import_v1 as C
    def output(relative):
        p=ROOT/relative
        if p.resolve()!=p or not p.is_relative_to(ROOT/'output')or any(q.is_symlink()for q in [p,*p.parents]):raise ValueError('Choose an ordinary fresh output path')
        return p
    plan_path=output(args.plan)
    if args.command=='prepare':
        plan=C.build(C.ref(ROOT/args.state),root=ROOT)
        if plan_path.exists():
            if json.loads(plan_path.read_text())!=plan:raise ValueError('Preserve the existing import plan')
        else:
            plan_path.parent.mkdir(parents=True,exist_ok=True);plan_path.write_text(json.dumps(plan,indent=2)+'\n')
        print(json.dumps({'status':'prepared_read_only','plan_ref':C.ref(plan_path),'candidate_run_id':plan['candidate_run_id'],'new_api_calls':0},indent=2))
    else:
        receipt=output(args.receipt)
        if receipt.exists():raise ValueError('Preserve the existing registration receipt')
        plan=json.loads(plan_path.read_text());result=C.register(plan,root=ROOT)
        receipt.parent.mkdir(parents=True,exist_ok=True);receipt.write_text(json.dumps({'plan_ref':C.ref(plan_path),**result},indent=2)+'\n')
        print(json.dumps(result,indent=2))


if __name__=='__main__':main()
