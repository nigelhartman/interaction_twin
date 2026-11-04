#!/usr/bin/env python3
"""Offline visual-quality packet preparation; exact approval gates paid execution."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import generation_visual_judge as J

def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
 prep=sub.add_parser('prepare');prep.add_argument('--program',required=True);prep.add_argument('--part-ledger',required=True);prep.add_argument('--component',required=True);prep.add_argument('--output',required=True);prep.add_argument('--agent-qa')
 check=sub.add_parser('verify');check.add_argument('--output',required=True)
 run=sub.add_parser('run');run.add_argument('--output',required=True);run.add_argument('--allow-paid-api',action='store_true',required=True);run.add_argument('--authorized-plan-sha256',required=True)
 a=p.parse_args()
 if a.command=='prepare':result=J.prepare(a.program,a.part_ledger,a.component,a.output,agent_qa_path=a.agent_qa)
 elif a.command=='verify':plan,_,_=J.verify_plan(a.output);result={'status':'verified_offline_no_upload','request_key':plan['request_key'],'new_api_calls':0}
 else:result=J.run_prepared(a.output,allow_paid_api=a.allow_paid_api,authorized_plan_sha256=a.authorized_plan_sha256)
 print(json.dumps(result,indent=2,allow_nan=False))
if __name__=='__main__':main()
