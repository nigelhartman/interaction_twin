#!/usr/bin/env python3
"""Explicit new text-review revision of unchanged saved candidate artifacts."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',required=True);p.add_argument('--revision',required=True)
    p.add_argument('--allow-paid-text-reviews',action='store_true',required=True)
    p.add_argument('--upgrade-accounted-output-cap',action='store_true')
    p.add_argument('--receipt',required=True);a=p.parse_args()
    out=ROOT/a.receipt
    if out.resolve()!=out or not out.is_relative_to(ROOT/'output') or out.exists():raise ValueError('New ordinary output receipt required')
    from admin_workflow import AdminWorkflow
    from admin_pipeline import QueueRunner
    from meta_budget import read_budget_status
    w=AdminWorkflow(ROOT)
    if any(j['state']in {'queued','running'} for j in w.snapshot()['jobs']):raise ValueError('Another owner job is queued or running')
    before=read_budget_status(ROOT/'.secrets/meta-budget.sqlite3')
    w.request_review(a.candidate,a.revision,processing_consent=True,allow_protocol_upgrade=a.upgrade_accounted_output_cap)
    runner=QueueRunner(ROOT);runner.execute();d=w.detail(a.candidate)
    after=read_budget_status(ROOT/'.secrets/meta-budget.sqlite3')
    current=d['revisions'][0]
    report={'format':'interaction-twin-explicit-review-revision-delivery','version':1,
        'candidate_id':a.candidate,'revision_id':current['id'],'candidate_status':d['candidate']['status'],
        'review':current['manifest']['review'],'new_gpu_calls':0,'new_geometry_generated':False,
        'human_approval_performed':False,'before_estimate':before,'after_estimate':after}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k]for k in ['candidate_status','revision_id','new_gpu_calls','human_approval_performed']}))
    print(json.dumps({k:report['review'][k]for k in ['hosted_roles_completed','hosted_roles_expected','hosted_review_status','blocking_failures']}))


if __name__=='__main__':main()
