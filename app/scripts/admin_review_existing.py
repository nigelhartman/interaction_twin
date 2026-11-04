#!/usr/bin/env python3
"""One explicit text-only four-role review of an exact curated example.

No new frames, model generation, public approval or GPU action. Existing native
geometry stays unchanged. The guarded reviewer retains every uncertain attempt.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interaction',required=True)
    parser.add_argument('--allow-paid-text-reviews',action='store_true',required=True)
    parser.add_argument('--receipt',required=True)
    args=parser.parse_args()
    out=ROOT/args.receipt
    if not out.is_relative_to(ROOT/'output') or out.resolve()!=out or any(p.is_symlink() for p in [out,*out.parents]):
        raise ValueError('Receipt must stay in ordinary output')
    if out.exists():raise ValueError('Preserve the existing delivery receipt')
    from admin_http import existing_example
    from admin_workflow import AdminWorkflow
    from admin_pipeline import FixedExecutor
    from admin_muse_reviews import build_reviewer
    from meta_budget import read_budget_status
    w=AdminWorkflow(ROOT);selected=existing_example(w,args.interaction)
    cid=selected['candidate']['id']
    before=read_budget_status(ROOT/'.secrets/meta-budget.sqlite3')
    if selected['candidate']['status']=='pending_consent':w.authorize_processing(cid)
    # The single local worker claim includes only this explicit queued import;
    # refuse to process unrelated queued uploads in a diagnostic CLI.
    snapshot=w.snapshot();queued=[j for j in snapshot['jobs'] if j['state']=='queued']
    if len(queued)!=1 or queued[0]['candidate_id']!=cid:
        raise ValueError('Expected exactly this queued existing-example review')
    result=w.worker_once(FixedExecutor(),hosted_reviewer_factory=lambda job:build_reviewer(
        job['memory_context'],root=ROOT,allow_paid_api=True))
    after=read_budget_status(ROOT/'.secrets/meta-budget.sqlite3')
    detail=w.detail(cid)
    receipt={'format':'interaction-twin-owner-review-delivery','version':1,'candidate_id':cid,
        'status':result['status'],'candidate_status':detail['candidate']['status'],
        'revision_id':detail['candidate']['current_revision_id'],'text_only':True,
        'new_frame_uploads':0,'new_gpu_calls':0,'public_approval_performed':False,
        'before_estimate':before,'after_estimate':after,
        'roles':detail['revisions'][0]['manifest']['review'] if detail['revisions'] else None}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ['status','candidate_status','candidate_id','revision_id','text_only','new_gpu_calls','public_approval_performed']}))


if __name__=='__main__':main()
