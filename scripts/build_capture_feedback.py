#!/usr/bin/env python3
"""Build immutable, local-only source capture feedback or an index."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import capture_feedback as feedback


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interaction')
    parser.add_argument('--case')
    parser.add_argument('--out')
    parser.add_argument('--base-feedback')
    parser.add_argument('--plan',action='append',default=[])
    parser.add_argument('--scope-audit',action='append',default=[])
    parser.add_argument('--index-out')
    parser.add_argument('--report',action='append',default=[])
    args=parser.parse_args()
    if args.index_out:
        if not args.report: raise ValueError('An index needs explicit feedback reports')
        rows=[]
        for value in args.report:
            path=feedback.checked(ROOT,value);report=feedback.read(path)
            if report.get('format')!='source-capture-feedback': raise ValueError('Unsupported feedback report')
            feedback.published(ROOT,report['interaction_path'],report['case_id'])
            if feedback.sha(feedback.data_path(ROOT,report['interaction_path']))!=report['interaction_sha256']:
                raise ValueError('Feedback publication changed')
            rows.append({k:report[k] for k in ['case_id','interaction_path','interaction_sha256']}|
                {'path':path.relative_to(ROOT).as_posix(),'sha256':feedback.sha(path),
                 'component_count':len(report['components']),'episode_count':len(report['episodes'])})
        output=feedback.data_path(ROOT,args.index_out)
        if output.exists() or not output.name.startswith('capture-feedback-index'): raise ValueError('Use a fresh feedback index')
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open('x') as stream:
            json.dump({'format':'source-capture-feedback-index','version':1,'reports':rows,'new_api_calls':0,
                       'new_model_inference_calls':0,'mps_used':False},stream,indent=2);stream.write('\n')
        print(json.dumps({'path':str(output),'sha256':feedback.sha(output),'reports':len(rows)}))
        return
    if not args.out: raise ValueError('A fresh --out is required')
    if args.base_feedback:
        report=feedback.attach_episodes(ROOT,args.base_feedback,args.plan,scope_audits=args.scope_audit)
    else:
        if args.plan or args.scope_audit: raise ValueError('Attach plans/audits to a frozen --base-feedback in a separate derivative')
        if not args.interaction and not args.case: raise ValueError('Select a registered interaction or case')
        report=feedback.build(ROOT,interaction=args.interaction,case_id=args.case)
    print(json.dumps(feedback.save(ROOT,report,args.out)),flush=True)


if __name__=='__main__':main()
