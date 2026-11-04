#!/usr/bin/env python3
"""Plan or execute a local-only component upgrade using saved reasoning."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from upgrade_runner import UpgradeRunner, plan

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['plan','run'])
    parser.add_argument('--twin-id',required=True); parser.add_argument('--track-id',required=True)
    parser.add_argument('--base-sha256',required=True)
    parser.add_argument('--mode',choices=['refit_existing','refit_native_cpu','local_triposr','imported_sam3d'],default='refit_existing')
    parser.add_argument('--imported-result-id'); parser.add_argument('--resume-run-id')
    args=parser.parse_args(); request={'twin_id':args.twin_id,'track_id':args.track_id,'base_sha256':args.base_sha256,'mode':args.mode}
    if args.imported_result_id: request['imported_result_id']=args.imported_result_id
    if args.action=='plan': result=plan(request)
    else: result=UpgradeRunner(request,run_id=args.resume_run_id).execute(request)
    print(json.dumps(result,indent=2))
    if result.get('status') in {'failed','cancelled'} or result.get('can_run') is False: raise SystemExit(1)

if __name__=='__main__': main()
