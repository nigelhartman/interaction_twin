#!/usr/bin/env python3
"""Prepare/adopt the exact precomputed laptop revision; never infer or approve."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import laptop_cached_revision_adoption_20261103 as C
from admin_completed_import_v1 import ref


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'adopt'])
    parser.add_argument('--plan', required=True)
    parser.add_argument('--receipt')
    parser.add_argument('--local-structural-review-only', action='store_true')
    args = parser.parse_args()
    def output(name):
        path = ROOT / name
        C.require(path.resolve() == path and path.is_relative_to(ROOT/'output') and not any(p.is_symlink() for p in [path,*path.parents]), 'Ordinary output path required')
        return path
    plan_path = output(args.plan)
    if args.action == 'prepare':
        plan = C.build()
        C.require(not plan_path.exists(), 'Preserve existing adoption plans')
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(plan, indent=2)+'\n')
        print(json.dumps({'status': 'prepared_read_only', 'plan_ref': ref(plan_path), 'artifact_count': len(plan['result']['artifacts']), 'blocking_failures': plan['structural_preflight']['blocking_failures']}))
    else:
        C.require(args.local_structural_review_only and args.receipt, 'Explicit local-only adoption and fresh receipt required')
        receipt = output(args.receipt)
        C.require(not receipt.exists(), 'Preserve existing adoption receipts')
        plan = json.loads(plan_path.read_text())
        result = C.adopt(plan)
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps({'plan_ref': ref(plan_path), **result}, indent=2)+'\n')
        print(json.dumps({k: result[k] for k in ['status','candidate_id','base_revision_id','revision_id','fingerprint','candidate_status','new_api_calls','new_gpu_calls','automatic_approval','automatic_publication']}, indent=2))


if __name__ == '__main__':
    main()
