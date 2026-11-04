#!/usr/bin/env python3
"""Explicitly audit one preserved admin text transport bound; never resend it.

Default behavior only validates. --apply requires a separately reviewed proof and
authorization reference; every reserved micro-dollar remains held as unknown.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request-id',required=True)
    parser.add_argument('--proof',required=True)
    parser.add_argument('--authorization-reference')
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    path=ROOT/args.proof
    if (not path.is_relative_to(ROOT/'output') or path.resolve()!=path or
            any(p.is_symlink() for p in [path,*path.parents]) or not path.is_file() or path.stat().st_size>16384):
        raise ValueError('Audit proof must be a bounded ordinary output JSON file')
    from meta_budget import BudgetLedger,validate_bounded_admin_text_audit
    proof=json.loads(path.read_text())
    ledger_path=ROOT/'.secrets/meta-budget.sqlite3'
    if args.apply:
        if not args.authorization_reference:parser.error('--apply requires --authorization-reference')
        # Validate before opening a writable ledger, including before table setup.
        validate_bounded_admin_text_audit(ledger_path,args.request_id,proof=proof,project_root=ROOT)
        result=BudgetLedger(ledger_path).acknowledge_bounded_admin_text_uncertainty(args.request_id,
            proof=proof,authorization_reference=args.authorization_reference,project_root=ROOT)
    else:
        result=validate_bounded_admin_text_audit(ledger_path,args.request_id,proof=proof,project_root=ROOT)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
