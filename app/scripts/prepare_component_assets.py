#!/usr/bin/env python3
"""Prepare/verify an offline component queue; never execute models or commands."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import component_asset_queue as Q
import upgrade_bundles as U


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='mode',required=True)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--twin',required=True,help='Existing program ID from workspace/cases.json')
    prepare.add_argument('--output',required=True,help='Fresh ordinary data/ directory')
    prepare.add_argument('--candidate-imports',help='Optional data/ JSON mapping track IDs to validated import references')
    prepare.add_argument('--prepare-jobs',action='store_true',help='Also prepare existing GPU handoff ZIPs locally; submits nothing')
    inspect = commands.add_parser('inspect')
    inspect.add_argument('--queue',required=True)
    args = parser.parse_args()
    if args.mode == 'prepare':
        references = U.read(U._local(args.candidate_imports,ROOT)) if args.candidate_imports else None
        result = Q.prepare(args.twin,args.output,candidate_imports=references,prepare_jobs=args.prepare_jobs)
    else:
        result = Q.inspect(args.queue)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
