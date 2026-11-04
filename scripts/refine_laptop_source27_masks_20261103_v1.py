#!/usr/bin/env python3
"""Prepare or explicitly dispatch one two-panel SAM source27 image request."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import source_mask_refinement_laptop27_v1 as M


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'verify', 'run'])
    parser.add_argument('--output', default=M.OUTPUT)
    parser.add_argument('--authorized-plan-sha256')
    parser.add_argument('--allow-paid-api', action='store_true')
    args = parser.parse_args()
    if args.action == 'prepare':
        value = M.prepare(output=args.output)
    elif args.action == 'verify':
        doc, folder = M.verified_plan(output=args.output)
        value = {'plan': M.R.ref(ROOT, folder / 'plan.json'), 'maximum_new_requests': 1,
                 'maximum_estimated_reservation_usd': doc['maximum_estimated_reservation_usd'],
                 'preflight': doc['adapter_preflight']}
    else:
        value = M.run(output=args.output, authorized_plan_sha256=args.authorized_plan_sha256, allow_paid_api=args.allow_paid_api)
    print(json.dumps(value), flush=True)


if __name__ == '__main__':
    main()
