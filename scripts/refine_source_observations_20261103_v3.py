#!/usr/bin/env python3
"""Explicit new review with room for reasoning after an accounted output cap."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import source_observation_refinement_v3 as R


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'verify', 'run'])
    parser.add_argument('--output', default=R.DEFAULT_OUTPUT)
    parser.add_argument('--authorized-plan-sha256')
    parser.add_argument('--allow-paid-api', action='store_true')
    args = parser.parse_args()
    if args.action == 'prepare':
        value = R.prepare(output=args.output)
    elif args.action == 'verify':
        doc, _, folder = R.verified_plan(output=args.output)
        value = {'plan': R.ref(ROOT, folder / 'plan.json'), 'images': len(doc['context']['images']), 'new_requests': 0,
                 'maximum_estimated_reservation_usd': doc['maximum_estimated_reservation_usd'],
                 'explicit_accounted_followup': doc['explicit_accounted_followup']}
    else:
        value = R.run(output=args.output, authorized_plan_sha256=args.authorized_plan_sha256, allow_paid_api=args.allow_paid_api)
    print(json.dumps(value), flush=True)


if __name__ == '__main__':
    main()
