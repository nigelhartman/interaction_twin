#!/usr/bin/env python3
"""Prepare a source-only packet, or explicitly send one root-reviewed revision."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import laptop_hinge_relocalization_v2 as R

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('action', choices=['prepare', 'verify', 'run'])
parser.add_argument('--authorized-plan-sha256')
parser.add_argument('--allow-paid-api', action='store_true')
args = parser.parse_args()
if args.action == 'prepare':
    value = R.prepare()
elif args.action == 'verify':
    doc, _, folder = R.verified_plan()
    value = {'plan': R.R.ref(folder / 'plan.json', ROOT), 'images': len(doc['context']['images']), 'no_requests': True}
else:
    value = R.run(authorized_plan_sha256=args.authorized_plan_sha256, allow_paid_api=args.allow_paid_api)
print(json.dumps(value), flush=True)
