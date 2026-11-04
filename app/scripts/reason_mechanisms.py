#!/usr/bin/env python3
"""Prepare offline, or explicitly run one bounded Muse mechanism request."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mechanism_reasoning as mechanism


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare', help='Offline hash-bound prompt/schema/evidence preparation; no credentials or API')
    prepare.add_argument('--interaction', required=True)
    prepare.add_argument('--output', required=True)
    prepare.add_argument('--max-images', type=int, default=16)
    prepare.add_argument('--max-output-tokens', type=int, default=10000)
    run = sub.add_parser('run', help='One paid request; requires explicit intent and existing task budget baseline')
    run.add_argument('--prepared', required=True)
    run.add_argument('--task-budget-plan', required=True)
    run.add_argument('--allow-paid-api', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        value = mechanism.prepare(args.interaction, args.output, max_images=args.max_images, max_output_tokens=args.max_output_tokens)
        print(json.dumps({k: value[k] for k in ['request_key', 'maximum_estimated_reservation_usd', 'paid_requests', 'automatic_retries', 'schema_diagnostics']}, indent=2))
    else:
        value = mechanism.run_prepared(args.prepared, allow_paid_api=args.allow_paid_api, task_budget_plan=args.task_budget_plan)
        print(json.dumps({'format': value['format'], 'joints': len(value['joints']), 'source_sha256': value['source_sha256'],
                          'physical_accuracy_verified': False, 'estimated_cost_usd': value['reasoning']['estimated_cost_usd']}, indent=2))


if __name__ == '__main__':
    main()
