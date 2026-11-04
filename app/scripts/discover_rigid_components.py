#!/usr/bin/env python3
"""Discover separate rigid component proposals with the budgeted Meta API."""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import rigid_discovery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['discover', 'prepare-recovery', 'recover'])
    parser.add_argument('--case', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--model', default='muse-spark-1.3', choices=['muse-spark-1.3'])
    parser.add_argument('--max-output-tokens', type=int, default=16000)
    parser.add_argument('--timeout-s', type=int, default=300)
    parser.add_argument('--increment-envelope', action='store_true')
    parser.add_argument('--initial-run')
    parser.add_argument('--repair-output')
    parser.add_argument('--prepared-repair')
    parser.add_argument('--authorized-plan-sha256')
    args = parser.parse_args()
    if not all(re.fullmatch(r'[a-z0-9][a-z0-9-]{0,95}', x) for x in [args.case, args.run]):
        parser.error('Case and run must be local slugs')
    if args.mode == 'prepare-recovery':
        import discovery_repair
        if not args.repair_output:
            parser.error('Preparation requires --repair-output in the private repair namespace')
        result = discovery_repair.prepare(ROOT / 'data' / args.case, args.run, args.repair_output)
        print(json.dumps(result, indent=2))
        return
    if args.mode == 'recover':
        if not args.prepared_repair or not args.authorized_plan_sha256 or args.increment_envelope:
            parser.error('Recovery requires exact prepared repair and plan SHA; its fixed budget guards are intrinsic')
        result = rigid_discovery.recover(ROOT / 'data' / args.case, args.run,
            prepared=args.prepared_repair, authorized_plan_sha256=args.authorized_plan_sha256,
            timeout=args.timeout_s)
        print(json.dumps({'status': result['status'], 'tracks': len(result['result']['tracks']),
            'tracking_config': result['tracking_config'], 'estimated_cost_usd': result['estimated_cost_usd'],
            'new_estimated_cost_usd': result['new_estimated_cost_usd']}))
        return
    client = None
    if args.increment_envelope:
        from increment_budget import IncrementMetaClient
        client = IncrementMetaClient(timeout=args.timeout_s)
    result = rigid_discovery.discover(ROOT / 'data' / args.case, args.run,
        model=args.model, max_output_tokens=args.max_output_tokens, timeout=args.timeout_s, client=client)
    print(json.dumps({'status': result['status'], 'tracks': len(result['result']['tracks']),
                      'tracking_config': result['tracking_config'], 'estimated_cost_usd': result['estimated_cost_usd']}))


if __name__ == '__main__':
    main()
