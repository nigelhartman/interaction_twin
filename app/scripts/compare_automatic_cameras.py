#!/usr/bin/env python3
"""Prepare then explicitly run one frozen ordinary-video camera comparison."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import automatic_camera as A


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--protocol-sha256')
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / 'data' / 'automatic-camera-20261101'):
        raise ValueError('Output must remain below the fixed comparison data root')
    if args.action == 'prepare':
        p = A.prepare(ROOT, output)
        print(json.dumps({'status': 'protocol_prepared_no_fit_run',
            'protocol_sha256': A.digest(output / 'protocol.json'),
            'support': p['support'], 'cached_prior_binding': p['cached_prior_binding']}))
    else:
        if not args.protocol_sha256 or len(args.protocol_sha256) != 64:
            raise ValueError('Explicit frozen protocol SHA-256 is required to run')
        r = A.run(ROOT, output, args.protocol_sha256)
        print(json.dumps({'status': r['status'], 'training_decision': r['training_decision']}))


if __name__ == '__main__':
    main()
