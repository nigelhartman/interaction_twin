#!/usr/bin/env python3
"""Run or resume the bounded evidence pipeline with the configured SAM 3D worker."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import Runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', default='meta_sam3d_default')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--input', type=Path)
    source.add_argument('--case')
    parser.add_argument('--run-id', help='Resume the identical request; receipts must still validate')
    args = parser.parse_args()
    runner = Runner(args.run_id)
    print(json.dumps({'run_id': runner.id, 'status_file': str(runner.path)}, indent=2), flush=True)
    result = runner.execute(args.profile, input_path=args.input, case_id=args.case)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['status'] in {'complete', 'complete_with_gaps'} else 1)


if __name__ == '__main__':
    main()
