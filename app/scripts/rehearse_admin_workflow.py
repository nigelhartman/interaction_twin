#!/usr/bin/env python3
"""Verify upload/review/publication offline, without touching the real owner state."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from admin_workflow_rehearsal_20261103 import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    target = args.receipt.resolve()
    if not target.is_relative_to(ROOT / 'output') or target.exists():
        parser.error('Choose a new receipt inside this project output directory')
    value = run(source_root=ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: value[k] for k in ['status', 'execution_kind', 'paid_api_transport_calls', 'gpu_calls', 'elapsed_s']}))


if __name__ == '__main__': main()
