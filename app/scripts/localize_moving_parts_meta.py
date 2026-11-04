#!/usr/bin/env python3
"""Prepare/run one visual localization and up to two bounded SAM box requests."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import localized_moving_parts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'run'])
    parser.add_argument('--output', default='data/moving-parts-sam31-20261031/jabra-localized-parts-v2')
    args = parser.parse_args()
    result = (localized_moving_parts.prepare if args.mode == 'prepare' else localized_moving_parts.run)(ROOT, ROOT / args.output)
    print({'status': result['status'], 'requests_attempted': result.get('requests_attempted', 0), 'output': args.output})


if __name__ == '__main__':
    main()
