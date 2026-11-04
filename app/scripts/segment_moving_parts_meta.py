#!/usr/bin/env python3
"""Prepare or execute the explicitly bounded eight-image SAM experiment."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import moving_parts_sam


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'run'])
    parser.add_argument('--output', default='data/moving-parts-sam31-20261031/jabra-new-parts-v1')
    args = parser.parse_args()
    result = (moving_parts_sam.prepare if args.mode == 'prepare' else moving_parts_sam.run)(ROOT, ROOT / args.output)
    print({'status': result['status'], 'requests_attempted': result.get('requests_attempted', 0), 'output': args.output})


if __name__ == '__main__':
    main()
