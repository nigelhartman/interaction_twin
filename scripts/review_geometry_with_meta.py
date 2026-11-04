#!/usr/bin/env python3
"""Prepare or run exactly one reviewed-transport Meta geometry advisory."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import geometry_intelligence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'run'])
    parser.add_argument('--output', default='data/geometry-intelligence-20261031/reconstruction-review-v1')
    args = parser.parse_args()
    output = ROOT / args.output
    result = (geometry_intelligence.prepare if args.mode == 'prepare' else geometry_intelligence.run)(ROOT, output)
    print({'status': result['status'], 'output': str(output.relative_to(ROOT))})


if __name__ == '__main__':
    main()
