#!/usr/bin/env python3
"""Prepare/run two offline MPS model previews from genuine SAM part masks."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import moving_parts_reconstruction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'run'])
    parser.add_argument('--output', default='data/moving-parts-reconstruction-20261031/jabra-parts-triposr-v1')
    args = parser.parse_args()
    result = (moving_parts_reconstruction.prepare if args.mode == 'prepare' else moving_parts_reconstruction.run)(ROOT, ROOT / args.output)
    print({'status': result['status'], 'output': args.output})


if __name__ == '__main__':
    main()
