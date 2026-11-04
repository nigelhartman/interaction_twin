#!/usr/bin/env python3
"""Run a fresh, local-only, train-only observed stereo diagnostic."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import observed_stereo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sparse-input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    observed_stereo.run(ROOT / args.sparse_input, ROOT / args.output, root=ROOT)


if __name__ == '__main__':
    main()
