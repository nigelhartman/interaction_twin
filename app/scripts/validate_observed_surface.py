#!/usr/bin/env python3
"""Run one frozen observed-surface held-out RGB diagnostic locally."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import observed_surface_validation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--surface-input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    observed_surface_validation.run(ROOT / args.surface_input, ROOT / args.output, root=ROOT)


if __name__ == '__main__':
    main()
