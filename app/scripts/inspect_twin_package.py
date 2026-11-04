#!/usr/bin/env python3
"""Validate portable directory/ZIP without executing it; optionally import fresh."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import portable_twin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package', type=Path)
    parser.add_argument('--extract-to', type=Path, help='Import into this new directory after validation')
    args = parser.parse_args()
    print(json.dumps(portable_twin.inspect_package(args.package, extract_to=args.extract_to), indent=2))


if __name__ == '__main__':
    main()
