#!/usr/bin/env python3
"""Export a catalog-validated saved twin; no inference or uploads."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import portable_twin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--twin-id', required=True)
    parser.add_argument('--output', type=Path, required=True, help='New directory; parent must exist')
    parser.add_argument('--include-video', action='store_true', help='Explicitly bundle the private full recording')
    parser.add_argument('--no-masks', action='store_true')
    parser.add_argument('--zip-output', type=Path)
    args = parser.parse_args()
    result = portable_twin.export_case(args.twin_id, args.output, include_video=args.include_video,
                                      include_masks=not args.no_masks, zip_output=args.zip_output)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
