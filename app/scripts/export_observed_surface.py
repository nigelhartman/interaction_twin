#!/usr/bin/env python3
"""Export preserved observed patches locally, with no inference or network."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from observed_surface_export import export_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/observed-surface-exports-20261031')
    parser.add_argument('--run', action='append')
    args = parser.parse_args()
    output = ROOT / args.output
    if not output.resolve().is_relative_to(ROOT / 'data'):
        raise ValueError('Exports must be written inside project data/')
    result = export_all(ROOT, output, args.run or ['camera-body-stereo-v1', 'camera-body-stereo-refined-k-v2'])
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
