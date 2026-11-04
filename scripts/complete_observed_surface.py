#!/usr/bin/env python3
"""Create one optional local inferred layer; preserve original observed inputs."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from observed_surface_completion import complete


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='Saved observed stereo directory under data/')
    parser.add_argument('--output', required=True, help='Fresh output directory under data/')
    parser.add_argument('--exclude-component-mask', action='append', default=[], help='Exact source-frame /files/data/ mask URL for a different component')
    args = parser.parse_args()
    try:
        report = complete(ROOT, args.source, args.output, args.exclude_component_mask)
    except Exception as error:
        output = (ROOT / args.output).resolve()
        if output.is_relative_to(ROOT / 'data') and output.is_dir() and (output / 'protocol.json').is_file():
            (output / 'failure.json').write_text(json.dumps({'status': 'failed', 'error_type': type(error).__name__,
                                                          'message': str(error), 'new_api_calls': 0})+'\n')
        raise
    print(json.dumps({'status': report['status'], 'counts': report['counts'], 'rejected_regions': report['rejected_regions']}))
