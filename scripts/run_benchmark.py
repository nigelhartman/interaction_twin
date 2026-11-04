#!/usr/bin/env python3
"""Inspect saved evidence and emit a versioned JSON manifest plus offline HTML."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import benchmark


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='benchmarks/local-v1.json', help='Project-relative suite configuration')
    p.add_argument('--output', default='output/benchmark-local-v1', help='Project-relative output directory')
    p.add_argument('--workspace-output', default='workspace/benchmark', help='Project-relative workspace mirror; regenerated with every run')
    args = p.parse_args()
    output = benchmark.path_in(ROOT, args.output)
    report = benchmark.build(ROOT, args.config)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (output / 'index.html').write_text(benchmark.render(report, os.path.relpath(ROOT, output) + '/'))
    mirror = benchmark.path_in(ROOT, args.workspace_output)
    mirror.mkdir(parents=True, exist_ok=True)
    (mirror / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (mirror / 'index.html').write_text(benchmark.render(report, os.path.relpath(ROOT, mirror) + '/'))
    print(json.dumps({'output': str(output / 'index.html'), 'cases': len(report['cases']),
                      'workspace_mirror': str(mirror / 'index.html'),
                      'case_statuses': {c['id']: c['status'] for c in report['cases']},
                      'preserved_failures': len(report['preserved_failures']), 'new_inference': False}, indent=2))
    return 1 if any(c['status'] != 'valid_saved_evidence' for c in report['cases']) else 0


if __name__ == '__main__':
    raise SystemExit(main())
