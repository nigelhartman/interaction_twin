#!/usr/bin/env python3
"""Prepare local GPU handoffs or import their contract-validated returned ZIPs."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import upgrade_bundles as bundles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    context = commands.add_parser('context')
    context.add_argument('--twin-id', required=True)
    context.add_argument('--track-id', required=True)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--twin-id', required=True)
    prepare.add_argument('--track-id', required=True)
    prepare.add_argument('--backend', choices=tuple(bundles.BACKENDS), required=True)
    prepare.add_argument('--base-sha256', required=True)
    prepare.add_argument('--prompt')
    prepare.add_argument('--appearance', choices=['vertex-color', 'uv-baked'])
    returned = commands.add_parser('import')
    returned.add_argument('--job-id', required=True)
    returned.add_argument('--zip', type=Path, required=True)
    commands.add_parser('list')
    args = parser.parse_args()
    if args.command == 'context':
        value = bundles.context(args.twin_id, args.track_id)
    elif args.command == 'prepare':
        body = {key: getattr(args, key) for key in ('twin_id', 'track_id', 'backend', 'base_sha256')}
        if args.prompt is not None:
            body['prompt'] = args.prompt
        if args.appearance is not None:
            body['appearance'] = args.appearance
        value = bundles.prepare_job(body)
    elif args.command == 'import':
        value = bundles.import_result(args.job_id, args.zip)
    else:
        value = {'jobs': bundles.list_jobs(), 'imports': bundles.list_imports()}
    print(json.dumps(value, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
