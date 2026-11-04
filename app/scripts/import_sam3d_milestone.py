#!/usr/bin/env python3
"""Inspect/import fresh SAM3D output locally. No models, network or model switch."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sam3d_milestone as milestone


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('inspect-input', 'check', 'import'))
    parser.add_argument('--input-bundle', type=Path, required=True)
    parser.add_argument('--adapter-output', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.command != 'inspect-input' and args.adapter_output is None:
        parser.error('check/import requires --adapter-output with unchanged copied output')
    if args.command == 'import' and args.output is None:
        parser.error('import requires a fresh --output directory')
    if args.command != 'import' and args.output is not None:
        parser.error('--output is only for import; check never writes files')
    try:
        if args.command == 'inspect-input':
            value = milestone.inspect_input_bundle(args.input_bundle)
            report = {'status': 'frozen_input_verified_awaiting_native_output', 'binding': value['binding'], 'learned_inference_executed': False, 'files': milestone._records(value['files'])}
        elif args.command == 'check':
            report = milestone.inspect_output(args.input_bundle, args.adapter_output)[0]
        else:
            report = milestone.import_output(args.input_bundle, args.adapter_output, args.output)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({'status': 'unsupported_or_changed_artifact', 'error': str(exc)}, indent=2))
        return 2
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
