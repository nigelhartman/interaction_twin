#!/usr/bin/env python3
"""Check/import a first SAM3D result locally, without an existing 3D baseline."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import initial_sam3d_import as initial


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('inspect-input', 'check', 'import'))
    parser.add_argument('--handoff', type=Path, required=True)
    parser.add_argument('--adapter-output', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.command != 'inspect-input' and args.adapter_output is None:
        parser.error('check/import requires --adapter-output with the unchanged returned directory')
    if args.command == 'import' and args.output is None:
        parser.error('import requires a fresh --output directory')
    if args.command != 'import' and args.output is not None:
        parser.error('--output is only for import; check never writes files')
    try:
        if args.command == 'inspect-input':
            value = initial.inspect_input(args.handoff)
            report = {'status': 'initial_input_verified_awaiting_native_output', 'handoff': value['handoff'],
                      'source': value['binding']['source'], 'component_id': value['binding']['component_id'],
                      'baseline_required': False, 'model_inference_ran_during_import': False}
        elif args.command == 'check':
            report = initial.inspect_output(args.handoff, args.adapter_output)[0]
        else:
            report = initial.import_output(args.handoff, args.adapter_output, args.output)
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps({'status': 'unsupported_or_changed_initial_artifact', 'error': str(error)}, indent=2))
        return 2
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
