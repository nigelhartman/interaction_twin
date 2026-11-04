#!/usr/bin/env python3
"""Dispatch or resume a source-bound SAM3D upload on the existing GPU worker."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sam3d_cluster_bridge import configuration, reconstruct, SSHTransport, WorkerUnavailable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    status = sub.add_parser('status')
    status.add_argument('--diagnose', action='store_true', help='Show bounded error categories without raw remote output or credentials')
    run = sub.add_parser('reconstruct')
    run.add_argument('--handoff', required=True, type=Path)
    run.add_argument('--destination', required=True, type=Path)
    args = parser.parse_args()
    if args.action == 'status':
        config = configuration(ROOT)
        if config is None:
            result = {'status': 'awaiting_gpu_worker', 'message': 'Optional SAM3D cluster worker is not enabled.'}
        else:
            try:
                result = SSHTransport(config).request({'action': 'probe'})
            except WorkerUnavailable as error:
                result = {'status': 'awaiting_gpu_worker', 'message': str(error)}
                if args.diagnose:
                    result['diagnostic'] = error.diagnostic
    else:
        result = reconstruct(args.handoff, args.destination, root=ROOT)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
