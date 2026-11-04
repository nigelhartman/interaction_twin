#!/usr/bin/env python3
"""Freeze/inspect an initial source anchor without TripoSR or model execution."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import initial_sam3d_handoff as H


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_subparsers(dest='mode',required=True)
    prepare=modes.add_parser('prepare')
    prepare.add_argument('--input-manifest',required=True)
    prepare.add_argument('--selection',required=True)
    prepare.add_argument('--output',required=True)
    inspect=modes.add_parser('inspect');inspect.add_argument('--handoff',required=True)
    args=parser.parse_args()
    result=H.prepare(args.input_manifest,args.selection,args.output) if args.mode == 'prepare' else H.inspect(args.handoff)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__ == '__main__':main()
