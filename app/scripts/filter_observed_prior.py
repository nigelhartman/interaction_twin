#!/usr/bin/env python3
"""Filter a saved prior with cached exact-frame SAM results; no inference."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from observed_prior_mask_filter import run


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True)
    parser.add_argument('--native-manifest',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    try:
        report=run(ROOT,args.source,args.native_manifest,args.output)
    except Exception as error:
        output=(ROOT/args.output).resolve()
        if output.is_relative_to(ROOT/'data') and output.is_dir() and (output/'protocol.json').is_file():
            (output/'failure.json').write_text(json.dumps({'status':'failed','error_type':type(error).__name__,
                                                          'message':str(error),'new_api_calls':0})+'\n')
        raise
    print(json.dumps(report))
