#!/usr/bin/env python3
"""One bounded offline MoGe/MPS completion candidate; never retry or retune."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from observed_depth_completion import run


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--exclude-component-mask',action='append',default=[])
    args=parser.parse_args()
    try:
        report=run(ROOT,args.source,args.output,args.exclude_component_mask)
    except Exception as error:
        output=(ROOT/args.output).resolve()
        if output.is_relative_to(ROOT/'data') and output.is_dir() and (output/'protocol.json').is_file():
            (output/'failure.json').write_text(json.dumps({'status':'failed','error_type':type(error).__name__,
                                                          'message':str(error),'new_api_calls':0})+'\n')
        raise
    print(json.dumps({'status':report['status'],'counts':report['counts'],'alignment':report['alignment']}))
