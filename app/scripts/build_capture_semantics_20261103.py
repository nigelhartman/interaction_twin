#!/usr/bin/env python3
"""Private fixed four-capture follow-on. Run once the named pipelines finish."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import capture_semantic_batch_v1 as B

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',action='append',choices=list(B.RUNS))
    parser.add_argument('--output-dir',default=B.DEFAULT_OUTPUT)
    parser.add_argument('--skip-cable-traces',action='store_true')
    args=parser.parse_args()
    ref,value=B.build_batch(capture_ids=args.capture,output_dir=args.output_dir,trace_cables=not args.skip_cable_traces)
    print(json.dumps({'manifest':ref,'captures':[r['capture_id']for r in value['records']],
                      'catalog':value['relation_catalog_ref'],'public_library_changed':False,'new_model_calls':0},indent=2))

if __name__=='__main__':main()
