#!/usr/bin/env python3
"""Read-only dry run by default. Root/operator alone may explicitly --commit."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from incremental_observed_publication_v1 import prepare,commit
def main():
 p=argparse.ArgumentParser();p.add_argument('--expected',required=True,type=Path);p.add_argument('--selections',required=True,type=Path);p.add_argument('--receipt',required=True,type=Path);p.add_argument('--commit',action='store_true');args=p.parse_args();assert not args.receipt.exists(),'Preserve prior receipt';expected=json.loads(args.expected.read_text());selections=json.loads(args.selections.read_text());result=commit(expected=expected,selections=selections,root=ROOT)if args.commit else prepare(expected=expected,selections=selections,root=ROOT).receipt;args.receipt.parent.mkdir(parents=True,exist_ok=True)
 with args.receipt.open('x')as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'status':result['status'],'catalog':result['catalog'],'receipt':str(args.receipt),'new_API_calls':0,'new_GPU_calls':0}))
if __name__=='__main__':main()
