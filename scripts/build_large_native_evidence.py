#!/usr/bin/env python3
"""Preserve an oversized SAM3D return and build an explicitly derived preview.
No inference, transport, canonical import, approval or publication.
"""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import large_native_evidence_v1 as L

def main():
 parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
 create=commands.add_parser('build');create.add_argument('--job',required=True);create.add_argument('--output',required=True);create.add_argument('--target-faces',type=int,default=350000)
 verify=commands.add_parser('verify');verify.add_argument('--report',required=True)
 args=parser.parse_args()
 if args.command=='build':
  report,_,_=L.build(args.job,args.output,target_faces=args.target_faces);print(json.dumps({'status':report['status'],'report':L.reference(Path(args.output)/'evidence.json'),'original':report['native_glb'],'preview':report['preview_glb'],'legacy_canonical_import_completed':False},indent=2))
 else:print(json.dumps(L.verify(L.reference(args.report)),indent=2))
if __name__=='__main__':main()
