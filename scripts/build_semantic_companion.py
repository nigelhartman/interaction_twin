#!/usr/bin/env python3
"""Create private additive semantic companions; no models or publication."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import semantic_asset_companion as S

def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    build=commands.add_parser('build');build.add_argument('--program',required=True);build.add_argument('--owner-context');build.add_argument('--output-dir',required=True)
    catalog=commands.add_parser('catalog');catalog.add_argument('--companion',action='append',required=True);catalog.add_argument('--output-dir',required=True)
    feedback=commands.add_parser('feedback');feedback.add_argument('--companion',required=True);feedback.add_argument('--feedback',required=True);feedback.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    if args.command=='build':ref=S.write_companion(S.build(args.program,owner_context_path=args.owner_context),args.output_dir)
    elif args.command=='catalog':ref=S.write_catalog(S.build_catalog(args.companion),args.output_dir)
    else:ref=S.write_companion(S.append_feedback(args.companion,args.feedback),args.output_dir)
    print(json.dumps({'artifact':ref,'new_model_calls':0,'public_library_changed':False},indent=2))
if __name__=='__main__':main()
