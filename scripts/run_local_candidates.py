#!/usr/bin/env python3
"""Run frozen cached-evidence local candidates; no hosted reasoning or downloads."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import local_candidates as C
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--case',default='recording-jabra-5fps-v1');p.add_argument('--prepare-only',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--max-runtime-s',type=int,default=1800);a=p.parse_args()
 out=a.output.resolve();case=(ROOT/'data'/a.case).resolve()
 if not out.is_relative_to(case) or out==case:p.error('Fresh output must stay inside the cached source case')
 result=C.prepare(out,case,'automatic-components-v1') if a.prepare_only else C.run(out,case,max_runtime_s=a.max_runtime_s,resume=a.resume)
 print(json.dumps({'output':str(out),'status':result['status'],'parts':[{ 'track_id':r['track_id'],'candidates':len(r['candidates']),'winner':r.get('selected_candidate_id')} for r in result['parts']]},indent=2))
