#!/usr/bin/env python3
"""Prepare locally, or explicitly execute the reviewed root-owned Meta/MCP loop."""
from pathlib import Path
import argparse,asyncio,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mcp_assessment as A
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','execute']);p.add_argument('--out',required=True);p.add_argument('--prepared');p.add_argument('--execute-reviewed-meta',action='store_true');a=p.parse_args()
 if a.action=='prepare':r=A.prepare(ROOT/a.out,root=ROOT);print(json.dumps({'status':'prepared_no_api_calls','protocol_sha256':A.sha(ROOT/a.out/'protocol.json')}))
 else:
  if not a.execute_reviewed_meta or not a.prepared:p.error('Explicit root review flag and prepared protocol are required')
  r=asyncio.run(A.execute(ROOT/a.prepared,ROOT/a.out,root=ROOT));print(json.dumps({'status':r['status'],'rounds':len(r['rounds']),'tool_calls':len(r['tool_receipts'])}))
