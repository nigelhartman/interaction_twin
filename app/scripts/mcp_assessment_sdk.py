#!/usr/bin/env python3
"""Fixed official-SDK MCP broker; JSON stdin chooses only pinned read-only tools.

This runs in the already prepared MCP runtime. It has no Meta client, model,
resource-reading operation, configurable command, or arbitrary root parameter.
"""
from pathlib import Path
import argparse,asyncio,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mcp_assessment as A
import jsonschema
from mcp import Client,StdioServerParameters
from demo_twin_agent import RecordedMCP, ROBOT_TOOLS

async def main(path,digest):
 p=Path(path).resolve();A.require(p.is_relative_to(ROOT/'data') and A.sha(p)==digest,'Expected pinned local protocol')
 protocol=A.read(p);A.verify_pins(protocol,ROOT)
 params=StdioServerParameters(command=str(ROOT/'.venv-mcp/bin/python'),args=['-B',str(ROOT/'scripts/serve_twin_mcp.py')],cwd=ROOT,
  env={'PATH':'/opt/homebrew/bin:/usr/bin:/bin','PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1'})
 async with Client(params,mode='auto',read_timeout_seconds=60) as client:
  tools=(await client.list_tools()).tools
  names={t.name for t in tools}
  A.require(names in [A.TOOLS,A.TOOLS|{'get_library_facets','get_generated_candidate','get_kinematics'},A.TOOLS|{'get_library_facets','get_generated_candidate','get_kinematics'}|ROBOT_TOOLS],'Unexpected live SDK tool surface')
  A.require(all(t.annotations and t.annotations.read_only_hint is True and t.annotations.destructive_hint is False and t.annotations.open_world_hint is False for t in tools),'Only read-only tools may be exposed')
  # Preserve the frozen assessment planner vocabulary. The new read-only facet
  # tool is checked above but not offered as an additional model capability.
  schemas={t.name:t.input_schema for t in tools if t.name in A.TOOLS};mcp=RecordedMCP(client)
  print(json.dumps({'ok':True,'official_sdk_client':True,'transport':'local_subprocess_stdio','schemas':schemas}),flush=True)
  count=0
  while True:
   line=await asyncio.to_thread(sys.stdin.readline,10001)
   if not line:return
   A.require(len(line)<=10000 and line.endswith('\n'),'Bounded tool request required')
   request=json.loads(line);A.require(set(request)=={'tool','arguments'} and request['tool'] in A.TOOLS,'Only named read-only tool calls are permitted')
   A.require(count<A.POLICY['maximum_tool_calls'],'SDK tool-call cap reached')
   jsonschema.validate(request['arguments'],A.arg_schemas(protocol)[request['tool']]);jsonschema.validate(request['arguments'],schemas[request['tool']])
   if request['tool'] in {'get_reconstruction','get_reconstruction_report'}:A.require(request['arguments']['version_id'] in protocol['allowed_reconstruction_ids'],'Undeclared reconstruction')
   try:value=await mcp.tool(request['tool'],request['arguments'])
   except Exception as error:
    print(json.dumps({'ok':False,'error_type':type(error).__name__,'message':'Read-only SDK tool rejected request.',
     'sdk_receipt':mcp.receipts[-1] if mcp.receipts else None}),flush=True)
    return
   count+=1
   A.require(len(A.encoded(value))<=A.POLICY['maximum_raw_tool_result_bytes'],'SDK result exceeds fixed allowance')
   print(json.dumps({'ok':True,'value':value,'sdk_receipt':mcp.receipts[-1]},allow_nan=False),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--protocol-sha256',required=True);a=p.parse_args()
 try:asyncio.run(main(a.protocol,a.protocol_sha256))
 except Exception as error:
  print(json.dumps({'ok':False,'error_type':type(error).__name__,'message':'Read-only SDK broker failed; see preserved local stderr.'}),flush=True)
  raise
