#!/usr/bin/env python3
"""Record an actual read-only stdio MCP charging-task workflow; no models or robot execution."""
from __future__ import annotations
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from mcp import Client,StdioServerParameters
from demo_twin_agent import RecordedMCP,encoded,require,sha

PHONE='flow-capture-phone-20261103-v1--flow-procedure-1'
CHARGER='flow-capture-charger-20261103-v1--flow-procedure-1'
DEFAULT=ROOT/'showcase/demos/charge-phone-v1'


def write_new(path,value):
    with path.open('x')as stream:json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')


async def collect(client):
    mcp=RecordedMCP(client)
    await mcp.discover()
    discovered={}
    for query,target in [('phone',PHONE),('charger',CHARGER)]:
        response=await mcp.tool('search_twins',{'query':query,'limit':25})
        require(response['access']=='owner_curated_public_snapshot','Cross-recording demo must use the curated library')
        require(any(r['twin_id']==target for r in response['twins']),'Source recording was not discovered by actual MCP search')
        discovered[target]=response
    programs={};interfaces={};descriptions={};procedures={}
    for twin_id in [PHONE,CHARGER]:
        descriptions[twin_id]=await mcp.tool('describe_twin',{'twin_id':twin_id})
        raw=await mcp.resource(f'twin://{twin_id}/program','application/json',maximum_bytes=1024*1024)
        programs[twin_id]=json.loads(raw)
        procedures[twin_id]=await mcp.tool('get_procedure',{'twin_id':twin_id})
        require(procedures[twin_id]['steps']==programs[twin_id]['steps'],'Returned procedure differs from retrieved program')
        interfaces[twin_id]=await mcp.tool('get_connection_interfaces',{'twin_id':twin_id})
    task=await mcp.tool('prepare_cross_recording_task',{'phone_twin_id':PHONE,'charger_twin_id':CHARGER,
        'instruction':'Prepare a task to charge this phone.'})
    require(task['status']=='draft_requires_validation'and task['robot_actions']==[],'Task must remain a non-executable draft')
    verified_assets=[];verified_evidence=[];total_download=0
    for source in task['sources']:
        tid=source['twin_id'];doc=programs[tid]
        require(source['program_sha256']==descriptions[tid]['program_sha256']==interfaces[tid]['program_sha256'],'Wrong source-program binding')
        require(source['source']==doc['source'],'Wrong source video binding')
        for asset in source['assets']:
            descriptor=await mcp.tool('get_asset',{'twin_id':tid,'track_id':asset['track_id']})
            require(descriptor['sha256']==asset['sha256']and descriptor['uri']==asset['uri'],'Wrong declared native asset')
            blob=await mcp.resource(asset['uri'],'model/gltf-binary',maximum_bytes=16*1024*1024)
            require(blob[:4]==b'glTF'and sha(blob)==asset['sha256'],'Native GLB bytes changed')
            total_download+=len(blob)
            require(total_download<=64*1024*1024,'Actual resource retrieval exceeds bounded demonstration budget')
            verified_assets.append({'twin_id':tid,'component_id':asset['track_id'],'sha256':sha(blob),'bytes':len(blob),'uri':asset['uri']})
        evidence={e['id']:e for e in doc['evidence']}
        used=sorted({eid for i in interfaces[tid]['interfaces']for eid in i['evidence_ids']})
        for eid in used:
            descriptor=await mcp.tool('get_evidence',{'twin_id':tid,'evidence_id':eid})
            require(descriptor['evidence']==evidence[eid],'Source evidence metadata changed')
            blob=await mcp.resource(descriptor['frame_uri'],'image/jpeg',maximum_bytes=2*1024*1024)
            require(blob[:2]==b'\xff\xd8'and sha(blob)==evidence[eid]['frame_sha256'],'Source JPEG bytes changed')
            verified_evidence.append({'twin_id':tid,'evidence_id':eid,'timestamp_s':evidence[eid]['timestamp_s'],
                'sha256':sha(blob),'bytes':len(blob),'uri':descriptor['frame_uri']})
    receipt={'format':'interaction-twin-actual-mcp-workflow','version':1,'created_utc':datetime.now(timezone.utc).isoformat(),
        'transport':'actual_official_sdk_subprocess_stdio','sdk':'mcp==2.3.0','workflow':'deterministic_scripted_client_not_live_llm',
        'instruction':task['instruction'],'source_recordings':2,'native_asset_bytes_verified':verified_assets,
        'source_evidence_bytes_verified':verified_evidence,'total_native_download_bytes':total_download,
        'original_program_file_bytes_included':False,'program_resource_encoding':'MCP JSON serialization; source program SHA remains declared original binding',
        'task_draft_sha256':task['draft_sha256'],'calls':mcp.receipts,
        'claims':{'read_only_mcp':True,'model_inference':False,'new_api_calls':0,'robot_execution':False,'physical_success':False}}
    return task,receipt


async def main(output):
    output=output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    if any((output/n).exists()for n in ['task.json','trace.json','summary.json']):raise ValueError('Existing demo is immutable; choose a new output directory')
    params=StdioServerParameters(command=str(ROOT/'.venv-mcp/bin/python'),args=[str(ROOT/'scripts/serve_twin_mcp.py'),'--curated'],cwd=ROOT)
    async with Client(params,mode='auto',read_timeout_seconds=90)as client:task,receipt=await collect(client)
    write_new(output/'task.json',task);write_new(output/'trace.json',receipt)
    summary={'format':'interaction-twin-charge-phone-demo','version':1,'status':'saved_actual_mcp_workflow',
        'source_recordings':2,'connection_candidates':len(task['connection_candidates']),'draft_steps':len(task['task_draft']['steps']),
        'mcp_calls':len(receipt['calls']),'verified_native_models':len(receipt['native_asset_bytes_verified']),
        'verified_evidence_frames':len(receipt['source_evidence_bytes_verified']),
        'task_sha256':hashlib.sha256((output/'task.json').read_bytes()).hexdigest(),
        'trace_sha256':hashlib.sha256((output/'trace.json').read_bytes()).hexdigest(),
        'live_llm_agent':False,'new_api_calls':0,'physical_success_verified':False,'robot_execution_available':False}
    write_new(output/'summary.json',summary);print(json.dumps(summary))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT)
    asyncio.run(main(parser.parse_args().output))
