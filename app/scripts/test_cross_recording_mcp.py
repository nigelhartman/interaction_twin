#!/usr/bin/env python3
"""Exercise new tools via actual official SDK, including private-owner boundary."""
from __future__ import annotations
import argparse
import asyncio
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from mcp import Client,StdioServerParameters
from serve_twin_mcp import create_server
from build_cross_recording_demo import PHONE,CHARGER


def structured(result):
    assert not result.is_error
    return result.structured_content or json.loads(result.content[0].text)


async def main(output):
    params=StdioServerParameters(command=str(ROOT/'.venv-mcp/bin/python'),args=[str(ROOT/'scripts/serve_twin_mcp.py'),'--curated'],cwd=ROOT)
    async with Client(params,mode='auto',read_timeout_seconds=60)as client:
        tools={t.name:t for t in (await client.list_tools()).tools}
        for name in ['get_connection_interfaces','prepare_cross_recording_task']:
            assert tools[name].annotations.read_only_hint and not tools[name].annotations.destructive_hint and not tools[name].annotations.open_world_hint
        interfaces=structured(await client.call_tool('get_connection_interfaces',{'twin_id':PHONE}))
        assert len(interfaces['interfaces'])==2 and interfaces['new_api_calls']==0
        task=structured(await client.call_tool('prepare_cross_recording_task',{'phone_twin_id':PHONE,'charger_twin_id':CHARGER}))
        assert task['status']=='draft_requires_validation'and task['robot_actions']==[]and task['claims']['simulation_ready']is False
        rejects=[]
        for name,args in [('get_connection_interfaces',{'twin_id':'../../private'}),
            ('get_connection_interfaces',{'twin_id':'flow-capture-bottle-20261103-v1--flow-procedure-1'}),
            ('prepare_cross_recording_task',{'phone_twin_id':PHONE,'charger_twin_id':CHARGER,'instruction':'Execute robot now'}),
            ('prepare_cross_recording_task',{'phone_twin_id':PHONE,'charger_twin_id':PHONE})]:
            assert (await client.call_tool(name,args)).is_error;rejects.append(name)
    working=json.loads((ROOT/'workspace/cases-index.json').read_text())['cases']
    public={r['id']for r in json.loads((ROOT/'showcase/public/cases-index.json').read_text())['cases']}
    private_id=next(r['id']for r in working if r['id']not in public)
    async with Client(create_server(private_owner=True),mode='auto',read_timeout_seconds=60)as client:
        assert(await client.call_tool('get_connection_interfaces',{'twin_id':private_id})).is_error
        assert(await client.call_tool('prepare_cross_recording_task',{'phone_twin_id':private_id,'charger_twin_id':CHARGER})).is_error
    report={'status':'passed','transport':'official_sdk_stdio_and_in_process_private_boundary','new_tools_read_only':True,
        'source_recordings':2,'draft_steps':len(task['task_draft']['steps']),'connection_candidates':len(task['connection_candidates']),
        'invalid_requests_rejected':len(rejects),'private_owner_mode_still_curated':True,
        'new_api_calls':0,'robot_execution_available':False,'physical_success_verified':False}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x')as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);asyncio.run(main(parser.parse_args().output))
