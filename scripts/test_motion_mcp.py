#!/usr/bin/env python3
"""Real read-only motion tools/resources in official SDK modern/legacy transports."""
import argparse,asyncio,base64,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from mcp import Client,StdioServerParameters
from serve_twin_mcp import create_server
from test_twin_mcp import structured
from rehearse_motion_handoff import DEFAULT_TWIN,DEFAULT_EXPERIMENT


async def exercise(client,label):
    tools=(await client.list_tools()).tools
    assert {t.name for t in tools}==__import__('test_twin_mcp').TOOLS and all(t.annotations.read_only_hint and not t.annotations.destructive_hint and not t.annotations.open_world_hint for t in tools)
    listing=structured(await client.call_tool('list_motion_experiments',{'twin_id':DEFAULT_TWIN}))
    assert {r['experiment_id'] for r in listing['experiments']}=={'jabra-case-base-3fps-v3','jabra-case-lid-3fps-v1','jabra-earbud-a-3fps-v1'}
    d=structured(await client.call_tool('get_motion_experiment',{'twin_id':DEFAULT_TWIN,'experiment_id':DEFAULT_EXPERIMENT}))
    exact=await client.read_resource(f'twin://{DEFAULT_TWIN}/motion/{DEFAULT_EXPERIMENT}')
    assert json.loads(exact.contents[0].text)==d and d['policy']['simulation_ready'] is False and d['policy']['pose_interpolation'] is False
    data={}
    for key,b in d['artifacts'].items():
        result=(await client.read_resource(b['uri'])).contents[0]
        assert result.mime_type=='application/octet-stream'
        data[key]=base64.b64decode(result.blob,validate=True)
        assert hashlib.sha256(data[key]).hexdigest()==b['sha256']
    assert json.loads(data['normalized'])==d['motion']
    assert data['source-image-0'][:2]==b'\xff\xd8' and data['source-mask-0'][:8]==b'\x89PNG\r\n\x1a\n'
    found=structured(await client.call_tool('search_twins',{'limit':25}))
    old=next(t['twin_id'] for t in found['twins'] if t['twin_id']!=DEFAULT_TWIN)
    empty=structured(await client.call_tool('list_motion_experiments',{'twin_id':old}));assert empty['experiments']==[]
    rejected=0
    for args in [{'twin_id':DEFAULT_TWIN,'experiment_id':'not-registered'}, {'twin_id':old,'experiment_id':DEFAULT_EXPERIMENT}, {'twin_id':DEFAULT_TWIN,'experiment_id':'../../private'}]:
        result=await client.call_tool('get_motion_experiment',args);assert result.is_error;rejected+=1
    for uri in [f'twin://{DEFAULT_TWIN}/motion/{DEFAULT_EXPERIMENT}/artifacts/unknown',f'twin://{old}/motion/{DEFAULT_EXPERIMENT}/artifacts/raw']:
        try:await client.read_resource(uri)
        except Exception:rejected+=1
        else:raise AssertionError('Unknown/cross-program resource was exposed')
    return {'transport':label,'protocol':client.session.protocol_version,'experiments':len(listing['experiments']),
        'new_tools':['list_motion_experiments','get_motion_experiment'],'resources_verified':len(data)+1,
        'bytes_verified':sum(len(v) for v in data.values()),'invalid_requests_rejected':rejected,
        'program_sha256':d['program_sha256'],'exact_source':d['inspection_source'],'model_inference':False,'simulation_ready':False}


async def run(output):
    results=[]
    async with Client(create_server(private_owner=True),mode='auto',read_timeout_seconds=90) as client:results.append(await exercise(client,'in_process_auto'))
    p=StdioServerParameters(command=str(ROOT/'.venv-mcp/bin/python'),args=[str(ROOT/'scripts/serve_twin_mcp.py'),'--private-owner'],cwd=ROOT)
    for mode in ['auto','legacy']:
        async with Client(p,mode=mode,read_timeout_seconds=90) as client:results.append(await exercise(client,'subprocess_stdio_'+mode))
    report={'format':'actual-readonly-motion-mcp-transport-tests','version':1,'status':'passed','official_sdk':'mcp==2.3.0','checks':results,'new_api_calls':0,'new_model_calls':0}
    with Path(output).open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);asyncio.run(run(parser.parse_args().output))
