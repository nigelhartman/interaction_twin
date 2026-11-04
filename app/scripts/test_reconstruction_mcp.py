#!/usr/bin/env python3
"""Actual modern/legacy MCP version, evidence and quality resource checks."""
import asyncio
import base64
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mcp import Client, StdioServerParameters
from scripts.serve_twin_mcp import create_server

TWIN = 'flow-meta-video-jabra-20261030-v1--flow-procedure-scope-normalized-v1'
VERSION = 'case-lid-triposr-v1'


def structured(result):
    assert not result.is_error
    return result.structured_content or json.loads(result.content[0].text)


async def exercise(client, label):
    listing = structured(await client.call_tool('list_reconstructions', {'twin_id': TWIN, 'selection': 'diagnostic'}))
    assert {r['component_id'] for r in listing['versions']} == {'case_base', 'case_lid'}
    accepted = structured(await client.call_tool('list_reconstructions', {'twin_id': TWIN, 'selection': 'accepted'}))
    assert accepted['versions'] == []
    descriptor = structured(await client.call_tool('get_reconstruction', {'twin_id': TWIN, 'version_id': VERSION}))
    assert descriptor['parent_track_id'] == 'charging_case' and descriptor['component_binding'] == 'model_proposed_subpart_of_existing_tracklet'
    assert descriptor['gauge']['metric'] is False and descriptor['simulation_ready'] is False
    version_resource = json.loads((await client.read_resource(f'twin://{TWIN}/reconstructions/{VERSION}')).contents[0].text)
    assert version_resource == descriptor
    report = structured(await client.call_tool('get_reconstruction_report', {'twin_id': TWIN, 'version_id': VERSION}))
    verified = {}
    for key in ['glb', 'quality', 'source-image-0', 'source-mask-0']:
        artifact = descriptor['artifacts'][key]
        content = (await client.read_resource(artifact['uri'])).contents[0]
        assert content.mime_type == artifact['resource_mime_type'] == 'application/octet-stream'
        data = base64.b64decode(content.blob, validate=True)
        assert hashlib.sha256(data).hexdigest() == artifact['sha256']
        verified[key] = {'sha256': artifact['sha256'], 'bytes': len(data), 'declared_media_mime': artifact['mime']}
        if key == 'quality':
            assert json.loads(data) == report['report']
    for version in ['camera-body-stereo-v1', '../../outside']:
        response = await client.call_tool('get_reconstruction', {'twin_id': TWIN, 'version_id': version})
        assert response.is_error
    return {'transport': label, 'protocol_version': client.session.protocol_version, 'diagnostic_versions': 2,
            'accepted_versions': 0, 'verified_resources': verified, 'wrong_source_and_path_requests_rejected': 2,
            'simulation_ready': False, 'new_model_or_api_calls': 0}


async def main():
    checks = []
    async with Client(create_server(private_owner=True), read_timeout_seconds=90) as client:
        checks.append(await exercise(client, 'in_process'))
    params = StdioServerParameters(command=str(ROOT / '.venv-mcp/bin/python'),
                                   args=[str(ROOT / 'scripts/serve_twin_mcp.py'),'--private-owner'], cwd=ROOT)
    for mode in ['auto', 'legacy']:
        async with Client(params, mode=mode, read_timeout_seconds=90) as client:
            checks.append(await exercise(client, 'subprocess_stdio_' + mode))
    path = ROOT / 'output/mcp/reconstruction-smoke-20261031.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'status': 'passed', 'checks': checks}, indent=2) + '\n')
    print({'status': 'passed', 'transports': len(checks), 'output': str(path.relative_to(ROOT))})


if __name__ == '__main__':
    asyncio.run(main())
