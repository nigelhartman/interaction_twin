#!/usr/bin/env python3
"""Actual SDK verification of current curated derived bytes; never inference.

Default uses in-process/modern+legacy stdio/private-owner sessions. --network
additionally starts ONLY its own ephemeral loopback HTTP/SSE servers. SSE reads
small binary patches and metadata; its documented SDK event cap excludes GLBs.
"""
from __future__ import annotations
import argparse
import asyncio
import base64
from contextlib import asynccontextmanager
import hashlib
import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from mcp import Client, StdioServerParameters
from mcp.client.sse import sse_client
from curated_twin_catalog import CuratedTwinCatalog
from serve_twin_mcp import create_server


def structured(result):
    assert not result.is_error, str(result.content)
    return result.structured_content or json.loads(result.content[0].text)


async def exercise(client, label, *, small_only=False):
    public = CuratedTwinCatalog(); records, _ = public._snapshot()
    public_ids = {r['id'] for r in records}
    working = json.loads((ROOT / 'workspace/cases-index.json').read_bytes())['cases']
    private_id = next(r['id'] for r in working if r['id'] not in public_ids)
    tools = (await client.list_tools()).tools
    assert {'list_derived_artifacts','get_observed_surfaces'} <= {t.name for t in tools}
    assert all(t.annotations.read_only_hint and not t.annotations.destructive_hint for t in tools)
    assert any(t.uri_template=='twin://{twin_id}/observed-surfaces/{artifact_id}' for t in (await client.list_resource_templates()).resource_templates)
    cases = []; reads = []; first_token = None; first_twin = None
    for record in records:
        twin_id = record['id']
        result = structured(await client.call_tool('list_derived_artifacts', {'twin_id': twin_id}))
        expected = public.derived_assembly(twin_id)
        assert result['access'] == 'owner_curated_public_snapshot'
        if expected is None:
            assert result['status'] == 'unavailable' and not result['artifacts']
            continue
        assert result['declaration'] == expected['artifact'] and result['status'] == 'available'
        items = result['artifacts']; assert 1 <= len(items) <= 128
        by_path = {a['path']: a for a in items}; assert len(by_path) == len(items)
        if first_token is None:
            first_token = items[0]['artifact_id']; first_twin = twin_id
        elif twin_id != first_twin:
            try:
                await client.read_resource(f'twin://{twin_id}/derived-artifacts/{first_token}')
            except Exception:
                pass
            else:
                raise AssertionError('A different twin accepted an artifact ID')
        wanted = {expected['artifact']['path']}
        for component in expected['document']['component_assets']:
            if not small_only:
                wanted.add(component['native_glb']['path'])
        # Verify every (bounded <=8) appearance patch, plus one exact exemplar
        # of other extension roles. Never infer paths or rewrite patch bytes.
        roles = ['source_appearance_states', 'appearance_previous_declaration',
                 'appearance:probe_native_vertex_ids_ref', 'appearance:keyframe:patch_ref',
                 'appearance:keyframe:source_rgb_ref', 'appearance:keyframe:source_mask_ref',
                 'nonrigid_observations', 'nonrigid_source_pixel_trace']
        for role in roles:
            matches = [i for i in items if role in i['roles']]
            selected = matches if role == 'appearance:keyframe:patch_ref' else matches[:1]
            for item in selected:
                if not small_only or item['bytes'] < 650000:
                    wanted.add(item['path'])
        for path in sorted(wanted):
            item = by_path[path]
            response = (await client.read_resource(item['uri'])).contents
            assert len(response) == 1 and hasattr(response[0], 'blob')
            raw = base64.b64decode(response[0].blob, validate=True)
            assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256']
            assert raw == public._bytes({k: item[k] for k in ('path', 'sha256', 'bytes')},
                                        maximum=16*1024*1024 if path.endswith('.glb') else 8*1024*1024)
            if 'current_declaration' in item['roles']:
                assert json.loads(raw) == expected['document']
            reads.append({'twin_id': twin_id, **{k: item[k] for k in ('sha256', 'bytes', 'roles')}})
        cases.append({'twin_id': twin_id, 'declaration_sha256': result['declaration']['sha256'],
                      'artifacts': len(items),
                      'appearance_available': bool(expected['document'].get('source_appearance_states_ref'))})
    assert len(cases) == 3
    for twin_id in [private_id, '../../.secrets/key', 'unknown-curated--procedure']:
        assert (await client.call_tool('list_derived_artifacts', {'twin_id': twin_id})).is_error
        try:
            await client.read_resource(f'twin://{twin_id}/derived-artifacts/{first_token}')
        except Exception:
            pass
        else:
            raise AssertionError('An unapproved ID returned derived bytes')
    # The SDK ignores additional tool arguments. They must not grant path
    # authority or alter the bounded inventory for an otherwise valid twin.
    ordinary = structured(await client.call_tool('list_derived_artifacts', {'twin_id': first_twin}))
    extra = structured(await client.call_tool('list_derived_artifacts',
                                             {'twin_id': first_twin, 'path': 'data/private'}))
    assert extra == ordinary
    assert not any(n.split('.')[0] in {'numpy', 'cv2', 'PIL', 'scipy', 'torch'} for n in sys.modules)
    return {'transport': label, 'current_declarations': cases, 'exact_binary_reads': reads,
            'private_owner_boundary_retained': True, 'arbitrary_paths_not_resource_authority': True,
            'cross_record_token_rejected': True, 'heavy_runtime_imports': [],
            'large_binary_scope': 'not_exercised_SDK_SSE_1MiB_event_limit' if small_only else 'all_current_GLBS',
            'new_api_calls': 0}


@asynccontextmanager
async def endpoint(transport):
    import uvicorn
    model = create_server()
    app = model.streamable_http_app() if transport == 'http' else model.sse_app()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port,
        log_level='warning', access_log=False, log_config=None))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        for _ in range(100):
            if server.started: break
            if task.done(): await task; raise RuntimeError('Own test server stopped')
            await asyncio.sleep(.05)
        else: raise RuntimeError('Own test server failed readiness')
        yield f'http://127.0.0.1:{port}' + ('/mcp' if transport == 'http' else '/sse')
    finally:
        server.should_exit = True
        try: await asyncio.wait_for(task, 10)
        finally: sock.close()


async def main(args):
    results = []
    if args.network:
        for transport, mode in [('http', 'auto'), ('sse', 'legacy')]:
            async with endpoint(transport) as url:
                connection = url if transport == 'http' else sse_client(url)
                async with Client(connection, mode=mode, read_timeout_seconds=60) as client:
                    results.append(await exercise(client, transport, small_only=transport == 'sse'))
    else:
        async with Client(create_server(), read_timeout_seconds=60) as client:
            results.append(await exercise(client, 'in_process'))
        params = StdioServerParameters(command=str(ROOT / '.venv-mcp/bin/python'),
            args=[str(ROOT / 'scripts/serve_twin_mcp.py')], cwd=ROOT)
        for mode in ['auto', 'legacy']:
            async with Client(params, mode=mode, read_timeout_seconds=60) as client:
                results.append(await exercise(client, 'stdio_' + mode))
        async with Client(create_server(private_owner=True), read_timeout_seconds=60) as client:
            results.append(await exercise(client, 'explicit_private_owner_curated_derivatives'))
    report = {'format': 'interaction-twin-derived-artifact-mcp-proof', 'version': 1,
              'status': 'passed', 'checks': results, 'new_api_calls': 0, 'SDK_modified': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': 'passed', 'sessions': len(results), 'output': str(args.output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--network', action='store_true')
    asyncio.run(main(parser.parse_args()))
