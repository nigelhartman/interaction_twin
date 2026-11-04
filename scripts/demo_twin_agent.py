#!/usr/bin/env python3
"""Scripted MCP consumer: inspect a saved twin and build an offline portable package.

All source-program, asset and image data comes from real MCP calls. There is no
LLM, model inference, network service, robot execution or direct source-file read.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_TWIN = 'jabra-stage6-v1--procedure-v1'
ID = re.compile(r'[a-z][a-z0-9_-]{0,159}\Z')
HASH = re.compile(r'[a-f0-9]{64}\Z')
SUPPORTED = {'supported_conditional_motion_candidate', 'reference_candidate_unverified_geometry_and_pose'}
TOOLS = {'search_twins', 'describe_twin', 'get_procedure', 'query_state',
         'get_asset', 'get_evidence', 'get_simulation_manifest',
         'list_reconstructions', 'get_reconstruction', 'get_reconstruction_report'}
MOTION_TOOLS = {'list_motion_experiments', 'get_motion_experiment'}
DISCOVERY_TOOLS = {'get_library_facets', 'get_generated_candidate', 'get_kinematics'}
CONNECTION_TOOLS = {'get_connection_interfaces','prepare_cross_recording_task','get_derived_assembly'}
ROBOT_TOOLS = {'list_robot_kits','get_robot_kit','plan_robot_task','validate_robot_task','export_robot_task_bundle'}


class DemoError(ValueError):
    pass


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False, ensure_ascii=False).encode('utf-8')


def require(condition, message):
    if not condition:
        raise DemoError(message)


def identifier(value):
    require(isinstance(value, str) and ID.fullmatch(value), 'Invalid catalog identifier')
    return value


def hash_value(value):
    require(isinstance(value, str) and HASH.fullmatch(value), 'Missing or invalid artifact hash')
    return value


class RecordedMCP:
    """Record actual parsed SDK responses and decoded resource hashes, not invented calls."""
    def __init__(self, client):
        self.client = client
        self.receipts = []

    async def discover(self):
        started = time.monotonic()
        at = datetime.now(timezone.utc).isoformat()
        result = await self.client.list_tools()
        serialized = result.model_dump(by_alias=True, mode='json')
        self.receipts.append({'sequence': len(self.receipts)+1, 'method': 'tools/list',
                              'started_at': at, 'elapsed_s': time.monotonic()-started,
                              'sdk_result_json_sha256': sha(encoded(serialized)),
                              'tools': sorted(t.name for t in result.tools)})
        names={t.name for t in result.tools}
        require(names in [TOOLS, TOOLS | MOTION_TOOLS, TOOLS | MOTION_TOOLS | DISCOVERY_TOOLS, TOOLS | MOTION_TOOLS | DISCOVERY_TOOLS | ROBOT_TOOLS, TOOLS | MOTION_TOOLS | DISCOVERY_TOOLS | ROBOT_TOOLS | CONNECTION_TOOLS], 'Unexpected local MCP tool surface')
        require(all(t.annotations is not None and t.annotations.read_only_hint is True and
                    t.annotations.destructive_hint is False and t.annotations.open_world_hint is False
                    for t in result.tools), 'MCP tools must declare the bounded read-only catalog policy')
        self.discovery_tools = names

    async def tool(self, name, arguments):
        started = time.monotonic()
        at = datetime.now(timezone.utc).isoformat()
        result = await self.client.call_tool(name, arguments)
        serialized = result.model_dump(by_alias=True, mode='json')
        receipt = {'sequence': len(self.receipts)+1, 'method': 'tools/call', 'name': name,
                   'arguments': arguments, 'started_at': at, 'elapsed_s': time.monotonic()-started,
                   'sdk_result_json_sha256': sha(encoded(serialized)), 'is_error': bool(result.is_error)}
        self.receipts.append(receipt)
        require(not result.is_error, 'MCP rejected tool call: ' + name)
        value = result.structured_content
        if value is None:
            require(len(result.content) == 1 and getattr(result.content[0], 'type', None) == 'text', 'Unexpected MCP tool content')
            value = json.loads(result.content[0].text)
        require(isinstance(value, dict), 'MCP tool must return a JSON object')
        receipt['structured_response_sha256'] = sha(encoded(value))
        return value

    async def resource(self, uri, mime, *, maximum_bytes):
        started = time.monotonic()
        at = datetime.now(timezone.utc).isoformat()
        result = await self.client.read_resource(uri)
        require(len(result.contents) == 1, 'Expected exactly one resource content item')
        content = result.contents[0]
        require(content.mime_type == mime, 'Resource MIME type does not match its declared use')
        require(str(content.uri) == uri, 'Resource URI differs from request')
        if hasattr(content, 'blob'):
            try:
                data = base64.b64decode(content.blob, validate=True)
            except (ValueError, TypeError) as exc:
                raise DemoError('Invalid MCP binary resource encoding') from exc
            kind = 'blob'
        else:
            require(hasattr(content, 'text'), 'Resource is neither text nor a binary blob')
            data = content.text.encode('utf-8')
            kind = 'text'
        require(len(data) <= maximum_bytes, 'Resource exceeds the scripted download budget')
        self.receipts.append({'sequence': len(self.receipts)+1, 'method': 'resources/read', 'uri': uri,
                              'started_at': at, 'elapsed_s': time.monotonic()-started,
                              'mime_type': mime, 'content_kind': kind, 'decoded_bytes': len(data), 'decoded_sha256': sha(data)})
        return data


def index_unique(rows, key):
    result = {}
    for row in rows:
        require(isinstance(row, dict) and key in row, 'Malformed declared artifact record')
        value = identifier(row[key])
        require(value not in result, 'Duplicate declared artifact identifier')
        result[value] = row
    return result


def expected_states(document, timestamp_s):
    evidence = {e['id']: e for e in document['evidence']}
    states = {v['id']: {'value': 'unknown', 'status': 'unknown', 'evidence_ids': []} for v in document['variables']}
    for assertion in document['assertions']:
        require(assertion['evidence_id'] in evidence and assertion['variable_id'] in states, 'Unlinked state assertion')
        if abs(evidence[assertion['evidence_id']]['timestamp_s']-timestamp_s) <= 1e-7:
            states[assertion['variable_id']] = {'value': assertion['value'], 'status': assertion['status'],
                                                 'evidence_ids': [assertion['evidence_id']]}
    return states


def verify_state(document, answer, timestamp_s, *, gap=False):
    require(answer.get('timestamp_s') == timestamp_s, 'State response has the wrong source time')
    expected = expected_states(document, timestamp_s)
    require(answer.get('states') == expected, 'State response differs from exact cited observations')
    poses = [m for m in document['motion'] if abs(m['timestamp_s']-timestamp_s) <= 1e-7 and
             m['status'] in SUPPORTED and m['camera_from_native_mesh'] is not None]
    require(answer.get('conditional_poses') == poses, 'State response invents or changes conditional poses')
    if gap:
        require(all(s['value'] == 'unknown' for s in expected.values()) and not poses,
                'Chosen gap unexpectedly contains an exact observation')


async def collect_twin(client, *, twin_id=DEFAULT_TWIN, query='', max_evidence=6):
    """Fetch bounded declared resources only; return bytes for the portable builder."""
    identifier(twin_id)
    require(isinstance(max_evidence, int) and not isinstance(max_evidence, bool) and 1 <= max_evidence <= 12,
            'Choose between one and twelve evidence frames')
    mcp = RecordedMCP(client)
    await mcp.discover()
    found = await mcp.tool('search_twins', {'query': query, 'limit': 25})
    if not any(t['twin_id'] == twin_id for t in found.get('twins', [])) and DISCOVERY_TOOLS <= mcp.discovery_tools:
        # The automatic library hides historic substitute-model studies by
        # default. An explicitly requested old ID may still be inspected through
        # a bounded read-only variant search; never guess a path or bypass MCP.
        offset = 0
        for _ in range(4):
            legacy = await mcp.tool('search_twins', {'query': query, 'limit': 25, 'offset': offset,
                                                    'group_by': 'variant', 'include_legacy': True})
            found['twins'].extend(legacy.get('twins', []))
            if any(t['twin_id'] == twin_id for t in found['twins']) or not legacy.get('has_more'):
                break
            next_offset = legacy.get('next_offset')
            require(type(next_offset) is int and offset < next_offset <= 100, 'Invalid bounded discovery pagination')
            offset = next_offset
    require(any(t['twin_id'] == twin_id for t in found.get('twins', [])), 'Requested twin was not discovered; choose a matching query or twin ID')
    description = await mcp.tool('describe_twin', {'twin_id': twin_id})
    require(description.get('twin_id') == twin_id, 'Description has the wrong twin ID')
    require(description.get('policy', {}).get('simulation_ready') is False and
            description['policy'].get('physical_accuracy_verified') is False, 'Catalog must retain unverified physics policy')
    original_hash = hash_value(description.get('program_sha256'))
    program_bytes = await mcp.resource(f'twin://{twin_id}/program', 'application/json', maximum_bytes=8*1024*1024)
    document = json.loads(program_bytes)
    require(document.get('format') == 'interaction-twin-evidence-procedure', 'Unexpected interaction program format')
    require(document.get('runtime_policy', {}).get('simulation_ready') is False and
            document['runtime_policy'].get('pose_interpolation') is False, 'Program cannot silently enable simulation or pose interpolation')
    evidence = index_unique(document['evidence'], 'id')
    require(len(evidence) >= 2, 'Two distinct source observations are needed to demonstrate unknown gaps')
    assets = index_unique(document['assets'], 'track_id')
    descriptors = index_unique(description['assets'], 'track_id')
    require(set(assets) == set(descriptors) and len(assets) <= 8, 'Description/program assets differ or exceed download budget')
    procedure = await mcp.tool('get_procedure', {'twin_id': twin_id})
    require(procedure.get('twin_id') == twin_id and procedure.get('program_sha256') == original_hash and
            procedure.get('steps') == document['steps'], 'Procedure differs from the cited program')
    require(all(e['id'] in evidence and e == evidence[e['id']] for e in procedure.get('evidence', [])), 'Procedure contains unlinked source evidence')
    require(bool(document['steps']), 'A descriptive procedure is required for this demo')
    first_step = document['steps'][0]
    one_step = await mcp.tool('get_procedure', {'twin_id': twin_id, 'step_id': first_step['id']})
    require(one_step.get('twin_id') == twin_id and one_step.get('program_sha256') == original_hash and
            one_step.get('steps') == [first_step], 'Selected procedure step differs from the program')
    ordered = sorted(evidence.values(), key=lambda e: e['timestamp_s'])
    require(all(math.isfinite(e['timestamp_s']) and e['timestamp_s'] >= 0 for e in ordered), 'Invalid source evidence time')
    require(all(b['timestamp_s'] > a['timestamp_s'] for a, b in zip(ordered, ordered[1:])), 'Source evidence times must be unique and ordered')
    exact_evidence_id = document['assertions'][0]['evidence_id'] if document['assertions'] else ordered[0]['id']
    exact_time = evidence[exact_evidence_id]['timestamp_s']
    # The midpoint is explicitly an unknown inspection time, never a simulated action.
    gap_time = (ordered[0]['timestamp_s']+ordered[1]['timestamp_s'])/2
    require(min(abs(e['timestamp_s']-gap_time) for e in ordered) > 1e-7, 'No usable unobserved gap in this program')
    exact = await mcp.tool('query_state', {'twin_id': twin_id, 'timestamp_s': exact_time})
    gap = await mcp.tool('query_state', {'twin_id': twin_id, 'timestamp_s': gap_time})
    for result, timestamp, is_gap in [(exact, exact_time, False), (gap, gap_time, True)]:
        require(result.get('twin_id') == twin_id and result.get('program_sha256') == original_hash,
                'State response cites a different twin or program')
        verify_state(document, result, timestamp, gap=is_gap)
    asset_bytes = {}
    for track_id, declared in assets.items():
        descriptor = await mcp.tool('get_asset', {'twin_id': twin_id, 'track_id': track_id})
        uri = f'twin://{twin_id}/assets/{track_id}'
        require(descriptor.get('track_id') == track_id and descriptor.get('uri') == uri and
                descriptor.get('sha256') == declared['sha256'] == descriptors[track_id]['sha256'], 'Asset descriptor or hash does not match the program')
        data = await mcp.resource(uri, 'model/gltf-binary', maximum_bytes=16*1024*1024)
        require(sha(data) == hash_value(declared['sha256']) and data[:4] == b'glTF', 'Generated GLB hash or header mismatch')
        asset_bytes[track_id] = data
    selected_ids = []
    for eid in [first_step['evidence_ids'][0], first_step['evidence_ids'][-1], exact_evidence_id]:
        require(eid in evidence, 'Procedure references an unknown evidence ID')
        if eid not in selected_ids: selected_ids.append(eid)
    for track_id in assets:
        reference = next((m for m in document['motion'] if m['track_id'] == track_id and
                          m['status'] == 'reference_candidate_unverified_geometry_and_pose'), None)
        if reference:
            frame = next((e for e in ordered if e['source_index'] == reference['source_sample_index']), None)
            require(frame is not None, 'Reference pose has no declared source observation')
            if frame['id'] not in selected_ids: selected_ids.append(frame['id'])
    evidence_bytes = {}
    for eid in selected_ids[:max_evidence]:
        descriptor = await mcp.tool('get_evidence', {'twin_id': twin_id, 'evidence_id': eid})
        uri = f'twin://{twin_id}/evidence/{eid}'
        require(descriptor.get('twin_id') == twin_id and descriptor.get('evidence') == evidence[eid] and
                descriptor.get('frame_uri') == uri, 'Evidence descriptor differs from the cited source')
        data = await mcp.resource(uri, 'image/jpeg', maximum_bytes=8*1024*1024)
        require(sha(data) == hash_value(evidence[eid]['frame_sha256']) and data[:2] == b'\xff\xd8', 'Source JPEG hash or header mismatch')
        evidence_bytes[eid] = data
    require(sum(map(len, asset_bytes.values()))+sum(map(len, evidence_bytes.values()))+len(program_bytes) <= 64*1024*1024,
            'Portable resource download budget exceeded')
    simulation = await mcp.tool('get_simulation_manifest', {'twin_id': twin_id})
    require(simulation.get('twin_id') == twin_id and simulation.get('program_sha256') == original_hash and
            simulation.get('conditional_motion_samples') == document['motion'] and
            simulation.get('demonstration_steps') == document['steps'] and
            simulation.get('policy', {}).get('simulation_ready') is False,
            'Simulation handoff changed the program or claims physics readiness')
    return {'twin_id': twin_id, 'title': description['title'], 'document': document, 'program_bytes': program_bytes,
            'source_program_sha256': original_hash, 'program_bytes_kind': 'mcp_json_serialization',
            'asset_bytes': asset_bytes, 'evidence_bytes': evidence_bytes,
            'receipt': {'format': 'interaction-twin-scripted-mcp-consumer', 'version': '0.1',
                        'created_at': datetime.now(timezone.utc).isoformat(), 'twin_id': twin_id,
                        'planning': 'deterministic_scripted_workflow_no_LLM', 'transport': 'local_subprocess_stdio',
                        'model_inference': False, 'network_or_cluster_contact': False, 'simulation_executed': False,
                        'source_program_sha256': original_hash, 'program_resource_sha256': sha(program_bytes),
                        'original_program_file_bytes_included': False,
                        'source_video_included': False, 'masks_included': False,
                        'assets_verified': {k: sha(v) for k, v in asset_bytes.items()},
                        'evidence_frames_verified': {k: sha(v) for k, v in evidence_bytes.items()},
                        'exact_state_response': exact, 'gap_state_response': gap,
                        'state_checks_meaning': 'Matches saved model proposals only; not semantic correctness or physical success',
                        'calls': mcp.receipts}}


def write_package(collected, output):
    """Only the collected in-memory MCP bytes are passed to the package builder."""
    from portable_twin import build_package
    output = Path(output)
    require(not output.exists(), 'Choose a fresh output directory; existing demos are preserved')
    receipt_path = Path(str(output)+'-scripted-agent-receipt.json')
    require(not receipt_path.exists(), 'Choose a fresh receipt path; existing demos are preserved')
    receipt_bytes = (json.dumps(collected['receipt'], indent=2, allow_nan=False)+'\n').encode('utf-8')
    result = build_package(output, document=collected['document'], program_bytes=collected['program_bytes'],
                           asset_bytes=collected['asset_bytes'], evidence_bytes=collected['evidence_bytes'],
                           source_program_sha256=collected['source_program_sha256'],
                           program_bytes_kind=collected['program_bytes_kind'], twin_id=collected['twin_id'],
                           title=collected['title'], provenance={'consumer': 'scripted_official_MCP_client',
                                                               'model_inference': False, 'network': False,
                                                               'scripted_receipt_sha256': sha(receipt_bytes)})
    # Keep the execution receipt alongside the package. Adding an undeclared
    # file inside a completed package would invalidate its directory manifest.
    with receipt_path.open('xb') as stream:
        stream.write(receipt_bytes)
    return result | {'scripted_receipt': str(receipt_path.absolute())}


async def collect_reconstruction_inspection(client, *, twin_id, version_id, query=''):
    """Use actual MCP responses to bind a diagnostic part to its source task.

    The resulting task is for visual inspection/review, never robot commands.
    No source path embedded in model/catalog text is opened by this client.
    """
    identifier(version_id)
    collected = await collect_twin(client, twin_id=twin_id, query=query, max_evidence=3)
    mcp = RecordedMCP(client)
    mcp.receipts = collected['receipt']['calls']
    listing = await mcp.tool('list_reconstructions', {'twin_id': twin_id, 'selection': 'all'})
    require(any(r['version_id'] == version_id for r in listing['versions']), 'Version not discovered for this source program')
    descriptor = await mcp.tool('get_reconstruction', {'twin_id': twin_id, 'version_id': version_id})
    require(descriptor['interaction_sha256'] == collected['source_program_sha256'] and
            descriptor['source_video_sha256'] == collected['document']['source']['sha256'],
            'Reconstruction source/program identity differs')
    require(descriptor['gauge']['metric'] is False and descriptor['physical_accuracy_verified'] is False and
            descriptor['simulation_ready'] is False, 'Reconstruction cannot imply metric or physical readiness')
    descriptor_bytes = await mcp.resource(f'twin://{twin_id}/reconstructions/{version_id}', 'application/json', maximum_bytes=2*1024*1024)
    require(json.loads(descriptor_bytes) == descriptor, 'Tool/resource reconstruction descriptors differ')
    quality = await mcp.tool('get_reconstruction_report', {'twin_id': twin_id, 'version_id': version_id})
    require(quality['program_sha256'] == collected['source_program_sha256'] and quality['policy']['simulation_ready'] is False,
            'Quality report identity or physics policy differs')
    downloads = {}
    for artifact_id in ['glb', 'quality', 'source-image-0', 'source-mask-0']:
        require(artifact_id in descriptor['artifacts'], 'Required reconstruction inspection evidence is absent')
        artifact = descriptor['artifacts'][artifact_id]
        expected_uri = f'twin://{twin_id}/reconstructions/{version_id}/artifacts/{artifact_id}'
        require(artifact['uri'] == expected_uri, 'Unexpected reconstruction resource URI')
        data = await mcp.resource(expected_uri, artifact['resource_mime_type'], maximum_bytes=16*1024*1024)
        require(sha(data) == hash_value(artifact['sha256']), 'Reconstruction artifact hash differs from descriptor')
        downloads[artifact_id] = data
    require(downloads['glb'][:4] == b'glTF' and downloads['source-mask-0'][:8] == b'\x89PNG\r\n\x1a\n',
            'Reconstruction asset or source mask header differs')
    require(json.loads(downloads['quality']) == quality['report'], 'Quality resource differs from saved report tool output')
    source = descriptor['source_evidence'][0]
    require(sha(downloads['source-image-0']) == source['image_sha256'] and sha(downloads['source-mask-0']) == source['mask_sha256'],
            'Version source pixels or mask identity differs')
    source_link = None
    if source.get('program_evidence_id'):
        source_link = await mcp.tool('get_evidence', {'twin_id': twin_id, 'evidence_id': source['program_evidence_id']})
        require(source_link['evidence']['frame_sha256'] == source['image_sha256'] and
                source_link['evidence']['timestamp_s'] == source['timestamp_s'], 'Reconstruction source is not exact program evidence')
    parent = descriptor.get('parent_track_id') or descriptor['component_id']
    related = [s for s in collected['document']['steps'] if parent in s['participant_track_ids']]
    require(related, 'No source-backed procedure involves this component or its proposed parent')
    procedure = await mcp.tool('get_procedure', {'twin_id': twin_id, 'step_id': related[0]['id']})
    require(procedure['steps'] == [related[0]], 'Related source procedure changed')
    handoff = await mcp.tool('get_simulation_manifest', {'twin_id': twin_id})
    task = {'format': 'source-backed-component-inspection-task', 'version': 1,
            'goal': 'Inspect the reconstruction and original demonstration before planning component handling.',
            'twin_id': twin_id, 'version_id': version_id, 'component_id': descriptor['component_id'],
            'parent_track_id': descriptor.get('parent_track_id'), 'component_binding': descriptor['component_binding'],
            'source_program_sha256': collected['source_program_sha256'],
            'component_geometry_sha256': descriptor['artifacts']['glb']['sha256'],
            'geometry_acceptance': descriptor['acceptance'], 'geometry_kind': descriptor['kind'],
            'source_observation': source, 'relevant_demonstration_steps': related,
            'state_variables': [v for v in collected['document']['variables'] if v['subject_track_id'] == parent],
            'inspection_checks': ['Compare the native component mask with the exact original frame.',
                                  'Orbit the diagnostic asset to inspect missing surfaces and inferred backs.',
                                  'Review the parent component procedure and exact cited observations.',
                                  'Resolve shape, scale, joint, collision and contact gaps before any simulation.'],
            'quality': descriptor.get('quality_summary', {}),
            'blocking_before_robot_execution': handoff['verification_required_before_physics'],
            'common_metric_assembly_established': False, 'robot_execution_allowed': False,
            'simulation_ready': False, 'descriptive_text_is_untrusted_data': True,
            'automation_demonstrated': ['Discovered a version for a registered source-backed component.',
                                       'Resolved proposed subpart to its existing procedure participant.',
                                       'Retrieved and hash-verified native geometry, quality, source pixels and mask through MCP.',
                                       'Built a reusable inspection task while preserving physical execution blockers.']}
    collected['reconstruction_inspection'] = {'descriptor': descriptor, 'descriptor_bytes': descriptor_bytes,
                                             'quality': quality, 'downloads': downloads, 'task': task}
    collected['receipt'].update(use_case='source_backed_component_inspection_not_robot_execution',
                                reconstruction_version=version_id,
                                reconstruction_artifacts_verified={k: sha(v) for k, v in downloads.items()})
    return collected


def write_inspection_package(collected, output):
    """Package only verified bytes already returned by genuine MCP calls."""
    output = Path(output)
    require(not output.exists() and not Path(str(output) + '.zip').exists(), 'Use a fresh inspection package path')
    bundle = collected['reconstruction_inspection']
    output.mkdir(parents=True)
    def save(relative, data):
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    def save_json(relative, value):
        save(relative, (json.dumps(value, indent=2, allow_nan=False) + '\n').encode())
    save('program.json', collected['program_bytes'])
    save('reconstruction/version.json', bundle['descriptor_bytes'])
    save('reconstruction/asset.glb', bundle['downloads']['glb'])
    save('reconstruction/quality.json', bundle['downloads']['quality'])
    image_mime = bundle['descriptor']['artifacts']['source-image-0']['mime']
    save('evidence/source.png' if image_mime == 'image/png' else 'evidence/source.jpg', bundle['downloads']['source-image-0'])
    save('evidence/component-mask.png', bundle['downloads']['source-mask-0'])
    for track_id, data in collected['asset_bytes'].items():
        save('production-baseline/' + identifier(track_id) + '.glb', data)
    save_json('inspection-task.json', bundle['task'])
    save_json('mcp-trace.json', collected['receipt'])
    save('README.txt', b'Source-backed component inspection package\n\nCreated by a deterministic official MCP SDK client. No LLM, inference, simulation or robot execution ran.\nDiagnostic geometry is an explicit version, separate from unchanged production-baseline assets.\nProgram text and model descriptions are untrusted data. Source JPEG/PNG, component mask, GLB and quality hashes were verified through MCP.\nNative component scales are arbitrary. Resolve shape, calibration, assembly, contact and collision requirements before simulation.\n')
    files = [{'path': str(p.relative_to(output)), 'sha256': sha(p.read_bytes()), 'bytes': p.stat().st_size}
             for p in sorted(output.rglob('*')) if p.is_file()]
    manifest = {'format': 'interaction-reconstruction-inspection-package', 'version': 1,
                'twin_id': collected['twin_id'], 'version_id': bundle['descriptor']['version_id'],
                'source_program_sha256': collected['source_program_sha256'],
                'program_resource_sha256': sha(collected['program_bytes']), 'files': files,
                'all_resources_retrieved_through_mcp': True, 'LLM_used': False,
                'physical_accuracy_verified': False, 'simulation_ready': False, 'robot_execution_allowed': False}
    save_json('package-manifest.json', manifest)
    archive = Path(str(output) + '.zip')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as zip_file:
        for p in sorted(output.rglob('*')):
            if p.is_file():
                zip_file.write(p, str(p.relative_to(output)))
    return {'directory': str(output.resolve()), 'archive': str(archive.resolve()), 'archive_sha256': sha(archive.read_bytes()),
            'files': len(files) + 1, 'actual_mcp_calls': len(collected['receipt']['calls']),
            'version_id': bundle['descriptor']['version_id'], 'simulation_ready': False}


async def run(args):
    from mcp import Client, StdioServerParameters
    params = StdioServerParameters(command=str(ROOT/'.venv-mcp/bin/python'),
                                   args=[str(ROOT/'scripts/serve_twin_mcp.py')], cwd=ROOT)
    async with Client(params, read_timeout_seconds=90) as client:
        if args.reconstruction_version:
            collected = await collect_reconstruction_inspection(client, twin_id=args.twin_id,
                         version_id=args.reconstruction_version, query=args.query)
        else:
            collected = await collect_twin(client, twin_id=args.twin_id, query=args.query, max_evidence=args.max_evidence)
        collected['receipt']['negotiated_protocol'] = client.session.protocol_version
    output = args.output or ROOT/'data'/('demo-agent-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
    result = write_inspection_package(collected, output) if args.reconstruction_version else write_package(collected, output)
    print(json.dumps({'status': 'complete', 'twin_id': collected['twin_id'], 'output': str(Path(output).resolve()),
                      'assets': len(collected['asset_bytes']), 'evidence_frames': len(collected['evidence_bytes']),
                      'actual_mcp_calls': len(collected['receipt']['calls']), 'LLM_used': False, 'simulation_ready': False,
                      'package_result': result}, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--twin-id', default=DEFAULT_TWIN)
    parser.add_argument('--query', default='')
    parser.add_argument('--max-evidence', type=int, default=6)
    parser.add_argument('--output', type=Path, help='Fresh output directory; default is data/demo-agent-TIMESTAMP-ID')
    parser.add_argument('--reconstruction-version', help='Explicit diagnostic version ID; build a source-backed inspection task package through MCP.')
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except (DemoError, ValueError, OSError) as exc:
        print(json.dumps({'status': 'failed', 'reason': str(exc), 'LLM_used': False}), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
